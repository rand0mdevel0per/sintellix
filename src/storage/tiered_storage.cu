#include "sintellix/storage/tiered_storage.cuh"
#include <fstream>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <cstring>

namespace sintellix {

TieredStorageManager::TieredStorageManager(const NeuronConfig& config)
    : config_(config)
    , gpu_cache_size_(config.storage().gpu_cache_size_mb() * 1024 * 1024)
    , ram_cache_size_(config.storage().ram_cache_size_mb() * 1024 * 1024)
    , disk_cache_path_(config.storage().disk_cache_path())
    , eviction_threshold_(config.storage().eviction_threshold())
    , current_gpu_usage_(0)
    , current_ram_usage_(0)
{
    // Create disk cache directory if it doesn't exist
    std::filesystem::create_directories(disk_cache_path_);
}

TieredStorageManager::~TieredStorageManager() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Free all GPU and RAM memory
    for (auto& pair : blocks_) {
        auto& block = pair.second;
        if (block->gpu_ptr) {
            cudaFree(block->gpu_ptr);
        }
        if (block->ram_ptr) {
            free(block->ram_ptr);
        }
    }
}

uint64_t TieredStorageManager::get_timestamp() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

bool TieredStorageManager::store(const std::string& key, const void* data, size_t size, bool is_device_ptr) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if block already exists
    auto it = blocks_.find(key);
    if (it != blocks_.end()) {
        // Update existing block
        auto& block = it->second;
        block->size = size;
        block->access_count++;
        block->last_access_time = get_timestamp();
        block->is_dirty = true;

        // Update data
        if (block->gpu_ptr) {
            if (is_device_ptr) {
                cudaMemcpy(block->gpu_ptr, data, size, cudaMemcpyDeviceToDevice);
            } else {
                cudaMemcpy(block->gpu_ptr, data, size, cudaMemcpyHostToDevice);
            }
        }
        return true;
    }

    // Create new block
    auto block = std::make_shared<DataBlock>();
    block->key = key;
    block->size = size;
    block->access_count = 1;
    block->last_access_time = get_timestamp();
    block->is_dirty = true;
    block->gpu_ptr = nullptr;
    block->ram_ptr = nullptr;

    // Try to allocate on GPU first
    if (current_gpu_usage_ + size <= gpu_cache_size_) {
        cudaMalloc(&block->gpu_ptr, size);
        if (block->gpu_ptr) {
            if (is_device_ptr) {
                cudaMemcpy(block->gpu_ptr, data, size, cudaMemcpyDeviceToDevice);
            } else {
                cudaMemcpy(block->gpu_ptr, data, size, cudaMemcpyHostToDevice);
            }
            block->current_tier = StorageTier::GPU;
            current_gpu_usage_ += size;
        }
    } else {
        // GPU full, evict cold data
        evict_cold_data();

        // Try again
        cudaMalloc(&block->gpu_ptr, size);
        if (block->gpu_ptr) {
            if (is_device_ptr) {
                cudaMemcpy(block->gpu_ptr, data, size, cudaMemcpyDeviceToDevice);
            } else {
                cudaMemcpy(block->gpu_ptr, data, size, cudaMemcpyHostToDevice);
            }
            block->current_tier = StorageTier::GPU;
            current_gpu_usage_ += size;
        } else {
            // Still can't allocate, store in RAM
            block->ram_ptr = malloc(size);
            if (block->ram_ptr) {
                if (is_device_ptr) {
                    cudaMemcpy(block->ram_ptr, data, size, cudaMemcpyDeviceToHost);
                } else {
                    memcpy(block->ram_ptr, data, size);
                }
                block->current_tier = StorageTier::RAM;
                current_ram_usage_ += size;
            } else {
                return false;
            }
        }
    }

    blocks_[key] = block;
    return true;
}

bool TieredStorageManager::load(const std::string& key, void** gpu_ptr) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = blocks_.find(key);
    if (it == blocks_.end()) {
        return false;
    }

    auto& block = it->second;
    block->access_count++;
    block->last_access_time = get_timestamp();

    // If already on GPU, return directly
    if (block->gpu_ptr) {
        *gpu_ptr = block->gpu_ptr;
        return true;
    }

    // Need to promote to GPU
    return promote_to_gpu(key);
}

void* TieredStorageManager::access(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = blocks_.find(key);
    if (it == blocks_.end()) {
        return nullptr;
    }

    auto& block = it->second;
    block->access_count++;
    block->last_access_time = get_timestamp();

    // If on GPU, return directly
    if (block->gpu_ptr) {
        return block->gpu_ptr;
    }

    // Promote to GPU
    if (promote_to_gpu(key)) {
        return block->gpu_ptr;
    }

    return nullptr;
}

void TieredStorageManager::remove(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = blocks_.find(key);
    if (it == blocks_.end()) {
        return;
    }

    auto& block = it->second;

    // Free GPU memory
    if (block->gpu_ptr) {
        cudaFree(block->gpu_ptr);
        current_gpu_usage_ -= block->size;
    }

    // Free RAM memory
    if (block->ram_ptr) {
        free(block->ram_ptr);
        current_ram_usage_ -= block->size;
    }

    // Remove disk file
    if (!block->disk_path.empty()) {
        std::filesystem::remove(block->disk_path);
    }

    blocks_.erase(it);
}

void TieredStorageManager::evict_cold_data() {
    // Find cold data blocks (low access count, old access time)
    std::vector<std::pair<std::string, uint64_t>> candidates;

    for (auto& pair : blocks_) {
        auto& block = pair.second;
        if (block->access_count < eviction_threshold_) {
            // Calculate coldness score (lower is colder)
            uint64_t coldness = block->access_count * 1000000 + block->last_access_time;
            candidates.push_back({pair.first, coldness});
        }
    }

    // Sort by coldness (coldest first)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // Evict coldest blocks until we have enough space
    for (auto& candidate : candidates) {
        auto& block = blocks_[candidate.first];

        if (block->current_tier == StorageTier::GPU) {
            // Evict from GPU to RAM
            if (evict_to_ram(candidate.first)) {
                // Check if we have enough space now
                if (current_gpu_usage_ < gpu_cache_size_ * 0.8) {
                    break;
                }
            }
        } else if (block->current_tier == StorageTier::RAM) {
            // Evict from RAM to Disk
            if (evict_to_disk(candidate.first)) {
                // Check if we have enough space now
                if (current_ram_usage_ < ram_cache_size_ * 0.8) {
                    break;
                }
            }
        }
    }
}

bool TieredStorageManager::evict_to_ram(const std::string& key) {
    auto it = blocks_.find(key);
    if (it == blocks_.end() || !it->second->gpu_ptr) {
        return false;
    }

    auto& block = it->second;

    // Check if RAM has space
    if (current_ram_usage_ + block->size > ram_cache_size_) {
        // Evict some RAM data to disk first
        evict_cold_data();
    }

    // Allocate RAM
    block->ram_ptr = malloc(block->size);
    if (!block->ram_ptr) {
        return false;
    }

    // Copy from GPU to RAM
    cudaMemcpy(block->ram_ptr, block->gpu_ptr, block->size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(block->gpu_ptr);
    block->gpu_ptr = nullptr;
    current_gpu_usage_ -= block->size;

    // Update tier
    block->current_tier = StorageTier::RAM;
    current_ram_usage_ += block->size;

    return true;
}

bool TieredStorageManager::evict_to_disk(const std::string& key) {
    auto it = blocks_.find(key);
    if (it == blocks_.end() || !it->second->ram_ptr) {
        return false;
    }

    auto& block = it->second;

    // Create disk file path
    block->disk_path = disk_cache_path_ + "/" + key + ".bin";

    // Write to disk
    std::ofstream file(block->disk_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.write(reinterpret_cast<const char*>(block->ram_ptr), block->size);
    file.close();

    // Free RAM memory
    free(block->ram_ptr);
    block->ram_ptr = nullptr;
    current_ram_usage_ -= block->size;

    // Update tier
    block->current_tier = StorageTier::DISK;

    return true;
}

bool TieredStorageManager::promote_to_gpu(const std::string& key) {
    auto it = blocks_.find(key);
    if (it == blocks_.end()) {
        return false;
    }

    auto& block = it->second;

    // If already on GPU, nothing to do
    if (block->gpu_ptr) {
        return true;
    }

    // Check if GPU has space
    if (current_gpu_usage_ + block->size > gpu_cache_size_) {
        // Evict cold data from GPU
        evict_cold_data();
    }

    // Allocate GPU memory
    cudaMalloc(&block->gpu_ptr, block->size);
    if (!block->gpu_ptr) {
        return false;
    }

    // Copy data to GPU
    if (block->ram_ptr) {
        // From RAM to GPU
        cudaMemcpy(block->gpu_ptr, block->ram_ptr, block->size, cudaMemcpyHostToDevice);

        // Free RAM
        free(block->ram_ptr);
        block->ram_ptr = nullptr;
        current_ram_usage_ -= block->size;
    } else if (!block->disk_path.empty()) {
        // From Disk to GPU (via RAM)
        if (!promote_to_ram(key)) {
            cudaFree(block->gpu_ptr);
            block->gpu_ptr = nullptr;
            return false;
        }

        // Now copy from RAM to GPU
        cudaMemcpy(block->gpu_ptr, block->ram_ptr, block->size, cudaMemcpyHostToDevice);

        // Free RAM
        free(block->ram_ptr);
        block->ram_ptr = nullptr;
        current_ram_usage_ -= block->size;
    }

    // Update tier
    block->current_tier = StorageTier::GPU;
    current_gpu_usage_ += block->size;

    return true;
}

bool TieredStorageManager::promote_to_ram(const std::string& key) {
    auto it = blocks_.find(key);
    if (it == blocks_.end()) {
        return false;
    }

    auto& block = it->second;

    // If already on RAM or GPU, nothing to do
    if (block->ram_ptr || block->gpu_ptr) {
        return true;
    }

    // Must be on disk
    if (block->disk_path.empty()) {
        return false;
    }

    // Check if RAM has space
    if (current_ram_usage_ + block->size > ram_cache_size_) {
        // Evict cold data from RAM
        evict_cold_data();
    }

    // Allocate RAM
    block->ram_ptr = malloc(block->size);
    if (!block->ram_ptr) {
        return false;
    }

    // Read from disk
    std::ifstream file(block->disk_path, std::ios::binary);
    if (!file.is_open()) {
        free(block->ram_ptr);
        block->ram_ptr = nullptr;
        return false;
    }

    file.read(reinterpret_cast<char*>(block->ram_ptr), block->size);
    file.close();

    // Update tier
    block->current_tier = StorageTier::RAM;
    current_ram_usage_ += block->size;

    // Optionally remove disk file
    // std::filesystem::remove(block->disk_path);
    // block->disk_path.clear();

    return true;
}

TieredStorageManager::Statistics TieredStorageManager::get_statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);

    Statistics stats;
    stats.gpu_usage_bytes = current_gpu_usage_;
    stats.ram_usage_bytes = current_ram_usage_;
    stats.disk_usage_bytes = 0;
    stats.total_blocks = blocks_.size();
    stats.gpu_blocks = 0;
    stats.ram_blocks = 0;
    stats.disk_blocks = 0;

    for (const auto& pair : blocks_) {
        const auto& block = pair.second;
        switch (block->current_tier) {
            case StorageTier::GPU:
                stats.gpu_blocks++;
                break;
            case StorageTier::RAM:
                stats.ram_blocks++;
                break;
            case StorageTier::DISK:
                stats.disk_blocks++;
                stats.disk_usage_bytes += block->size;
                break;
        }
    }

    return stats;
}

} // namespace sintellix
