#pragma once

#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>
#include "sintellix/core/config.hpp"

namespace sintellix {

/**
 * Storage tier levels
 */
enum class StorageTier {
    GPU,    // Hot data - frequently accessed
    RAM,    // Warm data - moderately accessed
    DISK    // Cold data - rarely accessed
};

/**
 * Data block metadata
 */
struct DataBlock {
    std::string key;              // Unique identifier
    void* gpu_ptr;                // GPU memory pointer (nullptr if not in GPU)
    void* ram_ptr;                // RAM memory pointer (nullptr if not in RAM)
    std::string disk_path;        // Disk file path (empty if not on disk)
    size_t size;                  // Data size in bytes
    StorageTier current_tier;     // Current storage tier
    uint64_t access_count;        // Access frequency
    uint64_t last_access_time;    // Last access timestamp
    bool is_dirty;                // Whether data has been modified
};

/**
 * Tiered storage manager
 * Manages data across GPU → RAM → Disk hierarchy
 */
class TieredStorageManager {
public:
    TieredStorageManager(const NeuronConfig& config);
    ~TieredStorageManager();

    // Disable copy
    TieredStorageManager(const TieredStorageManager&) = delete;
    TieredStorageManager& operator=(const TieredStorageManager&) = delete;

    /**
     * Store data in the tiered storage system
     * @param key Unique identifier
     * @param data Data pointer (host or device)
     * @param size Data size in bytes
     * @param is_device_ptr Whether data is on device
     * @return Success status
     */
    bool store(const std::string& key, const void* data, size_t size, bool is_device_ptr = false);

    /**
     * Load data from tiered storage to GPU
     * @param key Unique identifier
     * @param gpu_ptr Output GPU pointer
     * @return Success status
     */
    bool load(const std::string& key, void** gpu_ptr);

    /**
     * Access data (updates access statistics)
     * @param key Unique identifier
     * @return GPU pointer to data
     */
    void* access(const std::string& key);

    /**
     * Remove data from all tiers
     * @param key Unique identifier
     */
    void remove(const std::string& key);

    /**
     * Evict cold data to lower tiers
     */
    void evict_cold_data();

    /**
     * Get current statistics
     */
    struct Statistics {
        size_t gpu_usage_bytes;
        size_t ram_usage_bytes;
        size_t disk_usage_bytes;
        size_t total_blocks;
        size_t gpu_blocks;
        size_t ram_blocks;
        size_t disk_blocks;
    };
    Statistics get_statistics() const;

private:
    NeuronConfig config_;
    std::unordered_map<std::string, std::shared_ptr<DataBlock>> blocks_;
    mutable std::mutex mutex_;

    size_t gpu_cache_size_;       // GPU cache size in bytes
    size_t ram_cache_size_;       // RAM cache size in bytes
    std::string disk_cache_path_; // Disk cache directory
    uint32_t eviction_threshold_; // Access count threshold

    size_t current_gpu_usage_;
    size_t current_ram_usage_;

    // Helper methods
    bool evict_to_ram(const std::string& key);
    bool evict_to_disk(const std::string& key);
    bool promote_to_gpu(const std::string& key);
    bool promote_to_ram(const std::string& key);
    uint64_t get_timestamp() const;
};

} // namespace sintellix
