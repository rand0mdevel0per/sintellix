#include "sintellix/core/neuron_model.cuh"
#include <cuda_runtime.h>
#include <fstream>
#include <zstd.h>
#include "model_state.pb.h"

namespace sintellix {

// ============================================================================
// KFEManager Implementation
// ============================================================================

KFEManager::KFEManager(size_t max_slots)
    : max_slots_(max_slots)
{
}

KFEManager::~KFEManager() {
    // Free all GPU memory
    for (auto& pair : kfe_storage_) {
        if (pair.second.gpu_ptr) {
            cudaFree(pair.second.gpu_ptr);
        }
    }
}

bool KFEManager::store_kfe(const std::string& key, const double* kfe_matrix, size_t dim) {
    // Check if already exists
    auto it = kfe_storage_.find(key);
    if (it != kfe_storage_.end()) {
        // Update existing KFE
        cudaMemcpy(it->second.gpu_ptr, kfe_matrix, dim * dim * sizeof(double), cudaMemcpyDeviceToDevice);
        it->second.access_count++;
        it->second.last_access = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        return true;
    }

    // Check if we have space
    if (kfe_storage_.size() >= max_slots_) {
        // Evict least recently used
        std::string lru_key;
        uint64_t min_access = UINT64_MAX;
        for (const auto& p : kfe_storage_) {
            if (p.second.last_access < min_access) {
                min_access = p.second.last_access;
                lru_key = p.first;
            }
        }

        // Remove LRU
        cudaFree(kfe_storage_[lru_key].gpu_ptr);
        kfe_storage_.erase(lru_key);
    }

    // Create new slot
    KFESlot slot;
    slot.dim = dim;
    slot.access_count = 1;
    slot.last_access = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    cudaMalloc(&slot.gpu_ptr, dim * dim * sizeof(double));
    cudaMemcpy(slot.gpu_ptr, kfe_matrix, dim * dim * sizeof(double), cudaMemcpyDeviceToDevice);

    kfe_storage_[key] = slot;
    return true;
}

bool KFEManager::retrieve_kfe(const std::string& key, double* kfe_matrix, size_t dim) {
    auto it = kfe_storage_.find(key);
    if (it == kfe_storage_.end()) {
        return false;
    }

    cudaMemcpy(kfe_matrix, it->second.gpu_ptr, dim * dim * sizeof(double), cudaMemcpyDeviceToDevice);
    it->second.access_count++;
    it->second.last_access = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    return true;
}

bool KFEManager::has_kfe(const std::string& key) const {
    return kfe_storage_.find(key) != kfe_storage_.end();
}

// ============================================================================
// NeuronModel Implementation
// ============================================================================

NeuronModel::NeuronModel(const NeuronConfig& config)
    : config_(config)
    , kfe_manager_(10000)
    , grid_x_(config.grid_size().x())
    , grid_y_(config.grid_size().y())
    , grid_z_(config.grid_size().z())
    , total_neurons_(grid_x_ * grid_y_ * grid_z_)
    , dim_(config.dim())
{
    // Create tiered storage manager
    storage_manager_ = std::make_unique<TieredStorageManager>(config);

    // Create CUDA streams for parallel execution
    int num_streams = 8;
    streams_.resize(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams_[i]);
    }
}

NeuronModel::~NeuronModel() {
    // Destroy CUDA streams
    for (auto stream : streams_) {
        cudaStreamDestroy(stream);
    }
}

bool NeuronModel::initialize() {
    neurons_.reserve(total_neurons_);

    // Create all neurons
    for (uint32_t x = 0; x < grid_x_; x++) {
        for (uint32_t y = 0; y < grid_y_; y++) {
            for (uint32_t z = 0; z < grid_z_; z++) {
                int neuron_id = get_neuron_index(x, y, z);
                auto neuron = std::make_unique<Neuron>(config_, neuron_id, x, y, z);

                if (!neuron->initialize()) {
                    return false;
                }

                neurons_.push_back(std::move(neuron));
            }
        }
    }

    return true;
}

void NeuronModel::forward(const double* input, double* output) {
    // Parallel forward pass through all neurons
    int stream_idx = 0;

    for (size_t i = 0; i < neurons_.size(); i++) {
        cudaStream_t stream = streams_[stream_idx % streams_.size()];

        // Each neuron processes the input
        neurons_[i]->forward(input, output, stream);

        stream_idx++;
    }

    // Wait for all streams to complete
    for (auto stream : streams_) {
        cudaStreamSynchronize(stream);
    }

    // Global aggregation (if enabled)
    if (config_.modules().enable_global_aggregation()) {
        global_aggregation();
    }
}

void NeuronModel::backward(const double* grad_output, double* grad_input) {
    // Parallel backward pass through all neurons
    int stream_idx = 0;

    for (size_t i = 0; i < neurons_.size(); i++) {
        cudaStream_t stream = streams_[stream_idx % streams_.size()];

        neurons_[i]->backward(grad_output, grad_input, stream);

        stream_idx++;
    }

    // Wait for all streams to complete
    for (auto stream : streams_) {
        cudaStreamSynchronize(stream);
    }
}

void NeuronModel::update_parameters(float learning_rate) {
    // Parallel parameter update for all neurons
    int stream_idx = 0;

    for (size_t i = 0; i < neurons_.size(); i++) {
        cudaStream_t stream = streams_[stream_idx % streams_.size()];

        neurons_[i]->update_parameters(learning_rate, stream);

        stream_idx++;
    }

    // Wait for all streams to complete
    for (auto stream : streams_) {
        cudaStreamSynchronize(stream);
    }
}

void NeuronModel::global_aggregation() {
    // Simplified global aggregation
    // In full implementation, this would use sparse attention across neurons
    // For now, just a placeholder
}

void NeuronModel::replay_context(const std::vector<const double*>& history, bool fast_mode) {
    // Fast context replay for state injection
    // This allows quick restoration of model state from historical inputs

    double* temp_output;
    cudaMalloc(&temp_output, dim_ * dim_ * sizeof(double));

    for (size_t i = 0; i < history.size(); i++) {
        const double* input = history[i];

        if (fast_mode) {
            // Fast mode: Skip output generation, only update internal state
            int stream_idx = 0;

            for (size_t j = 0; j < neurons_.size(); j++) {
                cudaStream_t stream = streams_[stream_idx % streams_.size()];

                // Forward pass (internal state update only)
                neurons_[j]->forward(input, temp_output, stream);

                stream_idx++;
            }

            // Wait for completion
            for (auto stream : streams_) {
                cudaStreamSynchronize(stream);
            }
        } else {
            // Normal mode: Full forward pass
            forward(input, temp_output);
        }

        // Store KFE for this input
        std::string kfe_key = "context_" + std::to_string(i);
        kfe_manager_.store_kfe(kfe_key, temp_output, dim_);
    }

    cudaFree(temp_output);
}

bool NeuronModel::save_state(const std::string& path) {
    // Create ModelState protobuf
    ModelState model_state;

    // Set configuration
    *model_state.mutable_config() = config_;

    // Set metadata
    model_state.set_version("0.1.0");
    model_state.set_save_timestamp(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count());

    // Serialize all neurons (simplified - full implementation would serialize all state)
    // For now, just save basic statistics
    auto* stats = model_state.mutable_stats();
    stats->set_total_steps(0);
    stats->set_total_tokens(0);
    stats->set_average_loss(0.0f);

    // Serialize to string
    std::string serialized;
    if (!model_state.SerializeToString(&serialized)) {
        return false;
    }

    // Compress with zstd
    size_t compressed_bound = ZSTD_compressBound(serialized.size());
    std::vector<char> compressed(compressed_bound);

    size_t compressed_size = ZSTD_compress(
        compressed.data(), compressed_bound,
        serialized.data(), serialized.size(),
        3  // Compression level
    );

    if (ZSTD_isError(compressed_size)) {
        return false;
    }

    // Write to file
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.write(compressed.data(), compressed_size);
    file.close();

    return true;
}

bool NeuronModel::load_state(const std::string& path) {
    // Read compressed file
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> compressed(file_size);
    file.read(compressed.data(), file_size);
    file.close();

    // Decompress
    unsigned long long decompressed_size = ZSTD_getFrameContentSize(compressed.data(), file_size);
    if (decompressed_size == ZSTD_CONTENTSIZE_ERROR || decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
        return false;
    }

    std::vector<char> decompressed(decompressed_size);
    size_t actual_size = ZSTD_decompress(
        decompressed.data(), decompressed_size,
        compressed.data(), file_size
    );

    if (ZSTD_isError(actual_size)) {
        return false;
    }

    // Deserialize protobuf
    ModelState model_state;
    if (!model_state.ParseFromArray(decompressed.data(), actual_size)) {
        return false;
    }

    // Load configuration
    config_ = model_state.config();

    // Reinitialize model with loaded config
    // (Full implementation would restore all neuron states)

    return true;
}

} // namespace sintellix
