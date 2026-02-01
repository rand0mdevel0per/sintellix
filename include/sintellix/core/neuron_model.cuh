#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "sintellix/core/neuron.cuh"
#include "sintellix/core/config.hpp"
#include "sintellix/core/multi_gpu.cuh"
#include "sintellix/storage/tiered_storage.cuh"

namespace sintellix {

/**
 * KFE Manager for managing Knowledge Feature Encoding storage
 */
class KFEManager {
public:
    KFEManager(size_t max_slots = 10000);
    ~KFEManager();

    // Store KFE
    bool store_kfe(const std::string& key, const double* kfe_matrix, size_t dim);

    // Retrieve KFE
    bool retrieve_kfe(const std::string& key, double* kfe_matrix, size_t dim);

    // Check if KFE exists
    bool has_kfe(const std::string& key) const;

    // Get statistics
    size_t get_slot_count() const { return kfe_storage_.size(); }

private:
    struct KFESlot {
        double* gpu_ptr;
        size_t dim;
        uint64_t access_count;
        uint64_t last_access;
    };

    std::unordered_map<std::string, KFESlot> kfe_storage_;
    size_t max_slots_;
};

/**
 * NeuronModel class - manages the entire neuron grid
 */
class NeuronModel {
public:
    /**
     * Constructor
     * @param config Model configuration
     */
    NeuronModel(const NeuronConfig& config);

    /**
     * Destructor
     */
    ~NeuronModel();

    // Disable copy
    NeuronModel(const NeuronModel&) = delete;
    NeuronModel& operator=(const NeuronModel&) = delete;

    /**
     * Initialize model (allocate all neurons)
     */
    bool initialize();

    /**
     * Forward pass through entire model
     * @param input Input data [dim × dim]
     * @param output Output data [dim × dim]
     */
    void forward(const double* input, double* output);

    /**
     * Backward pass for training
     * @param grad_output Gradient of output
     * @param grad_input Gradient of input
     */
    void backward(const double* grad_output, double* grad_input);

    /**
     * Update all parameters
     * @param learning_rate Learning rate
     */
    void update_parameters(float learning_rate);

    /**
     * Fast context replay for state injection
     * @param history Vector of historical inputs
     * @param fast_mode Skip output generation (default: true)
     */
    void replay_context(const std::vector<const double*>& history, bool fast_mode = true);

    /**
     * Save model state to file (Protobuf + zstd)
     * @param path Output file path
     */
    bool save_state(const std::string& path);

    /**
     * Load model state from file
     * @param path Input file path
     */
    bool load_state(const std::string& path);

    /**
     * Get configuration
     */
    const NeuronConfig& get_config() const { return config_; }

    /**
     * Get KFE Manager
     */
    KFEManager& get_kfe_manager() { return kfe_manager_; }

    /**
     * Enable multi-GPU mode
     * @param device_ids Vector of GPU device IDs to use (empty = use all)
     * @return true if successful
     */
    bool enable_multi_gpu(const std::vector<int>& device_ids = {});

    /**
     * Check if multi-GPU mode is enabled
     */
    bool is_multi_gpu_enabled() const { return multi_gpu_enabled_; }

    /**
     * Get number of GPUs being used
     */
    int get_gpu_count() const;

private:
    NeuronConfig config_;
    std::vector<std::unique_ptr<Neuron>> neurons_;
    std::unique_ptr<TieredStorageManager> storage_manager_;
    KFEManager kfe_manager_;

    uint32_t grid_x_, grid_y_, grid_z_;
    uint32_t total_neurons_;
    uint32_t dim_;

    // CUDA streams for parallel execution
    std::vector<cudaStream_t> streams_;

    // Multi-GPU support
    std::unique_ptr<MultiGPUManager> multi_gpu_manager_;
    bool multi_gpu_enabled_;
    std::vector<std::pair<int, int>> neuron_distribution_;  // (start_idx, count) per device

    // Helper methods
    int get_neuron_index(int x, int y, int z) const {
        return x * grid_y_ * grid_z_ + y * grid_z_ + z;
    }

    void global_aggregation();
};

} // namespace sintellix
