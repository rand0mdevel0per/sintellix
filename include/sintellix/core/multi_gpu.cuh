#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <mutex>

namespace sintellix {

/**
 * @brief Multi-GPU Manager with Unified Memory Support
 *
 * Manages multiple GPUs using CUDA Unified Memory for automatic
 * data migration between devices. Simplifies multi-GPU programming
 * by letting CUDA handle data transfers automatically.
 */
class MultiGPUManager {
public:
    /**
     * @brief Initialize multi-GPU manager
     * @param device_ids Vector of GPU device IDs to use (empty = use all available)
     */
    explicit MultiGPUManager(const std::vector<int>& device_ids = {});
    ~MultiGPUManager();

    /**
     * @brief Get number of available GPUs
     */
    int get_device_count() const { return device_count_; }

    /**
     * @brief Get list of active device IDs
     */
    const std::vector<int>& get_device_ids() const { return device_ids_; }

    /**
     * @brief Allocate unified memory accessible from all GPUs
     * @param size Size in bytes
     * @return Pointer to unified memory
     */
    void* allocate_unified(size_t size);

    /**
     * @brief Free unified memory
     * @param ptr Pointer to unified memory
     */
    void free_unified(void* ptr);

    /**
     * @brief Set preferred location for unified memory
     * @param ptr Pointer to unified memory
     * @param device_id Device ID (-1 for CPU)
     */
    void set_preferred_location(void* ptr, int device_id, size_t size);

    /**
     * @brief Prefetch data to specific device
     * @param ptr Pointer to unified memory
     * @param size Size in bytes
     * @param device_id Target device ID
     */
    void prefetch_to_device(void* ptr, size_t size, int device_id);

    /**
     * @brief Enable peer access between GPUs
     */
    bool enable_peer_access();

    /**
     * @brief Synchronize all devices
     */
    void synchronize_all();

    /**
     * @brief Get CUDA stream for specific device
     * @param device_id Device ID
     * @return CUDA stream
     */
    cudaStream_t get_stream(int device_id);

private:
    int device_count_;
    std::vector<int> device_ids_;
    std::vector<cudaStream_t> streams_;
    std::mutex mutex_;

    void initialize_devices();
    void cleanup_devices();
};

/**
 * @brief RAII wrapper for setting CUDA device
 */
class DeviceGuard {
public:
    explicit DeviceGuard(int device_id);
    ~DeviceGuard();

private:
    int previous_device_;
};

/**
 * @brief Distribute neurons across multiple GPUs
 *
 * @param total_neurons Total number of neurons
 * @param num_devices Number of GPUs
 * @return Vector of (start_idx, count) pairs for each device
 */
std::vector<std::pair<int, int>> distribute_neurons(
    int total_neurons,
    int num_devices
);

} // namespace sintellix
