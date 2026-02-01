#include "sintellix/core/multi_gpu.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace sintellix {

MultiGPUManager::MultiGPUManager(const std::vector<int>& device_ids) {
    // Get total number of available GPUs
    cudaGetDeviceCount(&device_count_);

    if (device_count_ == 0) {
        throw std::runtime_error("No CUDA devices available");
    }

    // Use specified devices or all available devices
    if (device_ids.empty()) {
        // Use all available devices
        device_ids_.resize(device_count_);
        for (int i = 0; i < device_count_; i++) {
            device_ids_[i] = i;
        }
    } else {
        // Validate specified device IDs
        for (int id : device_ids) {
            if (id < 0 || id >= device_count_) {
                throw std::runtime_error("Invalid device ID: " + std::to_string(id));
            }
        }
        device_ids_ = device_ids;
    }

    initialize_devices();
}

MultiGPUManager::~MultiGPUManager() {
    cleanup_devices();
}

void MultiGPUManager::initialize_devices() {
    // Create CUDA stream for each device
    streams_.resize(device_ids_.size());

    for (size_t i = 0; i < device_ids_.size(); i++) {
        DeviceGuard guard(device_ids_[i]);
        cudaStreamCreate(&streams_[i]);
    }

    // Enable peer access between all GPUs
    enable_peer_access();
}

void MultiGPUManager::cleanup_devices() {
    for (size_t i = 0; i < streams_.size(); i++) {
        if (streams_[i]) {
            DeviceGuard guard(device_ids_[i]);
            cudaStreamDestroy(streams_[i]);
        }
    }
    streams_.clear();
}

void* MultiGPUManager::allocate_unified(size_t size) {
    void* ptr = nullptr;

    // Allocate unified memory accessible from all devices
    cudaError_t err = cudaMallocManaged(&ptr, size);

    if (err != cudaSuccess) {
        throw std::runtime_error(
            "Failed to allocate unified memory: " +
            std::string(cudaGetErrorString(err))
        );
    }

    return ptr;
}

void MultiGPUManager::free_unified(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

void MultiGPUManager::set_preferred_location(void* ptr, int device_id, size_t size) {
    if (device_id >= 0) {
        // Set preferred location to specific GPU
        cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device_id);
    } else {
        // Set preferred location to CPU
        cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    }
}

void MultiGPUManager::prefetch_to_device(void* ptr, size_t size, int device_id) {
    DeviceGuard guard(device_id);
    cudaMemPrefetchAsync(ptr, size, device_id);
}

bool MultiGPUManager::enable_peer_access() {
    bool success = true;

    // Enable peer access between all pairs of GPUs
    for (size_t i = 0; i < device_ids_.size(); i++) {
        DeviceGuard guard(device_ids_[i]);

        for (size_t j = 0; j < device_ids_.size(); j++) {
            if (i == j) continue;

            int can_access = 0;
            cudaDeviceCanAccessPeer(&can_access, device_ids_[i], device_ids_[j]);

            if (can_access) {
                cudaError_t err = cudaDeviceEnablePeerAccess(device_ids_[j], 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                    success = false;
                }
            }
        }
    }

    return success;
}

void MultiGPUManager::synchronize_all() {
    for (int device_id : device_ids_) {
        DeviceGuard guard(device_id);
        cudaDeviceSynchronize();
    }
}

cudaStream_t MultiGPUManager::get_stream(int device_id) {
    for (size_t i = 0; i < device_ids_.size(); i++) {
        if (device_ids_[i] == device_id) {
            return streams_[i];
        }
    }
    return nullptr;
}

// DeviceGuard implementation
DeviceGuard::DeviceGuard(int device_id) {
    cudaGetDevice(&previous_device_);
    cudaSetDevice(device_id);
}

DeviceGuard::~DeviceGuard() {
    cudaSetDevice(previous_device_);
}

// Utility function to distribute neurons across devices
std::vector<std::pair<int, int>> distribute_neurons(
    int total_neurons,
    int num_devices
) {
    std::vector<std::pair<int, int>> distribution;

    int neurons_per_device = total_neurons / num_devices;
    int remainder = total_neurons % num_devices;

    int start_idx = 0;
    for (int i = 0; i < num_devices; i++) {
        int count = neurons_per_device + (i < remainder ? 1 : 0);
        distribution.push_back({start_idx, count});
        start_idx += count;
    }

    return distribution;
}

} // namespace sintellix
