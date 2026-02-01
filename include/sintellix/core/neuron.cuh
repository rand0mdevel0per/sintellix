#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "sintellix/core/config.hpp"

namespace sintellix {

/**
 * KFE (Knowledge Feature Encoding) short-term memory slot
 */
struct KFE_STM_Slot {
    double* kfe_matrix;      // dim × dim matrix (dynamically allocated)
    char key[256];           // KFE identifier
    uint64_t access_count;   // Access frequency
    uint64_t last_access;    // Last access timestamp
    bool is_valid;           // Slot validity
};

/**
 * Neuron class with dynamic memory allocation
 * Supports configurable dimensions and module switches
 */
class Neuron {
public:
    /**
     * Constructor
     * @param config Neuron configuration
     * @param neuron_id Unique neuron identifier
     * @param grid_x Grid X coordinate
     * @param grid_y Grid Y coordinate
     * @param grid_z Grid Z coordinate
     */
    Neuron(const NeuronConfig& config, int neuron_id, int grid_x, int grid_y, int grid_z);

    /**
     * Destructor - frees all dynamically allocated memory
     */
    ~Neuron();

    // Disable copy constructor and assignment
    Neuron(const Neuron&) = delete;
    Neuron& operator=(const Neuron&) = delete;

    /**
     * Initialize neuron (allocate GPU memory)
     */
    bool initialize();

    /**
     * Forward pass with fused kernel
     */
    void forward(const double* input, double* output, cudaStream_t stream = 0);

    /**
     * Backward pass for training
     */
    void backward(const double* grad_output, double* grad_input, cudaStream_t stream = 0);

    /**
     * Update parameters with Adam optimizer
     */
    void update_parameters(float learning_rate, cudaStream_t stream = 0);

private:
    // Configuration
    NeuronConfig config_;
    int neuron_id_;
    int grid_x_, grid_y_, grid_z_;
    uint32_t dim_;           // Neuron dimension
    uint32_t num_heads_;     // Number of attention heads
    uint32_t head_dim_;      // Dimension per head (dim / num_heads)

    // Core state matrices (all dynamically allocated)
    double* P_Matrix_;       // Current state [dim × dim]
    double* P_stable_;       // Stable state [dim × dim]
    double* W_predict_;      // Prediction weights [dim × dim]
    double* M_KFE_;          // KFE matrix [dim × dim]
    double* Deviation_;      // Deviation matrix [dim × dim]
    double* PS_aggregate_;   // Aggregated state [dim × dim]

    // Temporal history (8 frames)
    double** P_history_;     // Array of pointers [8][dim × dim]
    int history_index_;      // Current history index

    // Port transformation matrices
    double** port_in_matrices_;   // 4 input ports [4][dim × dim]
    double** port_out_matrices_;  // 4 output ports [4][dim × dim]

    // Convolution kernels (4 ports × 8 kernels)
    double*** conv_kernels_;      // [4][8][8 × 8]

    // Multi-head attention weights (if enabled)
    double** Q_heads_;       // Query weights [num_heads][head_dim × head_dim]
    double** K_heads_;       // Key weights [num_heads][head_dim × head_dim]
    double** V_heads_;       // Value weights [num_heads][head_dim × head_dim]
    double* O_weight_;       // Output projection [dim × dim]

    // SSM state
    double* h_state_;        // Hidden state [dim]
    double* A_ssm_;          // SSM A matrix [dim × dim]
    double* B_ssm_;          // SSM B matrix [dim × dim]
    double* C_ssm_;          // SSM C matrix [dim × dim]

    // RWKV state
    double* wkv_state_;      // WKV state [dim]
    double* R_rwkv_;         // R matrix [dim × dim]
    double* K_rwkv_;         // K matrix [dim × dim]
    double* V_rwkv_;         // V matrix [dim × dim]

    // Adam optimizer state
    double* adam_m_;         // First moment [dim × dim]
    double* adam_v_;         // Second moment [dim × dim]
    uint64_t adam_step_;     // Optimizer step count

    // KFE local storage (16 slots)
    KFE_STM_Slot* kfe_local_;     // [16] slots
    static constexpr int KFE_LOCAL_SIZE = 16;

    // DDPM noise schedule
    double* noise_schedule_; // [16] noise levels
    static constexpr int DDPM_STEPS = 16;

    // KV-Cache (if enabled)
    double* kv_cache_keys_;  // [kv_cache_size × dim]
    double* kv_cache_values_;// [kv_cache_size × dim]
    int kv_cache_size_;
    int kv_cache_pos_;

    // Noise filter state (if enabled)
    double* ema_mean_;       // EMA mean [dim]
    double* ema_std_;        // EMA std [dim]
    float ema_alpha_;

    // cuBLAS handle
    cublasHandle_t cublas_handle_;

    // Helper methods
    bool allocate_memory();
    void free_memory();
    void initialize_weights();
};

} // namespace sintellix
