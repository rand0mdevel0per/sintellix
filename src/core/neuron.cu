#include "sintellix/core/neuron.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstring>
#include <cmath>

namespace sintellix {

Neuron::Neuron(const NeuronConfig& config, int neuron_id, int grid_x, int grid_y, int grid_z)
    : config_(config)
    , neuron_id_(neuron_id)
    , grid_x_(grid_x)
    , grid_y_(grid_y)
    , grid_z_(grid_z)
    , dim_(config.dim())
    , num_heads_(config.num_heads())
    , head_dim_(config.dim() / config.num_heads())
    , history_index_(0)
    , adam_step_(0)
    , kv_cache_pos_(0)
    , ema_alpha_(config.noise_filter().ema_alpha())
{
    // Initialize all pointers to nullptr
    P_Matrix_ = nullptr;
    P_stable_ = nullptr;
    W_predict_ = nullptr;
    M_KFE_ = nullptr;
    Deviation_ = nullptr;
    PS_aggregate_ = nullptr;

    P_history_ = nullptr;
    port_in_matrices_ = nullptr;
    port_out_matrices_ = nullptr;
    conv_kernels_ = nullptr;

    Q_heads_ = nullptr;
    K_heads_ = nullptr;
    V_heads_ = nullptr;
    O_weight_ = nullptr;

    h_state_ = nullptr;
    A_ssm_ = nullptr;
    B_ssm_ = nullptr;
    C_ssm_ = nullptr;

    wkv_state_ = nullptr;
    R_rwkv_ = nullptr;
    K_rwkv_ = nullptr;
    V_rwkv_ = nullptr;

    adam_m_ = nullptr;
    adam_v_ = nullptr;

    kfe_local_ = nullptr;
    noise_schedule_ = nullptr;

    kv_cache_keys_ = nullptr;
    kv_cache_values_ = nullptr;
    kv_cache_size_ = config.optimization().kv_cache_size();

    ema_mean_ = nullptr;
    ema_std_ = nullptr;

    // Create cuBLAS handle
    cublasCreate(&cublas_handle_);
}

Neuron::~Neuron() {
    free_memory();
    cublasDestroy(cublas_handle_);
}

bool Neuron::initialize() {
    if (!allocate_memory()) {
        return false;
    }
    initialize_weights();
    return true;
}

bool Neuron::allocate_memory() {
    size_t matrix_size = dim_ * dim_ * sizeof(double);
    size_t vector_size = dim_ * sizeof(double);

    // Allocate core state matrices
    cudaMalloc(&P_Matrix_, matrix_size);
    cudaMalloc(&P_stable_, matrix_size);
    cudaMalloc(&W_predict_, matrix_size);
    cudaMalloc(&M_KFE_, matrix_size);
    cudaMalloc(&Deviation_, matrix_size);
    cudaMalloc(&PS_aggregate_, matrix_size);

    // Allocate temporal history
    uint32_t temporal_frames = config_.temporal_frames();
    P_history_ = new double*[temporal_frames];
    for (uint32_t i = 0; i < temporal_frames; i++) {
        cudaMalloc(&P_history_[i], matrix_size);
    }

    // Allocate port matrices
    port_in_matrices_ = new double*[4];
    port_out_matrices_ = new double*[4];
    for (int i = 0; i < 4; i++) {
        cudaMalloc(&port_in_matrices_[i], matrix_size);
        cudaMalloc(&port_out_matrices_[i], matrix_size);
    }

    // Allocate convolution kernels (4 ports × 8 kernels × 8×8)
    conv_kernels_ = new double**[4];
    for (int i = 0; i < 4; i++) {
        conv_kernels_[i] = new double*[8];
        for (int j = 0; j < 8; j++) {
            cudaMalloc(&conv_kernels_[i][j], 64 * sizeof(double)); // 8×8
        }
    }

    // Allocate multi-head attention weights (if enabled)
    if (config_.modules().enable_multi_head()) {
        Q_heads_ = new double*[num_heads_];
        K_heads_ = new double*[num_heads_];
        V_heads_ = new double*[num_heads_];

        size_t head_matrix_size = head_dim_ * head_dim_ * sizeof(double);
        for (uint32_t i = 0; i < num_heads_; i++) {
            cudaMalloc(&Q_heads_[i], head_matrix_size);
            cudaMalloc(&K_heads_[i], head_matrix_size);
            cudaMalloc(&V_heads_[i], head_matrix_size);
        }

        cudaMalloc(&O_weight_, matrix_size);
    }

    // Allocate SSM state (if enabled)
    if (config_.modules().enable_ssm()) {
        cudaMalloc(&h_state_, vector_size);
        cudaMalloc(&A_ssm_, matrix_size);
        cudaMalloc(&B_ssm_, matrix_size);
        cudaMalloc(&C_ssm_, matrix_size);
    }

    // Allocate RWKV state (if enabled)
    if (config_.modules().enable_rwkv()) {
        cudaMalloc(&wkv_state_, vector_size);
        cudaMalloc(&R_rwkv_, matrix_size);
        cudaMalloc(&K_rwkv_, matrix_size);
        cudaMalloc(&V_rwkv_, matrix_size);
    }

    // Allocate Adam optimizer state
    cudaMalloc(&adam_m_, matrix_size);
    cudaMalloc(&adam_v_, matrix_size);

    // Allocate KFE local storage
    kfe_local_ = new KFE_STM_Slot[KFE_LOCAL_SIZE];
    for (int i = 0; i < KFE_LOCAL_SIZE; i++) {
        cudaMalloc(&kfe_local_[i].kfe_matrix, matrix_size);
        kfe_local_[i].is_valid = false;
        kfe_local_[i].access_count = 0;
        kfe_local_[i].last_access = 0;
    }

    // Allocate DDPM noise schedule (if enabled)
    if (config_.modules().enable_ddpm()) {
        cudaMalloc(&noise_schedule_, DDPM_STEPS * sizeof(double));
    }

    // Allocate KV-Cache (if enabled)
    if (config_.optimization().use_kv_cache()) {
        cudaMalloc(&kv_cache_keys_, kv_cache_size_ * dim_ * sizeof(double));
        cudaMalloc(&kv_cache_values_, kv_cache_size_ * dim_ * sizeof(double));
    }

    // Allocate noise filter state (if enabled)
    if (config_.modules().enable_noise_filter()) {
        cudaMalloc(&ema_mean_, vector_size);
        cudaMalloc(&ema_std_, vector_size);
    }

    return true;
}

void Neuron::free_memory() {
    // Free core matrices
    if (P_Matrix_) cudaFree(P_Matrix_);
    if (P_stable_) cudaFree(P_stable_);
    if (W_predict_) cudaFree(W_predict_);
    if (M_KFE_) cudaFree(M_KFE_);
    if (Deviation_) cudaFree(Deviation_);
    if (PS_aggregate_) cudaFree(PS_aggregate_);

    // Free temporal history
    if (P_history_) {
        for (uint32_t i = 0; i < config_.temporal_frames(); i++) {
            if (P_history_[i]) cudaFree(P_history_[i]);
        }
        delete[] P_history_;
    }

    // Free port matrices
    if (port_in_matrices_) {
        for (int i = 0; i < 4; i++) {
            if (port_in_matrices_[i]) cudaFree(port_in_matrices_[i]);
        }
        delete[] port_in_matrices_;
    }

    if (port_out_matrices_) {
        for (int i = 0; i < 4; i++) {
            if (port_out_matrices_[i]) cudaFree(port_out_matrices_[i]);
        }
        delete[] port_out_matrices_;
    }

    // Free convolution kernels
    if (conv_kernels_) {
        for (int i = 0; i < 4; i++) {
            if (conv_kernels_[i]) {
                for (int j = 0; j < 8; j++) {
                    if (conv_kernels_[i][j]) cudaFree(conv_kernels_[i][j]);
                }
                delete[] conv_kernels_[i];
            }
        }
        delete[] conv_kernels_;
    }

    // Free multi-head attention
    if (Q_heads_) {
        for (uint32_t i = 0; i < num_heads_; i++) {
            if (Q_heads_[i]) cudaFree(Q_heads_[i]);
        }
        delete[] Q_heads_;
    }

    if (K_heads_) {
        for (uint32_t i = 0; i < num_heads_; i++) {
            if (K_heads_[i]) cudaFree(K_heads_[i]);
        }
        delete[] K_heads_;
    }

    if (V_heads_) {
        for (uint32_t i = 0; i < num_heads_; i++) {
            if (V_heads_[i]) cudaFree(V_heads_[i]);
        }
        delete[] V_heads_;
    }

    if (O_weight_) cudaFree(O_weight_);

    // Free SSM state
    if (h_state_) cudaFree(h_state_);
    if (A_ssm_) cudaFree(A_ssm_);
    if (B_ssm_) cudaFree(B_ssm_);
    if (C_ssm_) cudaFree(C_ssm_);

    // Free RWKV state
    if (wkv_state_) cudaFree(wkv_state_);
    if (R_rwkv_) cudaFree(R_rwkv_);
    if (K_rwkv_) cudaFree(K_rwkv_);
    if (V_rwkv_) cudaFree(V_rwkv_);

    // Free Adam state
    if (adam_m_) cudaFree(adam_m_);
    if (adam_v_) cudaFree(adam_v_);

    // Free KFE local storage
    if (kfe_local_) {
        for (int i = 0; i < KFE_LOCAL_SIZE; i++) {
            if (kfe_local_[i].kfe_matrix) cudaFree(kfe_local_[i].kfe_matrix);
        }
        delete[] kfe_local_;
    }

    // Free DDPM
    if (noise_schedule_) cudaFree(noise_schedule_);

    // Free KV-Cache
    if (kv_cache_keys_) cudaFree(kv_cache_keys_);
    if (kv_cache_values_) cudaFree(kv_cache_values_);

    // Free noise filter
    if (ema_mean_) cudaFree(ema_mean_);
    if (ema_std_) cudaFree(ema_std_);
}

// CUDA kernel for Xavier initialization
__global__ void xavier_init_kernel(double* matrix, int rows, int cols, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        double scale = sqrt(2.0 / (rows + cols));
        matrix[idx] = curand_normal_double(&state) * scale;
    }
}

// CUDA kernel for zero initialization
__global__ void zero_init_kernel(double* matrix, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        matrix[idx] = 0.0;
    }
}

void Neuron::initialize_weights() {
    int threads = 256;
    size_t matrix_size = dim_ * dim_;
    int blocks = (matrix_size + threads - 1) / threads;

    unsigned long long seed = neuron_id_ * 12345ULL;

    // Initialize core matrices
    xavier_init_kernel<<<blocks, threads>>>(P_Matrix_, dim_, dim_, seed++);
    xavier_init_kernel<<<blocks, threads>>>(W_predict_, dim_, dim_, seed++);
    zero_init_kernel<<<blocks, threads>>>(P_stable_, matrix_size);
    zero_init_kernel<<<blocks, threads>>>(M_KFE_, matrix_size);
    zero_init_kernel<<<blocks, threads>>>(Deviation_, matrix_size);
    zero_init_kernel<<<blocks, threads>>>(PS_aggregate_, matrix_size);

    // Initialize temporal history
    for (uint32_t i = 0; i < config_.temporal_frames(); i++) {
        zero_init_kernel<<<blocks, threads>>>(P_history_[i], matrix_size);
    }

    // Initialize port matrices
    for (int i = 0; i < 4; i++) {
        xavier_init_kernel<<<blocks, threads>>>(port_in_matrices_[i], dim_, dim_, seed++);
        xavier_init_kernel<<<blocks, threads>>>(port_out_matrices_[i], dim_, dim_, seed++);
    }

    // Initialize convolution kernels
    int conv_blocks = (64 + threads - 1) / threads;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            xavier_init_kernel<<<conv_blocks, threads>>>(conv_kernels_[i][j], 8, 8, seed++);
        }
    }

    // Initialize multi-head attention (if enabled)
    if (config_.modules().enable_multi_head()) {
        size_t head_matrix_size = head_dim_ * head_dim_;
        int head_blocks = (head_matrix_size + threads - 1) / threads;

        for (uint32_t i = 0; i < num_heads_; i++) {
            xavier_init_kernel<<<head_blocks, threads>>>(Q_heads_[i], head_dim_, head_dim_, seed++);
            xavier_init_kernel<<<head_blocks, threads>>>(K_heads_[i], head_dim_, head_dim_, seed++);
            xavier_init_kernel<<<head_blocks, threads>>>(V_heads_[i], head_dim_, head_dim_, seed++);
        }

        xavier_init_kernel<<<blocks, threads>>>(O_weight_, dim_, dim_, seed++);
    }

    // Initialize SSM matrices (if enabled)
    if (config_.modules().enable_ssm()) {
        xavier_init_kernel<<<blocks, threads>>>(A_ssm_, dim_, dim_, seed++);
        xavier_init_kernel<<<blocks, threads>>>(B_ssm_, dim_, dim_, seed++);
        xavier_init_kernel<<<blocks, threads>>>(C_ssm_, dim_, dim_, seed++);

        int vec_blocks = (dim_ + threads - 1) / threads;
        zero_init_kernel<<<vec_blocks, threads>>>(h_state_, dim_);
    }

    // Initialize RWKV matrices (if enabled)
    if (config_.modules().enable_rwkv()) {
        xavier_init_kernel<<<blocks, threads>>>(R_rwkv_, dim_, dim_, seed++);
        xavier_init_kernel<<<blocks, threads>>>(K_rwkv_, dim_, dim_, seed++);
        xavier_init_kernel<<<blocks, threads>>>(V_rwkv_, dim_, dim_, seed++);

        int vec_blocks = (dim_ + threads - 1) / threads;
        zero_init_kernel<<<vec_blocks, threads>>>(wkv_state_, dim_);
    }

    // Initialize Adam optimizer state
    zero_init_kernel<<<blocks, threads>>>(adam_m_, matrix_size);
    zero_init_kernel<<<blocks, threads>>>(adam_v_, matrix_size);

    // Initialize DDPM noise schedule (if enabled)
    if (config_.modules().enable_ddpm()) {
        double h_noise_schedule[DDPM_STEPS];
        for (int i = 0; i < DDPM_STEPS; i++) {
            double t = (double)i / (DDPM_STEPS - 1);
            h_noise_schedule[i] = 0.0001 + (0.02 - 0.0001) * t;
        }
        cudaMemcpy(noise_schedule_, h_noise_schedule, DDPM_STEPS * sizeof(double), cudaMemcpyHostToDevice);
    }

    // Initialize noise filter (if enabled)
    if (config_.modules().enable_noise_filter()) {
        int vec_blocks = (dim_ + threads - 1) / threads;
        zero_init_kernel<<<vec_blocks, threads>>>(ema_mean_, dim_);
        zero_init_kernel<<<vec_blocks, threads>>>(ema_std_, dim_);
    }

    cudaDeviceSynchronize();
}

// ============================================================================
// Core CUDA Kernels
// ============================================================================

// Multi-head attention kernel
__global__ void multi_head_attention_kernel(
    const double* input,
    double** Q_heads,
    double** K_heads,
    double** V_heads,
    double* output,
    int dim,
    int num_heads,
    int head_dim
) {
    int head_idx = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (head_idx >= num_heads || row >= head_dim) return;

    // Compute Q, K, V for this head
    double q_val = 0.0, k_val = 0.0, v_val = 0.0;
    for (int i = 0; i < head_dim; i++) {
        int input_idx = head_idx * head_dim + i;
        q_val += Q_heads[head_idx][row * head_dim + i] * input[input_idx];
        k_val += K_heads[head_idx][row * head_dim + i] * input[input_idx];
        v_val += V_heads[head_idx][row * head_dim + i] * input[input_idx];
    }

    // Attention score (simplified, full version needs softmax)
    double score = q_val * k_val / sqrt((double)head_dim);
    double attn_output = score * v_val;

    // Write to output
    output[head_idx * head_dim + row] = attn_output;
}

// Convolution feature extraction kernel
__global__ void conv_feature_kernel(
    const double* input,
    double** conv_kernels,
    double* output,
    int dim,
    int num_kernels
) {
    int kernel_idx = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (kernel_idx >= num_kernels || idx >= dim) return;

    double sum = 0.0;
    // Apply 8x8 convolution (simplified)
    for (int i = 0; i < 64; i++) {
        int input_idx = (idx + i) % dim;
        sum += input[input_idx] * conv_kernels[kernel_idx][i];
    }

    output[kernel_idx * dim + idx] = sum;
}

// GEMM + DRC iteration kernel
__global__ void gemm_drc_kernel(
    double* P_Matrix,
    const double* W_predict,
    double* Deviation,
    int dim,
    int iteration
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= dim || col >= dim) return;

    // Compute prediction error
    double predicted = 0.0;
    for (int k = 0; k < dim; k++) {
        predicted += P_Matrix[row * dim + k] * W_predict[k * dim + col];
    }

    // Dynamic recalibration correction
    double error = P_Matrix[row * dim + col] - predicted;
    double correction = error * 0.1 / (iteration + 1);

    // Update P_Matrix
    P_Matrix[row * dim + col] += correction;

    // Update Deviation
    Deviation[row * dim + col] = error;
}

// SSM state update kernel
__global__ void ssm_update_kernel(
    double* h_state,
    const double* A_ssm,
    const double* B_ssm,
    const double* input,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    // h_t = A * h_{t-1} + B * x_t
    double new_h = 0.0;
    for (int i = 0; i < dim; i++) {
        new_h += A_ssm[idx * dim + i] * h_state[i];
        new_h += B_ssm[idx * dim + i] * input[i];
    }

    h_state[idx] = new_h;
}

// RWKV WKV computation kernel
__global__ void rwkv_wkv_kernel(
    double* wkv_state,
    const double* R_rwkv,
    const double* K_rwkv,
    const double* V_rwkv,
    const double* input,
    double* output,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    // Compute R, K, V
    double r = 0.0, k = 0.0, v = 0.0;
    for (int i = 0; i < dim; i++) {
        r += R_rwkv[idx * dim + i] * input[i];
        k += K_rwkv[idx * dim + i] * input[i];
        v += V_rwkv[idx * dim + i] * input[i];
    }

    // WKV update
    double wkv = wkv_state[idx] * exp(-k) + v;
    wkv_state[idx] = wkv;

    // Output
    output[idx] = r * wkv;
}

// Adaptive noise filter kernel with EMA
__global__ void adaptive_noise_filter_kernel(
    double* data,
    double* ema_mean,
    double* ema_std,
    float alpha,
    float threshold_multiplier,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    double val = data[idx];

    // Update EMA statistics
    double old_mean = ema_mean[idx];
    double new_mean = alpha * val + (1.0 - alpha) * old_mean;
    ema_mean[idx] = new_mean;

    double deviation = val - new_mean;
    double old_std = ema_std[idx];
    double new_std = alpha * (deviation * deviation) + (1.0 - alpha) * (old_std * old_std);
    ema_std[idx] = sqrt(new_std);

    // Apply threshold filtering
    double threshold = new_mean + threshold_multiplier * sqrt(new_std);
    if (fabs(val) < threshold) {
        data[idx] = 0.0;  // Filter out noise
    }
}

// Temporal attention kernel (8 frames)
__global__ void temporal_attention_kernel(
    double** P_history,
    double* output,
    int dim,
    int num_frames,
    int current_frame
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= dim || col >= dim) return;

    // Compute attention weights over temporal frames
    double sum = 0.0;
    double weight_sum = 0.0;

    for (int t = 0; t < num_frames; t++) {
        // Exponential decay for older frames
        double weight = exp(-0.1 * abs(t - current_frame));
        sum += P_history[t][row * dim + col] * weight;
        weight_sum += weight;
    }

    output[row * dim + col] = sum / weight_sum;
}

// FXAA-like edge detection and smoothing kernel
__global__ void fxaa_auxiliary_kernel(
    const double* input,
    double* output,
    int dim
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= dim || col >= dim) return;

    int idx = row * dim + col;

    // Edge detection (simplified Sobel-like)
    double edge_strength = 0.0;
    if (row > 0 && row < dim - 1 && col > 0 && col < dim - 1) {
        double gx = input[(row - 1) * dim + (col + 1)] - input[(row - 1) * dim + (col - 1)]
                  + 2.0 * input[row * dim + (col + 1)] - 2.0 * input[row * dim + (col - 1)]
                  + input[(row + 1) * dim + (col + 1)] - input[(row + 1) * dim + (col - 1)];

        double gy = input[(row + 1) * dim + (col - 1)] - input[(row - 1) * dim + (col - 1)]
                  + 2.0 * input[(row + 1) * dim + col] - 2.0 * input[(row - 1) * dim + col]
                  + input[(row + 1) * dim + (col + 1)] - input[(row - 1) * dim + (col + 1)];

        edge_strength = sqrt(gx * gx + gy * gy);
    }

    // Adaptive smoothing based on edge strength
    double smoothed = input[idx];
    if (edge_strength < 0.5) {  // Low edge strength -> apply smoothing
        double sum = 0.0;
        int count = 0;
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                int nr = row + dr;
                int nc = col + dc;
                if (nr >= 0 && nr < dim && nc >= 0 && nc < dim) {
                    sum += input[nr * dim + nc];
                    count++;
                }
            }
        }
        smoothed = sum / count;
    }

    output[idx] = smoothed;
}

// DDPM denoising kernel (reverse diffusion)
__global__ void ddpm_denoise_kernel(
    double* data,
    const double* noise_schedule,
    int dim,
    int step,
    int total_steps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;

    // Reverse diffusion step
    double noise_level = noise_schedule[step];
    double alpha = 1.0 - noise_level;
    double alpha_bar = 1.0;
    for (int i = 0; i <= step; i++) {
        alpha_bar *= (1.0 - noise_schedule[i]);
    }

    // Denoise
    double x_t = data[idx];
    double x_0_pred = x_t / sqrt(alpha_bar);

    // Update with predicted x_0
    if (step > 0) {
        double noise_level_prev = noise_schedule[step - 1];
        double alpha_bar_prev = alpha_bar / (1.0 - noise_level);
        data[idx] = sqrt(alpha_bar_prev) * x_0_pred;
    } else {
        data[idx] = x_0_pred;
    }
}

// Global sparse attention aggregation kernel
// This will be called from NeuronModel layer with all neurons' data
__global__ void global_aggregation_kernel(
    const double* local_output,
    const double** all_neurons_output,
    double* aggregated_output,
    const int* top_k_indices,
    int dim,
    int top_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;

    double sum = local_output[idx];

    // Aggregate from top-K important neurons
    for (int k = 0; k < top_k; k++) {
        int neuron_idx = top_k_indices[k];
        sum += all_neurons_output[neuron_idx][idx] * 0.1;  // Weighted aggregation
    }

    aggregated_output[idx] = sum;
}

// ============================================================================
// Forward Pass Implementation
// ============================================================================

void Neuron::forward(const double* input, double* output, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (dim_ * dim_ + threads - 1) / threads;
    const int vec_blocks = (dim_ + threads - 1) / threads;

    // Temporary buffers
    double* temp_buffer1;
    double* temp_buffer2;
    cudaMalloc(&temp_buffer1, dim_ * dim_ * sizeof(double));
    cudaMalloc(&temp_buffer2, dim_ * dim_ * sizeof(double));

    // Step 1: Port input transformation
    // Use cuBLAS for matrix multiplication: temp = port_in_matrices[0] * input
    const double alpha = 1.0;
    const double beta = 0.0;
    cublasDgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, dim_, dim_,
                &alpha,
                port_in_matrices_[0], dim_,
                input, dim_,
                &beta,
                temp_buffer1, dim_);

    // Step 2: Convolution feature extraction
    dim3 conv_blocks((dim_ + threads - 1) / threads, 4);
    conv_feature_kernel<<<conv_blocks, threads, 0, stream>>>(
        temp_buffer1, conv_kernels_[0], temp_buffer2, dim_, 8
    );

    // Step 3: Multi-head attention (if enabled)
    if (config_.modules().enable_multi_head()) {
        dim3 attn_blocks((head_dim_ + threads - 1) / threads, num_heads_);
        multi_head_attention_kernel<<<attn_blocks, threads, 0, stream>>>(
            temp_buffer2, Q_heads_, K_heads_, V_heads_,
            temp_buffer1, dim_, num_heads_, head_dim_
        );

        // Output projection
        cublasDgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                    dim_, dim_, dim_,
                    &alpha, O_weight_, dim_,
                    temp_buffer1, dim_,
                    &beta, temp_buffer2, dim_);
    }

    // Step 4: GEMM + DRC iterations (16 iterations)
    cudaMemcpy(P_Matrix_, temp_buffer2, dim_ * dim_ * sizeof(double), cudaMemcpyDeviceToDevice);

    dim3 gemm_blocks((dim_ + 15) / 16, (dim_ + 15) / 16);
    dim3 gemm_threads(16, 16);

    for (int iter = 0; iter < 16; iter++) {
        gemm_drc_kernel<<<gemm_blocks, gemm_threads, 0, stream>>>(
            P_Matrix_, W_predict_, Deviation_, dim_, iter
        );
    }

    // Step 5: SSM state update (if enabled)
    if (config_.modules().enable_ssm()) {
        ssm_update_kernel<<<vec_blocks, threads, 0, stream>>>(
            h_state_, A_ssm_, B_ssm_, P_Matrix_, dim_
        );

        // Apply SSM output: C * h_state
        cublasDgemv(cublas_handle_, CUBLAS_OP_N,
                    dim_, dim_,
                    &alpha, C_ssm_, dim_,
                    h_state_, 1,
                    &beta, temp_buffer1, 1);

        // Add to P_Matrix
        cublasDaxpy(cublas_handle_, dim_ * dim_,
                    &alpha, temp_buffer1, 1,
                    P_Matrix_, 1);
    }

    // Step 6: RWKV computation (if enabled)
    if (config_.modules().enable_rwkv()) {
        rwkv_wkv_kernel<<<vec_blocks, threads, 0, stream>>>(
            wkv_state_, R_rwkv_, K_rwkv_, V_rwkv_,
            P_Matrix_, temp_buffer1, dim_
        );

        // Add RWKV output to P_Matrix
        cublasDaxpy(cublas_handle_, dim_ * dim_,
                    &alpha, temp_buffer1, 1,
                    P_Matrix_, 1);
    }

    // Step 7: Adaptive noise filtering (if enabled)
    if (config_.modules().enable_noise_filter()) {
        adaptive_noise_filter_kernel<<<vec_blocks, threads, 0, stream>>>(
            P_Matrix_, ema_mean_, ema_std_,
            ema_alpha_,
            config_.noise_filter().threshold_multiplier(),
            dim_
        );
    }

    // Step 8: Update temporal history
    if (config_.modules().enable_temporal_attention()) {
        // Shift history
        history_index_ = (history_index_ + 1) % config_.temporal_frames();
        cudaMemcpy(P_history_[history_index_], P_Matrix_,
                   dim_ * dim_ * sizeof(double), cudaMemcpyDeviceToDevice);

        // Temporal attention aggregation
        temporal_attention_kernel<<<gemm_blocks, gemm_threads, 0, stream>>>(
            P_history_, temp_buffer1, dim_,
            config_.temporal_frames(), history_index_
        );

        // Blend with current state
        const double blend_alpha = 0.7;
        const double blend_beta = 0.3;
        cublasDaxpy(cublas_handle_, dim_ * dim_,
                    &blend_alpha, temp_buffer1, 1,
                    P_Matrix_, 1);
    }

    // Step 9: FXAA-like auxiliary prediction (if enabled)
    if (config_.modules().enable_fxaa_layer()) {
        fxaa_auxiliary_kernel<<<gemm_blocks, gemm_threads, 0, stream>>>(
            P_Matrix_, temp_buffer1, dim_
        );

        // Blend FXAA output
        const double fxaa_weight = 0.2;
        cublasDaxpy(cublas_handle_, dim_ * dim_,
                    &fxaa_weight, temp_buffer1, 1,
                    P_Matrix_, 1);
    }

    // Step 10: DDPM denoising (if enabled)
    if (config_.modules().enable_ddpm()) {
        for (int step = DDPM_STEPS - 1; step >= 0; step--) {
            ddpm_denoise_kernel<<<blocks, threads, 0, stream>>>(
                P_Matrix_, noise_schedule_, dim_, step, DDPM_STEPS
            );
        }
    }

    // Step 11: Update stable state (exponential moving average)
    const double stable_alpha = 0.99;
    const double stable_beta = 0.01;
    cublasDscal(cublas_handle_, dim_ * dim_, &stable_alpha, P_stable_, 1);
    cublasDaxpy(cublas_handle_, dim_ * dim_, &stable_beta, P_Matrix_, 1, P_stable_, 1);

    // Step 12: Port output transformation
    cublasDgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, dim_, dim_,
                &alpha, port_out_matrices_[0], dim_,
                P_Matrix_, dim_,
                &beta, output, dim_);

    // Cleanup
    cudaFree(temp_buffer1);
    cudaFree(temp_buffer2);

    cudaStreamSynchronize(stream);
}

// ============================================================================
// Adam Optimizer Update Kernel
// ============================================================================

__global__ void adam_update_kernel(
    double* param,
    const double* grad,
    double* m,
    double* v,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    int step,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Update biased first moment estimate
    m[idx] = beta1 * m[idx] + (1.0 - beta1) * grad[idx];

    // Update biased second raw moment estimate
    v[idx] = beta2 * v[idx] + (1.0 - beta2) * grad[idx] * grad[idx];

    // Compute bias-corrected first moment estimate
    double m_hat = m[idx] / (1.0 - pow(beta1, step));

    // Compute bias-corrected second raw moment estimate
    double v_hat = v[idx] / (1.0 - pow(beta2, step));

    // Update parameters
    param[idx] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
}

void Neuron::update_parameters(float learning_rate, cudaStream_t stream) {
    adam_step_++;

    const int threads = 256;
    const int blocks = (dim_ * dim_ + threads - 1) / threads;

    // Adam hyperparameters
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-8f;

    // Update W_predict (assuming gradients are stored in Deviation_)
    adam_update_kernel<<<blocks, threads, 0, stream>>>(
        W_predict_, Deviation_, adam_m_, adam_v_,
        learning_rate, beta1, beta2, epsilon,
        adam_step_, dim_ * dim_
    );

    // Update multi-head attention weights (if enabled)
    if (config_.modules().enable_multi_head()) {
        size_t head_matrix_size = head_dim_ * head_dim_;
        int head_blocks = (head_matrix_size + threads - 1) / threads;

        for (uint32_t i = 0; i < num_heads_; i++) {
            // Note: Need separate m/v for each head in full implementation
            adam_update_kernel<<<head_blocks, threads, 0, stream>>>(
                Q_heads_[i], Deviation_, adam_m_, adam_v_,
                learning_rate, beta1, beta2, epsilon,
                adam_step_, head_matrix_size
            );
        }
    }

    cudaStreamSynchronize(stream);
}

// ============================================================================
// Backward Pass Gradient Kernels
// ============================================================================

// Multi-head attention backward kernel
__global__ void multi_head_attention_backward_kernel(
    const double* grad_output,
    const double* input,
    double** Q_heads,
    double** K_heads,
    double** V_heads,
    double** grad_Q,
    double** grad_K,
    double** grad_V,
    double* grad_input,
    int dim,
    int num_heads,
    int head_dim
) {
    int head_idx = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (head_idx >= num_heads || row >= head_dim) return;

    // Simplified gradient computation
    double grad_out = grad_output[head_idx * head_dim + row];

    // Backprop through attention
    for (int i = 0; i < head_dim; i++) {
        int input_idx = head_idx * head_dim + i;
        atomicAdd(&grad_Q[head_idx][row * head_dim + i], grad_out * input[input_idx]);
        atomicAdd(&grad_K[head_idx][row * head_dim + i], grad_out * input[input_idx]);
        atomicAdd(&grad_V[head_idx][row * head_dim + i], grad_out * input[input_idx]);
        atomicAdd(&grad_input[input_idx], grad_out * Q_heads[head_idx][row * head_dim + i]);
    }
}

// GEMM + DRC backward kernel
__global__ void gemm_drc_backward_kernel(
    const double* grad_output,
    const double* P_Matrix,
    const double* W_predict,
    double* grad_P,
    double* grad_W,
    int dim
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= dim || col >= dim) return;

    double grad = grad_output[row * dim + col];

    // Gradient w.r.t P_Matrix
    atomicAdd(&grad_P[row * dim + col], grad);

    // Gradient w.r.t W_predict
    for (int k = 0; k < dim; k++) {
        atomicAdd(&grad_W[k * dim + col], grad * P_Matrix[row * dim + k]);
    }
}

// SSM backward kernel
__global__ void ssm_backward_kernel(
    const double* grad_output,
    const double* h_state,
    const double* A_ssm,
    const double* B_ssm,
    const double* input,
    double* grad_h,
    double* grad_A,
    double* grad_B,
    double* grad_input,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    double grad = grad_output[idx];

    // Gradient w.r.t A, B, and input
    for (int i = 0; i < dim; i++) {
        atomicAdd(&grad_A[idx * dim + i], grad * h_state[i]);
        atomicAdd(&grad_B[idx * dim + i], grad * input[i]);
        atomicAdd(&grad_input[i], grad * B_ssm[idx * dim + i]);
    }
}

// RWKV backward kernel
__global__ void rwkv_backward_kernel(
    const double* grad_output,
    const double* wkv_state,
    const double* R_rwkv,
    const double* K_rwkv,
    const double* V_rwkv,
    const double* input,
    double* grad_R,
    double* grad_K,
    double* grad_V,
    double* grad_input,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    double grad = grad_output[idx];

    // Simplified gradient computation
    for (int i = 0; i < dim; i++) {
        atomicAdd(&grad_R[idx * dim + i], grad * wkv_state[idx] * input[i]);
        atomicAdd(&grad_K[idx * dim + i], grad * input[i]);
        atomicAdd(&grad_V[idx * dim + i], grad * input[i]);
        atomicAdd(&grad_input[i], grad * R_rwkv[idx * dim + i]);
    }
}

// ============================================================================
// Backward Pass Implementation
// ============================================================================

void Neuron::backward(const double* grad_output, double* grad_input, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (dim_ * dim_ + threads - 1) / threads;
    const int vec_blocks = (dim_ + threads - 1) / threads;

    // Temporary gradient buffers
    double* grad_temp1;
    double* grad_temp2;
    cudaMalloc(&grad_temp1, dim_ * dim_ * sizeof(double));
    cudaMalloc(&grad_temp2, dim_ * dim_ * sizeof(double));
    cudaMemset(grad_temp1, 0, dim_ * dim_ * sizeof(double));
    cudaMemset(grad_temp2, 0, dim_ * dim_ * sizeof(double));

    // Initialize gradient accumulator
    cudaMemcpy(grad_temp1, grad_output, dim_ * dim_ * sizeof(double), cudaMemcpyDeviceToDevice);

    const double alpha = 1.0;
    const double beta = 0.0;

    // Step 1: Backprop through port output transformation
    cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, dim_, dim_,
                &alpha, port_out_matrices_[0], dim_,
                grad_temp1, dim_,
                &beta, grad_temp2, dim_);

    // Step 2: Backprop through DDPM (if enabled)
    if (config_.modules().enable_ddpm()) {
        // DDPM backward is complex, simplified here
        // In full implementation, need to backprop through all denoising steps
    }

    // Step 3: Backprop through FXAA layer (if enabled)
    if (config_.modules().enable_fxaa_layer()) {
        // FXAA backward (simplified)
        // Edge-aware gradient propagation
    }

    // Step 4: Backprop through temporal attention (if enabled)
    if (config_.modules().enable_temporal_attention()) {
        // Distribute gradients to temporal history
        for (uint32_t t = 0; t < config_.temporal_frames(); t++) {
            double weight = exp(-0.1 * abs((int)t - history_index_));
            cublasDaxpy(cublas_handle_, dim_ * dim_,
                       &weight, grad_temp2, 1,
                       grad_temp1, 1);
        }
    }

    // Step 5: Backprop through noise filter (if enabled)
    if (config_.modules().enable_noise_filter()) {
        // Gradient flows through non-filtered elements only
    }

    // Step 6: Backprop through RWKV (if enabled)
    if (config_.modules().enable_rwkv()) {
        dim3 rwkv_blocks((dim_ + threads - 1) / threads, 1);
        rwkv_backward_kernel<<<rwkv_blocks, threads, 0, stream>>>(
            grad_temp2, wkv_state_, R_rwkv_, K_rwkv_, V_rwkv_,
            P_Matrix_, grad_temp1, grad_temp1, grad_temp1, grad_temp2, dim_
        );
    }

    // Step 7: Backprop through SSM (if enabled)
    if (config_.modules().enable_ssm()) {
        ssm_backward_kernel<<<vec_blocks, threads, 0, stream>>>(
            grad_temp2, h_state_, A_ssm_, B_ssm_, P_Matrix_,
            grad_temp1, grad_temp1, grad_temp1, grad_temp2, dim_
        );
    }

    // Step 8: Backprop through GEMM + DRC
    dim3 gemm_blocks((dim_ + 15) / 16, (dim_ + 15) / 16);
    dim3 gemm_threads(16, 16);

    gemm_drc_backward_kernel<<<gemm_blocks, gemm_threads, 0, stream>>>(
        grad_temp2, P_Matrix_, W_predict_,
        grad_temp1, Deviation_, dim_
    );

    // Step 9: Backprop through multi-head attention (if enabled)
    if (config_.modules().enable_multi_head()) {
        // Backprop through output projection
        cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                    dim_, dim_, dim_,
                    &alpha, O_weight_, dim_,
                    grad_temp1, dim_,
                    &beta, grad_temp2, dim_);

        // Backprop through attention heads
        dim3 attn_blocks((head_dim_ + threads - 1) / threads, num_heads_);

        // Allocate gradient buffers for heads
        double** grad_Q_heads;
        double** grad_K_heads;
        double** grad_V_heads;
        cudaMalloc(&grad_Q_heads, num_heads_ * sizeof(double*));
        cudaMalloc(&grad_K_heads, num_heads_ * sizeof(double*));
        cudaMalloc(&grad_V_heads, num_heads_ * sizeof(double*));

        for (uint32_t i = 0; i < num_heads_; i++) {
            double* grad_q;
            double* grad_k;
            double* grad_v;
            cudaMalloc(&grad_q, head_dim_ * head_dim_ * sizeof(double));
            cudaMalloc(&grad_k, head_dim_ * head_dim_ * sizeof(double));
            cudaMalloc(&grad_v, head_dim_ * head_dim_ * sizeof(double));
            cudaMemset(grad_q, 0, head_dim_ * head_dim_ * sizeof(double));
            cudaMemset(grad_k, 0, head_dim_ * head_dim_ * sizeof(double));
            cudaMemset(grad_v, 0, head_dim_ * head_dim_ * sizeof(double));
            cudaMemcpy(&grad_Q_heads[i], &grad_q, sizeof(double*), cudaMemcpyHostToDevice);
            cudaMemcpy(&grad_K_heads[i], &grad_k, sizeof(double*), cudaMemcpyHostToDevice);
            cudaMemcpy(&grad_V_heads[i], &grad_v, sizeof(double*), cudaMemcpyHostToDevice);
        }

        multi_head_attention_backward_kernel<<<attn_blocks, threads, 0, stream>>>(
            grad_temp2, P_Matrix_, Q_heads_, K_heads_, V_heads_,
            grad_Q_heads, grad_K_heads, grad_V_heads,
            grad_temp1, dim_, num_heads_, head_dim_
        );

        // Cleanup gradient buffers
        for (uint32_t i = 0; i < num_heads_; i++) {
            double* grad_q;
            double* grad_k;
            double* grad_v;
            cudaMemcpy(&grad_q, &grad_Q_heads[i], sizeof(double*), cudaMemcpyDeviceToHost);
            cudaMemcpy(&grad_k, &grad_K_heads[i], sizeof(double*), cudaMemcpyDeviceToHost);
            cudaMemcpy(&grad_v, &grad_V_heads[i], sizeof(double*), cudaMemcpyDeviceToHost);
            cudaFree(grad_q);
            cudaFree(grad_k);
            cudaFree(grad_v);
        }
        cudaFree(grad_Q_heads);
        cudaFree(grad_K_heads);
        cudaFree(grad_V_heads);
    }

    // Step 10: Backprop through convolution
    // (Simplified - full implementation needs conv backward)

    // Step 11: Backprop through port input transformation
    cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                dim_, dim_, dim_,
                &alpha, port_in_matrices_[0], dim_,
                grad_temp1, dim_,
                &beta, grad_input, dim_);

    // Cleanup
    cudaFree(grad_temp1);
    cudaFree(grad_temp2);

    cudaStreamSynchronize(stream);
}

} // namespace sintellix
