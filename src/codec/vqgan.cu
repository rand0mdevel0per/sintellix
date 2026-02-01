#include "sintellix/codec/vqgan.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <fstream>
#include <cmath>
#include <limits>

namespace sintellix {

// ============================================================================
// VQCodebook Implementation
// ============================================================================

VQCodebook::VQCodebook(size_t codebook_size, size_t embedding_dim)
    : codebook_size_(codebook_size)
    , embedding_dim_(embedding_dim)
    , codebook_gpu_(nullptr)
{
}

VQCodebook::~VQCodebook() {
    if (codebook_gpu_) {
        cudaFree(codebook_gpu_);
    }
}

// CUDA kernel for codebook initialization
__global__ void codebook_init_kernel(double* codebook, size_t codebook_size, size_t embedding_dim, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = codebook_size * embedding_dim;

    if (idx < total) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Xavier initialization
        double scale = sqrt(2.0 / embedding_dim);
        codebook[idx] = curand_normal_double(&state) * scale;
    }
}

bool VQCodebook::initialize() {
    // Allocate GPU memory
    size_t total_size = codebook_size_ * embedding_dim_ * sizeof(double);
    cudaMalloc(&codebook_gpu_, total_size);

    if (!codebook_gpu_) {
        return false;
    }

    // Initialize with random values
    int threads = 256;
    int blocks = (codebook_size_ * embedding_dim_ + threads - 1) / threads;

    unsigned long long seed = 42ULL;
    codebook_init_kernel<<<blocks, threads>>>(codebook_gpu_, codebook_size_, embedding_dim_, seed);

    cudaDeviceSynchronize();
    return true;
}

bool VQCodebook::load_from_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read codebook size and embedding dim
    size_t file_codebook_size, file_embedding_dim;
    file.read(reinterpret_cast<char*>(&file_codebook_size), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&file_embedding_dim), sizeof(size_t));

    if (file_codebook_size != codebook_size_ || file_embedding_dim != embedding_dim_) {
        return false;
    }

    // Read codebook data
    size_t total_size = codebook_size_ * embedding_dim_;
    std::vector<double> codebook_host(total_size);
    file.read(reinterpret_cast<char*>(codebook_host.data()), total_size * sizeof(double));
    file.close();

    // Copy to GPU
    if (!codebook_gpu_) {
        cudaMalloc(&codebook_gpu_, total_size * sizeof(double));
    }
    cudaMemcpy(codebook_gpu_, codebook_host.data(), total_size * sizeof(double), cudaMemcpyHostToDevice);

    return true;
}

bool VQCodebook::save_to_file(const std::string& path) {
    if (!codebook_gpu_) {
        return false;
    }

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Write codebook size and embedding dim
    file.write(reinterpret_cast<const char*>(&codebook_size_), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&embedding_dim_), sizeof(size_t));

    // Copy from GPU to host
    size_t total_size = codebook_size_ * embedding_dim_;
    std::vector<double> codebook_host(total_size);
    cudaMemcpy(codebook_host.data(), codebook_gpu_, total_size * sizeof(double), cudaMemcpyDeviceToHost);

    // Write codebook data
    file.write(reinterpret_cast<const char*>(codebook_host.data()), total_size * sizeof(double));
    file.close();

    return true;
}

// CUDA kernel for vector quantization
// Find nearest codebook entry for each input vector
__global__ void quantize_kernel(
    const double* vectors,
    const double* codebook,
    int* codes,
    size_t batch_size,
    size_t embedding_dim,
    size_t codebook_size
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    const double* vector = vectors + batch_idx * embedding_dim;

    // Find nearest codebook entry
    double min_distance = INFINITY;
    int best_code = 0;

    for (int code_idx = 0; code_idx < codebook_size; code_idx++) {
        const double* code_vector = codebook + code_idx * embedding_dim;

        // Compute L2 distance
        double distance = 0.0;
        for (int d = 0; d < embedding_dim; d++) {
            double diff = vector[d] - code_vector[d];
            distance += diff * diff;
        }

        if (distance < min_distance) {
            min_distance = distance;
            best_code = code_idx;
        }
    }

    codes[batch_idx] = best_code;
}

void VQCodebook::quantize(const double* vectors, int* codes, size_t batch_size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    quantize_kernel<<<blocks, threads, 0, stream>>>(
        vectors, codebook_gpu_, codes,
        batch_size, embedding_dim_, codebook_size_
    );

    if (stream == 0) {
        cudaDeviceSynchronize();
    }
}

// CUDA kernel for dequantization
// Lookup codebook entries for given codes
__global__ void dequantize_kernel(
    const int* codes,
    const double* codebook,
    double* vectors,
    size_t batch_size,
    size_t embedding_dim
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int code = codes[batch_idx];
    const double* code_vector = codebook + code * embedding_dim;
    double* output_vector = vectors + batch_idx * embedding_dim;

    // Copy codebook entry to output
    for (int d = 0; d < embedding_dim; d++) {
        output_vector[d] = code_vector[d];
    }
}

void VQCodebook::dequantize(const int* codes, double* vectors, size_t batch_size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    dequantize_kernel<<<blocks, threads, 0, stream>>>(
        codes, codebook_gpu_, vectors,
        batch_size, embedding_dim_
    );

    if (stream == 0) {
        cudaDeviceSynchronize();
    }
}

// ============================================================================
// VQGANEncoder Implementation
// ============================================================================

VQGANEncoder::VQGANEncoder(size_t input_dim, size_t hidden_dim, std::shared_ptr<VQCodebook> codebook)
    : input_dim_(input_dim)
    , hidden_dim_(hidden_dim)
    , codebook_(codebook)
    , W_proj_(nullptr)
    , b_proj_(nullptr)
{
}

VQGANEncoder::~VQGANEncoder() {
    if (W_proj_) cudaFree(W_proj_);
    if (b_proj_) cudaFree(b_proj_);
}

// CUDA kernel for Xavier initialization
__global__ void xavier_init_encoder_kernel(double* matrix, int rows, int cols, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        double scale = sqrt(2.0 / (rows + cols));
        matrix[idx] = curand_normal_double(&state) * scale;
    }
}

bool VQGANEncoder::initialize() {
    // Allocate projection weights
    cudaMalloc(&W_proj_, input_dim_ * hidden_dim_ * sizeof(double));
    cudaMalloc(&b_proj_, hidden_dim_ * sizeof(double));

    if (!W_proj_ || !b_proj_) {
        return false;
    }

    // Initialize weights
    int threads = 256;
    int blocks_W = (input_dim_ * hidden_dim_ + threads - 1) / threads;
    int blocks_b = (hidden_dim_ + threads - 1) / threads;

    unsigned long long seed = 123ULL;
    xavier_init_encoder_kernel<<<blocks_W, threads>>>(W_proj_, input_dim_, hidden_dim_, seed);

    // Initialize bias to zero
    cudaMemset(b_proj_, 0, hidden_dim_ * sizeof(double));

    cudaDeviceSynchronize();
    return true;
}

// CUDA kernel for encoder projection
__global__ void encoder_projection_kernel(
    const double* input,
    const double* W_proj,
    const double* b_proj,
    double* output,
    size_t batch_size,
    size_t input_dim,
    size_t hidden_dim
) {
    int batch_idx = blockIdx.y;
    int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || hidden_idx >= hidden_dim) return;

    const double* input_vec = input + batch_idx * input_dim;
    double sum = b_proj[hidden_idx];

    // Matrix-vector multiplication
    for (int i = 0; i < input_dim; i++) {
        sum += input_vec[i] * W_proj[i * hidden_dim + hidden_idx];
    }

    // ReLU activation
    output[batch_idx * hidden_dim + hidden_idx] = fmax(0.0, sum);
}

void VQGANEncoder::encode(const double* input, int* codes, size_t batch_size, cudaStream_t stream) {
    // Allocate temporary buffer for projected vectors
    double* projected;
    cudaMalloc(&projected, batch_size * hidden_dim_ * sizeof(double));

    // Project input to hidden dimension
    int threads = 256;
    dim3 blocks((hidden_dim_ + threads - 1) / threads, batch_size);

    encoder_projection_kernel<<<blocks, threads, 0, stream>>>(
        input, W_proj_, b_proj_, projected,
        batch_size, input_dim_, hidden_dim_
    );

    // Quantize projected vectors
    codebook_->quantize(projected, codes, batch_size, stream);

    // Cleanup
    cudaFree(projected);

    if (stream == 0) {
        cudaDeviceSynchronize();
    }
}

// ============================================================================
// VQGANDecoder Implementation
// ============================================================================

VQGANDecoder::VQGANDecoder(size_t output_dim, size_t hidden_dim, std::shared_ptr<VQCodebook> codebook)
    : output_dim_(output_dim)
    , hidden_dim_(hidden_dim)
    , codebook_(codebook)
    , W_proj_(nullptr)
    , b_proj_(nullptr)
{
}

VQGANDecoder::~VQGANDecoder() {
    if (W_proj_) cudaFree(W_proj_);
    if (b_proj_) cudaFree(b_proj_);
}

bool VQGANDecoder::initialize() {
    // Allocate projection weights
    cudaMalloc(&W_proj_, hidden_dim_ * output_dim_ * sizeof(double));
    cudaMalloc(&b_proj_, output_dim_ * sizeof(double));

    if (!W_proj_ || !b_proj_) {
        return false;
    }

    // Initialize weights
    int threads = 256;
    int blocks_W = (hidden_dim_ * output_dim_ + threads - 1) / threads;

    unsigned long long seed = 456ULL;
    xavier_init_encoder_kernel<<<blocks_W, threads>>>(W_proj_, hidden_dim_, output_dim_, seed);

    // Initialize bias to zero
    cudaMemset(b_proj_, 0, output_dim_ * sizeof(double));

    cudaDeviceSynchronize();
    return true;
}

// CUDA kernel for decoder projection
__global__ void decoder_projection_kernel(
    const double* input,
    const double* W_proj,
    const double* b_proj,
    double* output,
    size_t batch_size,
    size_t hidden_dim,
    size_t output_dim
) {
    int batch_idx = blockIdx.y;
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || output_idx >= output_dim) return;

    const double* input_vec = input + batch_idx * hidden_dim;
    double sum = b_proj[output_idx];

    // Matrix-vector multiplication
    for (int i = 0; i < hidden_dim; i++) {
        sum += input_vec[i] * W_proj[i * output_dim + output_idx];
    }

    // Tanh activation for output
    output[batch_idx * output_dim + output_idx] = tanh(sum);
}

void VQGANDecoder::decode(const int* codes, double* output, size_t batch_size, cudaStream_t stream) {
    // Allocate temporary buffer for dequantized vectors
    double* dequantized;
    cudaMalloc(&dequantized, batch_size * hidden_dim_ * sizeof(double));

    // Dequantize codes to vectors
    codebook_->dequantize(codes, dequantized, batch_size, stream);

    // Project to output dimension
    int threads = 256;
    dim3 blocks((output_dim_ + threads - 1) / threads, batch_size);

    decoder_projection_kernel<<<blocks, threads, 0, stream>>>(
        dequantized, W_proj_, b_proj_, output,
        batch_size, hidden_dim_, output_dim_
    );

    // Cleanup
    cudaFree(dequantized);

    if (stream == 0) {
        cudaDeviceSynchronize();
    }
}

} // namespace sintellix
