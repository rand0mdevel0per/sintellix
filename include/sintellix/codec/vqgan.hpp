#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace sintellix {

/**
 * VQ-GAN Codebook
 * Stores discrete codes for vector quantization
 */
class VQCodebook {
public:
    /**
     * Constructor
     * @param codebook_size Number of codes in codebook
     * @param embedding_dim Dimension of each code vector
     */
    VQCodebook(size_t codebook_size, size_t embedding_dim);
    ~VQCodebook();

    // Disable copy
    VQCodebook(const VQCodebook&) = delete;
    VQCodebook& operator=(const VQCodebook&) = delete;

    /**
     * Initialize codebook with random values
     */
    bool initialize();

    /**
     * Load codebook from file
     * @param path Path to codebook file
     */
    bool load_from_file(const std::string& path);

    /**
     * Save codebook to file
     * @param path Path to save codebook
     */
    bool save_to_file(const std::string& path);

    /**
     * Quantize continuous vectors to discrete codes
     * @param vectors Input vectors [batch_size × embedding_dim]
     * @param codes Output codes [batch_size]
     * @param batch_size Number of vectors
     * @param stream CUDA stream
     */
    void quantize(const double* vectors, int* codes, size_t batch_size, cudaStream_t stream = 0);

    /**
     * Dequantize discrete codes to continuous vectors
     * @param codes Input codes [batch_size]
     * @param vectors Output vectors [batch_size × embedding_dim]
     * @param batch_size Number of codes
     * @param stream CUDA stream
     */
    void dequantize(const int* codes, double* vectors, size_t batch_size, cudaStream_t stream = 0);

    /**
     * Get codebook size
     */
    size_t get_codebook_size() const { return codebook_size_; }

    /**
     * Get embedding dimension
     */
    size_t get_embedding_dim() const { return embedding_dim_; }

    /**
     * Get GPU pointer to codebook
     */
    double* get_codebook_ptr() { return codebook_gpu_; }

private:
    size_t codebook_size_;      // Number of codes (e.g., 2048, 4096, 8192)
    size_t embedding_dim_;      // Dimension of each code vector
    double* codebook_gpu_;      // GPU memory for codebook [codebook_size × embedding_dim]
};

/**
 * VQ-GAN Encoder
 * Encodes continuous vectors to discrete codes
 */
class VQGANEncoder {
public:
    /**
     * Constructor
     * @param input_dim Input dimension (e.g., 1024 for E5-Large)
     * @param hidden_dim Hidden layer dimension
     * @param codebook Shared codebook
     */
    VQGANEncoder(size_t input_dim, size_t hidden_dim, std::shared_ptr<VQCodebook> codebook);
    ~VQGANEncoder();

    // Disable copy
    VQGANEncoder(const VQGANEncoder&) = delete;
    VQGANEncoder& operator=(const VQGANEncoder&) = delete;

    /**
     * Initialize encoder weights
     */
    bool initialize();

    /**
     * Encode input to discrete codes
     * @param input Input vectors [batch_size × input_dim]
     * @param codes Output codes [batch_size]
     * @param batch_size Number of inputs
     * @param stream CUDA stream
     */
    void encode(const double* input, int* codes, size_t batch_size, cudaStream_t stream = 0);

    /**
     * Get input dimension
     */
    size_t get_input_dim() const { return input_dim_; }

private:
    size_t input_dim_;          // Input dimension
    size_t hidden_dim_;         // Hidden dimension
    std::shared_ptr<VQCodebook> codebook_;  // Shared codebook

    // Encoder weights
    double* W_proj_;            // Projection matrix [input_dim × hidden_dim]
    double* b_proj_;            // Projection bias [hidden_dim]
};

/**
 * VQ-GAN Decoder
 * Decodes discrete codes to continuous output
 */
class VQGANDecoder {
public:
    /**
     * Constructor
     * @param output_dim Output dimension (e.g., 256×256 for neuron input)
     * @param hidden_dim Hidden layer dimension
     * @param codebook Shared codebook
     */
    VQGANDecoder(size_t output_dim, size_t hidden_dim, std::shared_ptr<VQCodebook> codebook);
    ~VQGANDecoder();

    // Disable copy
    VQGANDecoder(const VQGANDecoder&) = delete;
    VQGANDecoder& operator=(const VQGANDecoder&) = delete;

    /**
     * Initialize decoder weights
     */
    bool initialize();

    /**
     * Decode discrete codes to output
     * @param codes Input codes [batch_size]
     * @param output Output vectors [batch_size × output_dim]
     * @param batch_size Number of codes
     * @param stream CUDA stream
     */
    void decode(const int* codes, double* output, size_t batch_size, cudaStream_t stream = 0);

    /**
     * Get output dimension
     */
    size_t get_output_dim() const { return output_dim_; }

private:
    size_t output_dim_;         // Output dimension
    size_t hidden_dim_;         // Hidden dimension
    std::shared_ptr<VQCodebook> codebook_;  // Shared codebook

    // Decoder weights
    double* W_proj_;            // Projection matrix [hidden_dim × output_dim]
    double* b_proj_;            // Projection bias [output_dim]
};

} // namespace sintellix
