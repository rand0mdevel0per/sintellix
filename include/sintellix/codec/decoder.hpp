#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include "sintellix/codec/vqgan.hpp"
#include "sintellix/codec/cic_data.hpp"

namespace sintellix {

/**
 * Semantic Decoder
 * Decodes VQ codes to CIC data (universal output)
 */
class SemanticDecoder {
public:
    /**
     * Constructor
     * @param codebook Shared VQ codebook
     * @param output_dim Output dimension (e.g., 256×256 for neuron input)
     */
    SemanticDecoder(std::shared_ptr<VQCodebook> codebook, size_t output_dim);
    ~SemanticDecoder();

    // Disable copy
    SemanticDecoder(const SemanticDecoder&) = delete;
    SemanticDecoder& operator=(const SemanticDecoder&) = delete;

    /**
     * Initialize decoder
     */
    bool initialize();

    /**
     * Decode VQ codes to CIC data (contains emb only)
     * @param codes Input VQ codes
     * @param cic_data Output CIC data container (will contain emb)
     * @return Success status
     */
    bool decode(const std::vector<int>& codes, CICData& cic_data);

    /**
     * Decode VQ codes to embedding directly
     * @param codes Input VQ codes
     * @param emb Output embedding [512]
     * @return Success status
     */
    bool decode_to_emb(const std::vector<int>& codes, std::vector<double>& emb);

    /**
     * Decode VQ codes to text
     * @param codes Input VQ codes
     * @param text Output text
     * @return Success status
     */
    bool decode_to_text(const std::vector<int>& codes, std::string& text);

    /**
     * Decode batch of VQ codes to matrices
     * @param codes_batch Input VQ codes [batch_size × num_codes]
     * @param outputs Output matrices [batch_size × output_dim]
     * @return Success status
     */
    bool decode_batch_to_matrix(const std::vector<std::vector<int>>& codes_batch,
                                 std::vector<std::vector<double>>& outputs);

    /**
     * Get output dimension
     */
    size_t get_output_dim() const { return output_dim_; }

private:
    std::shared_ptr<VQCodebook> codebook_;
    std::unique_ptr<VQGANDecoder> vqgan_decoder_;
    size_t output_dim_;

    // GPU buffers
    int* codes_gpu_;           // VQ codes buffer
    double* output_gpu_;       // Output buffer

    // Vocabulary for text decoding (placeholder, may be removed)
    std::vector<std::string> vocabulary_;

    // Helper functions
    bool load_vocabulary();
};

} // namespace sintellix
