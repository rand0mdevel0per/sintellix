#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include "sintellix/codec/vqgan.hpp"
#include "sintellix/codec/cic_data.hpp"

namespace sintellix {

/**
 * Semantic Encoder
 * Encodes text/image/multimodal to discrete VQ codes using CLIP
 */
class SemanticEncoder {
public:
    /**
     * Constructor
     * @param model_path Path to CLIP model (ONNX format)
     * @param codebook Shared VQ codebook
     */
    SemanticEncoder(const std::string& model_path, std::shared_ptr<VQCodebook> codebook);
    ~SemanticEncoder();

    // Disable copy
    SemanticEncoder(const SemanticEncoder&) = delete;
    SemanticEncoder& operator=(const SemanticEncoder&) = delete;

    /**
     * Initialize encoder (load CLIP model)
     */
    bool initialize();

    /**
     * Encode CIC data to VQ codes
     * CIC must contain emb (512-dim CLIP embedding)
     * If CIC has nested data, will fuse all embeddings
     * @param cic_data CIC data container (must have emb or nested)
     * @param codes Output VQ codes
     * @param max_codes Maximum number of codes to generate
     * @return Number of codes generated
     */
    size_t encode(const CICData& cic_data, std::vector<int>& codes, size_t max_codes = 256);

    /**
     * Encode embedding directly to VQ codes
     * @param embedding Input embedding [512]
     * @param codes Output VQ codes
     * @param max_codes Maximum number of codes to generate
     * @return Number of codes generated
     */
    size_t encode_from_emb(const std::vector<double>& embedding, std::vector<int>& codes, size_t max_codes = 256);

    /**
     * Encode text to VQ codes
     * @param text Input text
     * @param codes Output VQ codes
     * @param max_codes Maximum number of codes to generate
     * @return Number of codes generated
     */
    size_t encode_text(const std::string& text, std::vector<int>& codes, size_t max_codes = 256);

    /**
     * Encode text batch to VQ codes
     * @param texts Input texts
     * @param codes Output VQ codes [batch_size × max_codes]
     * @param max_codes Maximum number of codes per text
     * @return Number of codes generated per text
     */
    std::vector<size_t> encode_text_batch(const std::vector<std::string>& texts,
                                           std::vector<std::vector<int>>& codes,
                                           size_t max_codes = 256);

private:
    std::string model_path_;
    std::shared_ptr<VQCodebook> codebook_;
    std::unique_ptr<VQGANEncoder> vqgan_encoder_;

    // CLIP model handle (ONNX Runtime)
    void* clip_session_;  // Opaque pointer to ONNX session

    // GPU buffers
    double* clip_output_gpu_;  // CLIP output buffer [batch_size × 512]
    int* codes_gpu_;           // VQ codes buffer [batch_size × max_codes]

    // Helper functions
    bool load_clip_model();
    bool run_clip_text_inference(const std::vector<std::string>& texts, double* output, size_t batch_size);
    bool run_clip_image_inference(const uint8_t* image_data, int width, int height, double* output);

    // CIC data processing helpers
    bool extract_embedding(const CICData& cic_data, std::vector<double>& embedding);
    bool fuse_embeddings(const std::vector<std::vector<double>>& embeddings, std::vector<double>& fused);
};

} // namespace sintellix
