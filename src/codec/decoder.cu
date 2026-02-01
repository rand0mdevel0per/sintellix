#include "sintellix/codec/decoder.hpp"
#include <cuda_runtime.h>
#include <iostream>

namespace sintellix {

SemanticDecoder::SemanticDecoder(std::shared_ptr<VQCodebook> codebook, size_t output_dim)
    : codebook_(codebook)
    , output_dim_(output_dim)
    , codes_gpu_(nullptr)
    , output_gpu_(nullptr)
{
}

SemanticDecoder::~SemanticDecoder() {
    if (codes_gpu_) cudaFree(codes_gpu_);
    if (output_gpu_) cudaFree(output_gpu_);
}

bool SemanticDecoder::initialize() {
    // Create VQ-GAN decoder
    size_t hidden_dim = codebook_->get_embedding_dim();
    vqgan_decoder_ = std::make_unique<VQGANDecoder>(output_dim_, hidden_dim, codebook_);

    if (!vqgan_decoder_->initialize()) {
        return false;
    }

    // Allocate GPU buffers
    cudaMalloc(&codes_gpu_, 256 * sizeof(int));  // Max 256 codes
    cudaMalloc(&output_gpu_, output_dim_ * sizeof(double));

    // Load vocabulary for text decoding
    return load_vocabulary();
}

bool SemanticDecoder::load_vocabulary() {
    // TODO: Load vocabulary from file
    // For now, use placeholder vocabulary
    vocabulary_ = {"<pad>", "<unk>", "<sos>", "<eos>"};

    // Add some common words as placeholder
    std::vector<std::string> common_words = {
        "the", "a", "an", "is", "are", "was", "were",
        "hello", "world", "test", "example", "data"
    };

    vocabulary_.insert(vocabulary_.end(), common_words.begin(), common_words.end());

    std::cout << "Vocabulary loaded: " << vocabulary_.size() << " tokens" << std::endl;
    return true;
}

// ============================================================================
// CIC-based interface (simplified)
// ============================================================================

bool SemanticDecoder::decode(const std::vector<int>& codes, CICData& cic_data) {
    // Decode to embedding
    std::vector<double> emb;
    if (!decode_to_emb(codes, emb)) {
        return false;
    }

    // Create CIC with embedding only
    cic_data.emb = emb;
    return true;
}

bool SemanticDecoder::decode_to_emb(const std::vector<int>& codes, std::vector<double>& emb) {
    if (codes.empty()) {
        return false;
    }

    // Copy codes to GPU
    cudaMemcpy(codes_gpu_, codes.data(), codes.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Decode using VQ-GAN decoder
    vqgan_decoder_->decode(codes_gpu_, output_gpu_, 1);

    // Copy decoded output back to host (this is the embedding)
    emb.resize(output_dim_);
    cudaMemcpy(emb.data(), output_gpu_, output_dim_ * sizeof(double), cudaMemcpyDeviceToHost);

    return true;
}

// Legacy convenience methods (may be removed later)
bool SemanticDecoder::decode_to_text(const std::vector<int>& codes, std::string& text) {
    // TODO: This should be handled by external modules
    // For now, use simple vocabulary lookup
    text.clear();
    for (int code : codes) {
        if (code >= 0 && code < vocabulary_.size()) {
            text += vocabulary_[code] + " ";
        } else {
            text += "<unk> ";
        }
    }
    return true;
}

} // namespace sintellix
