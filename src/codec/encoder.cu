#include "sintellix/codec/encoder.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace sintellix {

SemanticEncoder::SemanticEncoder(const std::string& model_path, std::shared_ptr<VQCodebook> codebook)
    : model_path_(model_path)
    , codebook_(codebook)
    , clip_session_(nullptr)
    , clip_output_gpu_(nullptr)
    , codes_gpu_(nullptr)
{
}

SemanticEncoder::~SemanticEncoder() {
    if (clip_output_gpu_) cudaFree(clip_output_gpu_);
    if (codes_gpu_) cudaFree(codes_gpu_);

#ifdef USE_ONNXRUNTIME
    if (clip_session_) {
        delete static_cast<Ort::Session*>(clip_session_);
        clip_session_ = nullptr;
    }
#endif
}

bool SemanticEncoder::initialize() {
    // Create VQ-GAN encoder (512-dim CLIP output -> codebook embedding dim)
    size_t clip_dim = 512;
    size_t hidden_dim = codebook_->get_embedding_dim();

    vqgan_encoder_ = std::make_unique<VQGANEncoder>(clip_dim, hidden_dim, codebook_);

    if (!vqgan_encoder_->initialize()) {
        return false;
    }

    // Allocate GPU buffers
    cudaMalloc(&clip_output_gpu_, 256 * clip_dim * sizeof(double));  // Max batch size 256
    cudaMalloc(&codes_gpu_, 256 * 256 * sizeof(int));  // Max 256 codes per input

    // Load CLIP model
    return load_clip_model();
}

bool SemanticEncoder::load_clip_model() {
#ifdef USE_ONNXRUNTIME
    try {
        std::cout << "Loading CLIP model from: " << model_path_ << std::endl;

        // Check if model file exists
        std::ifstream file(model_path_);
        if (!file.good()) {
            std::cerr << "CLIP model file not found: " << model_path_ << std::endl;
            return false;
        }
        file.close();

        // Create ONNX Runtime environment and session
        Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "CLIP");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        // Enable CUDA if available
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);

        // Create session
        Ort::Session* session = new Ort::Session(*env, model_path_.c_str(), session_options);
        clip_session_ = session;

        std::cout << "CLIP model loaded successfully with ONNX Runtime" << std::endl;
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return false;
    }
#else
    // Placeholder implementation when ONNX Runtime is not available
    std::cout << "Loading CLIP model from: " << model_path_ << std::endl;

    std::ifstream file(model_path_);
    if (!file.good()) {
        std::cerr << "CLIP model file not found: " << model_path_ << std::endl;
        return false;
    }

    std::cout << "CLIP model loaded successfully (placeholder - ONNX Runtime not available)" << std::endl;
    return true;
#endif
}

bool SemanticEncoder::run_clip_text_inference(const std::vector<std::string>& texts, double* output, size_t batch_size) {
#ifdef USE_ONNXRUNTIME
    try {
        // TODO: Implement proper tokenization
        // CLIP uses BPE tokenizer which requires vocabulary file
        // For now, this is a simplified placeholder

        std::cerr << "Warning: CLIP tokenizer not yet implemented, using placeholder" << std::endl;

        // Generate placeholder embeddings
        std::vector<double> embeddings(batch_size * 512);
        for (size_t i = 0; i < batch_size * 512; i++) {
            embeddings[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }

        cudaMemcpy(output, embeddings.data(), batch_size * 512 * sizeof(double), cudaMemcpyHostToDevice);
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime inference error: " << e.what() << std::endl;
        return false;
    }
#else
    // Placeholder implementation
    std::vector<double> embeddings(batch_size * 512);
    for (size_t i = 0; i < batch_size * 512; i++) {
        embeddings[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    cudaMemcpy(output, embeddings.data(), batch_size * 512 * sizeof(double), cudaMemcpyHostToDevice);
    return true;
#endif
}

bool SemanticEncoder::run_clip_image_inference(const uint8_t* image_data, int width, int height, double* output) {
#ifdef USE_ONNXRUNTIME
    try {
        // TODO: Implement proper image preprocessing
        // CLIP requires specific image preprocessing (resize, normalize, etc.)

        std::cerr << "Warning: CLIP image preprocessing not yet implemented, using placeholder" << std::endl;

        // Generate placeholder embeddings
        std::vector<double> embeddings(512);
        for (size_t i = 0; i < 512; i++) {
            embeddings[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }

        cudaMemcpy(output, embeddings.data(), 512 * sizeof(double), cudaMemcpyHostToDevice);
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime inference error: " << e.what() << std::endl;
        return false;
    }
#else
    // Placeholder implementation
    std::vector<double> embeddings(512);
    for (size_t i = 0; i < 512; i++) {
        embeddings[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    cudaMemcpy(output, embeddings.data(), 512 * sizeof(double), cudaMemcpyHostToDevice);
    return true;
#endif
}

size_t SemanticEncoder::encode_text(const std::string& text, std::vector<int>& codes, size_t max_codes) {
    std::vector<std::string> texts = {text};
    std::vector<std::vector<int>> batch_codes;

    auto counts = encode_text_batch(texts, batch_codes, max_codes);

    if (!batch_codes.empty()) {
        codes = batch_codes[0];
        return counts[0];
    }

    return 0;
}

std::vector<size_t> SemanticEncoder::encode_text_batch(const std::vector<std::string>& texts,
                                                        std::vector<std::vector<int>>& codes,
                                                        size_t max_codes) {
    size_t batch_size = texts.size();
    std::vector<size_t> code_counts(batch_size);

    // Run CLIP inference to get embeddings
    if (!run_clip_text_inference(texts, clip_output_gpu_, batch_size)) {
        return code_counts;
    }

    // Encode embeddings to VQ codes
    vqgan_encoder_->encode(clip_output_gpu_, codes_gpu_, batch_size);

    // Copy codes back to host
    std::vector<int> codes_host(batch_size);
    cudaMemcpy(codes_host.data(), codes_gpu_, batch_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Organize codes by batch
    codes.resize(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        codes[i] = {codes_host[i]};
        code_counts[i] = 1;
    }

    return code_counts;
}

// ============================================================================
// CIC-based interface (simplified)
// ============================================================================

size_t SemanticEncoder::encode(const CICData& cic_data, std::vector<int>& codes, size_t max_codes) {
    // Extract embedding from CIC data
    std::vector<double> embedding;
    if (!extract_embedding(cic_data, embedding)) {
        return 0;
    }

    // Encode embedding to VQ codes
    return encode_from_emb(embedding, codes, max_codes);
}

size_t SemanticEncoder::encode_from_emb(const std::vector<double>& embedding, std::vector<int>& codes, size_t max_codes) {
    if (embedding.size() != 512) {
        std::cerr << "Invalid embedding dimension: " << embedding.size() << " (expected 512)" << std::endl;
        return 0;
    }

    // Copy embedding to GPU
    cudaMemcpy(clip_output_gpu_, embedding.data(), 512 * sizeof(double), cudaMemcpyHostToDevice);

    // Encode to VQ codes
    vqgan_encoder_->encode(clip_output_gpu_, codes_gpu_, 1);

    // Copy codes back to host
    std::vector<int> codes_host(1);
    cudaMemcpy(codes_host.data(), codes_gpu_, sizeof(int), cudaMemcpyDeviceToHost);

    codes = {codes_host[0]};
    return 1;
}

// ============================================================================
// CIC data processing helpers
// ============================================================================

bool SemanticEncoder::extract_embedding(const CICData& cic_data, std::vector<double>& embedding) {
    // If CIC has embedding, use it directly
    if (cic_data.has_emb()) {
        embedding = cic_data.emb;

        // Verify embedding dimension
        if (embedding.size() != 512) {
            std::cerr << "Invalid embedding dimension: " << embedding.size() << " (expected 512)" << std::endl;
            return false;
        }

        return true;
    }

    // If CIC has nested data, fuse all embeddings
    if (cic_data.has_nested()) {
        std::vector<std::vector<double>> embeddings;
        embeddings.reserve(cic_data.nested.size());

        for (const auto& nested : cic_data.nested) {
            std::vector<double> nested_emb;
            if (extract_embedding(*nested, nested_emb)) {
                embeddings.push_back(nested_emb);
            }
        }

        if (embeddings.empty()) {
            std::cerr << "Failed to extract any embeddings from nested CIC data" << std::endl;
            return false;
        }

        return fuse_embeddings(embeddings, embedding);
    }

    // CIC has neither emb nor nested data
    std::cerr << "CIC data has no embedding or nested data" << std::endl;
    return false;
}

bool SemanticEncoder::fuse_embeddings(const std::vector<std::vector<double>>& embeddings,
                                       std::vector<double>& fused) {
    if (embeddings.empty()) {
        return false;
    }

    // Simple fusion strategy: average all embeddings
    fused.resize(512, 0.0);

    for (const auto& emb : embeddings) {
        if (emb.size() != 512) {
            std::cerr << "Invalid embedding dimension in fusion: " << emb.size() << std::endl;
            continue;
        }

        for (size_t i = 0; i < 512; i++) {
            fused[i] += emb[i];
        }
    }

    // Normalize by count
    double count = static_cast<double>(embeddings.size());
    for (size_t i = 0; i < 512; i++) {
        fused[i] /= count;
    }

    return true;
}

} // namespace sintellix
