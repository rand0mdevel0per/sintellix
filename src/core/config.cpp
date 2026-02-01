#include "sintellix/core/config.hpp"
#include <fstream>
#include <iostream>

namespace sintellix {

NeuronConfig ConfigManager::createDefault() {
    NeuronConfig config;

    // Basic dimensions
    config.set_dim(256);
    config.set_num_heads(8);
    config.set_temporal_frames(8);

    // Grid size
    auto* grid = config.mutable_grid_size();
    grid->set_x(32);
    grid->set_y(32);
    grid->set_z(32);

    // Module switches (all enabled by default)
    auto* modules = config.mutable_modules();
    modules->set_enable_multi_head(true);
    modules->set_enable_global_aggregation(true);
    modules->set_enable_noise_filter(true);
    modules->set_enable_temporal_attention(true);
    modules->set_enable_fxaa_layer(true);
    modules->set_enable_kfe_memory(true);
    modules->set_enable_ssm(true);
    modules->set_enable_rwkv(true);
    modules->set_enable_ddpm(true);

    // Storage configuration
    auto* storage = config.mutable_storage();
    storage->set_gpu_cache_size_mb(8192);
    storage->set_ram_cache_size_mb(32768);
    storage->set_disk_cache_path("/tmp/sintellix_cache");
    storage->set_eviction_threshold(100);

    // Optimization configuration
    auto* opt = config.mutable_optimization();
    opt->set_use_tensor_cores(true);
    opt->set_use_fused_kernels(true);
    opt->set_use_kv_cache(true);
    opt->set_kv_cache_size(512);

    // Noise filter configuration
    auto* noise = config.mutable_noise_filter();
    noise->set_ema_alpha(0.1f);
    noise->set_threshold_multiplier(2.0f);

    // Global aggregation configuration
    auto* global = config.mutable_global_aggregation();
    global->set_top_k(64);
    global->set_attention_dropout(0.1f);

    return config;
}

bool ConfigManager::loadFromJson(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << path << std::endl;
        return false;
    }

    nlohmann::json j;
    file >> j;

    jsonToProto(j, config_);
    return true;
}

bool ConfigManager::saveToJson(const std::string& path) const {
    nlohmann::json j = protoToJson(config_);

    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << path << std::endl;
        return false;
    }

    file << j.dump(2);
    return true;
}

void ConfigManager::jsonToProto(const nlohmann::json& j, NeuronConfig& config) {
    // Basic dimensions
    if (j.contains("neuron")) {
        auto& n = j["neuron"];
        if (n.contains("dim")) config.set_dim(n["dim"]);
        if (n.contains("num_heads")) config.set_num_heads(n["num_heads"]);
        if (n.contains("temporal_frames")) config.set_temporal_frames(n["temporal_frames"]);

        if (n.contains("grid_size")) {
            auto* grid = config.mutable_grid_size();
            grid->set_x(n["grid_size"][0]);
            grid->set_y(n["grid_size"][1]);
            grid->set_z(n["grid_size"][2]);
        }
    }

    // Module switches
    if (j.contains("modules")) {
        auto& m = j["modules"];
        auto* modules = config.mutable_modules();
        if (m.contains("enable_multi_head"))
            modules->set_enable_multi_head(m["enable_multi_head"]);
        if (m.contains("enable_global_aggregation"))
            modules->set_enable_global_aggregation(m["enable_global_aggregation"]);
        if (m.contains("enable_noise_filter"))
            modules->set_enable_noise_filter(m["enable_noise_filter"]);
        if (m.contains("enable_temporal_attention"))
            modules->set_enable_temporal_attention(m["enable_temporal_attention"]);
        if (m.contains("enable_fxaa_layer"))
            modules->set_enable_fxaa_layer(m["enable_fxaa_layer"]);
    }

    // Storage configuration
    if (j.contains("storage")) {
        auto& s = j["storage"];
        auto* storage = config.mutable_storage();
        if (s.contains("gpu_cache_size_mb"))
            storage->set_gpu_cache_size_mb(s["gpu_cache_size_mb"]);
        if (s.contains("ram_cache_size_mb"))
            storage->set_ram_cache_size_mb(s["ram_cache_size_mb"]);
        if (s.contains("disk_cache_path"))
            storage->set_disk_cache_path(s["disk_cache_path"]);
    }
}

nlohmann::json ConfigManager::protoToJson(const NeuronConfig& config) const {
    nlohmann::json j;

    // Basic dimensions
    j["neuron"]["dim"] = config.dim();
    j["neuron"]["num_heads"] = config.num_heads();
    j["neuron"]["temporal_frames"] = config.temporal_frames();
    j["neuron"]["grid_size"] = {
        config.grid_size().x(),
        config.grid_size().y(),
        config.grid_size().z()
    };

    // Module switches
    j["modules"]["enable_multi_head"] = config.modules().enable_multi_head();
    j["modules"]["enable_global_aggregation"] = config.modules().enable_global_aggregation();
    j["modules"]["enable_noise_filter"] = config.modules().enable_noise_filter();
    j["modules"]["enable_temporal_attention"] = config.modules().enable_temporal_attention();
    j["modules"]["enable_fxaa_layer"] = config.modules().enable_fxaa_layer();

    // Storage configuration
    j["storage"]["gpu_cache_size_mb"] = config.storage().gpu_cache_size_mb();
    j["storage"]["ram_cache_size_mb"] = config.storage().ram_cache_size_mb();
    j["storage"]["disk_cache_path"] = config.storage().disk_cache_path();

    return j;
}

bool ConfigManager::loadFromProto(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open proto file: " << path << std::endl;
        return false;
    }

    return config_.ParseFromIstream(&file);
}

bool ConfigManager::saveToProto(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << path << std::endl;
        return false;
    }

    return config_.SerializeToOstream(&file);
}

} // namespace sintellix
