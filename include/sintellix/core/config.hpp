#pragma once

#include <string>
#include <memory>
#include <nlohmann/json.hpp>
#include "neuron_config.pb.h"

namespace sintellix {

/**
 * Configuration manager for Sintellix
 * Supports loading from JSON/TOML and Protobuf
 */
class ConfigManager {
public:
    ConfigManager() = default;
    ~ConfigManager() = default;

    // Load configuration from JSON file
    bool loadFromJson(const std::string& path);

    // Load configuration from Protobuf
    bool loadFromProto(const std::string& path);

    // Save configuration to JSON file
    bool saveToJson(const std::string& path) const;

    // Save configuration to Protobuf
    bool saveToProto(const std::string& path) const;

    // Get configuration
    const NeuronConfig& getConfig() const { return config_; }
    NeuronConfig& getConfig() { return config_; }

    // Create default configuration
    static NeuronConfig createDefault();

private:
    NeuronConfig config_;

    // Convert JSON to Protobuf
    void jsonToProto(const nlohmann::json& j, NeuronConfig& config);

    // Convert Protobuf to JSON
    nlohmann::json protoToJson(const NeuronConfig& config) const;
};

} // namespace sintellix
