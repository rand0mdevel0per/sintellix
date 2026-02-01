#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace sintellix {

/**
 * CIC (Channel-in-Channel) Data Container
 * Universal data structure for NMDB communication
 *
 * CIC只包含两部分：
 * 1. src: 源数据（任意格式的原始bytes）
 * 2. emb: 语义embedding（512-dim CLIP embedding）
 *
 * 具体的模态转换（text/image/audio）由外围模块处理
 */
struct CICData {
    // 源数据（原始bytes，可以是文本、图像、音频等任意格式）
    std::vector<uint8_t> src;

    // 语义embedding（512-dim CLIP embedding）
    std::vector<double> emb;

    // 元数据（可选，用于描述src的格式等信息）
    std::map<std::string, std::string> metadata;

    // 嵌套CIC数据（可选，用于channel-in-channel）
    std::vector<std::shared_ptr<CICData>> nested;

    // Constructors
    CICData() = default;

    // Create CIC with src only
    static std::shared_ptr<CICData> create_with_src(const std::vector<uint8_t>& src_data);

    // Create CIC with emb only
    static std::shared_ptr<CICData> create_with_emb(const std::vector<double>& embedding);

    // Create CIC with both src and emb
    static std::shared_ptr<CICData> create(const std::vector<uint8_t>& src_data,
                                            const std::vector<double>& embedding);

    // Create nested CIC
    static std::shared_ptr<CICData> create_nested(const std::vector<std::shared_ptr<CICData>>& children);

    // Helper methods
    bool has_src() const { return !src.empty(); }
    bool has_emb() const { return !emb.empty(); }
    bool has_nested() const { return !nested.empty(); }

    bool has_metadata(const std::string& key) const;
    std::string get_metadata(const std::string& key, const std::string& default_value = "") const;
    void set_metadata(const std::string& key, const std::string& value);
};

} // namespace sintellix
