#include "sintellix/codec/cic_data.hpp"

namespace sintellix {

std::shared_ptr<CICData> CICData::create_with_src(const std::vector<uint8_t>& src_data) {
    auto cic = std::make_shared<CICData>();
    cic->src = src_data;
    return cic;
}

std::shared_ptr<CICData> CICData::create_with_emb(const std::vector<double>& embedding) {
    auto cic = std::make_shared<CICData>();
    cic->emb = embedding;
    return cic;
}

std::shared_ptr<CICData> CICData::create(const std::vector<uint8_t>& src_data,
                                          const std::vector<double>& embedding) {
    auto cic = std::make_shared<CICData>();
    cic->src = src_data;
    cic->emb = embedding;
    return cic;
}

std::shared_ptr<CICData> CICData::create_nested(const std::vector<std::shared_ptr<CICData>>& children) {
    auto cic = std::make_shared<CICData>();
    cic->nested = children;
    return cic;
}

bool CICData::has_metadata(const std::string& key) const {
    return metadata.find(key) != metadata.end();
}

std::string CICData::get_metadata(const std::string& key, const std::string& default_value) const {
    auto it = metadata.find(key);
    if (it != metadata.end()) {
        return it->second;
    }
    return default_value;
}

void CICData::set_metadata(const std::string& key, const std::string& value) {
    metadata[key] = value;
}

} // namespace sintellix
