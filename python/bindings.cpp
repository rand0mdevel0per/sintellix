#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "sintellix/codec/cic_data.hpp"
#include "sintellix/codec/vqgan.hpp"
#include "sintellix/codec/encoder.hpp"
#include "sintellix/codec/decoder.hpp"
#include "sintellix/core/config.hpp"
#include "sintellix/core/neuron_model.cuh"

namespace py = pybind11;

PYBIND11_MODULE(_sintellix_native, m) {
    m.doc() = "Sintellix native C++/CUDA bindings";

    // CICData bindings
    py::class_<sintellix::CICData, std::shared_ptr<sintellix::CICData>>(m, "CICData")
        .def(py::init<>())
        .def_readwrite("src", &sintellix::CICData::src)
        .def_readwrite("emb", &sintellix::CICData::emb)
        .def_readwrite("metadata", &sintellix::CICData::metadata)
        .def_readwrite("nested", &sintellix::CICData::nested)
        .def("has_src", &sintellix::CICData::has_src)
        .def("has_emb", &sintellix::CICData::has_emb)
        .def("has_nested", &sintellix::CICData::has_nested)
        .def_static("create_with_src", &sintellix::CICData::create_with_src)
        .def_static("create_with_emb", &sintellix::CICData::create_with_emb)
        .def_static("create", &sintellix::CICData::create);

    // VQCodebook bindings
    py::class_<sintellix::VQCodebook, std::shared_ptr<sintellix::VQCodebook>>(m, "VQCodebook")
        .def(py::init<size_t, size_t>(), py::arg("codebook_size"), py::arg("embedding_dim"))
        .def("initialize", &sintellix::VQCodebook::initialize)
        .def("load_from_file", &sintellix::VQCodebook::load_from_file)
        .def("save_to_file", &sintellix::VQCodebook::save_to_file)
        .def("get_codebook_size", &sintellix::VQCodebook::get_codebook_size)
        .def("get_embedding_dim", &sintellix::VQCodebook::get_embedding_dim);

    // VQGANEncoder bindings
    py::class_<sintellix::VQGANEncoder>(m, "VQGANEncoder")
        .def(py::init<size_t, size_t, std::shared_ptr<sintellix::VQCodebook>>(),
             py::arg("input_dim"), py::arg("hidden_dim"), py::arg("codebook"))
        .def("initialize", &sintellix::VQGANEncoder::initialize)
        .def("get_input_dim", &sintellix::VQGANEncoder::get_input_dim);

    // VQGANDecoder bindings
    py::class_<sintellix::VQGANDecoder>(m, "VQGANDecoder")
        .def(py::init<size_t, size_t, std::shared_ptr<sintellix::VQCodebook>>(),
             py::arg("output_dim"), py::arg("hidden_dim"), py::arg("codebook"))
        .def("initialize", &sintellix::VQGANDecoder::initialize)
        .def("get_output_dim", &sintellix::VQGANDecoder::get_output_dim);

    // SemanticEncoder bindings
    py::class_<sintellix::SemanticEncoder>(m, "SemanticEncoder")
        .def(py::init<const std::string&, std::shared_ptr<sintellix::VQCodebook>>(),
             py::arg("model_path"), py::arg("codebook"))
        .def("initialize", &sintellix::SemanticEncoder::initialize)
        .def("encode", &sintellix::SemanticEncoder::encode)
        .def("encode_from_emb", &sintellix::SemanticEncoder::encode_from_emb)
        .def("encode_text", &sintellix::SemanticEncoder::encode_text)
        .def("encode_text_batch", &sintellix::SemanticEncoder::encode_text_batch);

    // SemanticDecoder bindings
    py::class_<sintellix::SemanticDecoder>(m, "SemanticDecoder")
        .def(py::init<std::shared_ptr<sintellix::VQCodebook>, size_t>(),
             py::arg("codebook"), py::arg("output_dim"))
        .def("initialize", &sintellix::SemanticDecoder::initialize)
        .def("decode", &sintellix::SemanticDecoder::decode)
        .def("decode_to_emb", &sintellix::SemanticDecoder::decode_to_emb)
        .def("decode_to_text", &sintellix::SemanticDecoder::decode_to_text)
        .def("get_output_dim", &sintellix::SemanticDecoder::get_output_dim);

    // ConfigManager bindings
    py::class_<sintellix::ConfigManager>(m, "ConfigManager")
        .def(py::init<>())
        .def("load_from_json", &sintellix::ConfigManager::loadFromJson)
        .def("load_from_proto", &sintellix::ConfigManager::loadFromProto)
        .def("save_to_json", &sintellix::ConfigManager::saveToJson)
        .def("save_to_proto", &sintellix::ConfigManager::saveToProto)
        .def("get_config", py::overload_cast<>(&sintellix::ConfigManager::getConfig),
             py::return_value_policy::reference_internal)
        .def_static("create_default", &sintellix::ConfigManager::createDefault);

    // KFEManager bindings
    py::class_<sintellix::KFEManager>(m, "KFEManager")
        .def(py::init<size_t>(), py::arg("max_slots") = 10000)
        .def("has_kfe", &sintellix::KFEManager::has_kfe)
        .def("get_slot_count", &sintellix::KFEManager::get_slot_count);

    // NeuronModel bindings
    py::class_<sintellix::NeuronModel>(m, "NeuronModel")
        .def(py::init<const sintellix::NeuronConfig&>())
        .def("initialize", &sintellix::NeuronModel::initialize)
        .def("update_parameters", &sintellix::NeuronModel::update_parameters)
        .def("save_state", &sintellix::NeuronModel::save_state)
        .def("load_state", &sintellix::NeuronModel::load_state)
        .def("get_kfe_manager", &sintellix::NeuronModel::get_kfe_manager,
             py::return_value_policy::reference_internal);
}
