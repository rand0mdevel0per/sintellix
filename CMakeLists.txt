cmake_minimum_required(VERSION 3.18)
project(sintellix VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Fetch nmdb as external dependency
include(FetchContent)
FetchContent_Declare(
    nmdb
    GIT_REPOSITORY https://github.com/rand0mdevel0per/nmdb.git
    GIT_TAG master
)
FetchContent_MakeAvailable(nmdb)

# Add nmdb include directories
include_directories(${nmdb_SOURCE_DIR}/include)
include_directories(${nmdb_BINARY_DIR})

# Options
option(USE_HIP "Use HIP instead of CUDA" OFF)
option(BUILD_PYTHON "Build Python bindings" OFF)
option(BUILD_TESTS "Build tests" OFF)

# Find dependencies
find_package(CUDAToolkit REQUIRED)

# Optional dependencies
find_package(Protobuf)
find_package(nlohmann_json 3.2.0)
find_package(zstd)

# ONNX Runtime (for CLIP model inference)
find_package(onnxruntime CONFIG)
if(NOT onnxruntime_FOUND)
    message(STATUS "ONNX Runtime not found, will use placeholder implementation")
endif()

# Protobuf generation (optional)
if(Protobuf_FOUND)
    set(PROTO_FILES
        proto/neuron_config.proto
        proto/model_state.proto
        proto/embedding_config.proto
    )
    protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
    message(STATUS "Protobuf found, enabling serialization support")
else()
    set(PROTO_SRCS "")
    set(PROTO_HDRS "")
    message(WARNING "Protobuf not found, serialization support disabled")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PROTOBUF_INCLUDE_DIRS}
)

# Source files
set(SINTELLIX_SOURCES
    # Core neuron implementation
    src/core/neuron.cu
    src/core/neuron_model.cu
    src/core/multi_gpu.cu
    src/core/config.cpp

    # Storage
    src/storage/tiered_storage.cu

    # Codec files
    src/codec/cic_data.cpp
    src/codec/vqgan.cu
    src/codec/encoder.cu
    src/codec/decoder.cu

    # CIC channel operations
    src/cic_channel_ops.cpp
    src/model_cic_processor.cpp
    src/embedding_config_loader.cpp

    ${PROTO_SRCS}
)

set(SINTELLIX_HEADERS
    include/sintellix/core/neuron.cuh
    include/sintellix/core/neuron_model.cuh
    include/sintellix/core/multi_gpu.cuh
    include/sintellix/core/config.hpp
    include/sintellix/codec/cic_data.hpp
    include/sintellix/codec/vqgan.hpp
    include/sintellix/codec/encoder.hpp
    include/sintellix/codec/decoder.hpp
    include/sintellix/storage/tiered_storage.cuh
    include/sintellix/cic_channel_ops.h
    include/sintellix/model_cic_processor.h
    include/sintellix/embedding_config_loader.h
)

# Main library
add_library(sintellix SHARED ${SINTELLIX_SOURCES} ${SINTELLIX_HEADERS})

target_link_libraries(sintellix
    CUDA::cudart
    CUDA::cublas
)

# Optional libraries
if(Protobuf_FOUND)
    target_link_libraries(sintellix protobuf::libprotobuf)
endif()

if(nlohmann_json_FOUND)
    target_link_libraries(sintellix nlohmann_json::nlohmann_json)
endif()

if(zstd_FOUND)
    target_link_libraries(sintellix zstd::libzstd_shared)
endif()

if(onnxruntime_FOUND)
    target_link_libraries(sintellix onnxruntime::onnxruntime)
    target_compile_definitions(sintellix PRIVATE USE_ONNXRUNTIME)
endif()

# CUDA compilation flags
set_target_properties(sintellix PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86;89"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

target_compile_options(sintellix PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --expt-relaxed-constexpr
        -Xcompiler=-fPIC
    >
)

# Installation
install(TARGETS sintellix
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/sintellix
    DESTINATION include
)

# Python bindings
if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
