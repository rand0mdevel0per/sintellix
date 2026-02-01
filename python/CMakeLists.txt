cmake_minimum_required(VERSION 3.18)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Python extension module
pybind11_add_module(_sintellix_native
    bindings.cpp
)

target_link_libraries(_sintellix_native PRIVATE
    sintellix
    CUDA::cudart
    CUDA::cublas
)

target_include_directories(_sintellix_native PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Install to python package directory
install(TARGETS _sintellix_native
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/python/sintellix
)
