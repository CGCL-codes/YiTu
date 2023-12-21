cmake_minimum_required(VERSION 2.8.12)
project(YiTu_GP LANGUAGES CXX CUDA)

add_subdirectory(pybind11)

include_directories(
    "/usr/lib/x86_64-linux-gnu/"
    "/usr/lib/x86_64-linux-gnu/openmpi/include"
)

set(dso_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/shared/globals.hpp
    ${PROJECT_SOURCE_DIR}/shared/gpu_kernels.cu
    ${PROJECT_SOURCE_DIR}/shared/partitioner.cu
    ${PROJECT_SOURCE_DIR}/shared/subgraph_generator.cu
    ${PROJECT_SOURCE_DIR}/shared/subgraph.cu
    ${PROJECT_SOURCE_DIR}/shared/subway_utilities.cpp
    ${PROJECT_SOURCE_DIR}/shared/argument_parsing.cu
    ${PROJECT_SOURCE_DIR}/shared/graph.cu
    ${PROJECT_SOURCE_DIR}/shared/timer.cpp

    ${PROJECT_SOURCE_DIR}/ndp/bc_sig_async.cu
    ${PROJECT_SOURCE_DIR}/ndp/bfs_sig_async.cu
    ${PROJECT_SOURCE_DIR}/ndp/cc_sig_async.cu
    ${PROJECT_SOURCE_DIR}/ndp/pr_sig_async.cu
    ${PROJECT_SOURCE_DIR}/ndp/sssp_sig_async.cu
    ${PROJECT_SOURCE_DIR}/ndp/sswp_sig_async.cu
)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
set(CMAKE_CUDA_ARCHITECTURES 50 52 53)
endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)

pybind11_add_module(YiTu_GP YiTu_GP.cpp ${dso_SOURCE_FILES})
pybind11_add_module(Tools ${PROJECT_SOURCE_DIR}/tools/converter.cpp ${PROJECT_SOURCE_DIR}/shared/globals.hpp)

target_link_libraries(YiTu_GP PUBLIC /usr/lib/x86_64-linux-gnu/libmpi.so)