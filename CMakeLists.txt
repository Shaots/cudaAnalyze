cmake_minimum_required(VERSION 3.15)
project(cudaAnalyze CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(coalesced)

add_executable(cudaAnalyze main.cu info.cu test.cu)

target_link_libraries(cudaAnalyze PUBLIC Coalesced)

set_target_properties(cudaAnalyze PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
