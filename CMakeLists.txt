cmake_minimum_required(VERSION 3.15)
project(cudaAnalyze CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)

add_executable(cudaAnalyze main.cu info.cu )


set_target_properties(cudaAnalyze PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
