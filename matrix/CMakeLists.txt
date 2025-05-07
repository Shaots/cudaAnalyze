add_library(Matrix matrix.cu)

target_include_directories(Matrix INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})