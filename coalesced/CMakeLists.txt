add_library(Coalesced coalesced.cu)

target_include_directories(Coalesced
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)