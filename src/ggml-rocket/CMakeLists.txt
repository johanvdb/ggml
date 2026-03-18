message(STATUS "Using Rocket backend")

# Find librocket first
find_library(LIBROCKET rocket HINTS /usr/local/lib /usr/lib)

if(NOT LIBROCKET)
    message(FATAL_ERROR "librocket not found - Install from https://github.com/johanvdb/librocket")
endif()

message(STATUS "Found librocket: ${LIBROCKET}")

# Use the standard GGML backend library macro
ggml_add_backend_library(ggml-rocket
                         ggml-rocket.cpp
                        )

# Add include directories
target_include_directories(ggml-rocket PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    /usr/local/include
)

# Link against librocket
target_link_libraries(ggml-rocket PUBLIC ${LIBROCKET})

