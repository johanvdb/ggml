message(STATUS "Using Rocket backend")

# Find librocket using pkg-config
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(LIBROCKET QUIET rocket)
endif()

# Fallback to find_library if pkg-config doesn't find it
if(NOT LIBROCKET_FOUND)
    find_library(LIBROCKET_LIBRARIES rocket HINTS /usr/local/lib /usr/lib)
    find_path(LIBROCKET_INCLUDE_DIRS rocket_interface.h HINTS /usr/local/include /usr/include)

    if(NOT LIBROCKET_LIBRARIES)
        message(FATAL_ERROR "librocket not found - Install from https://github.com/johanvdb/librocket")
    endif()
endif()

message(STATUS "Found librocket: ${LIBROCKET_LIBRARIES}")

# Use the standard GGML backend library macro
ggml_add_backend_library(ggml-rocket
                         ggml-rocket.cpp
                         ggml-rocket-ops.cpp
                        )

# Add include directories
target_include_directories(ggml-rocket PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${LIBROCKET_INCLUDE_DIRS}
)

# Link against librocket
target_link_libraries(ggml-rocket PUBLIC ${LIBROCKET_LIBRARIES})

