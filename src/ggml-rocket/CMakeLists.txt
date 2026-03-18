set(TARGET ggml-rocket)

add_library(${TARGET} OBJECT
    ggml-rocket.cpp
)

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    /usr/local/include
)

# Find librocket
find_library(LIBROCKET rocket HINTS /usr/local/lib /usr/lib)

if(LIBROCKET)
    message(STATUS "Found librocket: ${LIBROCKET}")
    target_link_libraries(${TARGET} PUBLIC ${LIBROCKET})
else()
    message(FATAL_ERROR "librocket not found - Install from https://github.com/johanvdb/librocket")
endif()

# Export for parent
set(GGML_ROCKET_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ggml-rocket.cpp PARENT_SCOPE)
set(GGML_ROCKET_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/ggml-rocket.h PARENT_SCOPE)

