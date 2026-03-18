set(TARGET ggml-rocket)

add_library(${TARGET} OBJECT
    ggml-rocket.cpp
)

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Try to find librocket
find_library(LIBROCKET rocket HINTS /usr/lib /usr/local/lib)

if(LIBROCKET)
    message(STATUS "Found librocket: ${LIBROCKET}")
    target_link_libraries(${TARGET} PUBLIC ${LIBROCKET})
else()
    message(WARNING "librocket not found - Rocket backend will not work without it")
    message(STATUS "Install librocket from https://github.com/johanvdb/librocket")
endif()

# Export for parent
set(GGML_ROCKET_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ggml-rocket.cpp PARENT_SCOPE)
set(GGML_ROCKET_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/ggml-rocket.h PARENT_SCOPE)

