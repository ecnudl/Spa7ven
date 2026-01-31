cmake_minimum_required(VERSION 3.14)
project(ann_lib VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

# OpenMP support (optional)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling parallel support")
else()
    message(STATUS "OpenMP not found, using std::thread fallback")
endif()

# Header-only library
add_library(ann INTERFACE)
target_include_directories(ann INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(ann INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(ann INTERFACE ANN_USE_OPENMP)
endif()

# Sanity test executable
add_executable(sanity_test tests/sanity_test.cpp)
target_link_libraries(sanity_test PRIVATE ann)

if(OpenMP_CXX_FOUND)
    target_link_libraries(sanity_test PRIVATE OpenMP::OpenMP_CXX)
endif()

enable_testing()
add_test(NAME sanity_test COMMAND sanity_test)
