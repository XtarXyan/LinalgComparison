cmake_minimum_required(VERSION 3.10)

project(LinalgComparison)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math -funroll-loops")

# single-threaded execution
add_definitions(-DEIGEN_DONT_PARALLELIZE)

find_package(Eigen3 CONFIG REQUIRED)

add_executable(LinalgComparison benchmark.cpp)

target_link_libraries(LinalgComparison PRIVATE Eigen3::Eigen)

# Define DATA_DIR as a quoted C string literal so we can use it in C++ source code.
target_compile_definitions(LinalgComparison PRIVATE "DATA_DIR=\"${CMAKE_SOURCE_DIR}/Data\"")

# additional Eigen optimizations
target_compile_definitions(LinalgComparison PRIVATE 
    EIGEN_NO_DEBUG
    EIGEN_FAST_MATH
    EIGEN_UNROLLING_LIMIT=100
)

# MSVC: heavy template instantiation (Eigen) can produce object files with
# many COMDAT sections and trigger "number of sections exceeded" (C1128).
# Add /bigobj to allow more sections in object files for MSVC builds.
if (MSVC)
    target_compile_options(LinalgComparison PRIVATE /bigobj)
endif()