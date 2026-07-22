# Standalone build of the CSLLM C++ engine, extracted from an export bundle.
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
#
# No third-party dependencies. Accelerate provides BLAS on Apple platforms;
# elsewhere the portable GEMM in src/gemm.cpp is used instead.
cmake_minimum_required(VERSION 3.20)
project(csllm_engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CSLLM_BLAS_BACKEND "naive")
if(APPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate)
  if(ACCELERATE_FRAMEWORK)
    set(CSLLM_BLAS_BACKEND "Accelerate")
  endif()
endif()

file(GLOB CSLLM_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(csllm_engine STATIC ${CSLLM_SOURCES})
target_include_directories(csllm_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(Threads REQUIRED)
target_link_libraries(csllm_engine PUBLIC Threads::Threads)

target_compile_definitions(csllm_engine PUBLIC
  CSLLM_VERSION="${PROJECT_VERSION}"
  CSLLM_BLAS_BACKEND="${CSLLM_BLAS_BACKEND}")

if(CSLLM_BLAS_BACKEND STREQUAL "Accelerate")
  target_link_libraries(csllm_engine PUBLIC ${ACCELERATE_FRAMEWORK})
  # ACCELERATE_NEW_LAPACK selects the modern (non-deprecated) cblas_* prototypes;
  # without it recent SDKs fail the build on deprecation.
  target_compile_definitions(csllm_engine PUBLIC
    CSLLM_USE_ACCELERATE=1 ACCELERATE_NEW_LAPACK=1)
else()
  target_compile_definitions(csllm_engine PUBLIC CSLLM_USE_ACCELERATE=0)
endif()

# -ffast-math is deliberately NOT set: it licenses reassociation and breaks the
# engine's NaN/Inf guards and reproducibility.
target_compile_options(csllm_engine PRIVATE -O3 -funroll-loops -fno-math-errno -Wall -Wextra)

message(STATUS "csllm_engine: BLAS backend = ${CSLLM_BLAS_BACKEND}")
