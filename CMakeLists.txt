cmake_minimum_required(VERSION 3.20)

project(csllm
  VERSION 1.0.0
  DESCRIPTION "CSLLM — an autoregressive Transformer built from scratch in C++"
  LANGUAGES CXX)

# ─────────────────────────────────────────────────────────────────────────────
# Global settings
# ─────────────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(CSLLM_NATIVE_ARCH "Tune for the host CPU (-mcpu=native)" ON)
option(CSLLM_BUILD_TESTS "Build the C++ smoke-test executable"  OFF)

# NOTE: -ffast-math is deliberately NOT used. It licenses reassociation, which
# breaks NaN/Inf guards and makes results non-reproducible — both fatal for the
# hand-written autograd this project depends on. See memory-bank/rules.md.
add_library(csllm_flags INTERFACE)
target_compile_options(csllm_flags INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -fno-math-errno>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -funroll-loops>)

if(CSLLM_NATIVE_ARCH)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mcpu=native" CSLLM_HAS_MCPU_NATIVE)
  if(CSLLM_HAS_MCPU_NATIVE)
    target_compile_options(csllm_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-mcpu=native>)
  else()
    check_cxx_compiler_flag("-march=native" CSLLM_HAS_MARCH_NATIVE)
    if(CSLLM_HAS_MARCH_NATIVE)
      target_compile_options(csllm_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
    endif()
  endif()
endif()

# ─────────────────────────────────────────────────────────────────────────────
# BLAS backend — Apple Accelerate, with a portable naive fallback
# ─────────────────────────────────────────────────────────────────────────────
set(CSLLM_BLAS_BACKEND "naive")
if(APPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate)
  if(ACCELERATE_FRAMEWORK)
    set(CSLLM_BLAS_BACKEND "Accelerate")
  endif()
endif()

add_library(csllm_blas INTERFACE)
if(CSLLM_BLAS_BACKEND STREQUAL "Accelerate")
  target_link_libraries(csllm_blas INTERFACE ${ACCELERATE_FRAMEWORK})
  # ACCELERATE_NEW_LAPACK selects the modern (ILP64-capable, non-deprecated)
  # prototypes; without it recent SDKs emit deprecation errors for cblas_*.
  target_compile_definitions(csllm_blas INTERFACE
    CSLLM_USE_ACCELERATE=1 ACCELERATE_NEW_LAPACK=1)
else()
  target_compile_definitions(csllm_blas INTERFACE CSLLM_USE_ACCELERATE=0)
endif()

find_package(Threads REQUIRED)

# ─────────────────────────────────────────────────────────────────────────────
# Core static library (shared by the Python module and any C++ test binary)
# ─────────────────────────────────────────────────────────────────────────────
add_library(csllm_core STATIC
  core/src/arena.cpp
  core/src/tensor.cpp
  core/src/autograd.cpp
  core/src/gemm.cpp
  core/src/json.cpp
  core/src/ops_norm.cpp
  core/src/ops_rope.cpp
  core/src/ops_activation.cpp
  core/src/ops_loss.cpp
  core/src/attention.cpp
  core/src/model.cpp
  core/src/optim.cpp
  core/src/sampler.cpp
  core/src/serialize.cpp
  core/src/threadpool.cpp
  core/src/build_info.cpp)

target_include_directories(csllm_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core/include>)

target_compile_definitions(csllm_core PUBLIC
  CSLLM_VERSION="${PROJECT_VERSION}"
  CSLLM_BLAS_BACKEND="${CSLLM_BLAS_BACKEND}")

target_link_libraries(csllm_core
  PUBLIC  csllm_blas Threads::Threads
  PRIVATE csllm_flags)

# ─────────────────────────────────────────────────────────────────────────────
# Python bindings
# ─────────────────────────────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Three-tier pybind11 discovery so the build never hard-blocks:
#   1. an already-installed CMake config package
#   2. the pip-installed `pybind11` module's bundled cmakedir
#   3. FetchContent from GitHub as a last resort
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET RESULT_VARIABLE _pybind11_probe)
  if(_pybind11_probe EQUAL 0 AND EXISTS "${_pybind11_cmakedir}")
    find_package(pybind11 CONFIG QUIET PATHS "${_pybind11_cmakedir}" NO_DEFAULT_PATH)
  endif()
endif()
if(NOT pybind11_FOUND)
  # pybind11 >= 3.0 is REQUIRED: the 2.13.x series predates Python 3.14 and does
  # not support its C-API changes. Do not lower this tag.
  message(STATUS "pybind11 not found locally — fetching from GitHub")
  include(FetchContent)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v3.0.4
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(_csllm_core MODULE core/bindings/py_module.cpp)
target_link_libraries(_csllm_core PRIVATE csllm_core csllm_flags)
target_compile_definitions(_csllm_core PRIVATE CSLLM_VERSION="${PROJECT_VERSION}")

# scikit-build-core packages whatever `install()` places into the wheel.
install(TARGETS _csllm_core LIBRARY DESTINATION csllm)

if(CSLLM_BUILD_TESTS)
  add_executable(csllm_smoke core/tests/smoke.cpp)
  target_link_libraries(csllm_smoke PRIVATE csllm_core csllm_flags)
endif()

message(STATUS "───────────────────────────────────────────────")
message(STATUS "  CSLLM ${PROJECT_VERSION}")
message(STATUS "  BLAS backend : ${CSLLM_BLAS_BACKEND}")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Python       : ${Python3_VERSION} (${Python3_EXECUTABLE})")
message(STATUS "───────────────────────────────────────────────")
