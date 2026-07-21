#pragma once
//
// Shared primitives: scalar aliases, error checking, and build introspection.
//
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace csllm {

using f32 = float;
using f64 = double;
using i32 = std::int32_t;
using i64 = std::int64_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// Thrown for all recoverable errors; pybind11 translates it to a Python
// RuntimeError at the binding boundary.
class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& what) : std::runtime_error(what) {}
};

#define CSLLM_CHECK(cond, msg)                                              \
  do {                                                                      \
    if (!(cond)) {                                                          \
      throw ::csllm::Error(std::string(__FILE__) + ":" +                     \
                           std::to_string(__LINE__) + " CSLLM_CHECK(" #cond \
                           ") failed: " + (msg));                           \
    }                                                                       \
  } while (0)

#define CSLLM_NOT_IMPLEMENTED() \
  throw ::csllm::Error(std::string(__func__) + " is not implemented yet (Phase 2)")

struct BuildInfo {
  std::string version;
  std::string blas_backend;   // "Accelerate" or "naive"
  bool accelerate_enabled;
  bool fast_math;             // always false — see rules.md
  unsigned hardware_threads;
  long cxx_standard;
};

BuildInfo build_info();

}  // namespace csllm
