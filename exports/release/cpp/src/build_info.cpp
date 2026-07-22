#include "csllm/common.hpp"

#include <thread>

#include "csllm/gemm.hpp"

namespace csllm {

BuildInfo build_info() {
  BuildInfo info;
  info.version = CSLLM_VERSION;
  info.blas_backend = blas_backend_name();
#if CSLLM_USE_ACCELERATE
  info.accelerate_enabled = true;
#else
  info.accelerate_enabled = false;
#endif
  // -ffast-math is banned project-wide; assert that nobody has re-enabled it.
#ifdef __FAST_MATH__
  info.fast_math = true;
#else
  info.fast_math = false;
#endif
  info.hardware_threads = std::thread::hardware_concurrency();
  info.cxx_standard = __cplusplus;
  return info;
}

}  // namespace csllm
