#include "csllm/gemm.hpp"

#if CSLLM_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace csllm {
namespace {

template <typename T>
void gemm_naive(bool trans_a, bool trans_b, i64 m, i64 n, i64 k, T alpha, const T* a, i64 lda,
                const T* b, i64 ldb, T beta, T* c, i64 ldc) {
  for (i64 i = 0; i < m; ++i) {
    for (i64 j = 0; j < n; ++j) {
      T sum = T(0);
      for (i64 p = 0; p < k; ++p) {
        const T av = trans_a ? a[p * lda + i] : a[i * lda + p];
        const T bv = trans_b ? b[j * ldb + p] : b[p * ldb + j];
        sum += av * bv;
      }
      T* dst = &c[i * ldc + j];
      *dst = alpha * sum + (beta == T(0) ? T(0) : beta * *dst);
    }
  }
}

}  // namespace

const char* blas_backend_name() { return CSLLM_BLAS_BACKEND; }

#if CSLLM_USE_ACCELERATE

template <>
void gemm<f32>(bool trans_a, bool trans_b, i64 m, i64 n, i64 k, f32 alpha, const f32* a, i64 lda,
               const f32* b, i64 ldb, f32 beta, f32* c, i64 ldc) {
  cblas_sgemm(CblasRowMajor, trans_a ? CblasTrans : CblasNoTrans,
              trans_b ? CblasTrans : CblasNoTrans, static_cast<int>(m), static_cast<int>(n),
              static_cast<int>(k), alpha, a, static_cast<int>(lda), b, static_cast<int>(ldb), beta,
              c, static_cast<int>(ldc));
}

template <>
void gemm<f64>(bool trans_a, bool trans_b, i64 m, i64 n, i64 k, f64 alpha, const f64* a, i64 lda,
               const f64* b, i64 ldb, f64 beta, f64* c, i64 ldc) {
  cblas_dgemm(CblasRowMajor, trans_a ? CblasTrans : CblasNoTrans,
              trans_b ? CblasTrans : CblasNoTrans, static_cast<int>(m), static_cast<int>(n),
              static_cast<int>(k), alpha, a, static_cast<int>(lda), b, static_cast<int>(ldb), beta,
              c, static_cast<int>(ldc));
}

#else

template <>
void gemm<f32>(bool trans_a, bool trans_b, i64 m, i64 n, i64 k, f32 alpha, const f32* a, i64 lda,
               const f32* b, i64 ldb, f32 beta, f32* c, i64 ldc) {
  gemm_naive<f32>(trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void gemm<f64>(bool trans_a, bool trans_b, i64 m, i64 n, i64 k, f64 alpha, const f64* a, i64 lda,
               const f64* b, i64 ldb, f64 beta, f64* c, i64 ldc) {
  gemm_naive<f64>(trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif

}  // namespace csllm
