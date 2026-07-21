#pragma once
//
// GEMM — the one place a third-party library is permitted.
//
// Dispatches to Apple Accelerate's cblas_sgemm/cblas_dgemm when available,
// otherwise a portable naive triple loop so the project builds anywhere.
// Everything else in the engine is written from scratch.
//
#include "csllm/common.hpp"

namespace csllm {

// Row-major C[M,N] = alpha * op(A) @ op(B) + beta * C
//   op(A) is [M,K] (or [K,M] transposed when trans_a), lda/ldb/ldc are row strides.
template <typename T>
void gemm(bool trans_a, bool trans_b, i64 m, i64 n, i64 k, T alpha, const T* a, i64 lda,
          const T* b, i64 ldb, T beta, T* c, i64 ldc);

// Convenience overload for tightly packed row-major matrices.
template <typename T>
void gemm_nn(i64 m, i64 n, i64 k, const T* a, const T* b, T* c) {
  gemm<T>(false, false, m, n, k, T(1), a, k, b, n, T(0), c, n);
}

const char* blas_backend_name();

}  // namespace csllm
