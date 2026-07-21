#include <algorithm>
#include <cmath>

#include "csllm/gemm.hpp"
#include "csllm/ops.hpp"

namespace csllm {

// ── matmul ──────────────────────────────────────────────────────────────────
// C[M,N] = X[M,K] @ W[K,N];  dX = dC·Wᵀ,  dW = Xᵀ·dC
template <typename T>
Tensor<T> matmul(Arena& arena, const Tensor<T>& x, const Tensor<T>& w) {
  CSLLM_CHECK(x.shape.ndim == 2 && w.shape.ndim == 2, "matmul expects 2-D tensors");
  const i64 M = x.shape[0], K = x.shape[1], N = w.shape[1];
  CSLLM_CHECK(w.shape[0] == K, "matmul inner dims disagree: " + x.shape.str() + " @ " +
                                   w.shape.str());

  const bool rg = any_requires_grad(x, w);
  Tensor<T> out = make_tensor<T>(arena, Shape{M, N}, rg);
  gemm<T>(false, false, M, N, K, T(1), x.data, K, w.data, N, T(0), out.data, N);

  if (rg) {
    out.node_id = record<T>("matmul", {x.node_id, w.node_id}, [x, w, out, M, N, K] {
      // dX[M,K] = dC[M,N] @ W[K,N]ᵀ   (beta=1 to accumulate)
      if (x.grad) {
        gemm<T>(false, true, M, K, N, T(1), out.grad, N, w.data, N, T(1), x.grad, K);
      }
      // dW[K,N] = X[M,K]ᵀ @ dC[M,N]
      if (w.grad) {
        gemm<T>(true, false, K, N, M, T(1), x.data, K, out.grad, N, T(1), w.grad, N);
      }
    });
  }
  return out;
}

// ── matmul_bt ───────────────────────────────────────────────────────────────
// C[M,N] = X[M,K] @ W[N,K]ᵀ;  dX = dC·W,  dW = dCᵀ·X
template <typename T>
Tensor<T> matmul_bt(Arena& arena, const Tensor<T>& x, const Tensor<T>& w) {
  CSLLM_CHECK(x.shape.ndim == 2 && w.shape.ndim == 2, "matmul_bt expects 2-D tensors");
  const i64 M = x.shape[0], K = x.shape[1], N = w.shape[0];
  CSLLM_CHECK(w.shape[1] == K, "matmul_bt inner dims disagree: " + x.shape.str() + " @ " +
                                   w.shape.str() + "ᵀ");

  const bool rg = any_requires_grad(x, w);
  Tensor<T> out = make_tensor<T>(arena, Shape{M, N}, rg);
  gemm<T>(false, true, M, N, K, T(1), x.data, K, w.data, K, T(0), out.data, N);

  if (rg) {
    out.node_id = record<T>("matmul_bt", {x.node_id, w.node_id}, [x, w, out, M, N, K] {
      // dX[M,K] = dC[M,N] @ W[N,K]
      if (x.grad) {
        gemm<T>(false, false, M, K, N, T(1), out.grad, N, w.data, K, T(1), x.grad, K);
      }
      // dW[N,K] = dC[M,N]ᵀ @ X[M,K]
      if (w.grad) {
        gemm<T>(true, false, N, K, M, T(1), out.grad, N, x.data, K, T(1), w.grad, K);
      }
    });
  }
  return out;
}

// ── add ─────────────────────────────────────────────────────────────────────
template <typename T>
Tensor<T> add(Arena& arena, const Tensor<T>& x, const Tensor<T>& y) {
  CSLLM_CHECK(x.shape == y.shape, "add: shape mismatch " + x.shape.str() + " vs " + y.shape.str());
  const i64 n = x.numel();

  const bool rg = any_requires_grad(x, y);
  Tensor<T> out = make_tensor<T>(arena, x.shape, rg);
  for (i64 i = 0; i < n; ++i) out.data[i] = x.data[i] + y.data[i];

  if (rg) {
    out.node_id = record<T>("add", {x.node_id, y.node_id}, [x, y, out, n] {
      if (x.grad) {
        for (i64 i = 0; i < n; ++i) x.grad[i] += out.grad[i];
      }
      if (y.grad) {
        for (i64 i = 0; i < n; ++i) y.grad[i] += out.grad[i];
      }
    });
  }
  return out;
}

// ── rmsnorm ─────────────────────────────────────────────────────────────────
// y = gain ⊙ x / r,  r = sqrt(mean(x²) + eps)
// s   = Σⱼ(dyⱼ·gainⱼ·xⱼ)
// dxⱼ = gainⱼ·dyⱼ/r − xⱼ·s/(C·r³)
// dgⱼ = Σᵢ dyᵢⱼ·xᵢⱼ/rᵢ
template <typename T>
Tensor<T> rmsnorm(Arena& arena, const Tensor<T>& x, const Tensor<T>& gain, T eps) {
  CSLLM_CHECK(x.shape.ndim == 2, "rmsnorm expects x to be 2-D");
  const i64 rows = x.shape[0], C = x.shape[1];
  CSLLM_CHECK(gain.numel() == C, "rmsnorm: gain length must equal x's last dim");

  const bool rg = any_requires_grad(x, gain);
  Tensor<T> out = make_tensor<T>(arena, x.shape, rg);

  // The per-row reciprocal RMS is saved for backward rather than recomputed.
  T* inv_r = arena.alloc_n<T>(static_cast<std::size_t>(rows));

  for (i64 i = 0; i < rows; ++i) {
    const T* xr = x.data + i * C;
    T sumsq = T(0);
    for (i64 j = 0; j < C; ++j) sumsq += xr[j] * xr[j];
    const T r = std::sqrt(sumsq / static_cast<T>(C) + eps);
    inv_r[i] = T(1) / r;
    T* orow = out.data + i * C;
    for (i64 j = 0; j < C; ++j) orow[j] = gain.data[j] * xr[j] * inv_r[i];
  }

  if (rg) {
    out.node_id = record<T>("rmsnorm", {x.node_id, gain.node_id},
                            [x, gain, out, inv_r, rows, C] {
                              for (i64 i = 0; i < rows; ++i) {
                                const T* xr = x.data + i * C;
                                const T* dy = out.grad + i * C;
                                const T ir = inv_r[i];

                                T s = T(0);
                                for (i64 j = 0; j < C; ++j) s += dy[j] * gain.data[j] * xr[j];

                                if (x.grad) {
                                  T* dx = x.grad + i * C;
                                  const T scale = s * ir * ir * ir / static_cast<T>(C);
                                  for (i64 j = 0; j < C; ++j) {
                                    dx[j] += gain.data[j] * dy[j] * ir - xr[j] * scale;
                                  }
                                }
                                if (gain.grad) {
                                  for (i64 j = 0; j < C; ++j) gain.grad[j] += dy[j] * xr[j] * ir;
                                }
                              }
                            });
  }
  return out;
}

// ── softmax_causal ──────────────────────────────────────────────────────────
// scores[n, T, T]; key j is visible to query i only when j ≤ i.
// dx = y ⊙ (dy − Σ(dy⊙y)), and masked entries stay exactly zero because y = 0
// there — which is also why the attention backward must re-apply the mask.
template <typename T>
Tensor<T> softmax_causal(Arena& arena, const Tensor<T>& scores) {
  CSLLM_CHECK(scores.shape.ndim == 3, "softmax_causal expects [n, T, T]");
  const i64 n = scores.shape[0], Tq = scores.shape[1], Tk = scores.shape[2];
  CSLLM_CHECK(Tq == Tk, "softmax_causal expects square attention matrices");

  const bool rg = any_requires_grad(scores);
  Tensor<T> out = make_tensor<T>(arena, scores.shape, rg);

  for (i64 m = 0; m < n; ++m) {
    for (i64 i = 0; i < Tq; ++i) {
      const T* s = scores.data + (m * Tq + i) * Tk;
      T* o = out.data + (m * Tq + i) * Tk;

      T maxv = s[0];
      for (i64 j = 1; j <= i; ++j) maxv = std::max(maxv, s[j]);

      T sum = T(0);
      for (i64 j = 0; j <= i; ++j) {
        o[j] = std::exp(s[j] - maxv);
        sum += o[j];
      }
      const T inv = T(1) / sum;
      for (i64 j = 0; j <= i; ++j) o[j] *= inv;
      for (i64 j = i + 1; j < Tk; ++j) o[j] = T(0);
    }
  }

  if (rg) {
    out.node_id = record<T>("softmax_causal", {scores.node_id}, [scores, out, n, Tq, Tk] {
      if (!scores.grad) return;
      for (i64 m = 0; m < n; ++m) {
        for (i64 i = 0; i < Tq; ++i) {
          const T* y = out.data + (m * Tq + i) * Tk;
          const T* dy = out.grad + (m * Tq + i) * Tk;
          T* dx = scores.grad + (m * Tq + i) * Tk;

          T dot = T(0);
          for (i64 j = 0; j <= i; ++j) dot += dy[j] * y[j];
          for (i64 j = 0; j <= i; ++j) dx[j] += y[j] * (dy[j] - dot);
          // j > i stays zero: masked positions must receive no gradient.
        }
      }
    });
  }
  return out;
}

#define CSLLM_INSTANTIATE(T)                                                                \
  template Tensor<T> matmul<T>(Arena&, const Tensor<T>&, const Tensor<T>&);                 \
  template Tensor<T> matmul_bt<T>(Arena&, const Tensor<T>&, const Tensor<T>&);              \
  template Tensor<T> add<T>(Arena&, const Tensor<T>&, const Tensor<T>&);                    \
  template Tensor<T> rmsnorm<T>(Arena&, const Tensor<T>&, const Tensor<T>&, T);             \
  template Tensor<T> softmax_causal<T>(Arena&, const Tensor<T>&);
CSLLM_INSTANTIATE(f32)
CSLLM_INSTANTIATE(f64)
#undef CSLLM_INSTANTIATE

}  // namespace csllm
