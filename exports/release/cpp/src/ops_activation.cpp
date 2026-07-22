#include <cmath>

#include "csllm/gemm.hpp"
#include "csllm/ops.hpp"

namespace csllm {
namespace {

template <typename T>
inline T sigmoid(T x) {
  return T(1) / (T(1) + std::exp(-x));
}

}  // namespace

// ── silu ────────────────────────────────────────────────────────────────────
// y = x·σ(x);   dx = dy·σ·(1 + x·(1−σ))
template <typename T>
Tensor<T> silu(Arena& arena, const Tensor<T>& x) {
  const i64 n = x.numel();
  const bool rg = any_requires_grad(x);
  Tensor<T> out = make_tensor<T>(arena, x.shape, rg);
  for (i64 i = 0; i < n; ++i) out.data[i] = x.data[i] * sigmoid(x.data[i]);

  if (rg) {
    out.node_id = record<T>("silu", {x.node_id}, [x, out, n] {
      if (!x.grad) return;
      for (i64 i = 0; i < n; ++i) {
        const T s = sigmoid(x.data[i]);
        x.grad[i] += out.grad[i] * s * (T(1) + x.data[i] * (T(1) - s));
      }
    });
  }
  return out;
}

// ── swiglu (fused) ──────────────────────────────────────────────────────────
//   a = x·wg,  b = x·wu,  h = silu(a) ⊙ b,  out = h·wd
// Fused rather than composed from primitives so the intermediates a, b, h are
// materialised once and the three matmul backwards share them.
template <typename T>
Tensor<T> swiglu(Arena& arena, const Tensor<T>& x, const Tensor<T>& wg, const Tensor<T>& wu,
                 const Tensor<T>& wd) {
  CSLLM_CHECK(x.shape.ndim == 2, "swiglu expects x to be 2-D");
  const i64 rows = x.shape[0], C = x.shape[1], Hf = wg.shape[1];
  CSLLM_CHECK(wg.shape[0] == C && wu.shape[0] == C, "swiglu: wg/wu first dim must equal C");
  CSLLM_CHECK(wu.shape[1] == Hf, "swiglu: wg and wu must share the hidden dim");
  CSLLM_CHECK(wd.shape[0] == Hf && wd.shape[1] == C, "swiglu: wd must be [H,C]");

  const bool rg = any_requires_grad(x, wg, wu, wd);

  T* a = arena.alloc_n<T>(static_cast<std::size_t>(rows * Hf));
  T* b = arena.alloc_n<T>(static_cast<std::size_t>(rows * Hf));
  T* h = arena.alloc_n<T>(static_cast<std::size_t>(rows * Hf));

  gemm<T>(false, false, rows, Hf, C, T(1), x.data, C, wg.data, Hf, T(0), a, Hf);
  gemm<T>(false, false, rows, Hf, C, T(1), x.data, C, wu.data, Hf, T(0), b, Hf);
  for (i64 i = 0; i < rows * Hf; ++i) h[i] = a[i] * sigmoid(a[i]) * b[i];

  Tensor<T> out = make_tensor<T>(arena, Shape{rows, C}, rg);
  gemm<T>(false, false, rows, C, Hf, T(1), h, Hf, wd.data, C, T(0), out.data, C);

  if (rg) {
    // Scratch for dh / da / db, reused across the backward.
    T* dh = arena.alloc_n<T>(static_cast<std::size_t>(rows * Hf));
    T* da = arena.alloc_n<T>(static_cast<std::size_t>(rows * Hf));
    T* db = arena.alloc_n<T>(static_cast<std::size_t>(rows * Hf));

    out.node_id = record<T>(
        "swiglu", {x.node_id, wg.node_id, wu.node_id, wd.node_id},
        [x, wg, wu, wd, out, a, b, h, dh, da, db, rows, C, Hf] {
          // dh[rows,Hf] = dout[rows,C] @ wd[Hf,C]ᵀ
          gemm<T>(false, true, rows, Hf, C, T(1), out.grad, C, wd.data, C, T(0), dh, Hf);
          // dwd[Hf,C] = h[rows,Hf]ᵀ @ dout[rows,C]
          if (wd.grad) {
            gemm<T>(true, false, Hf, C, rows, T(1), h, Hf, out.grad, C, T(1), wd.grad, C);
          }

          for (i64 i = 0; i < rows * Hf; ++i) {
            const T s = sigmoid(a[i]);
            const T sa = a[i] * s;                            // silu(a)
            const T dsilu = s * (T(1) + a[i] * (T(1) - s));   // silu'(a)
            da[i] = dh[i] * b[i] * dsilu;
            db[i] = dh[i] * sa;
          }

          if (x.grad) {
            // dx += da @ wgᵀ + db @ wuᵀ
            gemm<T>(false, true, rows, C, Hf, T(1), da, Hf, wg.data, Hf, T(1), x.grad, C);
            gemm<T>(false, true, rows, C, Hf, T(1), db, Hf, wu.data, Hf, T(1), x.grad, C);
          }
          if (wg.grad) {
            gemm<T>(true, false, C, Hf, rows, T(1), x.data, C, da, Hf, T(1), wg.grad, Hf);
          }
          if (wu.grad) {
            gemm<T>(true, false, C, Hf, rows, T(1), x.data, C, db, Hf, T(1), wu.grad, Hf);
          }
        });
  }
  return out;
}

template Tensor<f32> silu<f32>(Arena&, const Tensor<f32>&);
template Tensor<f64> silu<f64>(Arena&, const Tensor<f64>&);
template Tensor<f32> swiglu<f32>(Arena&, const Tensor<f32>&, const Tensor<f32>&, const Tensor<f32>&,
                                 const Tensor<f32>&);
template Tensor<f64> swiglu<f64>(Arena&, const Tensor<f64>&, const Tensor<f64>&, const Tensor<f64>&,
                                 const Tensor<f64>&);

}  // namespace csllm
