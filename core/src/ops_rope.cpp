#include <cmath>

#include "csllm/ops.hpp"

namespace csllm {

// Rotary position embedding.
//
// Convention: INTERLEAVED pairs — channels (2p, 2p+1) form each 2-D vector that
// gets rotated. (The other common convention splits the head in half, pairing d
// with d+Dh/2. They are not interchangeable; tests/reference.py must match this
// one.)
//
//   angle = (t + pos_offset) · theta^(−2p/Dh)
//   forward : y₀ = x₀c − x₁s,   y₁ = x₀s + x₁c
//   inverse : d₀ = g₀c + g₁s,   d₁ = −g₀s + g₁c
//
// The rotation matrix is orthogonal, so Rᵀ = R(−θ): the vector-Jacobian product
// is just this same kernel run with `inverse = true`. No Jacobian required.
template <typename T>
void rope_apply(T* x, i64 B, i64 H, i64 T_len, i64 head_dim, i64 pos_offset, T theta,
                bool inverse) {
  CSLLM_CHECK(head_dim % 2 == 0, "RoPE requires an even head_dim");
  const i64 half = head_dim / 2;

  for (i64 b = 0; b < B; ++b) {
    for (i64 h = 0; h < H; ++h) {
      for (i64 t = 0; t < T_len; ++t) {
        T* v = x + (((b * H + h) * T_len) + t) * head_dim;
        const T pos = static_cast<T>(t + pos_offset);
        for (i64 p = 0; p < half; ++p) {
          const T inv_freq =
              std::pow(theta, -static_cast<T>(2 * p) / static_cast<T>(head_dim));
          const T angle = pos * inv_freq;
          const T c = std::cos(angle);
          const T s = std::sin(angle);
          const T x0 = v[2 * p];
          const T x1 = v[2 * p + 1];
          if (inverse) {
            v[2 * p] = x0 * c + x1 * s;
            v[2 * p + 1] = -x0 * s + x1 * c;
          } else {
            v[2 * p] = x0 * c - x1 * s;
            v[2 * p + 1] = x0 * s + x1 * c;
          }
        }
      }
    }
  }
}

// Tape-aware wrapper so RoPE can be gradchecked in isolation.
template <typename T>
Tensor<T> rope(Arena& arena, const Tensor<T>& x, i64 pos_offset, T theta) {
  CSLLM_CHECK(x.shape.ndim == 4, "rope expects [B,H,T,Dh]");
  const i64 B = x.shape[0], H = x.shape[1], T_len = x.shape[2], dh = x.shape[3];

  const bool rg = any_requires_grad(x);
  Tensor<T> out = make_tensor<T>(arena, x.shape, rg);
  std::memcpy(out.data, x.data, static_cast<std::size_t>(x.numel()) * sizeof(T));
  rope_apply<T>(out.data, B, H, T_len, dh, pos_offset, theta, /*inverse=*/false);

  if (rg) {
    out.node_id = record<T>("rope", {x.node_id}, [x, out, B, H, T_len, dh, pos_offset, theta] {
      if (!x.grad) return;
      const auto n = static_cast<std::size_t>(x.numel());
      // Rotate the incoming gradient backwards, then accumulate.
      std::vector<T> tmp(out.grad, out.grad + n);
      rope_apply<T>(tmp.data(), B, H, T_len, dh, pos_offset, theta, /*inverse=*/true);
      for (std::size_t i = 0; i < n; ++i) x.grad[i] += tmp[i];
    });
  }
  return out;
}

template void rope_apply<f32>(f32*, i64, i64, i64, i64, i64, f32, bool);
template void rope_apply<f64>(f64*, i64, i64, i64, i64, i64, f64, bool);
template Tensor<f32> rope<f32>(Arena&, const Tensor<f32>&, i64, f32);
template Tensor<f64> rope<f64>(Arena&, const Tensor<f64>&, i64, f64);

}  // namespace csllm
