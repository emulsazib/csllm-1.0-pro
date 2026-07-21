#include <algorithm>
#include <cmath>

#include "csllm/ops.hpp"

namespace csllm {

// ── embedding ───────────────────────────────────────────────────────────────
// Gather rows of table[V,C]; backward is a scatter-add.
//
// Under weight tying this table's gradient ALSO receives the lm_head
// contribution from matmul_bt. Both paths accumulate into the same buffer
// (beta=1 / +=), which is exactly what makes tying correct — and dropping
// either path yields a model that still trains but converges wrong.
template <typename T>
Tensor<T> embedding(Arena& arena, const Tensor<T>& table, const i32* ids, i64 n) {
  CSLLM_CHECK(table.shape.ndim == 2, "embedding expects table[V,C]");
  const i64 V = table.shape[0], C = table.shape[1];

  // The caller's id buffer may not outlive the tape, so keep our own copy.
  i32* ids_copy = arena.alloc_n<i32>(static_cast<std::size_t>(n));
  for (i64 i = 0; i < n; ++i) {
    CSLLM_CHECK(ids[i] >= 0 && ids[i] < V,
                "embedding: token id " + std::to_string(ids[i]) + " out of range [0," +
                    std::to_string(V) + ")");
    ids_copy[i] = ids[i];
  }

  const bool rg = any_requires_grad(table);
  Tensor<T> out = make_tensor<T>(arena, Shape{n, C}, rg);
  for (i64 i = 0; i < n; ++i) {
    std::memcpy(out.data + i * C, table.data + static_cast<i64>(ids_copy[i]) * C,
                static_cast<std::size_t>(C) * sizeof(T));
  }

  if (rg) {
    out.node_id = record<T>("embedding", {table.node_id}, [table, out, ids_copy, n, C] {
      if (!table.grad) return;
      for (i64 i = 0; i < n; ++i) {
        T* dst = table.grad + static_cast<i64>(ids_copy[i]) * C;
        const T* src = out.grad + i * C;
        for (i64 j = 0; j < C; ++j) dst[j] += src[j];
      }
    });
  }
  return out;
}

// ── cross_entropy (fused log-softmax + NLL) ─────────────────────────────────
// Fused for numerical stability (max subtraction) and to avoid materialising a
// second [n,V] tensor. Returns a scalar tensor so the loss is an ordinary tape
// node the caller can seed with 1.0.
//   dlogits = (p − onehot)/n
template <typename T>
Tensor<T> cross_entropy(Arena& arena, const Tensor<T>& logits, const i32* targets, i64 n) {
  CSLLM_CHECK(logits.shape.ndim == 2, "cross_entropy expects logits[n,V]");
  CSLLM_CHECK(logits.shape[0] == n, "cross_entropy: logits rows must equal n");
  const i64 V = logits.shape[1];

  i32* tgt = arena.alloc_n<i32>(static_cast<std::size_t>(n));
  for (i64 i = 0; i < n; ++i) {
    CSLLM_CHECK(targets[i] >= 0 && targets[i] < V,
                "cross_entropy: target " + std::to_string(targets[i]) + " out of range");
    tgt[i] = targets[i];
  }

  const bool rg = any_requires_grad(logits);
  // Softmax probabilities are saved for backward.
  T* probs = arena.alloc_n<T>(static_cast<std::size_t>(n * V));

  T total = T(0);
  for (i64 i = 0; i < n; ++i) {
    const T* row = logits.data + i * V;
    T* p = probs + i * V;

    T maxv = row[0];
    for (i64 j = 1; j < V; ++j) maxv = std::max(maxv, row[j]);

    T sum = T(0);
    for (i64 j = 0; j < V; ++j) {
      p[j] = std::exp(row[j] - maxv);
      sum += p[j];
    }
    const T inv = T(1) / sum;
    for (i64 j = 0; j < V; ++j) p[j] *= inv;

    // loss_i = logsumexp(row) − row[target]
    total += std::log(sum) + maxv - row[tgt[i]];
  }

  Tensor<T> out = make_tensor<T>(arena, Shape{1}, rg);
  out.data[0] = total / static_cast<T>(n);

  if (rg) {
    out.node_id = record<T>("cross_entropy", {logits.node_id}, [logits, out, probs, tgt, n, V] {
      if (!logits.grad) return;
      const T scale = out.grad[0] / static_cast<T>(n);
      for (i64 i = 0; i < n; ++i) {
        const T* p = probs + i * V;
        T* dl = logits.grad + i * V;
        for (i64 j = 0; j < V; ++j) dl[j] += scale * p[j];
        dl[tgt[i]] -= scale;
      }
    });
  }
  return out;
}

template Tensor<f32> embedding<f32>(Arena&, const Tensor<f32>&, const i32*, i64);
template Tensor<f64> embedding<f64>(Arena&, const Tensor<f64>&, const i32*, i64);
template Tensor<f32> cross_entropy<f32>(Arena&, const Tensor<f32>&, const i32*, i64);
template Tensor<f64> cross_entropy<f64>(Arena&, const Tensor<f64>&, const i32*, i64);

}  // namespace csllm
