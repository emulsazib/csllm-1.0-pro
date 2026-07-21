#pragma once
//
// Tensor<T> — a non-owning view over Arena memory, plus autograd bookkeeping.
//
// Templated on the scalar type so the entire op set can be instantiated in
// double precision for finite-difference gradient checking. fp32 central
// differences are too noisy to distinguish a real gradient bug from rounding,
// which makes this template the project's most important testability decision.
// See memory-bank/design.md → D6.
//
#include <array>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <string>

#include "csllm/arena.hpp"
#include "csllm/common.hpp"

namespace csllm {

inline constexpr int kMaxDims = 4;

struct Shape {
  std::array<i64, kMaxDims> dims{};
  int ndim = 0;

  Shape() = default;
  Shape(std::initializer_list<i64> d) {
    CSLLM_CHECK(d.size() <= kMaxDims, "shape exceeds kMaxDims");
    ndim = static_cast<int>(d.size());
    int i = 0;
    for (i64 v : d) dims[i++] = v;
  }

  i64 numel() const noexcept {
    i64 n = 1;
    for (int i = 0; i < ndim; ++i) n *= dims[i];
    return n;
  }
  i64 operator[](int i) const noexcept { return dims[i]; }
  bool operator==(const Shape& o) const noexcept {
    if (ndim != o.ndim) return false;
    for (int i = 0; i < ndim; ++i) {
      if (dims[i] != o.dims[i]) return false;
    }
    return true;
  }
  std::string str() const;
};

template <typename T>
struct Tensor {
  T* data = nullptr;
  T* grad = nullptr;          // null when requires_grad == false
  Shape shape;
  bool requires_grad = false;
  int node_id = -1;           // index into the active Tape, or -1 if a leaf

  i64 numel() const noexcept { return shape.numel(); }
  bool defined() const noexcept { return data != nullptr; }

  // Convenience for the 2-D [rows, cols] layout most ops work in.
  i64 rows() const noexcept { return shape.ndim >= 1 ? shape[0] : 0; }
  i64 cols() const noexcept { return shape.ndim >= 2 ? shape[shape.ndim - 1] : 1; }

  T& operator[](i64 i) noexcept { return data[i]; }
  const T& operator[](i64 i) const noexcept { return data[i]; }
};

using TensorF = Tensor<f32>;
using TensorD = Tensor<f64>;

// Allocates data (uninitialised — ops must fully write it) and, when
// requires_grad, a zeroed gradient buffer from the same arena.
template <typename T>
Tensor<T> make_tensor(Arena& arena, const Shape& shape, bool requires_grad) {
  Tensor<T> t;
  t.shape = shape;
  t.requires_grad = requires_grad;
  const auto n = static_cast<std::size_t>(shape.numel());
  t.data = arena.alloc_n<T>(n);
  if (requires_grad) {
    t.grad = arena.alloc_n<T>(n);
    std::memset(t.grad, 0, n * sizeof(T));
  }
  return t;
}

template <typename T>
void zero_(T* p, i64 n) {
  std::memset(p, 0, static_cast<std::size_t>(n) * sizeof(T));
}

}  // namespace csllm
