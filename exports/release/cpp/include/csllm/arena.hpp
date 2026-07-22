#pragma once
//
// Bump allocator.
//
// Activations are allocated from an Arena and freed wholesale with reset()
// once per training step: no per-op malloc traffic, no fragmentation, and
// deallocation is a single pointer assignment. Parameters live in a separate,
// persistent Arena that is never reset.
//
#include <cstddef>

#include "csllm/common.hpp"

namespace csllm {

class Arena {
 public:
  static constexpr std::size_t kDefaultAlign = 64;  // cache line / SIMD friendly

  explicit Arena(std::size_t capacity_bytes);
  ~Arena();

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;
  Arena(Arena&&) noexcept;
  Arena& operator=(Arena&&) noexcept;

  // Throws Error if the arena is exhausted — deliberately loud, because a
  // silent fallback to malloc would hide a sizing bug.
  void* allocate(std::size_t bytes, std::size_t align = kDefaultAlign);

  template <typename T>
  T* alloc_n(std::size_t n) {
    return static_cast<T*>(allocate(n * sizeof(T), kDefaultAlign));
  }

  void reset() noexcept;

  std::size_t used() const noexcept { return offset_; }
  std::size_t capacity() const noexcept { return capacity_; }
  std::size_t high_water() const noexcept { return high_water_; }

 private:
  std::byte* base_ = nullptr;
  std::size_t capacity_ = 0;
  std::size_t offset_ = 0;
  std::size_t high_water_ = 0;
};

}  // namespace csllm
