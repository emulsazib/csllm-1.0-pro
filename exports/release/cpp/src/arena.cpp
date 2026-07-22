#include "csllm/arena.hpp"

#include <cstdlib>
#include <utility>

namespace csllm {
namespace {

std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1) & ~(a - 1); }

}  // namespace

Arena::Arena(std::size_t capacity_bytes) : capacity_(align_up(capacity_bytes, kDefaultAlign)) {
  if (capacity_ == 0) return;
  base_ = static_cast<std::byte*>(std::aligned_alloc(kDefaultAlign, capacity_));
  CSLLM_CHECK(base_ != nullptr, "arena allocation failed");
}

Arena::~Arena() { std::free(base_); }

Arena::Arena(Arena&& o) noexcept
    : base_(std::exchange(o.base_, nullptr)),
      capacity_(std::exchange(o.capacity_, 0)),
      offset_(std::exchange(o.offset_, 0)),
      high_water_(std::exchange(o.high_water_, 0)) {}

Arena& Arena::operator=(Arena&& o) noexcept {
  if (this != &o) {
    std::free(base_);
    base_ = std::exchange(o.base_, nullptr);
    capacity_ = std::exchange(o.capacity_, 0);
    offset_ = std::exchange(o.offset_, 0);
    high_water_ = std::exchange(o.high_water_, 0);
  }
  return *this;
}

void* Arena::allocate(std::size_t bytes, std::size_t align) {
  const std::size_t start = align_up(offset_, align);
  // Exhaustion throws rather than falling back to malloc: a silent fallback
  // would hide an arena-sizing bug behind a slow, fragmenting allocation.
  CSLLM_CHECK(start + bytes <= capacity_,
              "arena exhausted: requested " + std::to_string(bytes) + " B, " +
                  std::to_string(capacity_ - start) + " B free of " + std::to_string(capacity_));
  offset_ = start + bytes;
  if (offset_ > high_water_) high_water_ = offset_;
  return base_ + start;
}

void Arena::reset() noexcept { offset_ = 0; }

}  // namespace csllm
