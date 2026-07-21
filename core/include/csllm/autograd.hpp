#pragma once
//
// Reverse-mode autograd over an explicit tape.
//
// Each op appends a Node holding a closure that maps output gradients to input
// gradients. backward() walks the tape in reverse, accumulating into
// Tensor::grad. The graph is rebuilt every step, so control flow in the model
// is free.
//
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "csllm/common.hpp"
#include "csllm/tensor.hpp"

namespace csllm {

struct Node {
  std::function<void()> backward_fn;
  std::vector<int> parents;
  std::string name;  // op name, for debugging and graph dumps
};

template <typename T>
class Tape {
 public:
  int push(Node node);

  // Walks the tape in reverse from root_node_id, running each reachable
  // node's backward closure. The caller seeds the root gradient first.
  void backward(int root_node_id);

  void clear() noexcept { nodes_.clear(); }
  std::size_t size() const noexcept { return nodes_.size(); }
  const Node& at(int i) const { return nodes_.at(static_cast<std::size_t>(i)); }

  // The tape ops record into. Thread-local so independent models/sessions do
  // not interleave graphs.
  static Tape& active();

 private:
  std::vector<Node> nodes_;
};

// RAII guard disabling graph construction (inference / no_grad regions).
class NoGradGuard {
 public:
  NoGradGuard();
  ~NoGradGuard();
  static bool enabled() noexcept;

 private:
  bool prev_;
};

inline bool grad_enabled() noexcept { return NoGradGuard::enabled(); }

// True when an op should build a graph node for these inputs.
template <typename... Ts>
bool any_requires_grad(const Ts&... ts) {
  return grad_enabled() && (... || ts.requires_grad);
}

// Records a backward closure and returns the new node id.
template <typename T, typename F>
int record(const char* name, std::vector<int> parents, F&& fn) {
  return Tape<T>::active().push(Node{std::forward<F>(fn), std::move(parents), name});
}

}  // namespace csllm
