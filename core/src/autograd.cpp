#include "csllm/autograd.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace csllm {
namespace {

thread_local bool g_grad_enabled = true;

}  // namespace

NoGradGuard::NoGradGuard() : prev_(g_grad_enabled) { g_grad_enabled = false; }
NoGradGuard::~NoGradGuard() { g_grad_enabled = prev_; }
bool NoGradGuard::enabled() noexcept { return g_grad_enabled; }

template <typename T>
Tape<T>& Tape<T>::active() {
  // Thread-local so concurrent GenerationSessions never interleave graphs.
  static thread_local Tape<T> tape;
  return tape;
}

template <typename T>
int Tape<T>::push(Node node) {
  nodes_.push_back(std::move(node));
  return static_cast<int>(nodes_.size()) - 1;
}

template <typename T>
void Tape<T>::backward(int root_node_id) {
  CSLLM_CHECK(root_node_id >= 0 && root_node_id < static_cast<int>(nodes_.size()),
              "backward(): root node id out of range");

  // Nodes are appended in forward order, so a node's index always exceeds its
  // parents'. Walking indices downward from the root is therefore already a
  // valid reverse-topological order — no explicit sort needed. We only need to
  // know which nodes are reachable from the root.
  std::vector<bool> reachable(nodes_.size(), false);
  reachable[static_cast<std::size_t>(root_node_id)] = true;

  for (int i = root_node_id; i >= 0; --i) {
    const auto idx = static_cast<std::size_t>(i);
    if (!reachable[idx]) continue;
    for (int p : nodes_[idx].parents) {
      if (p >= 0) reachable[static_cast<std::size_t>(p)] = true;
    }
  }

  for (int i = root_node_id; i >= 0; --i) {
    const auto idx = static_cast<std::size_t>(i);
    if (reachable[idx] && nodes_[idx].backward_fn) nodes_[idx].backward_fn();
  }
}

template class Tape<f32>;
template class Tape<f64>;

}  // namespace csllm
