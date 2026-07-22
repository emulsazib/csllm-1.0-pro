#pragma once
//
// Minimal fork-join thread pool for parallelising over B×H in attention.
//
// NOTE: Accelerate already threads GEMM internally. Nesting this pool around
// BLAS calls risks core oversubscription — measure before doing so.
// See memory-bank/design.md → Open Questions #1.
//
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "csllm/common.hpp"

namespace csllm {

class ThreadPool {
 public:
  // n == 0 selects std::thread::hardware_concurrency().
  explicit ThreadPool(unsigned n = 0);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // Runs fn(i) for i in [0, n). Blocks until every index completes.
  // Executes inline on the calling thread when the pool is serial or n is small.
  void parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn);

  unsigned size() const noexcept { return static_cast<unsigned>(workers_.size()) + 1; }

  // Process-wide pool, sized from hardware concurrency.
  static ThreadPool& global();

 private:
  void worker_loop();

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;

  const std::function<void(std::size_t)>* task_ = nullptr;
  std::size_t task_n_ = 0;
  std::size_t next_index_ = 0;
  std::size_t completed_ = 0;
  u64 generation_ = 0;
  bool stop_ = false;
};

}  // namespace csllm
