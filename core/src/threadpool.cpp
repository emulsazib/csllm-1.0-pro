#include "csllm/threadpool.hpp"

#include <algorithm>

namespace csllm {

ThreadPool::ThreadPool(unsigned n) {
  if (n == 0) n = std::thread::hardware_concurrency();
  if (n == 0) n = 1;
  // The calling thread participates, so spawn n-1 workers.
  workers_.reserve(n - 1);
  for (unsigned i = 0; i + 1 < n; ++i) workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_start_.notify_all();
  for (auto& t : workers_) {
    if (t.joinable()) t.join();
  }
}

ThreadPool& ThreadPool::global() {
  static ThreadPool pool;
  return pool;
}

void ThreadPool::worker_loop() {
  u64 seen = 0;
  for (;;) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_start_.wait(lk, [&] { return stop_ || generation_ != seen; });
    if (stop_) return;
    seen = generation_;

    for (;;) {
      if (next_index_ >= task_n_) break;
      const std::size_t i = next_index_++;
      lk.unlock();
      (*task_)(i);
      lk.lock();
      ++completed_;
      if (completed_ == task_n_) cv_done_.notify_all();
    }
  }
}

void ThreadPool::parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn) {
  if (n == 0) return;
  // Serial pool, or too little work to be worth the handoff.
  if (workers_.empty() || n == 1) {
    for (std::size_t i = 0; i < n; ++i) fn(i);
    return;
  }

  std::unique_lock<std::mutex> lk(mu_);
  task_ = &fn;
  task_n_ = n;
  next_index_ = 0;
  completed_ = 0;
  ++generation_;
  cv_start_.notify_all();

  // The calling thread pulls work too rather than idling.
  for (;;) {
    if (next_index_ >= task_n_) break;
    const std::size_t i = next_index_++;
    lk.unlock();
    fn(i);
    lk.lock();
    ++completed_;
    if (completed_ == task_n_) cv_done_.notify_all();
  }

  cv_done_.wait(lk, [&] { return completed_ == task_n_; });
  task_ = nullptr;
}

}  // namespace csllm
