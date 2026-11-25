// A bounded HTTP request scheduler. A single worker owns the model and can
// interleave prefill and decode across independent request slots.
// Client threads drain per-job token queues and handle formatting and HTTP writes.
// The model worker never waits on a client. Cancellation is observed at generation
// boundaries, after which other jobs can use the slot.
#pragma once

#include "ingine/generator.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace ingine::engine {

using Clock = std::chrono::steady_clock;

// Output drained from a job: token ids in order, then a failure message and/or the end of generation.
struct JobBatch {
  std::vector<int32_t> tokens;
  std::optional<std::string> failure;
  bool end = false;
  bool empty() const { return tokens.empty() && !failure && !end; }
};

class Job {
 public:
  Job(std::vector<int32_t> prompt_ids, int max_new_tokens, model::Sampling sampling,
      std::vector<model::PromptImage> images = {});

  const std::string id;  // "chatcmpl-" + 32 hex digits
  const std::vector<int32_t> prompt_ids;
  const std::vector<model::PromptImage> images;  // the prompt's images, by position (see model::Prompt)
  const int max_new_tokens;
  const model::Sampling sampling;
  const Clock::time_point t_submit;

  void cancel() { cancel_.store(true, std::memory_order_release); }
  bool cancelled() const { return cancel_.load(std::memory_order_acquire); }
  std::optional<Clock::time_point> t_start() const;
  int cached_tokens() const { return cached_tokens_.load(); }

  // Waits up to `timeout` for output, then returns everything available (empty on timeout).
  JobBatch take(Clock::duration timeout);

 private:
  friend class Scheduler;
  void started();
  void push(const std::vector<int32_t>& tokens);
  void fail(std::string message);
  void finish();

  std::atomic<bool> cancel_{false};
  std::atomic<int> cached_tokens_{0};
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<int32_t> tokens_;
  std::optional<std::string> failure_;
  bool end_ = false;
  std::optional<Clock::time_point> t_start_;
};

struct SchedulerStats {
  int waiting = 0;
  bool running = false;
  int max_queue = 0;
  int active = 0;
  int max_batch = 1;
};

class Scheduler {
 public:
  Scheduler(model::Generator& generator, std::vector<int32_t> stop_ids, int max_queue);
  ~Scheduler();
  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;

  // False when closed or when the active and waiting capacity is exhausted.
  bool submit(std::shared_ptr<Job> job);
  // Cancels the job and drops it from the waiting list if it has not started.
  void withdraw(const std::shared_ptr<Job>& job);
  SchedulerStats stats() const;
  // Fails waiting jobs ("server shutting down"), cancels the running one and joins the worker.
  void close();

 private:
  void loop();
  void run(Job& job);
  void batch_loop();

  struct ActiveSlot {
    std::shared_ptr<Job> job;
    int slot;
    int offset = 0, emitted = 0;
    int32_t token = 0;
    model::BatchSampling sampling;
    bool solo = false, done = false;
  };

  model::Generator& generator_;
  const std::vector<int32_t> stop_list_;
  const std::unordered_set<int32_t> stop_ids_;
  const int max_queue_;
  const int max_batch_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Job>> waiting_;
  std::shared_ptr<Job> active_;
  std::vector<ActiveSlot> slots_;
  bool closed_ = false;
  std::thread thread_;
};

}  // namespace ingine::engine
