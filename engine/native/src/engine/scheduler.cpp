#include "engine/scheduler.h"

#include "engine/log.h"

#include <cxxabi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <span>
#include <typeinfo>
#include <utility>

namespace ingine::engine {
namespace {

// uuid.uuid4().hex: 32 random hex digits.
std::string random_hex32() {
  thread_local std::mt19937_64 rng{std::random_device{}() ^ (uint64_t(std::random_device{}()) << 32)};
  char buf[33];
  std::snprintf(buf, sizeof buf, "%016llx%016llx", static_cast<unsigned long long>(rng()),
                static_cast<unsigned long long>(rng()));
  return buf;
}

std::string type_name(const std::exception& e) {
  int status = 0;
  char* name = abi::__cxa_demangle(typeid(e).name(), nullptr, nullptr, &status);
  std::string out = status == 0 && name ? name : typeid(e).name();
  std::free(name);
  return out;
}

}  // namespace

Job::Job(std::vector<int32_t> prompt_ids_, int max_new_tokens_, model::Sampling sampling_,
         std::vector<model::PromptImage> images_)
    : id("chatcmpl-" + random_hex32()),
      prompt_ids(std::move(prompt_ids_)),
      images(std::move(images_)),
      max_new_tokens(max_new_tokens_),
      sampling(sampling_),
      t_submit(Clock::now()) {}

std::optional<Clock::time_point> Job::t_start() const {
  std::lock_guard lock(mutex_);
  return t_start_;
}

JobBatch Job::take(Clock::duration timeout) {
  std::unique_lock lock(mutex_);
  cv_.wait_for(lock, timeout, [&] { return !tokens_.empty() || failure_ || end_; });
  JobBatch batch;
  batch.tokens.assign(tokens_.begin(), tokens_.end());
  tokens_.clear();
  // The failure and the end are reported once, after every token pushed before them.
  batch.failure = std::exchange(failure_, std::nullopt);
  batch.end = std::exchange(end_, false);
  return batch;
}

void Job::started() {
  std::lock_guard lock(mutex_);
  t_start_ = Clock::now();
}

void Job::push(const std::vector<int32_t>& tokens) {
  if (tokens.empty()) return;
  {
    std::lock_guard lock(mutex_);
    tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
  }
  cv_.notify_all();
}

void Job::fail(std::string message) {
  {
    std::lock_guard lock(mutex_);
    failure_ = std::move(message);
  }
  cv_.notify_all();
}

void Job::finish() {
  {
    std::lock_guard lock(mutex_);
    end_ = true;
  }
  cv_.notify_all();
}

Scheduler::Scheduler(model::Generator& generator, std::vector<int32_t> stop_ids, int max_queue)
    : generator_(generator),
      stop_list_(std::move(stop_ids)),
      stop_ids_(stop_list_.begin(), stop_list_.end()),
      max_queue_(max_queue),
      max_batch_(generator.max_batch()),
      thread_([this] { loop(); }) {}

Scheduler::~Scheduler() { close(); }

bool Scheduler::submit(std::shared_ptr<Job> job) {
  {
    std::lock_guard lock(mutex_);
    const int active = max_batch_ == 1 ? int(active_ != nullptr) : int(slots_.size());
    if (closed_ || int(waiting_.size()) + active >= max_queue_ + max_batch_) return false;
    waiting_.push_back(std::move(job));
  }
  cv_.notify_all();
  return true;
}

void Scheduler::withdraw(const std::shared_ptr<Job>& job) {
  job->cancel();
  std::lock_guard lock(mutex_);
  auto it = std::find(waiting_.begin(), waiting_.end(), job);
  if (it != waiting_.end()) waiting_.erase(it);
}

SchedulerStats Scheduler::stats() const {
  std::lock_guard lock(mutex_);
  const int active = max_batch_ == 1 ? int(active_ != nullptr) : int(slots_.size());
  return {int(waiting_.size()), active != 0, max_queue_, active, max_batch_};
}

void Scheduler::close() {
  {
    std::lock_guard lock(mutex_);
    closed_ = true;
    for (auto& job : waiting_) {
      job->cancel();
      job->fail("server shutting down");
      job->finish();
    }
    waiting_.clear();
    if (active_) active_->cancel();
    for (auto& slot : slots_) slot.job->cancel();
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void Scheduler::loop() {
  if (max_batch_ > 1) {
    batch_loop();
    return;
  }
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [&] { return !waiting_.empty() || closed_; });
      if (closed_) return;
      job = active_ = waiting_.front();
      waiting_.pop_front();
    }
    run(*job);
    job->finish();
    std::lock_guard lock(mutex_);
    active_ = nullptr;
  }
}

void Scheduler::batch_loop() {
  const auto emit = [&](ActiveSlot& slot, std::span<const int32_t> tokens) {
    std::vector<int32_t> out;
    const int limit = std::min(slot.job->max_new_tokens,
                               generator_.max_model_len() - int(slot.job->prompt_ids.size()));
    for (const int32_t token : tokens) {
      if (slot.job->cancelled() || slot.done) break;
      out.push_back(token);
      slot.token = token;
      slot.done = ++slot.emitted >= limit || stop_ids_.contains(token);
    }
    slot.job->push(out);
  };
  for (;;) {
    {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [&] { return closed_ || !slots_.empty() || !waiting_.empty(); });
      std::erase_if(slots_, [&](const auto& slot) {
        if (!slot.done && !slot.job->cancelled()) return false;
        try {
          generator_.release_slot(slot.slot);
        } catch (const std::exception& e) {
          server_log().error("slot release failed: {}", e.what());
        }
        slot.job->finish();
        return true;
      });
      if (closed_) return;
      while (!waiting_.empty() && int(slots_.size()) < max_batch_) {
        auto job = waiting_.front();
        if (!job->images.empty() && !slots_.empty()) break;
        waiting_.pop_front();
        const bool idle = slots_.empty() && waiting_.empty();
        int best = -1, prefix = -1;
        for (int id = 0; id < max_batch_; ++id) {
          if (std::any_of(slots_.begin(), slots_.end(), [&](const auto& s) { return s.slot == id; })) continue;
          const int reuse = generator_.reusable_prefix(id, job->prompt_ids);
          if (reuse > prefix) {
            best = id;
            prefix = reuse;
          }
        }
        if (!job->images.empty()) best = 0;
        auto& slot = slots_.emplace_back(ActiveSlot{job, best});
        slot.sampling.params = job->sampling;
        slot.sampling.seed = job->sampling.seed ? *job->sampling.seed
            : (uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
        slot.solo = !job->images.empty() || (idle && best == 0);
        job->started();
        if (!job->images.empty()) break;
      }
    }

    if (slots_.empty()) continue;
    auto& first = slots_.front();
    if (first.solo) {
      first.solo = false;
      bool interrupted = false;
      try {
        const auto st = generator_.generate_interruptible(
            {first.job->prompt_ids, first.job->images}, first.job->max_new_tokens, first.sampling, stop_list_,
            [&](std::span<const int32_t> tokens) {
              emit(first, tokens);
              if (first.done || first.job->cancelled()) return false;
              std::lock_guard lock(mutex_);
              interrupted = !waiting_.empty() && first.job->images.empty() &&
                            waiting_.front()->images.empty();
              return !interrupted;
            });
        first.job->cached_tokens_.store(st.reused_tokens);
        first.offset = st.interrupted_prefill_offset ? st.interrupted_prefill_offset : int(first.job->prompt_ids.size());
        first.done = first.done || !interrupted;
        server_log().info("{}: reuse={} reused={} spec_rounds={} drafter={} interrupted={}",
                          first.job->id, model::to_string(st.reuse), st.reused_tokens, st.spec_rounds,
                          st.used_dflash2 ? "dflash2" : "target/dspark", interrupted);
      } catch (const std::exception& e) {
        first.job->fail(e.what());
        first.done = true;
      }
      continue;
    }

    std::vector<model::BatchSlotStep> rows;
    std::vector<ActiveSlot*> decoding;
    ActiveSlot* prefill = nullptr;
    for (auto& slot : slots_) {
      if (slot.job->cancelled()) continue;
      if (slot.emitted) {
        rows.push_back({slot.slot, slot.token, &slot.sampling});
        decoding.push_back(&slot);
      } else if (!prefill || slot.offset < prefill->offset) {
        prefill = &slot;
      }
    }
    if (prefill) {
      auto& slot = *prefill;
      try {
        const auto result = generator_.prefill_slot_chunk(
            slot.slot, slot.job->prompt_ids, slot.offset, slot.sampling);
        if (!slot.offset) slot.job->cached_tokens_.store(result.reused_tokens);
        slot.offset = result.offset;
        if (result.token) emit(slot, std::span(&*result.token, 1));
      } catch (const std::exception& e) {
        slot.job->fail(e.what());
        slot.done = true;
      }
    }
    if (!rows.empty()) {
      try {
        const auto tokens = generator_.decode_batch_step(rows);
        for (std::size_t i = 0; i < rows.size(); ++i) emit(*decoding[i], std::span(&tokens.at(i), 1));
      } catch (const std::exception& e) {
        for (auto* slot : decoding) {
          slot->job->fail(e.what());
          slot->done = true;
        }
      }
    }
  }
}

void Scheduler::run(Job& job) {
  if (job.cancelled()) return;
  job.started();
  int n = 0;
  std::vector<int32_t> out;
  const auto on_tokens = [&](std::span<const int32_t> tokens) -> bool {
    out.clear();
    bool more = true;
    for (int32_t tok : tokens) {
      if (job.cancelled()) {
        more = false;
        break;
      }
      out.push_back(tok);
      ++n;
      if (stop_ids_.contains(tok) || n >= job.max_new_tokens || job.cancelled()) {
        more = false;
        break;
      }
    }
    job.push(out);
    return more;
  };
  try {
    const model::GenerateStats st = generator_.generate(model::Prompt{job.prompt_ids, job.images}, job.max_new_tokens,
                                                        job.sampling, stop_list_, on_tokens);
    job.cached_tokens_.store(st.reused_tokens);
    if (st.used_dflash2) server_log().info("{}: drafter=dflash2", job.id);
    // prefix reuse of the prompt: the path, the tokens reused and prefilled, the prefill time
    server_log().info("{}: reuse={} reused={} prefilled={} prefill={:.3f}s", job.id, model::to_string(st.reuse),
                      st.reused_tokens, st.prompt_tokens - st.reused_tokens, st.prefill_seconds);
    if (!job.images.empty()) {  // images: all of the prompt, those the vision tower encoded now, its time
      server_log().info("{}: images={} image_tokens={} encoded={} vit={:.3f}s", job.id, job.images.size(),
                        st.image_tokens, st.images_encoded, st.vision_seconds);
    }
  } catch (const std::exception& e) {  // report to the client instead of killing the worker
    server_log().error("generator failed: {}: {}", type_name(e), e.what());
    job.fail("generation failed: " + type_name(e) + ": " + e.what());
  } catch (...) {
    server_log().error("generator failed: unknown exception");
    job.fail("generation failed: unknown exception");
  }
}

}  // namespace ingine::engine
