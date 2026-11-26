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

Scheduler::Scheduler(model::Generator& generator, std::vector<int32_t> stop_ids, int max_queue,
                     std::vector<int32_t> prefix_boundary_ids)
    : generator_(generator),
      stop_list_(std::move(stop_ids)),
      stop_ids_(stop_list_.begin(), stop_list_.end()),
      boundary_ids_(prefix_boundary_ids.begin(), prefix_boundary_ids.end()),
      max_queue_(max_queue),
      max_batch_(generator.max_batch()),
      thread_([this] { loop(); }) {
  boundary_ids_.insert(stop_ids_.begin(), stop_ids_.end());
}

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
  return {int(waiting_.size()), active != 0, max_queue_, active, max_batch_, generator_.batch_max_ctx(), generator_.batch_spec()};
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
  const auto needs_solo = [&](const std::shared_ptr<Job>& job) {
    return !job->images.empty() ||
           (generator_.batch_spec() > 0 && int(job->prompt_ids.size()) + generator_.batch_spec() + 2 >= generator_.batch_max_ctx());
  };
  const auto emit = [&](ActiveSlot& slot, std::span<const int32_t> tokens) {
    std::vector<int32_t> out;
    const int ctx_cap = (slot.solo || slot.force_solo) ? generator_.max_model_len() : generator_.batch_max_ctx();
    const int limit = std::min(slot.job->max_new_tokens,
                               std::max(0, ctx_cap - int(slot.job->prompt_ids.size())));
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
      constexpr int kMinAnchorPrefix = 256;
      const auto is_active_id = [&](int id) {
        return std::any_of(slots_.begin(), slots_.end(), [&](const auto& s) { return s.slot == id; });
      };
      const auto is_anchor_id = [&](int id) {
        return std::find(anchor_slots_.begin(), anchor_slots_.end(), id) != anchor_slots_.end();
      };
      const auto erase_anchor = [&](int id) {
        auto it = std::find(anchor_slots_.begin(), anchor_slots_.end(), id);
        if (it != anchor_slots_.end()) anchor_slots_.erase(it);
      };
      const auto touch_anchor = [&](int id) {
        erase_anchor(id);
        anchor_slots_.push_back(id);
      };
      while (!waiting_.empty() && int(slots_.size()) < max_batch_) {
        auto job = waiting_.front();
        const bool solo_only = needs_solo(job);
        if (solo_only && !slots_.empty()) break;
        waiting_.pop_front();
        const bool idle = slots_.empty() && waiting_.empty();
        int best_direct = -1, direct_prefix = -1;
        int best_anchor = -1, anchor_prefix = -1;
        int first_free_non_anchor = -1;
        for (int id = 0; id < max_batch_; ++id) {
          if (is_active_id(id)) continue;
          const int reuse = generator_.reusable_prefix(id, job->prompt_ids);
          if (is_anchor_id(id)) {
            if (reuse > anchor_prefix) {
              best_anchor = id;
              anchor_prefix = reuse;
            }
          } else {
            if (first_free_non_anchor < 0) first_free_non_anchor = id;
            if (reuse > direct_prefix) {
              best_direct = id;
              direct_prefix = reuse;
            }
          }
        }
        int chosen = -1;
        int pending_src = -1;
        int pending_len = 0;
        if (solo_only) {
          chosen = 0;
          erase_anchor(0);
        } else if (direct_prefix > 0 && direct_prefix >= anchor_prefix) {
          chosen = best_direct;
        } else if (generator_.batch_spec() > 0 && best_anchor >= 0 && anchor_prefix >= kMinAnchorPrefix) {
          if (first_free_non_anchor >= 0) {
            chosen = first_free_non_anchor;
            pending_src = best_anchor;
            pending_len = anchor_prefix;
            touch_anchor(best_anchor);
          } else {
            // Full capacity: steal the matching anchor in-place rather than blocking.
            chosen = best_anchor;
            erase_anchor(best_anchor);
          }
        } else if (first_free_non_anchor >= 0) {
          chosen = first_free_non_anchor;
        } else if (!anchor_slots_.empty()) {
          // Full capacity: evict the oldest LRU anchor slot.
          chosen = anchor_slots_.front();
          anchor_slots_.pop_front();
        } else {
          chosen = 0;
        }
        auto& slot = slots_.emplace_back(ActiveSlot{job, chosen});
        slot.sampling.params = job->sampling;
        slot.sampling.seed = job->sampling.seed ? *job->sampling.seed
            : (uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
        slot.force_solo = solo_only;
        slot.pending_anchor_src = pending_src;
        slot.pending_anchor_len = pending_len;
        // Keep solo mode when truly idle unless a warm persistent prefix anchor already holds >= 256 tokens.
        slot.solo = solo_only || (idle && chosen == 0 && pending_src < 0 && anchor_prefix < kMinAnchorPrefix);
        job->started();
        if (solo_only) break;
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
              interrupted = !first.force_solo && !waiting_.empty() && first.job->images.empty() &&
                            !needs_solo(waiting_.front()) &&
                            (generator_.batch_spec() == 0 ||
                             int(first.job->prompt_ids.size()) + first.emitted + generator_.batch_spec() + 2 < generator_.batch_max_ctx());
              return !interrupted;
            });
        first.job->cached_tokens_.store(st.reused_tokens);
        first.offset = st.interrupted_prefill_offset ? st.interrupted_prefill_offset : int(first.job->prompt_ids.size());
        first.done = first.done || !interrupted;
        if (interrupted && !first.done && !first.job->cancelled() && generator_.batch_spec() > 0) {
          if (!generator_.promote_solo_to_batch(first.slot, first.emitted > 0 ? first.token : -1)) {
            first.job->fail("solo-to-batch promotion exceeded batch context");
            first.done = true;
          }
        }
        server_log().info("{}: reuse={} reused={} spec_rounds={} drafter={} interrupted={}",
                          first.job->id, model::to_string(st.reuse), st.reused_tokens, st.spec_rounds,
                          st.used_dflash2 ? "dflash2" : "target/dspark", interrupted);
      } catch (const std::exception& e) {
        first.job->fail(e.what());
        first.done = true;
      }
      continue;
    }

    constexpr int kMinForkPrefix = 64;
    const auto common_prefix = [](std::span<const int32_t> a, std::span<const int32_t> b) -> int {
      const std::size_t n = std::min(a.size(), b.size());
      std::size_t i = 0;
      while (i < n && a[i] == b[i]) ++i;
      return int(i);
    };
    const auto fork_from = [&](const ActiveSlot& src) {
      if (src.offset < kMinForkPrefix || src.emitted || src.done || src.job->cancelled()) return 0;
      int cloned = 0;
      for (auto& other : slots_) {
        if (other.slot == src.slot || other.job->cancelled() || other.done || other.emitted) continue;
        if (other.offset >= src.offset) continue;
        if (int(other.job->prompt_ids.size()) <= src.offset) continue;
        const int lcp = std::min(common_prefix(src.job->prompt_ids, other.job->prompt_ids),
                                 int(std::min(src.job->prompt_ids.size(), other.job->prompt_ids.size())) - 1);
        // Only clone once src has reached the full shared-prefix boundary with `other` (avoids redundant mid-chunk clones).
        if (lcp != src.offset) continue;
        try {
          if (generator_.clone_slot(src.slot, other.slot)) {
            other.offset = src.offset;
            other.job->cached_tokens_.store(src.offset);
            ++cloned;
          }
        } catch (const std::exception& e) {
          server_log().error("shared-prefix clone failed ({} -> {}): {}", src.slot, other.slot, e.what());
        }
      }
      if (cloned > 0) {
        server_log().info("shared-prefix fork: src_slot={} cloned={} lanes at {} tokens",
                          src.slot, cloned, src.offset);
      }
      return cloned;
    };

    const int max_anchors = (generator_.batch_spec() > 0 && max_batch_ >= 4) ? std::min(4, max_batch_ / 4) : 0;
    const auto is_slot_active = [&](int id) {
      return std::any_of(slots_.begin(), slots_.end(), [&](const auto& s) { return s.slot == id; });
    };
    const auto has_matching_anchor = [&](std::span<const int32_t> prompt, int len) -> int {
      for (int aid : anchor_slots_) {
        if (is_slot_active(aid)) continue;
        if (generator_.reusable_prefix(aid, prompt) == len) return aid;
      }
      return -1;
    };
    const auto save_anchor = [&](const ActiveSlot& src) {
      if (max_anchors <= 0 || src.offset < kMinForkPrefix || src.emitted || src.done || src.job->cancelled()) return;
      if (const int existing = has_matching_anchor(src.job->prompt_ids, src.offset); existing >= 0) {
        auto it = std::find(anchor_slots_.begin(), anchor_slots_.end(), existing);
        if (it != anchor_slots_.end()) {
          anchor_slots_.erase(it);
          anchor_slots_.push_back(existing);
        }
        return;
      }
      int dst = -1;
      if (int(anchor_slots_.size()) < max_anchors) {
        // Prefer high-numbered idle non-anchor slots so slot 0 stays available for solo mode.
        for (int id = max_batch_ - 1; id >= 1; --id) {
          if (is_slot_active(id)) continue;
          if (std::find(anchor_slots_.begin(), anchor_slots_.end(), id) != anchor_slots_.end()) continue;
          dst = id;
          break;
        }
      }
      if (dst < 0 && !anchor_slots_.empty()) {
        for (auto it = anchor_slots_.begin(); it != anchor_slots_.end(); ++it) {
          if (!is_slot_active(*it)) {
            dst = *it;
            anchor_slots_.erase(it);
            break;
          }
        }
      }
      if (dst < 0) return;
      try {
        if (generator_.clone_slot(src.slot, dst)) {
          anchor_slots_.push_back(dst);
          server_log().info("prefix-anchor saved: src_slot={} -> anchor_slot={} at {} tokens (pool={}/{})",
                            src.slot, dst, src.offset, anchor_slots_.size(), max_anchors);
        }
      } catch (const std::exception& e) {
        server_log().error("prefix-anchor save failed ({} -> {}): {}", src.slot, dst, e.what());
      }
    };
    const auto find_system_boundary = [&](std::span<const int32_t> prompt) -> int {
      if (max_anchors <= 0 || boundary_ids_.empty() || int(prompt.size()) < kMinForkPrefix + 16) return 0;
      const int limit = int(prompt.size()) - 16;
      for (int i = 0; i < limit; ++i) {
        if (boundary_ids_.contains(prompt[std::size_t(i)])) {
          const int boundary = i + 1;
          if (boundary >= kMinForkPrefix) return boundary;
        }
      }
      return 0;
    };

    // Materialize any persistent prefix-anchor clones queued during admission (outside mutex_, before any prefill).
    if (generator_.batch_spec() > 0) {
      int anchor_hits = 0;
      for (auto& slot : slots_) {
        if (slot.pending_anchor_src < 0) continue;
        const int src_id = slot.pending_anchor_src;
        const int len = slot.pending_anchor_len;
        slot.pending_anchor_src = -1;
        slot.pending_anchor_len = 0;
        if (slot.done || slot.job->cancelled() || slot.offset > 0) continue;
        try {
          if (generator_.clone_slot(src_id, slot.slot)) {
            slot.offset = len;
            slot.job->cached_tokens_.store(len);
            ++anchor_hits;
          }
        } catch (const std::exception& e) {
          server_log().error("prefix-anchor clone failed ({} -> {}): {}", src_id, slot.slot, e.what());
        }
      }
      if (anchor_hits > 0) {
        server_log().info("prefix-anchor reuse: cloned {} lane(s) from persistent anchor pool", anchor_hits);
      }
      for (const auto& src : slots_) {
        if (src.offset >= kMinForkPrefix && !src.emitted && !src.done && !src.job->cancelled()) {
          if (fork_from(src) > 0) save_anchor(src);
        }
      }
    }

    int prefill_budget = generator_.batch_spec() > 0 ? 1024 : 1;
    while (prefill_budget > 0) {
      ActiveSlot* prefill = nullptr;
      if (generator_.batch_spec() > 0) {
        int best_peers = -1;
        for (auto& slot : slots_) {
          if (slot.job->cancelled() || slot.done || slot.emitted) continue;
          int peers = 0;
          for (const auto& other : slots_) {
            if (other.slot == slot.slot || other.job->cancelled() || other.done || other.emitted) continue;
            const int lcp = std::min(common_prefix(slot.job->prompt_ids, other.job->prompt_ids),
                                     int(std::min(slot.job->prompt_ids.size(), other.job->prompt_ids.size())) - 1);
            if (lcp >= kMinForkPrefix && other.offset < lcp && slot.offset < lcp) ++peers;
          }
          if (!prefill || peers > best_peers ||
              (peers == best_peers && peers > 0 && slot.offset > prefill->offset) ||
              (peers == best_peers && peers == 0 && slot.offset < prefill->offset)) {
            prefill = &slot;
            best_peers = peers;
          }
        }
      } else {
        for (auto& slot : slots_) {
          if (slot.job->cancelled() || slot.done || slot.emitted) continue;
          if (!prefill || slot.offset < prefill->offset) prefill = &slot;
        }
      }
      if (!prefill) break;
      auto& slot = *prefill;
      int fork_at = 0;
      bool anchor_boundary = false;
      if (generator_.batch_spec() > 0) {
        for (const auto& other : slots_) {
          if (other.slot == slot.slot || other.job->cancelled() || other.done || other.emitted) continue;
          const int lcp = std::min(common_prefix(slot.job->prompt_ids, other.job->prompt_ids),
                                   int(std::min(slot.job->prompt_ids.size(), other.job->prompt_ids.size())) - 1);
          if (lcp >= kMinForkPrefix && lcp > slot.offset && other.offset < lcp) {
            if (fork_at == 0 || lcp < fork_at) fork_at = lcp;
          }
        }
        const int sys_end = find_system_boundary(slot.job->prompt_ids);
        if (sys_end > slot.offset && has_matching_anchor(slot.job->prompt_ids, sys_end) < 0 &&
            int(slots_.size()) < max_batch_) {
          if (fork_at == 0 || sys_end <= fork_at) {
            fork_at = sys_end;
            anchor_boundary = true;
          }
        }
      }
      const int before = slot.offset;
      try {
        const auto result = generator_.prefill_slot_chunk(
            slot.slot, slot.job->prompt_ids, slot.offset, slot.sampling, fork_at);
        if (!slot.offset) slot.job->cached_tokens_.store(result.reused_tokens);
        slot.offset = result.offset;
        if (result.token) emit(slot, std::span(&*result.token, 1));
      } catch (const std::exception& e) {
        slot.job->fail(e.what());
        slot.done = true;
      }
      if (fork_at > 0 && slot.offset == fork_at && !slot.done && !slot.job->cancelled()) {
        const int cloned = fork_from(slot);
        if (cloned > 0 || anchor_boundary) save_anchor(slot);
        continue;
      }
      prefill_budget -= std::max(1, slot.offset - before);
      if (generator_.batch_spec() == 0 || slot.offset < int(slot.job->prompt_ids.size())) break;
    }
    std::vector<model::BatchSlotStep> rows;
    std::vector<ActiveSlot*> decoding;
    for (auto& slot : slots_) {
      if (slot.job->cancelled() || slot.done || !slot.emitted) continue;
      rows.push_back({slot.slot, slot.token, &slot.sampling});
      decoding.push_back(&slot);
    }
    if (!rows.empty()) {
      try {
        const auto emitted = generator_.decode_batch_step_multi(rows);
        for (std::size_t i = 0; i < rows.size(); ++i) emit(*decoding[i], emitted.at(i));
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
