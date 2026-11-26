// Scheduler unit tests with a generator that emits several tokens per callback (like a speculative round):
// the scheduler must cut a batch at a stop id, at max_new_tokens and on cancellation, run jobs in FIFO order,
// bound the waiting list and report generator failures.
#include "engine/scheduler.h"
#include "batch_generator.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using ingine::engine::Job;
using ingine::engine::JobBatch;
using ingine::engine::Scheduler;
using ingine::model::GenerateStats;
using ingine::model::Sampling;

int g_failures = 0;
#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                   \
    }                                                                                 \
  } while (0)

// Emits tokens 100, 101, ... in rounds of `round` tokens until on_tokens refuses; throws if prompt[0] < 0.
class RoundGenerator final : public ingine::model::Generator {
 public:
  explicit RoundGenerator(int round, std::chrono::milliseconds pause = {}) : round_(round), pause_(pause) {}
  std::string model_id() const override { return "round"; }
  int max_model_len() const override { return 1 << 20; }
  GenerateStats generate(std::span<const int32_t> prompt, int, const Sampling&, std::span<const int32_t>,
                         const std::function<bool(std::span<const int32_t>)>& on_tokens) override {
    if (!prompt.empty() && prompt[0] < 0) throw std::invalid_argument("prompt does not fit");
    order.push_back(prompt.empty() ? -1 : prompt[0]);
    for (int32_t next = 100;; next += round_) {
      std::vector<int32_t> toks;
      for (int i = 0; i < round_; ++i) toks.push_back(next + i);
      if (pause_.count()) std::this_thread::sleep_for(pause_);
      if (!on_tokens(toks)) break;
    }
    return {};
  }
  std::vector<int32_t> order;

 private:
  int round_;
  std::chrono::milliseconds pause_;
};

std::vector<int32_t> drain(Job& job, std::optional<std::string>* failure = nullptr) {
  std::vector<int32_t> out;
  for (;;) {
    JobBatch b = job.take(std::chrono::seconds(5));
    out.insert(out.end(), b.tokens.begin(), b.tokens.end());
    if (b.failure && failure) *failure = b.failure;
    if (b.end) return out;
    if (b.empty()) throw std::runtime_error("timeout");
  }
}

void test_cuts() {
  RoundGenerator gen(4);
  Scheduler s(gen, {105}, 4);
  auto stop = std::make_shared<Job>(std::vector<int32_t>{1}, 100, Sampling{});
  CHECK(s.submit(stop));
  CHECK((drain(*stop) == std::vector<int32_t>{100, 101, 102, 103, 104, 105}));  // stop id mid-round
  auto len = std::make_shared<Job>(std::vector<int32_t>{2}, 3, Sampling{});
  CHECK(s.submit(len));
  CHECK((drain(*len) == std::vector<int32_t>{100, 101, 102}));  // max_new_tokens mid-round
  CHECK(len->t_start().has_value());
}

void test_cancel_fifo_bound() {
  RoundGenerator gen(2, std::chrono::milliseconds(5));
  Scheduler s(gen, {}, 1);
  auto a = std::make_shared<Job>(std::vector<int32_t>{1}, 1000000, Sampling{});
  auto b = std::make_shared<Job>(std::vector<int32_t>{2}, 2, Sampling{});
  auto c = std::make_shared<Job>(std::vector<int32_t>{3}, 2, Sampling{});
  CHECK(s.submit(a));
  while (!s.stats().running) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  CHECK(s.submit(b));
  CHECK(!s.submit(c));  // running + 1 waiting > max_queue 1
  CHECK(s.stats().waiting == 1 && s.stats().max_queue == 1);
  s.withdraw(b);  // never runs
  CHECK(s.stats().waiting == 0);
  CHECK(s.submit(c));
  a->cancel();
  CHECK(drain(*a).size() < 1000);
  CHECK(drain(*c).size() == 2);
  CHECK((gen.order == std::vector<int32_t>{1, 3}));
}

void test_failure_and_close() {
  RoundGenerator gen(1, std::chrono::milliseconds(2));
  Scheduler s(gen, {}, 4);
  auto bad = std::make_shared<Job>(std::vector<int32_t>{-1}, 5, Sampling{});
  CHECK(s.submit(bad));
  std::optional<std::string> failure;
  CHECK(drain(*bad, &failure).empty());
  CHECK(failure && failure->find("generation failed: std::invalid_argument: prompt does not fit") == 0);
  auto running = std::make_shared<Job>(std::vector<int32_t>{1}, 1000000, Sampling{});
  auto waiting = std::make_shared<Job>(std::vector<int32_t>{2}, 5, Sampling{});
  CHECK(s.submit(running));
  CHECK(s.submit(waiting));
  s.close();
  failure.reset();
  CHECK(drain(*waiting, &failure).empty() && failure == "server shutting down");
  CHECK(drain(*running).size() < 1000000);
  CHECK(!s.submit(std::make_shared<Job>(std::vector<int32_t>{3}, 1, Sampling{})));
}

void test_batch_capacity_and_cancel() {
  BatchGenerator gen(100);
  Scheduler scheduler(gen, {}, 32);
  std::vector<std::shared_ptr<Job>> jobs;
  for (int i = 0; i < 40; ++i) {
    jobs.push_back(std::make_shared<Job>(std::vector<int32_t>(i ? 1 : 4100, i + 1), 48, Sampling{}));
    CHECK(scheduler.submit(jobs.back()));
  }
  CHECK(!scheduler.submit(std::make_shared<Job>(std::vector<int32_t>{41}, 48, Sampling{})));
  const auto first = jobs[1]->take(std::chrono::seconds(5));
  CHECK(!first.tokens.empty() && !first.end);
  scheduler.withdraw(jobs[1]);
  CHECK(drain(*jobs[1]).size() + first.tokens.size() < 48);
  for (auto& job : jobs) {
    if (job != jobs[1]) CHECK(drain(*job) == std::vector<int32_t>(48, 100));
  }
  CHECK(gen.peak.load() == 8 && gen.released.load() == 40);
}


void test_shared_prefix_fork() {
  BatchGenerator gen(100);
  gen.spec_ = 4;
  Scheduler scheduler(gen, {}, 16);
  std::vector<int32_t> shared(512, 42);
  std::vector<std::shared_ptr<Job>> jobs;
  for (int i = 0; i < 6; ++i) {
    auto p = shared;
    p.push_back(1000 + i);
    p.push_back(2000 + i);
    jobs.push_back(std::make_shared<Job>(std::move(p), 8, Sampling{}));
    CHECK(scheduler.submit(jobs.back()));
  }
  for (auto& job : jobs) CHECK(drain(*job) == std::vector<int32_t>(8, 100));
  CHECK(gen.cloned.load() >= 4);
  int cached_hits = 0;
  for (auto& job : jobs) {
    if (job->cached_tokens() == 512) ++cached_hits;
  }
  CHECK(cached_hits >= 4);
}


void test_persistent_prefix_anchor_across_waves() {
  BatchGenerator gen(100);
  gen.spec_ = 4;
  Scheduler scheduler(gen, {}, 16);
  std::vector<int32_t> shared(512, 77);

  // Wave 1: 4 concurrent requests share 512-token prefix -> forks + saves persistent anchor.
  {
    std::vector<std::shared_ptr<Job>> wave1;
    for (int i = 0; i < 4; ++i) {
      auto p = shared;
      p.push_back(1000 + i);
      p.push_back(2000 + i);
      wave1.push_back(std::make_shared<Job>(std::move(p), 4, Sampling{}));
      CHECK(scheduler.submit(wave1.back()));
    }
    for (auto& job : wave1) CHECK(drain(*job) == std::vector<int32_t>(4, 100));
  }
  const int prefilled_after_w1 = gen.prefilled_tokens.load();

  // Wave 2: 4 brand-new requests with DIFFERENT suffixes -> ALL 4 must hit persistent anchor (0 shared tokens prefilled!).
  {
    std::vector<std::shared_ptr<Job>> wave2;
    for (int i = 0; i < 4; ++i) {
      auto p = shared;
      p.push_back(3000 + i);
      p.push_back(4000 + i);
      wave2.push_back(std::make_shared<Job>(std::move(p), 4, Sampling{}));
      CHECK(scheduler.submit(wave2.back()));
    }
    for (auto& job : wave2) CHECK(drain(*job) == std::vector<int32_t>(4, 100));
    for (auto& job : wave2) CHECK(job->cached_tokens() == 512);
  }
  const int prefilled_w2 = gen.prefilled_tokens.load() - prefilled_after_w1;
  CHECK(prefilled_w2 == 8);  // only 2 suffix tokens x 4 lanes!

  // Wave 3: anchor must STILL be intact and reusable across waves.
  {
    std::vector<std::shared_ptr<Job>> wave3;
    for (int i = 0; i < 4; ++i) {
      auto p = shared;
      p.push_back(5000 + i);
      p.push_back(6000 + i);
      wave3.push_back(std::make_shared<Job>(std::move(p), 4, Sampling{}));
      CHECK(scheduler.submit(wave3.back()));
    }
    for (auto& job : wave3) CHECK(drain(*job) == std::vector<int32_t>(4, 100));
    for (auto& job : wave3) CHECK(job->cached_tokens() == 512);
  }
  const int prefilled_w3 = gen.prefilled_tokens.load() - (prefilled_after_w1 + prefilled_w2);
  CHECK(prefilled_w3 == 8);
}

void test_prefix_anchor_eviction_at_full_capacity() {
  BatchGenerator gen(100);
  gen.spec_ = 4;
  Scheduler scheduler(gen, {}, 16);
  std::vector<int32_t> shared(512, 88);
  {
    std::vector<std::shared_ptr<Job>> wave1;
    for (int i = 0; i < 3; ++i) {
      auto p = shared;
      p.push_back(100 + i);
      wave1.push_back(std::make_shared<Job>(std::move(p), 2, Sampling{}));
      CHECK(scheduler.submit(wave1.back()));
    }
    for (auto& job : wave1) CHECK(drain(*job) == std::vector<int32_t>(2, 100));
  }
  // Now saturate all 8 slots simultaneously -> anchor slot must be evicted cleanly so peak == 8.
  gen.peak.store(0);
  std::vector<std::shared_ptr<Job>> full;
  for (int i = 0; i < 8; ++i) {
    full.push_back(std::make_shared<Job>(std::vector<int32_t>{9000 + i, 9100 + i}, 12, Sampling{}));
    CHECK(scheduler.submit(full.back()));
  }
  for (auto& job : full) CHECK(drain(*job) == std::vector<int32_t>(12, 100));
  CHECK(gen.peak.load() == 8);
}

void test_system_prompt_anchor_boundary() {
  BatchGenerator gen(100);
  gen.spec_ = 4;
  constexpr int32_t kImEnd = 9999;
  Scheduler scheduler(gen, {kImEnd}, 16);
  std::vector<int32_t> sys(320, 55);
  sys.push_back(kImEnd);  // boundary at 321 tokens
  auto make_prompt = [&](int tag) {
    auto p = sys;
    for (int k = 0; k < 24; ++k) p.push_back(tag * 100 + k);
    return p;
  };
  // Two concurrent requests in Wave 1 with different user suffixes sharing the system prompt + single follow-up Wave 2.
  auto j1 = std::make_shared<Job>(make_prompt(1), 4, Sampling{});
  auto j2 = std::make_shared<Job>(make_prompt(2), 4, Sampling{});
  CHECK(scheduler.submit(j1));
  CHECK(scheduler.submit(j2));
  CHECK(drain(*j1) == std::vector<int32_t>(4, 100));
  CHECK(drain(*j2) == std::vector<int32_t>(4, 100));
  // Wave 2 (c=2 concurrent requests with different user suffixes) must reuse the 321-token system anchor.
  const int before = gen.prefilled_tokens.load();
  auto j3 = std::make_shared<Job>(make_prompt(3), 4, Sampling{});
  auto j4 = std::make_shared<Job>(make_prompt(4), 4, Sampling{});
  CHECK(scheduler.submit(j3));
  CHECK(scheduler.submit(j4));
  CHECK(drain(*j3) == std::vector<int32_t>(4, 100));
  CHECK(drain(*j4) == std::vector<int32_t>(4, 100));
  CHECK(j3->cached_tokens() == 321);
  CHECK(j4->cached_tokens() == 321);
  CHECK(gen.prefilled_tokens.load() - before == 48);

  // An idle c=1 request must route to fast Solo mode (never trapped in BatchDecoder by a batch anchor).
  auto j5 = std::make_shared<Job>(make_prompt(5), 4, Sampling{});
  CHECK(scheduler.submit(j5));
  CHECK(drain(*j5) == std::vector<int32_t>(4, 100));
  CHECK(scheduler.stats().decoding == 0);
}

}  // namespace

int main() {
  test_batch_capacity_and_cancel();
  test_shared_prefix_fork();
  test_persistent_prefix_anchor_across_waves();
  test_prefix_anchor_eviction_at_full_capacity();
  test_system_prompt_anchor_boundary();
  test_cuts();
  test_cancel_fifo_bound();
  test_failure_and_close();
  std::printf(g_failures ? "FAILED (%d checks)\n" : "ok\n", g_failures);
  return g_failures ? 1 : 0;
}
