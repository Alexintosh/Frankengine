#include "../../apps/serve/tp_control.h"

#include <cstdio>
#include <exception>
#include <thread>

namespace {

using namespace ingine::model;
using ingine::serve::TpControl;
using ingine::serve::TpGenerator;

struct PrefillInput {
  int slot, offset;
  std::vector<int32_t> prompt;
  float temperature, top_p;
  int top_k;

  bool operator==(const PrefillInput&) const = default;
};

struct PrefillCall {
  int max_steps;
  std::vector<PrefillInput> inputs;

  bool operator==(const PrefillCall&) const = default;
};

class RecordingGenerator final : public Generator {
 public:
  std::string model_id() const override { return "recording"; }
  int max_model_len() const override { return 4096; }
  GenerateStats generate(std::span<const int32_t>, int, const Sampling&, std::span<const int32_t>,
                         const std::function<bool(std::span<const int32_t>)>&) override { return {}; }
  bool supports_batch_short_prefill() const override { return true; }

  std::vector<PrefillChunkResult> prefill_batch_short(std::span<const BatchPrefillSlot> slots, int max_steps) override {
    PrefillCall call{max_steps, {}};
    std::vector<PrefillChunkResult> results;
    bool invalid = false;
    for (const auto& slot : slots) {
      const auto& sample = slot.sampling->params;
      call.inputs.push_back({slot.slot, slot.offset, {slot.prompt.begin(), slot.prompt.end()},
                             sample.temperature, sample.top_p, sample.top_k});
      invalid |= sample.top_k != 0 || sample.top_p < 1.f;
      results.push_back({int(slot.prompt.size()), slot.offset, slot.prompt.back()});
    }
    calls.push_back(std::move(call));
    if (invalid) throw std::invalid_argument("short prefill sampling");
    return results;
  }

  std::vector<PrefillCall> calls;
};

void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

void round_trip() {
  int fds[2];
  check(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "socketpair failed");
  auto control = std::make_unique<TpControl>(fds[0]);
  TpControl worker_socket(fds[1]);
  auto root = std::make_unique<RecordingGenerator>();
  RecordingGenerator* local = root.get();
  RecordingGenerator worker;
  std::exception_ptr worker_error;
  std::jthread thread([&] {
    try {
      ingine::serve::run_tp_worker(worker, worker_socket);
    } catch (...) {
      worker_error = std::current_exception();
    }
  });

  std::vector<PrefillCall> local_calls;
  {
    TpGenerator generator(std::move(root), std::move(control));
    check(generator.supports_batch_short_prefill(), "TP wrapper hid short-prefill capability");
    BatchSampling greedy, sampled;
    greedy.params.temperature = 0.f;
    sampled.params.temperature = 0.625f;
    sampled.params.top_p = 0.875f;
    sampled.params.top_k = 17;
    const std::vector<int32_t> a{1, 2, 3, 4}, b{42, 43};
    const BatchPrefillSlot slots[]{{3, a, 2, &greedy}, {1, b, 0, &sampled}};
    bool invalid = false;
    try {
      generator.prefill_batch_short(slots, 1);
    } catch (const std::invalid_argument&) {
      invalid = true;
    }
    check(invalid, "local generator did not receive invalid sampling parameters");

    sampled.params.top_p = 1.f;
    sampled.params.top_k = 0;
    const auto results = generator.prefill_batch_short(slots);
    check(results.size() == 2 && results[0].offset == 4 && results[0].reused_tokens == 2 &&
              results[0].token == 4 && results[1].offset == 2 && results[1].token == 43,
          "incorrect local short-prefill results");
    local_calls = std::move(local->calls);
  }

  thread.join();
  if (worker_error) std::rethrow_exception(worker_error);
  check(worker.calls.size() == 2 && worker.calls == local_calls, "worker short-prefill inputs differ from rank 0");
}

}  // namespace

int main() {
  try {
    round_trip();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
  return 0;
}
