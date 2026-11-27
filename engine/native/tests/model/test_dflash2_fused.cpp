#include "common/test_util.h"
#include "core/arena.h"
#include "ingine/generator.h"
#include "kernels/batch/dense.h"
#include "kernels/draft/dflash2_selector.h"
#include "kernels/draft/draft.h"
#include "kernels/head/nvfp4_head.h"
#include "model/dflash2.h"
#include "model/tp_layout.h"

#include <algorithm>
#include <array>
#include <bit>
#include <numeric>
#include <filesystem>
#include <memory>
#include <string_view>

namespace {
using namespace ingine;
using namespace ingine::model;
using namespace ingine::test;
namespace kb = kernels::batch;
namespace kd = kernels::draft;
namespace kh = kernels::head;
using Bf = __nv_bfloat16;
constexpr int H = HIDDEN, G = DFlash2::block - 1, S = kd::DFLASH2_SUPPORT;
constexpr int HeadRows = kd::CANDIDATES;
constexpr int LongPrompt = 123, DecodeRows = 3;

template <class T>
std::vector<T> copy_device(const void* device, int count, cudaStream_t stream) {
  std::vector<T> host(count);
  INGINE_TEST_CUDA(cudaMemcpyAsync(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  return host;
}

struct Inputs {
  DeviceBuffer codes{HeadRows * H / 2}, scales{HeadRows * H / 16};
  DeviceBuffer embed{std::size_t(DFlash2::mask_token + 1) * H * sizeof(Bf)}, norms{H * sizeof(float)};
  DeviceBuffer embeddings{DFlash2::block * H * sizeof(Bf)};
  DeviceBuffer taps{(LongPrompt + DecodeRows) * 4 * H * sizeof(Bf)};
  kh::Nvfp4Matrix head;

  explicit Inputs(cudaStream_t stream) {
    // Each nonzero dot has one term, so the raw and BF16-rounded heads
    // have the same ordering and scores. Each group contains an exact tie.
    Rng rng(92);
    std::vector<Bf> head_weights(HeadRows * H, float_to_bf16(0.f));
    for (int i = 0; i < S; ++i) {
      head_weights[(101 + i) * H] = float_to_bf16(1.f);
      head_weights[(301 + i) * H] = float_to_bf16(-1.f);
    }
    DeviceBuffer weights(head_weights.size() * sizeof(Bf));
    weights.copy_from_host(head_weights.data(), weights.bytes);
    auto* packed = static_cast<std::uint8_t*>(codes.p);
    auto* packed_scales = static_cast<std::uint8_t*>(scales.p);
    const float scale = kh::nvfp4_outer_scale(static_cast<const Bf*>(weights.p), head_weights.size(), stream);
    kh::nvfp4_pack_rows(static_cast<const Bf*>(weights.p), HeadRows, scale, packed, packed_scales, stream);
    head = {packed, packed_scales, scale, HeadRows, 0};
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));

    std::vector<Bf> rows(DFlash2::block * H);
    for (auto& x : rows) x = float_to_bf16(float(rng.uniform(-0.3, 0.3)));
    embeddings.copy_from_host(rows.data(), embeddings.bytes);
    embed.copy_from_host(rows.data(), 3 * H * sizeof(Bf));
    embed.copy_from_host(rows.data() + 3 * H, H * sizeof(Bf), std::size_t(DFlash2::mask_token) * H * sizeof(Bf));
    std::vector<float> norm(H, 1.f);
    norms.copy_from_host(norm.data(), norms.bytes);
    std::vector<Bf> context(taps.bytes / sizeof(Bf));
    for (auto& x : context) x = float_to_bf16(float(rng.uniform(-0.12, 0.12)));
    taps.copy_from_host(context.data(), taps.bytes);
  }
};

void sharded_head_offset(const Inputs& in, cudaStream_t stream) {
  auto shard = in.head;
  shard.v_begin = 100029;
  DeviceBuffer workspace(kb::head_topk16_nvfp4_workspace_bytes(G, shard.rows));
  DeviceBuffer values(G * S * sizeof(float)), ids(G * S * sizeof(int));
  kb::head_topk16_nvfp4(static_cast<const Bf*>(in.embeddings.p), H, shard, G, workspace.p,
                        static_cast<float*>(values.p), static_cast<int*>(ids.p), stream);
  const auto top_ids = copy_device<int>(ids.p, G * S, stream);
  const auto top_values = copy_device<float>(values.p, G * S, stream);
  const auto inputs = copy_device<Bf>(in.embeddings.p, G * H, stream);
  for (int row = 0; row < G; ++row) {
    const float x = bf16_to_float(inputs[row * H]);
    const int first = x > 0.f ? 101 : x < 0.f ? 301 : 0;
    for (int j = 0; j < S; ++j) {
      INGINE_CHECK(top_ids[row * S + j] == shard.v_begin + first + j);
      INGINE_CHECK(top_values[row * S + j] == std::abs(x));
    }
  }
}

// Exercise local sharded ranking and the selector without requiring a second
// process for the candidate all-gather.
void sharded_selector_parity(DFlash2& draft, const Inputs& in, cudaStream_t stream) {
  auto shard = in.head;
  shard.v_begin = 100029;
  const std::size_t ws_bytes = std::max(kh::nvfp4_topk16_small_workspace(G, shard.rows),
                                        kd::topk_workspace_bytes(G, shard.rows));
  DeviceBuffer workspace(ws_bytes), unary(G * shard.rows * sizeof(float));
  DeviceBuffer values(G * S * sizeof(float)), ids(G * S * sizeof(int)), anchor(sizeof(int));
  DeviceBuffer pool_values(G * kd::CANDIDATES * sizeof(float)), pool_ids(G * kd::CANDIDATES * sizeof(int));
  const int anchor_id = 1;
  anchor.copy_from_host(&anchor_id, sizeof(anchor_id));
  const auto* hidden = draft.forward(static_cast<const Bf*>(in.embeddings.p));
  kh::project_nvfp4(hidden, G, H, shard, static_cast<float*>(unary.p), stream);
  INGINE_TEST_CUDA(cudaMemsetAsync(workspace.p, 0, workspace.bytes, stream));
  kd::topk_candidates(static_cast<const float*>(unary.p), G, shard.rows, nullptr,
                      static_cast<float*>(pool_values.p), static_cast<int*>(pool_ids.p), workspace.p, stream);
  const auto legacy_values = copy_device<float>(pool_values.p, G * kd::CANDIDATES, stream);
  const auto legacy_ids = copy_device<int>(pool_ids.p, G * kd::CANDIDATES, stream);
  std::vector<int> order(kd::CANDIDATES), want_ids(G * S);
  std::vector<float> want_values(G * S);
  for (int m = 0; m < G; ++m) {
    const int start = m * kd::CANDIDATES;
    std::iota(order.begin(), order.end(), start);
    std::partial_sort(order.begin(), order.begin() + S, order.end(), [&](int a, int b) {
      return legacy_values[a] != legacy_values[b] ? legacy_values[a] > legacy_values[b]
                                                  : legacy_ids[a] < legacy_ids[b];
    });
    for (int k = 0; k < S; ++k) {
      want_ids[m * S + k] = legacy_ids[order[k]] + shard.v_begin;
      want_values[m * S + k] = legacy_values[order[k]];
    }
  }
  ids.copy_from_host(want_ids.data(), ids.bytes);
  values.copy_from_host(want_values.data(), values.bytes);
  auto select = [&] {
    return copy_device<int>(draft.select(static_cast<const int*>(ids.p), static_cast<const float*>(values.p),
                                          static_cast<const int*>(anchor.p)), G, stream);
  };
  const auto legacy_proposals = select();
  const auto legacy_conf = copy_device<float>(draft.confidence(), G, stream);
  kh::project_nvfp4_topk16_small(hidden, G, H, shard, nullptr, static_cast<float*>(values.p),
                                 static_cast<int*>(ids.p), workspace.p, stream);
  INGINE_CHECK(copy_device<int>(ids.p, G * S, stream) == want_ids);
  const auto got_values = copy_device<float>(values.p, G * S, stream);
  for (int i = 0; i < G * S; ++i)
    INGINE_CHECK(std::bit_cast<std::uint32_t>(got_values[i]) == std::bit_cast<std::uint32_t>(want_values[i]));
  INGINE_CHECK(select() == legacy_proposals);
  INGINE_CHECK(copy_device<float>(draft.confidence(), G, stream) == legacy_conf);
}

void wrapped_advance_parity(DFlash2& draft, const Inputs& in, cudaStream_t stream) {
  constexpr int Position = 2050, Width = DFlash2::block;
  constexpr int Next = 2;
  constexpr float Temperature = 0.83f;
  constexpr std::uint64_t Seed = 97;
  const auto* taps = static_cast<const Bf*>(in.taps.p);
  draft.reset();
  for (int start = 0; start < Position;) {
    const int count = std::min(LongPrompt, Position - start);
    draft.commit(taps, count, start);
    start += count;
  }
  DeviceBuffer prefix(draft.checkpoint_bytes()), ring(draft.checkpoint_bytes());
  draft.save_checkpoint(prefix.p);
  draft.commit(taps, DecodeRows, Position, true);
  const auto want = copy_device<int>(draft.propose(Next, Temperature, Seed), G, stream);
  const auto want_conf = copy_device<float>(draft.confidence(), G, stream);
  draft.restore_checkpoint(prefix.p, Position);

  DeviceBuffer control(3 * sizeof(int));
  const auto* args = static_cast<const int*>(control.p);
  kd::RoundInputs round;
  round.nwidths = 2;
  round.widths[0] = 4;
  round.widths[1] = Width;
  round.taps[0] = taps;
  round.taps[1] = taps + 4 * 4 * H;
  round.m = args;
  round.accepted = args + 1;
  round.next_greedy = args + 2;
  draft.bind_round(round);
  {
    const int inputs[] = {4, DecodeRows - 1, Next};
    control.copy_from_host(inputs, sizeof(inputs));
    INGINE_CHECK(draft.advance(Position, true, Temperature, Seed));
    draft.advanced(DecodeRows);
    INGINE_CHECK(draft.length() == Position + DecodeRows);
    INGINE_CHECK(copy_device<int>(draft.tokens_device(), G, stream) == want);
    INGINE_CHECK(copy_device<float>(draft.confidence(), G, stream) == want_conf);
  }
  draft.restore_checkpoint(prefix.p, Position);
  {
    const int inputs[] = {Width, Width - 1, Next};
    control.copy_from_host(inputs, sizeof(inputs));
    INGINE_CHECK(draft.advance(Position, true, 0.f, 0));
    draft.advanced(Width);
    draft.trim_advance(round.taps[1], Position, 2, Width);
    INGINE_CHECK(draft.length() == Position + 2);
    draft.save_checkpoint(ring.p);
    const auto trimmed = copy_device<std::uint16_t>(ring.p, ring.bytes / sizeof(std::uint16_t), stream);
    draft.adopt_checkpoint(Position);
    draft.restore_checkpoint(prefix.p, Position);
    draft.commit(round.taps[1], 2, Position, true);
    draft.save_checkpoint(ring.p);
    INGINE_CHECK(copy_device<std::uint16_t>(ring.p, ring.bytes / sizeof(std::uint16_t), stream) == trimmed);
  }
}

struct Outcome {
  std::vector<std::uint16_t> hidden;
  std::vector<int> tokens;
  std::vector<float> confidence;
  bool operator==(const Outcome&) const = default;
};

std::vector<Outcome> collect(const std::filesystem::path& checkpoint, const Inputs& in, cudaStream_t stream,
                             int tp, bool nvfp4, bool fused, bool small = false) {
  INGINE_CHECK(setenv("INGINE_DFLASH2_FUSED_PROJ", fused ? "1" : "0", 1) == 0);
  INGINE_CHECK(setenv("INGINE_DFLASH2_FUSED_HEAD", fused ? "1" : "0", 1) == 0);
  INGINE_CHECK(setenv("INGINE_DFLASH2_GEMV_TOPK", small ? "1" : "0", 1) == 0);
  DFlash2 rank0(checkpoint, stream, nvfp4, nullptr, tp, 0);
  rank0.bind_head(in.head, static_cast<const Bf*>(in.embed.p), static_cast<const float*>(in.norms.p));
  std::unique_ptr<DFlash2> rank1;
  if (tp == 2) rank1 = std::make_unique<DFlash2>(checkpoint, stream, nvfp4, nullptr, 2, 1);
  const auto* taps = static_cast<const Bf*>(in.taps.p);
  const auto* embeddings = static_cast<const Bf*>(in.embeddings.p);
  std::vector<Outcome> outcomes;
  for (int prefill : {3, LongPrompt}) {
    for (auto* rank : {&rank0, rank1.get()}) {
      if (!rank) continue;
      rank->reset();
      rank->commit(taps, prefill, 0);
      rank->commit(taps + prefill * 4 * H, DecodeRows, prefill, true);
    }
    const auto* hidden = rank1 ? DFlash2::forward_pair(rank0, *rank1, embeddings, true) : rank0.forward(embeddings);
    auto hidden_bits = copy_device<std::uint16_t>(hidden, G * H, stream);
    for (int anchor : {1, 2}) {
      const auto* greedy = rank1 ? DFlash2::propose_pair(rank0, *rank1, anchor, true) : rank0.propose(anchor);
      outcomes.push_back({hidden_bits, copy_device<int>(greedy, G, stream),
                          copy_device<float>(rank0.confidence(), G, stream)});
    }
    if (!rank1) {
      constexpr std::uint64_t seed = 0xd9119873aa502638ull;
      const auto sample = [&] {
        const auto tokens = copy_device<int>(rank0.propose(2, 0.83f, seed), G, stream);
        const auto ids = copy_device<int>(rank0.support_ids(), G * S, stream);
        const auto probs = copy_device<float>(rank0.support_probs(), G * S, stream);
        const auto picked = copy_device<int>(rank0.support_picked(), G, stream);
        for (int i = 0; i < G; ++i) {
          INGINE_CHECK(picked[i] >= 0 && picked[i] < S && ids[i * S + picked[i]] == tokens[i]);
          float total = 0.f;
          for (int j = 0; j < S; ++j) total += probs[i * S + j];
          INGINE_CHECK(std::abs(total - 1.f) < 2e-6);
        }
        return tokens;
      };
      const auto sampled = sample();
      INGINE_CHECK(copy_device<int>(rank0.propose(2), G, stream) == outcomes.back().tokens);
      INGINE_CHECK(sample() == sampled);
    }
  }
  if (tp == 1 && small && !nvfp4) {
    if (std::string_view(std::getenv("INKLING_DFLASH2_PRECISION")) == "fp8")
      sharded_selector_parity(rank0, in, stream);
    wrapped_advance_parity(rank0, in, stream);
  }
  return outcomes;
}
struct Generation {
  std::vector<int> tokens, batches;
  int rounds = 0, accepted = 0, reused = 0;
  bool operator==(const Generation&) const = default;
};

std::vector<Generation> generate(const std::filesystem::path& target, const std::filesystem::path& draft,
                                 bool device) {
  INGINE_CHECK(setenv("INGINE_DFLASH2_DEVICE_LOOP", device ? "1" : "0", 1) == 0);
  RuntimeOptions options;
  options.model_dir = target;
  options.dflash2_dir = draft;
  options.draft_type = "dflash2";
  options.spec = options.head_nvfp4 = options.kv_fp8 = true;
  options.spec_widths = SpecWidths::Fine;
  options.max_ctx = 512;
  options.prefill_chunk = 128;
  options.park_mb = 0;
  auto generator = make_generator(options);
  auto run = [&](const std::vector<int>& prompt, int limit, const Sampling& sampling) {
    Generation result;
    const auto stats = generator->generate(prompt, limit, sampling, {}, [&](std::span<const int> batch) {
      result.batches.push_back(int(batch.size()));
      result.tokens.insert(result.tokens.end(), batch.begin(), batch.end());
      return true;
    });
    INGINE_CHECK(stats.used_dflash2 && stats.generated_tokens == limit && stats.spec_trace.empty());
    result.rounds = stats.spec_rounds;
    result.accepted = stats.spec_accepted;
    result.reused = stats.reused_tokens;
    return result;
  };
  std::vector<Generation> results;
  for (float temperature : {0.f, 0.83f}) {
    Sampling sampling;
    sampling.temperature = temperature;
    sampling.seed = 97;
    // Repeated in-vocabulary IDs give the model a predictable continuation.
    std::vector<int> prompt(116, 198);
    prompt.front() = temperature > 0.f ? 17 : 23;

    auto full = run(prompt, 32, sampling);
    INGINE_CHECK(full.rounds > 1);
    int limit = 0;
    for (int batch : full.batches) {
      if (batch > 1) break;
      limit += batch;
    }
    limit = limit < 31 ? limit + 1 : 2;

    // Force a fresh prefill for the shortened run, then check the exact live
    // prefix reused by its next turn.
    run({998, 999}, 2, sampling);
    auto early = run(prompt, limit, sampling);
    INGINE_CHECK(std::equal(early.tokens.begin(), early.tokens.end(), full.tokens.begin()));
    auto follow = prompt;
    follow.insert(follow.end(), early.tokens.begin(), early.tokens.end());
    follow.push_back(209);
    auto continuation = run(follow, 12, sampling);
    INGINE_CHECK(continuation.reused == int(prompt.size() + early.tokens.size()) - 1);
    results.push_back(std::move(full));
    results.push_back(std::move(early));
    results.push_back(std::move(continuation));
  }
  return results;
}

void device_loop_parity(const std::filesystem::path& checkpoint) {
  const std::filesystem::path target(INGINE_MODEL_DIR);
  if (!std::filesystem::exists(target / "model.safetensors.index.json") &&
      !std::filesystem::exists(target / "model.safetensors")) {
    std::printf("skip: target checkpoint missing at %s\n", target.c_str());
    return;
  }
  INGINE_CHECK(setenv("INKLING_DFLASH2_PRECISION", "fp8", 1) == 0);
  INGINE_CHECK(setenv("INGINE_DFLASH2_FUSED_HEAD", "0", 1) == 0);
  INGINE_CHECK(setenv("INGINE_DFLASH2_GEMV_TOPK", "1", 1) == 0);
  INGINE_CHECK(setenv("INGINE_DFLASH2_EXACT_REJECT", "1", 1) == 0);
  INGINE_CHECK(setenv("INGINE_DFLASH2_ADAPTIVE_EMA", "1", 1) == 0);
  INGINE_CHECK(setenv("INGINE_DFLASH2_SAMPLED", "1", 1) == 0);
  INGINE_CHECK(setenv("INGINE_DFLASH2_SHORTLIST", "8192", 1) == 0);
  const auto host = generate(target, checkpoint, false);
  const auto device = generate(target, checkpoint, true);
  INGINE_CHECK(host == device);
  std::puts("PASS solo DFlash2 device loop greedy and sampled generation, early stop and live prefix");
}
}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  const char* dir = std::getenv("INKLING_DFLASH2_DIR");
  const auto checkpoint = dir && *dir ? std::filesystem::path(dir) :
      std::filesystem::path(INGINE_MODEL_DIR).parent_path() / "Inkling-2-Mini-DFlash2";
  if (!std::filesystem::exists(checkpoint)) {
    std::printf("skip: DFlash2 checkpoint missing at %s\n", checkpoint.c_str());
    return kSkip;
  }
  unsetenv("INGINE_DFLASH2_SHORTLIST");
  unsetenv("INKLING_DFLASH2_DUMP");
  unsetenv("INKLING_DFLASH2_CUMULATIVE_CONFIDENCE");
  unsetenv("INKLING_DFLASH2_PROFILE");
  cudaStream_t stream;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  {
    const Inputs inputs(stream);
    sharded_head_offset(inputs, stream);
    for (const char* precision : {"fp8", "hybrid", "nvfp4", "bf16"}) {
      const bool nvfp4 = std::string_view(precision) == "nvfp4";
      if (nvfp4) unsetenv("INKLING_DFLASH2_PRECISION");
      else INGINE_CHECK(setenv("INKLING_DFLASH2_PRECISION", std::string_view(precision) == "bf16" ? "" : precision, 1) == 0);
      for (int tp : {1, 2}) {
        const auto unfused = collect(checkpoint, inputs, stream, tp, nvfp4, false);
        const auto fused = collect(checkpoint, inputs, stream, tp, nvfp4, true);
        std::printf("DFlash2 fused parity, precision=%s TP=%d\n", precision, tp);
        INGINE_CHECK(fused == unfused);
        const auto small = collect(checkpoint, inputs, stream, tp, nvfp4, false, true);
        INGINE_CHECK(small == unfused);
      }
    }
  }
  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
  device_loop_parity(checkpoint);
  std::puts("PASS model_dflash2_fused");
}
