#include "common/test_util.h"
#include "core/arena.h"
#include "kernels/batch/dense.h"
#include "kernels/draft/dflash2_selector.h"
#include "kernels/draft/draft.h"
#include "kernels/head/nvfp4_head.h"
#include "model/dflash2.h"
#include "model/tp_layout.h"

#include <array>
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

struct Outcome {
  std::vector<std::uint16_t> hidden;
  std::vector<int> tokens;
  std::vector<float> confidence;
  bool operator==(const Outcome&) const = default;
};

std::vector<Outcome> collect(const std::filesystem::path& checkpoint, const Inputs& in, cudaStream_t stream,
                             int tp, bool nvfp4, bool fused) {
  INGINE_CHECK(setenv("INGINE_DFLASH2_FUSED_PROJ", fused ? "1" : "0", 1) == 0);
  INGINE_CHECK(setenv("INGINE_DFLASH2_FUSED_HEAD", fused ? "1" : "0", 1) == 0);
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
  return outcomes;
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
      }
    }
  }
  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
  std::puts("PASS model_dflash2_fused");
}
