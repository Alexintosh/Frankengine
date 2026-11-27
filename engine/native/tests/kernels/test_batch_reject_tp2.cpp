#include "common/test_util.h"
#include "kernels/batch/dense.h"
#include "kernels/common/pico12.cuh"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>

namespace {

using namespace ingine::test;
namespace batch = ingine::kernels::batch;
namespace head = ingine::kernels::head;
namespace pico12 = ingine::kernels::pico12;

constexpr int H = 2304, V = 286, K = batch::RejectSpec::kSupport;
constexpr float kNoScore = -std::numeric_limits<float>::infinity();

enum class Format { bf16, pico12, nvfp4 };

struct PackedShard {
  pico12::Packed host;
  DeviceArray<uint8_t> device;

  PackedShard(const __nv_bfloat16* weights, int rows)
      : host(pico12::pack_pico12_host(weights, rows, H)), device(host.bytes) {
    INGINE_TEST_CUDA(cudaMemcpy(device.get(), host.data.get(), host.bytes, cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  }
};

void check_shards(Format format, int rows, int split, cudaStream_t stream) {
  std::vector<__nv_bfloat16> weights(V * H, float_to_bf16(0.f));
  std::vector<uint8_t> codes(V * (H / 2));
  constexpr std::array dots{0.f, 1.f, 2.f, 3.f, -1.f, -2.f, -3.f};
  constexpr std::array<uint8_t, 7> fp4{0, 2, 4, 5, 10, 12, 13};
  for (int v = 0; v < V; ++v) {
    weights[v * H] = float_to_bf16(dots[v % dots.size()]);
    codes[v * (H / 2)] = fp4[v % fp4.size()];
  }
  // Two equally strong support tokens straddle the split. If rank 0's is rejected, rank 1's
  // has most of the residual mass at the lower temperature.
  for (const int v : {split - 1, split}) {
    weights[v * H + 1] = float_to_bf16(3.f);
    codes[v * (H / 2)] |= 5 << 4;
  }
  const DeviceArray<__nv_bfloat16> bf16_weights(weights);
  const PackedShard packed_full(weights.data(), V), packed_left(weights.data(), split),
      packed_right(weights.data() + split * H, V - split);
  const DeviceArray<uint8_t> nvfp4_codes(codes), nvfp4_scales(std::vector<uint8_t>(V * (H / 16), 0x38));

  std::vector<__nv_bfloat16> activations(rows * H, float_to_bf16(0.f));
  std::vector<float> temps(rows, 1.f), probs(rows * K, 0.f);
  std::vector<int32_t> ids(rows * K, -1), picked(rows, -1);
  std::vector<uint8_t> active(rows, 1);
  for (int r = 1; r < rows; ++r) activations[r * H] = float_to_bf16(1.f);
  active[0] = active[1] = 0;
  temps[0] = temps[2] = 0.f;
  for (int r = 3; r < rows; ++r) {
    ids[r * K] = split - 1;
    ids[r * K + 1] = split;
    picked[r] = 0;
    probs[r * K] = 1.f;
  }
  picked[3] = 1;
  probs[3 * K + 1] = 0x1p-20f;
  probs[3 * K] = 1.f - probs[3 * K + 1];  // Rank 1 acceptance probability is one.
  activations[4 * H] = float_to_bf16(0.f);
  activations[4 * H + 1] = float_to_bf16(4.f);
  temps[4] = 0.75f;
  if (split < K) {
    std::iota(ids.begin() + 5 * K, ids.begin() + 5 * K + split + 1, 0);
    // Every token in the first shard is in the proposal support.
  }

  const DeviceArray<__nv_bfloat16> input(activations);
  const DeviceArray<float> d_temps(temps), d_probs(probs);
  const DeviceArray<int32_t> d_ids(ids), d_picked(picked);
  const DeviceArray<uint8_t> d_active(active);
  DeviceArray<uint64_t> d_seed(1);
  DeviceArray<int32_t> full(rows), merged(rows);
  DeviceArray<batch::RejectShardPartial> shards(2 * rows);
  const std::size_t count = format == Format::nvfp4 ? batch::head_partials_nvfp4(rows, V)
                                                  : batch::head_partials(rows, V);
  DeviceArray<int2> partials(count);
  DeviceArray<float> scores(std::vector<float>(rows * K, 123.f)), lse(2 * count);
  const batch::RejectSpec reject{d_active.get(), d_ids.get(), d_probs.get(), d_picked.get(),
                                scores.get(), lse.get()};

  auto launch = [&](int begin, int width, const PackedShard& packed, int32_t* out,
                    batch::RejectShardPartial* shard_out, const uint64_t* seed) {
    if (format == Format::nvfp4) {
      const head::Nvfp4Matrix w{nvfp4_codes.get() + begin * (H / 2), nvfp4_scales.get() + begin * (H / 16),
                               1.f, width, begin};
      batch::head_sample_nvfp4(input.get(), H, w, rows, d_temps.get(), 123, nullptr, 0, partials.get(),
                              out, stream, nullptr, seed, &reject, shard_out);
    } else {
      const bool pico = format == Format::pico12;
      const void* w = pico ? static_cast<const void*>(packed.device.get()) : bf16_weights.get() + begin * H;
      batch::head_sample(input.get(), H, w, H, rows, width, H, d_temps.get(), 123, nullptr, 0, partials.get(),
                         out, stream, begin, nullptr, seed, pico ? &packed.host.layout : nullptr, &reject, shard_out);
    }
  };
  auto launch_sharded = [&](const uint64_t* seed) {
    launch(0, split, packed_left, nullptr, shards.get(), seed);
    launch(split, V - split, packed_right, nullptr, shards.get() + rows, seed);
    batch::head_reject_merge_tp2(shards.get(), split, d_temps.get(), 123, seed, reject, rows, merged.get(), stream);
  };

  launch(0, V, packed_full, full.get(), nullptr, nullptr);
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  const auto expected_scores = scores.download();
  launch_sharded(nullptr);
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  INGINE_CHECK(merged.download() == full.download());
  const auto records = shards.download();
  for (int r = 0; r < rows; ++r) {
    for (int j = 0; j < K; ++j) {
      const int i = r * K + j;
      const int id = ids[i];
      const bool scored = active[r] && temps[r] > 0.f && id >= 0;
      INGINE_CHECK(records[r].target_scores[j] == (scored && id < split ? expected_scores[i] : kNoScore));
      INGINE_CHECK(records[rows + r].target_scores[j] == (scored && id >= split ? expected_scores[i] : kNoScore));
    }
  }
  if (split < K) {
    INGINE_CHECK(records[5].best_token == std::numeric_limits<int32_t>::max());
    INGINE_CHECK(records[5].outside_max == kNoScore && records[5].outside_sum == 0.f);
  }

  cudaGraph_t graph;
  cudaGraphExec_t exec;
  INGINE_TEST_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  launch(0, V, packed_full, full.get(), nullptr, d_seed.get());
  launch_sharded(d_seed.get());
  INGINE_TEST_CUDA(cudaStreamEndCapture(stream, &graph));
  std::size_t nodes = 0;
  INGINE_TEST_CUDA(cudaGraphGetNodes(graph, nullptr, &nodes));
  INGINE_CHECK(nodes == 7);  // Three head passes, their reductions and the merge.
  INGINE_TEST_CUDA(cudaGraphInstantiate(&exec, graph, 0));
  Rng seeds(817);
  bool support_residual = false, outside_residual = false, seed_changed_token = false;
  int32_t first_sampled = -1;
  for (int i = 0; i < 24; ++i) {
    d_seed.upload({seeds.next()});
    INGINE_TEST_CUDA(cudaGraphLaunch(exec, stream));
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    const auto actual = merged.download();
    INGINE_CHECK(actual == full.download());
    INGINE_CHECK(actual[0] == 0 && actual[3] == split);
    if (i == 0) first_sampled = actual[1];
    seed_changed_token |= actual[1] != first_sampled;
    support_residual |= actual[4] == split;
    outside_residual |= actual[5] != split - 1 && actual[5] != split;
  }
  INGINE_CHECK(support_residual && outside_residual && seed_changed_token);
  INGINE_TEST_CUDA(cudaGraphExecDestroy(exec));
  INGINE_TEST_CUDA(cudaGraphDestroy(graph));
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  cudaStream_t stream;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (const Format format : {Format::bf16, Format::pico12, Format::nvfp4}) {
    check_shards(format, 6, 7, stream);
    check_shards(format, 17, 137, stream);
    check_shards(format, 43, 137, stream);
  }
  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
  std::printf("PASS: two-shard rejection sampling matches the unsharded head\n");
  return 0;
}
