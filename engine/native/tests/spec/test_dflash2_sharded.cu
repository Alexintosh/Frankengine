#include "common/test_util.h"
#include "core/arena.h"
#include "core/device.h"
#include "kernels/dense/dense.h"
#include "kernels/head/head.h"
#include "model/dflash2.h"
#include "model/prefill.h"
#include "model/runtime.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string_view>
#include <vector>

namespace {

using namespace ingine;
using namespace ingine::model;
using namespace ingine::test;
using Bf = __nv_bfloat16;

constexpr int H = HIDDEN;
constexpr int Block = DFlash2::block;
constexpr int Proposals = Block - 1;

__global__ void f32_to_bf16(const float* in, Bf* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16(in[i]);
}

__global__ void add_f32_to_bf16(const float* a, const float* b, Bf* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16_rn(a[i] + b[i]);
}

__global__ void add_bf16(const Bf* a, const Bf* b, Bf* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16_rn(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

std::filesystem::path find_dflash2_dir() {
  if (const char* env = std::getenv("INKLING_DFLASH2_DIR"); env && *env)
    return env;
  const auto sibling = std::filesystem::path(INGINE_MODEL_DIR).parent_path() / "Inkling-2-Mini-DFlash2";
  if (std::filesystem::exists(sibling)) return sibling;
  return {};
}

struct HiddenDiff {
  float max_abs = 0.f;
  float max_ref = 0.f;
  double rmse = 0.0;
  double rel_rmse = 0.0;
  int gt_1ulp = 0;
};

HiddenDiff compare_bf16_device(const Bf* a_dev, const Bf* b_dev, std::size_t count, cudaStream_t s) {
  std::vector<Bf> a(count), b(count);
  INGINE_TEST_CUDA(cudaMemcpyAsync(a.data(), a_dev, count * sizeof(Bf), cudaMemcpyDeviceToHost, s));
  INGINE_TEST_CUDA(cudaMemcpyAsync(b.data(), b_dev, count * sizeof(Bf), cudaMemcpyDeviceToHost, s));
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));
  HiddenDiff d;
  double sq = 0.0, ref_sq = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const float va = __bfloat162float(a[i]), vb = __bfloat162float(b[i]);
    if (std::fabs(va) > d.max_ref) d.max_ref = std::fabs(va);
    const float diff = std::fabs(va - vb);
    if (diff > d.max_abs) d.max_abs = diff;
    const float ulp = std::ldexp(1.f, std::ilogb(std::max(std::fabs(va), 1e-2f)) - 7);
    if (diff > ulp * 1.01f) ++d.gt_1ulp;
    sq += double(diff) * double(diff);
    ref_sq += double(va) * double(va);
  }
  d.rmse = std::sqrt(sq / double(count));
  d.rel_rmse = ref_sq > 0.0 ? std::sqrt(sq / ref_sq) : 0.0;
  return d;
}

std::vector<int> copy_proposals(const int* dev, cudaStream_t s) {
  std::vector<int> out(Proposals);
  INGINE_TEST_CUDA(cudaMemcpyAsync(out.data(), dev, Proposals * sizeof(int), cudaMemcpyDeviceToHost, s));
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));
  return out;
}

int count_matches(const std::vector<int>& a, const std::vector<int>& b) {
  int n = 0;
  for (int i = 0; i < Proposals; ++i)
    if (a[i] == b[i]) ++n;
  return n;
}

struct CapturedPrompt {
  DeviceBuffer taps;  // bf16 [len, 4, H]
  int len = 0;
  int anchor = 0;
};

CapturedPrompt capture_real_taps(const Weights& w, const Ablation& abl, Prefiller& prefiller,
                                 State& state, int prompt_len, cudaStream_t s) {
  reset_state(state, s);
  std::vector<std::int32_t> tokens(prompt_len);
  static constexpr std::int32_t seed_seq[] = {
      1, 200008, 371, 150409, 171960, 149887, 161043, 149879,
      1234, 5678, 9101, 42, 314, 2718, 8192, 16384,
      500, 1024, 2048, 4096, 8000, 12000, 24000, 48000};
  for (int i = 0; i < prompt_len; ++i)
    tokens[i] = seed_seq[i % (sizeof(seed_seq) / sizeof(seed_seq[0]))] + (i / 24) * 7 % 50000;
  DeviceBuffer tok_dev(std::size_t(prompt_len) * sizeof(std::int32_t));
  tok_dev.copy_from_host(tokens.data(), tok_dev.bytes);
  DeviceBuffer taps_buf(std::size_t(prompt_len) * NUM_TAPS * H * sizeof(Bf));
  DeviceBuffer logits_buf(std::size_t(UNPADDED_VOCAB) * sizeof(float));
  DeviceBuffer argmax_buf(sizeof(std::int32_t));
  const int chunk = prefiller.max_chunk();
  for (int pos = 0; pos < prompt_len; pos += chunk) {
    const int T = std::min(chunk, prompt_len - pos);
    Tensor taps_slice = rm(static_cast<Bf*>(taps_buf.p) + std::size_t(pos) * NUM_TAPS * H,
                           DType::BF16, {T, NUM_TAPS, H});
    prefiller.run(state, static_cast<const std::int32_t*>(tok_dev.p) + pos, T, pos,
                  static_cast<float*>(logits_buf.p), &taps_slice, {}, nullptr, pos + T == prompt_len);
  }
  kernels::head::argmax(static_cast<const float*>(logits_buf.p), 1, UNPADDED_VOCAB, UNPADDED_VOCAB,
                        static_cast<std::int32_t*>(argmax_buf.p), s);
  std::int32_t anchor = 0;
  INGINE_TEST_CUDA(cudaMemcpyAsync(&anchor, argmax_buf.p, sizeof(anchor), cudaMemcpyDeviceToHost, s));
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));
  return CapturedPrompt{std::move(taps_buf), prompt_len, anchor};
}

float bench_ms(cudaStream_t s, int warmup_iters, int timed_iters, auto&& fn) {
  for (int i = 0; i < warmup_iters; ++i) fn();
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));
  cudaEvent_t t0, t1;
  INGINE_TEST_CUDA(cudaEventCreate(&t0));
  INGINE_TEST_CUDA(cudaEventCreate(&t1));
  INGINE_TEST_CUDA(cudaEventRecord(t0, s));
  for (int i = 0; i < timed_iters; ++i) fn();
  INGINE_TEST_CUDA(cudaEventRecord(t1, s));
  INGINE_TEST_CUDA(cudaEventSynchronize(t1));
  float total_ms = 0.f;
  INGINE_TEST_CUDA(cudaEventElapsedTime(&total_ms, t0, t1));
  INGINE_TEST_CUDA(cudaEventDestroy(t0));
  INGINE_TEST_CUDA(cudaEventDestroy(t1));
  return total_ms / float(timed_iters);
}

// Kernel-level unit check: dflash2_fp8 split-K (4096 -> 2x2048, 6912 -> 2x3456) matches full K
// within 1 ULP under FP32 reduction and tightly under BF16 reduction.
void test_split_gemv_kernels(cudaStream_t s) {
  for (int K : {4096, 6912}) {
    const int R = 2304, M = 8, K2 = K / 2;
    std::vector<Bf> hx(M * K), hw(R * K);
    for (int i = 0; i < M * K; ++i) hx[i] = __float2bfloat16_rn(std::sin(i * 0.013f) * 0.25f);
    for (int i = 0; i < R * K; ++i) hw[i] = __float2bfloat16_rn(std::cos(i * 0.017f) * 0.1f);
    DeviceBuffer dx(M * K * 2), dw(R * K * 2);
    dx.copy_from_host(hx.data(), dx.bytes);
    dw.copy_from_host(hw.data(), dw.bytes);
    DeviceBuffer dq(R * K), dsc(R * 4);
    kernels::dense::fp8_rows(static_cast<const Bf*>(dw.p), R, K, dq.p, static_cast<float*>(dsc.p), s);
    DeviceBuffer dout_full(M * R * 2);
    kernels::dense::dflash2_fp8(static_cast<const Bf*>(dx.p), {dq.p, static_cast<const float*>(dsc.p), R, K},
                                static_cast<Bf*>(dout_full.p), s);
    DeviceBuffer dx0(M * K2 * 2), dx1(M * K2 * 2), dq0(R * K2), dq1(R * K2);
    INGINE_TEST_CUDA(cudaMemcpy2DAsync(dx0.p, K2 * 2, dx.p, K * 2, K2 * 2, M, cudaMemcpyDeviceToDevice, s));
    INGINE_TEST_CUDA(cudaMemcpy2DAsync(dx1.p, K2 * 2, static_cast<const Bf*>(dx.p) + K2, K * 2, K2 * 2, M, cudaMemcpyDeviceToDevice, s));
    INGINE_TEST_CUDA(cudaMemcpy2DAsync(dq0.p, K2, dq.p, K, K2, R, cudaMemcpyDeviceToDevice, s));
    INGINE_TEST_CUDA(cudaMemcpy2DAsync(dq1.p, K2, static_cast<const uint8_t*>(dq.p) + K2, K, K2, R, cudaMemcpyDeviceToDevice, s));
    DeviceBuffer p0_f32(M * R * 4), p1_f32(M * R * 4), p0_bf16(M * R * 2), p1_bf16(M * R * 2);
    DeviceBuffer dout_f32(M * R * 2), dout_bf16(M * R * 2);
    kernels::dense::dflash2_fp8_f32(static_cast<const Bf*>(dx0.p), {dq0.p, static_cast<const float*>(dsc.p), R, K2},
                                    static_cast<float*>(p0_f32.p), s);
    kernels::dense::dflash2_fp8_f32(static_cast<const Bf*>(dx1.p), {dq1.p, static_cast<const float*>(dsc.p), R, K2},
                                    static_cast<float*>(p1_f32.p), s);
    add_f32_to_bf16<<<(M * R + 127) / 128, 128, 0, s>>>(
        static_cast<const float*>(p0_f32.p), static_cast<const float*>(p1_f32.p),
        static_cast<Bf*>(dout_f32.p), M * R);
    kernels::dense::dflash2_fp8(static_cast<const Bf*>(dx0.p), {dq0.p, static_cast<const float*>(dsc.p), R, K2},
                                static_cast<Bf*>(p0_bf16.p), s);
    kernels::dense::dflash2_fp8(static_cast<const Bf*>(dx1.p), {dq1.p, static_cast<const float*>(dsc.p), R, K2},
                                static_cast<Bf*>(p1_bf16.p), s);
    add_bf16<<<(M * R + 127) / 128, 128, 0, s>>>(
        static_cast<const Bf*>(p0_bf16.p), static_cast<const Bf*>(p1_bf16.p),
        static_cast<Bf*>(dout_bf16.p), M * R);

    const HiddenDiff d_f32 = compare_bf16_device(static_cast<const Bf*>(dout_full.p),
                                                 static_cast<const Bf*>(dout_f32.p), M * R, s);
    const HiddenDiff d_bf16 = compare_bf16_device(static_cast<const Bf*>(dout_full.p),
                                                  static_cast<const Bf*>(dout_bf16.p), M * R, s);
    std::printf("KERNEL SPLIT [K=%d -> 2x%d]: fp32 gt_1ulp=%d rel_rmse=%.6f | bf16 gt_1ulp=%d rel_rmse=%.6f\n",
                K, K2, d_f32.gt_1ulp, d_f32.rel_rmse, d_bf16.gt_1ulp, d_bf16.rel_rmse);
    INGINE_CHECK(d_f32.gt_1ulp == 0 && d_f32.rel_rmse < 1e-4);
    INGINE_CHECK(d_bf16.rel_rmse < 5e-3);
  }
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  const std::filesystem::path model_dir(INGINE_MODEL_DIR);
  const std::filesystem::path dflash2_dir = find_dflash2_dir();
  if (!std::filesystem::exists(model_dir / "model.safetensors.index.json")) {
    std::printf("skip: target model missing at %s\n", model_dir.c_str());
    return kSkip;
  }
  if (dflash2_dir.empty() || !std::filesystem::exists(dflash2_dir)) {
    std::printf("skip: DFlash2 checkpoint missing\n");
    return kSkip;
  }

  INGINE_TEST_CUDA(cudaSetDevice(0));
  cudaStream_t stream = nullptr;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  // 1. Exact split-K FP8 GEMV kernel parity.
  test_split_gemv_kernels(stream);

  // 2. Capture real target taps at ctx=30 and ctx=3000 (+4 decode rows each) using Inkling-2-Mini.
  CapturedPrompt prompt30, prompt3000;
  DeviceBuffer embed_copy, embed_norm_copy;
  std::vector<DeviceBuffer> head_storage;
  kernels::head::Nvfp4Matrix head_nvfp4;
  {
    const Weights w = load_weights(model_dir, /*dense_fp8=*/true, stream, /*head_nvfp4=*/true);
    const Ablation abl = load_ablation({});
    State state = make_state(/*max_ctx=*/3200, /*kv_fp8=*/true);
    Prefiller prefiller(w, abl, /*max_chunk=*/512, stream);

    prompt30 = capture_real_taps(w, abl, prefiller, state, 34, stream);
    prompt3000 = capture_real_taps(w, abl, prefiller, state, 3004, stream);

    const std::size_t embed_bytes = w.embed.bytes();
    const std::size_t norm_bytes = w.embed_norm.bytes();
    embed_copy = DeviceBuffer(embed_bytes);
    INGINE_TEST_CUDA(cudaMemcpyAsync(embed_copy.p, w.embed.data, embed_bytes,
                                     cudaMemcpyDeviceToDevice, stream));
    embed_norm_copy = DeviceBuffer(norm_bytes);
    INGINE_TEST_CUDA(cudaMemcpyAsync(embed_norm_copy.p, w.embed_norm.data, norm_bytes,
                                     cudaMemcpyDeviceToDevice, stream));
    head_nvfp4 = load_head(model_dir, false, true, stream, head_storage).nvfp4;
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  }
  const auto* embed = static_cast<const Bf*>(embed_copy.p);
  const auto* embed_norm = static_cast<const float*>(embed_norm_copy.p);

  std::printf("Captured real target taps: ctx30 anchor=%d, ctx3000 anchor=%d\n",
              prompt30.anchor, prompt3000.anchor);

  // 3. 2-rank simulated parity test (FP8 default precision) at ctx=30+4 and ctx=3000+4.
  {
    DFlash2 ref(dflash2_dir, stream, /*nvfp4=*/false, nullptr, 1, 0);
    DFlash2 r0(dflash2_dir, stream, /*nvfp4=*/false, nullptr, 2, 0);
    DFlash2 r1(dflash2_dir, stream, /*nvfp4=*/false, nullptr, 2, 1);
    ref.bind_head(head_nvfp4, embed, embed_norm);
    r0.bind_head(head_nvfp4, embed, embed_norm);

    DeviceBuffer ids_buf(Block * sizeof(std::int64_t));
    DeviceBuffer emb_f32(Block * H * sizeof(float));
    DeviceBuffer emb_bf16(Block * H * sizeof(Bf));

    auto run_case = [&](const char* label, const CapturedPrompt& p) {
      const int prefill_n = p.len - 4;
      const auto* taps = static_cast<const Bf*>(p.taps.p);
      const auto* decode_taps = taps + std::size_t(prefill_n) * NUM_TAPS * H;

      ref.reset();
      r0.reset();
      r1.reset();
      // Exercise both prefill commit (BF16 GEMM) and decode commit (FP8 padded path).
      ref.commit(taps, prefill_n, 0, false);
      ref.commit(decode_taps, 4, prefill_n, true);
      r0.commit(taps, prefill_n, 0, false);
      r0.commit(decode_taps, 4, prefill_n, true);
      r1.commit(taps, prefill_n, 0, false);
      r1.commit(decode_taps, 4, prefill_n, true);

      // Test the primary real anchor plus 4 diverse follow-up anchors on the same committed KV ring.
      const int anchors[5] = {p.anchor, 200008, 371, 150409, 1234};
      for (int a_idx = 0; a_idx < 5; ++a_idx) {
        const int anchor = anchors[a_idx];
        std::int64_t host_ids[Block];
        host_ids[0] = anchor;
        for (int i = 1; i < Block; ++i) host_ids[i] = DFlash2::mask_token;
        ids_buf.copy_from_host(host_ids, sizeof(host_ids));
        kernels::dense::embed(static_cast<const std::int64_t*>(ids_buf.p), Block, embed, embed_norm,
                              static_cast<float*>(emb_f32.p), false, stream);
        f32_to_bf16<<<(Block * H + 127) / 128, 128, 0, stream>>>(
            static_cast<const float*>(emb_f32.p), static_cast<Bf*>(emb_bf16.p), Block * H);

        const Bf* ref_hidden = ref.forward(static_cast<const Bf*>(emb_bf16.p));
        DeviceBuffer ref_hidden_copy(Proposals * H * sizeof(Bf));
        INGINE_TEST_CUDA(cudaMemcpyAsync(ref_hidden_copy.p, ref_hidden, ref_hidden_copy.bytes,
                                         cudaMemcpyDeviceToDevice, stream));
        const auto ref_tokens = copy_proposals(ref.propose(anchor), stream);

        for (bool fp32_reduce : {false, true}) {
          const Bf* sharded_hidden =
              DFlash2::forward_pair(r0, r1, static_cast<const Bf*>(emb_bf16.p), fp32_reduce);
          const HiddenDiff diff = compare_bf16_device(
              static_cast<const Bf*>(ref_hidden_copy.p), sharded_hidden, Proposals * H, stream);
          const auto sharded_tokens =
              copy_proposals(DFlash2::propose_pair(r0, r1, anchor, fp32_reduce), stream);
          const int matched = count_matches(ref_tokens, sharded_tokens);
          if (a_idx == 0) {
            std::printf("PARITY [%s, anchor=%d, %s]: match=%d/7 rmse=%.6f rel_rmse=%.6f max_abs=%.4f (max_ref=%.2f)\n",
                        label, anchor, fp32_reduce ? "fp32_allreduce" : "bf16_allreduce",
                        matched, diff.rmse, diff.rel_rmse, diff.max_abs, diff.max_ref);
            INGINE_CHECK(matched == Proposals);
          } else {
            std::printf("  extra [%s, anchor=%d, %s]: match=%d/7 rel_rmse=%.6f\n",
                        label, anchor, fp32_reduce ? "fp32_allreduce" : "bf16_allreduce",
                        matched, diff.rel_rmse);
            INGINE_CHECK(matched >= 4);
          }
          // Relative RMSE across 5 BF16 layers with residual conv amplification (ref PyTorch vs CUDA = 0.064 RMSE).
          INGINE_CHECK(diff.rel_rmse < (a_idx == 0 ? 0.025 : 0.04));
        }
      }
    };

    run_case("ctx=30+4", prompt30);
    run_case("ctx=3000+4", prompt3000);
  }

  // 4. 2-rank NVFP4 parity check at ctx=30+4.
  {
    DFlash2 ref(dflash2_dir, stream, /*nvfp4=*/true, nullptr, 1, 0);
    DFlash2 r0(dflash2_dir, stream, /*nvfp4=*/true, nullptr, 2, 0);
    DFlash2 r1(dflash2_dir, stream, /*nvfp4=*/true, nullptr, 2, 1);
    ref.bind_head(head_nvfp4, embed, embed_norm);
    r0.bind_head(head_nvfp4, embed, embed_norm);
    const auto* taps = static_cast<const Bf*>(prompt30.taps.p);
    ref.commit(taps, 30, 0, false);
    ref.commit(taps + std::size_t(30) * NUM_TAPS * H, 4, 30, true);
    r0.commit(taps, 30, 0, false);
    r0.commit(taps + std::size_t(30) * NUM_TAPS * H, 4, 30, true);
    r1.commit(taps, 30, 0, false);
    r1.commit(taps + std::size_t(30) * NUM_TAPS * H, 4, 30, true);
    const auto ref_tokens = copy_proposals(ref.propose(prompt30.anchor), stream);
    const auto sharded_tokens = copy_proposals(DFlash2::propose_pair(r0, r1, prompt30.anchor, false), stream);
    const int matched = count_matches(ref_tokens, sharded_tokens);
    std::printf("PARITY [nvfp4 ctx=30+4, bf16_allreduce]: match=%d/7\n", matched);
    INGINE_CHECK(matched == Proposals);
  }

  // 5. Single-rank CUDA-graph latency benchmark: unsharded TP=1 vs half-shard TP=2 rank 0.
  {
    DFlash2 full_bb(dflash2_dir, stream, /*nvfp4=*/false, nullptr, 1, 0);
    DFlash2 half_bb(dflash2_dir, stream, /*nvfp4=*/false, nullptr, 2, 0);
    full_bb.bind_backbone_only(embed, embed_norm);
    half_bb.bind_backbone_only(embed, embed_norm);
    full_bb.warmup();
    half_bb.warmup();

    const auto* taps = static_cast<const Bf*>(prompt3000.taps.p);
    for (int ctx : {64, 2048}) {
      full_bb.reset();
      half_bb.reset();
      full_bb.commit(taps, ctx, 0, false);
      half_bb.commit(taps, ctx, 0, false);

      const float full_propose_ms = bench_ms(stream, 10, 50, [&] { full_bb.propose(prompt3000.anchor); });
      const float half_propose_ms = bench_ms(stream, 10, 50, [&] { half_bb.propose(prompt3000.anchor); });
      int pos_f = ctx, pos_h = ctx;
      const float full_c4_ms = bench_ms(stream, 5, 40, [&] {
        full_bb.commit(taps, 4, pos_f, true);
        pos_f += 4;
      });
      const float half_c4_ms = bench_ms(stream, 5, 40, [&] {
        half_bb.commit(taps, 4, pos_h, true);
        pos_h += 4;
      });

      std::printf(
          "BENCH [ctx=%4d]: backbone_propose unsharded=%.3f ms -> half_shard=%.3f ms (delta=-%.3f ms, +0.16ms RoCE => est=%.3f ms) | "
          "decode_commit(n=4) unsharded=%.3f ms -> half_shard=%.3f ms (delta=-%.3f ms)\n",
          ctx, full_propose_ms, half_propose_ms, full_propose_ms - half_propose_ms,
          half_propose_ms + 0.16f, full_c4_ms, half_c4_ms, full_c4_ms - half_c4_ms);
      INGINE_CHECK(half_propose_ms < full_propose_ms * 0.72f);
    }
  }

  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
  std::printf("PASS spec_dflash2_sharded\n");
  return 0;
}
