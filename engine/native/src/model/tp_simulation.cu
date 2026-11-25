#include "model/tp_simulation.h"

#include "core/device.h"
#include "kernels/attn/attn.h"
#include "kernels/dense/dense.h"
#include "kernels/moe/moe.h"
#include "model/runtime.h"

#include <type_traits>

namespace ingine::model {
namespace {

namespace attn = kernels::attn;
namespace dense = kernels::dense;
namespace moe = kernels::moe;
constexpr int H = HIDDEN, MaxM = 8;
constexpr TpLayout Local(2);
constexpr int LocalV = UNPADDED_VOCAB / 2;

dense::Matrix matrix(const Dense& d) {
  return {d.w.data, ptr<float>(d.scale), d.rows(), d.cols()};
}

void copy_rows(void* to, int stride, const void* from, int from_stride, int bytes, int count, cudaStream_t s) {
  CUDA_CHECK(cudaMemcpy2DAsync(to, stride, from, from_stride, bytes, count, cudaMemcpyDeviceToDevice, s));
}

DeviceBuffer segments(const TpLayout& tp, bool full, bool single) {
  std::vector<int> rows;
  for (int i = 0; i < num_segs(full); ++i) {
    rows.push_back(tp.weight_offset(i));
    rows.push_back(tp.weight_offset(i + 1));
    rows.push_back(single ? 0 : SEG_INPUT[i]);
  }
  DeviceBuffer b(rows.size() * sizeof(int));
  b.copy_from_host(rows.data(), b.bytes);
  return b;
}

template <typename T>
__global__ void sum_hidden(const T* a, const T* b, T* sum) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if constexpr (std::is_same_v<T, float>) sum[i] = a[i] + b[i];
  else sum[i] = __float2bfloat16_rn(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

__global__ void sum_experts(const float* a, const float* b, const float* weight, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = i / H, ch = i % H;
  float sum = 0.f;
  for (int k = 0; k < TOP_K; ++k) {
    const int slot = m * TOP_K + k;
    const int e = slot * H + ch;
    const float dot = __bfloat162float(__float2bfloat16_rn(a[e] + b[e]));
    sum = __fadd_rn(sum, __fmul_rn(weight[slot], dot));
  }
  out[i] = sum;
}

}  // namespace

struct TpSimulation::Rank {
  const TpLayout tp;
  State state;
  HeadWeights head;
  std::vector<Layer> layers;
  std::vector<DeviceBuffer> weights;
  std::array<DeviceBuffer, 3> seg;
  DeviceBuffer workspace, proj, attn_out, act, partial_f32, partial_bf16, experts, logits;

  Rank(int rank, const std::filesystem::path& dir, int max_ctx, bool dense_fp8, bool kv_fp8,
       bool head_nvfp4, cudaStream_t s)
      : tp(2, rank), state(make_state(max_ctx, kv_fp8, tp)),
        seg{segments(tp, false, true), segments(tp, false, false), segments(tp, true, false)},
        workspace(attn::workspace_bytes(MaxM, tp.hkv_local)),
        proj(MaxM * tp.projection_stride(true) * sizeof(float)),
        attn_out(MaxM * tp.q_dim_local * sizeof(__nv_bfloat16)),
        act(MaxM * TOP_K * tp.I_local * sizeof(__nv_bfloat16)),
        partial_f32(MaxM * H * sizeof(float)), partial_bf16(MaxM * H * sizeof(__nv_bfloat16)),
        experts(MaxM * TOP_K * H * sizeof(float)), logits(MaxM * LocalV * sizeof(float)) {
    workspace.fill(0);
    for (int l = 0; l < LAYERS; ++l) layers.push_back(load_layer(dir, l, dense_fp8, s, weights, tp));
    head = load_head(dir, dense_fp8, head_nvfp4, s, weights, tp);
  }
};

TpSimulation::TpSimulation(const std::filesystem::path& dir, int max_ctx, bool dense_fp8, bool kv_fp8,
                           bool head_nvfp4, cudaStream_t s)
    : sum_(MaxM * H * sizeof(float)), replica_(MaxM * H * sizeof(float)),
      replica_history_(MaxM * H * sizeof(float)) {
  for (int r = 0; r < 2; ++r)
    ranks_[r] = std::make_unique<Rank>(r, dir, max_ctx, dense_fp8, kv_fp8, head_nvfp4, s);
}

TpSimulation::~TpSimulation() = default;

void TpSimulation::reset(cudaStream_t s) {
  for (auto& rank : ranks_) reset_state(rank->state, s);
}

const State& TpSimulation::state(int rank) const { return ranks_.at(rank)->state; }

void TpSimulation::attention(const Layer& ref, Buffers& b, int visit, bool single, bool history,
                             const int64_t* pos, TpReduction reduction, cudaStream_t s) {
  const int m = b.m, q = Local.q_dim_local, kv = Local.kv_dim_local;
  const bool reconstruct = reduction == TpReduction::Reconstruct;
  const bool bf = reduction == TpReduction::Bf16;
  for (const auto& r : ranks_) {
    const Layer& layer = r->layers[ref.idx];
    const int seg = layer.full ? 2 : single ? 0 : 1;
    float* proj = static_cast<float*>(r->proj.p);
    dense::proj(ptr<__nv_bfloat16>(b.xin), m, matrix(layer.wproj), static_cast<const int*>(r->seg[seg].p),
                num_segs(layer.full), proj, false, s);

    attn::DecodeArgs a;
    a.proj = proj;
    a.M = m;
    a.hkv = r->tp.hkv_local;
    a.q_scale = ptr<float>(layer.q_scale);
    a.r_scale = ptr<float>(layer.r_scale);
    a.rel_proj = ptr<float>(layer.rel_proj);
    a.k_conv = ptr<float>(layer.full ? layer.k_static : layer.k_conv);
    a.v_conv = ptr<float>(layer.full ? layer.v_static : layer.v_conv);
    a.conv_k_state = r->state.conv_k_at(visit);
    a.conv_v_state = r->state.conv_v_at(visit);
    a.pos = pos;
    a.workspace = r->workspace.p;
    a.workspace_size = r->workspace.bytes;
    a.out = static_cast<__nv_bfloat16*>(r->attn_out.p);
    if (layer.full) {
      a.k_cache = r->state.k_full_at(FULL_INDEX[visit]).data;
      a.v_cache = r->state.v_full_at(FULL_INDEX[visit]).data;
      a.cache_rows = r->state.max_ctx;
      a.cache_fp8 = r->state.kv_fp8;
      a.log_scaling = true;
    } else {
      const int ring = RING_INDEX[visit];
      a.k_cache = r->state.k_ring_at(ring).data;
      a.v_cache = r->state.v_ring_at(ring).data;
      a.kpos = r->state.ring_pos_at(ring);
      a.cache_rows = a.window = WINDOW;
    }
    attn::attn_decode(a, s);

    if (history) {
      const int pitch = r->tp.projection_stride(layer.full);
      for (auto [dst, segment] : {std::pair{&b.hist_k, SEG_K}, std::pair{&b.hist_v, SEG_V}}) {
        auto* row = ptr<float>(*dst) + size_t(visit) * m * KV_DIM + kv * r->tp.tp_rank;
        copy_rows(row, KV_DIM * 4, proj + r->tp.activation_offset(segment), pitch * 4, kv * 4, m, s);
      }
    }
    if (reconstruct) {
      copy_rows(ptr<__nv_bfloat16>(b.attn_out) + r->tp.tp_rank * q, 4 * q, a.out, 2 * q, 2 * q, m, s);
    } else {
      dense::proj_raw(a.out, m, matrix(layer.wo), bf ? nullptr : static_cast<float*>(r->partial_f32.p),
                      bf ? static_cast<__nv_bfloat16*>(r->partial_bf16.p) : nullptr, false, s);
    }
  }

  const Rank& a = *ranks_[0];
  const Rank& z = *ranks_[1];
  auto* sum = static_cast<float*>(sum_.p);
  auto* sum16 = static_cast<__nv_bfloat16*>(sum_.p);
  if (bf)
    sum_hidden<<<m * H / 128, 128, 0, s>>>(static_cast<const __nv_bfloat16*>(a.partial_bf16.p),
                                          static_cast<const __nv_bfloat16*>(z.partial_bf16.p), sum16);
  else if (!reconstruct)
    sum_hidden<<<m * H / 128, 128, 0, s>>>(static_cast<const float*>(a.partial_f32.p),
                                          static_cast<const float*>(z.partial_f32.p), sum);
  CUDA_CHECK(cudaGetLastError());
  for (const auto& r : ranks_) {
    const bool first = r->tp.tp_rank == 0;
    float* out = first ? ptr<float>(b.y) : static_cast<float*>(replica_.p);
    float* hist = history ? (first ? ptr<float>(b.hist_attn) + size_t(visit) * m * H
                                  : static_cast<float*>(replica_history_.p)) : nullptr;
    if (reconstruct) {
      dense::proj_conv(ptr<__nv_bfloat16>(b.attn_out), m, matrix(ref.wo), ptr<float>(ref.attn_conv),
                       r->state.conv_attn_at(visit), out, hist, false, s);
    } else {
      dense::conv_residual_epilogue(bf ? nullptr : sum, bf ? sum16 : nullptr, m, dense::ConvKind::Attention,
                                    ptr<float>(r->layers[ref.idx].attn_conv), r->state.conv_attn_at(visit), out,
                                    hist, false, s);
    }
  }
}

void TpSimulation::mlp(const Layer& ref, Buffers& b, int visit, bool history, TpReduction reduction, cudaStream_t s) {
  const int m = b.m, I = Local.I_local;
  const auto fmt = static_cast<moe::Format>(ref.experts.fmt);
  const bool reconstruct = reduction == TpReduction::Reconstruct;
  const bool bf = reduction == TpReduction::Bf16, ordered = reduction == TpReduction::ExpertOrdered;
  const int* idx = ptr<int>(b.idx);
  const float* wts = ptr<float>(b.wts);
  for (const auto& r : ranks_) {
    const auto& ex = r->layers[ref.idx].experts;
    auto* act = static_cast<__nv_bfloat16*>(r->act.p);
    moe::moe_up(ptr<__nv_bfloat16>(b.xm), m, idx, ex.w13.data, ex.w13_scale.data,
                ptr<float>(ex.w13_factor), fmt, act, false, s, I);
    if (reconstruct) {
      copy_rows(ptr<__nv_bfloat16>(b.act) + r->tp.tp_rank * I, 4 * I, act, 2 * I, 2 * I, m * TOP_K, s);
    } else {
      moe::moe_down_partial(act, m, idx, wts, ex.w2.data, ex.w2_scale.data, ptr<float>(ex.w2_factor), fmt, I,
                            bf ? static_cast<__nv_bfloat16*>(r->partial_bf16.p) : nullptr,
                            !bf && !ordered ? static_cast<float*>(r->partial_f32.p) : nullptr,
                            ordered ? static_cast<float*>(r->experts.p) : nullptr, false, s);
    }
  }

  const Rank& a = *ranks_[0];
  const Rank& z = *ranks_[1];
  auto* sum = static_cast<float*>(sum_.p);
  auto* sum16 = static_cast<__nv_bfloat16*>(sum_.p);
  if (ordered)
    sum_experts<<<m * H / 128, 128, 0, s>>>(static_cast<const float*>(a.experts.p),
                                           static_cast<const float*>(z.experts.p), wts, sum);
  else if (bf)
    sum_hidden<<<m * H / 128, 128, 0, s>>>(static_cast<const __nv_bfloat16*>(a.partial_bf16.p),
                                          static_cast<const __nv_bfloat16*>(z.partial_bf16.p), sum16);
  else if (!reconstruct)
    sum_hidden<<<m * H / 128, 128, 0, s>>>(static_cast<const float*>(a.partial_f32.p),
                                          static_cast<const float*>(z.partial_f32.p), sum);
  CUDA_CHECK(cudaGetLastError());
  for (const auto& r : ranks_) {
    const bool first = r->tp.tp_rank == 0;
    float* out = first ? ptr<float>(b.y2) : static_cast<float*>(replica_.p);
    float* hist = history ? (first ? ptr<float>(b.hist_mlp) + size_t(visit) * m * H
                                  : static_cast<float*>(replica_history_.p)) : nullptr;
    if (reconstruct) {
      const auto& ex = ref.experts;
      moe::moe_down(ptr<__nv_bfloat16>(b.act), m, idx, wts, ex.w2.data, ex.w2_scale.data,
                    ptr<float>(ex.w2_factor), fmt, ptr<float>(ref.mlp_conv), r->state.conv_mlp_at(visit),
                    out, hist, false, s);
    } else {
      dense::conv_residual_epilogue(bf ? nullptr : sum, bf ? sum16 : nullptr, m, dense::ConvKind::Mlp,
                                    ptr<float>(r->layers[ref.idx].mlp_conv), r->state.conv_mlp_at(visit),
                                    out, hist, false, s);
    }
  }
}

void TpSimulation::head(Buffers& b, cudaStream_t s) {
  for (const auto& r : ranks_) {
    auto* local = static_cast<float*>(r->logits.p);
    if (r->head.nvfp4.codes)
      kernels::head::head_nvfp4(ptr<__nv_bfloat16>(b.xf), b.m, r->head.nvfp4, local, nullptr, nullptr, false, s);
    else
      dense::head(ptr<__nv_bfloat16>(b.xf), b.m, matrix(r->head.dense), local, false, s);
    copy_rows(ptr<float>(b.logits) + r->head.v_begin, UNPADDED_VOCAB * 4, local, LocalV * 4, LocalV * 4, b.m, s);
  }
}

}  // namespace ingine::model
