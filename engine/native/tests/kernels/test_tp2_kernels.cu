#include "common/test_util.h"

#include "kernels/attn/attn.h"
#include "kernels/dense/dense.h"
#include "kernels/head/nvfp4_head.h"
#include "kernels/moe/moe.h"
#include "model/runtime.h"
#include "model/prefill.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <limits>
#include <string_view>

using namespace ingine;
using namespace ingine::model;
using namespace ingine::test;
namespace attn = ingine::kernels::attn;
namespace dense = ingine::kernels::dense;
namespace moe = ingine::kernels::moe;
namespace head = ingine::kernels::head;

namespace {

constexpr int H = HIDDEN;
cudaStream_t stream;

template <class T>
void zero(DeviceArray<T>& b) { INGINE_TEST_CUDA(cudaMemsetAsync(b.get(), 0, b.count * sizeof(T), stream)); }

void copy_rows(void* dst, int dst_pitch, const void* src, int src_pitch, int bytes, int rows) {
  INGINE_TEST_CUDA(cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, bytes, rows, cudaMemcpyDeviceToDevice, stream));
}

__global__ void different(const uint8_t* dst, int dp, const uint8_t* src, int sp, int bytes, int rows, int* fail) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size_t(dp) * rows;
       i += size_t(gridDim.x) * blockDim.x) {
    const int offset = i % dp;
    if (dst[i] != (offset < bytes ? src[i / dp * sp + offset] : 0)) {
      atomicExch(fail, 1);
      return;
    }
  }
}

bool same_rows(const void* dst, int dp, const void* src, int sp, int bytes, int rows) {
  DeviceArray<int> fail(1);
  zero(fail);
  const auto n = std::min<size_t>(4096, (size_t(dp) * rows + 255) / 256);
  different<<<n, 256, 0, stream>>>(static_cast<const uint8_t*>(dst), dp, static_cast<const uint8_t*>(src), sp,
                                   bytes, rows, fail.get());
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  return fail.download()[0] == 0;
}

std::vector<float> values(size_t n, uint64_t seed, float amplitude = 1.f) {
  Rng rng(seed);
  std::vector<float> v(n);
  for (auto& x : v) x = float(rng.uniform(-amplitude, amplitude));
  return v;
}

std::vector<__nv_bfloat16> bf16_values(size_t n, uint64_t seed, float amplitude = 1.f) {
  const auto v = values(n, seed, amplitude);
  std::vector<__nv_bfloat16> out(n);
  std::transform(v.begin(), v.end(), out.begin(), float_to_bf16);
  return out;
}

const float* f32(const Tensor& t) { return static_cast<const float*>(t.data); }
const __nv_bfloat16* bf16(const Tensor& t) { return static_cast<const __nv_bfloat16*>(t.data); }
dense::Matrix matrix(const Dense& d) { return {d.w.data, f32(d.scale), d.rows(), d.cols()}; }

void check_packed(const Layer& full, const Layer& rank, const TpLayout& tp) {
  const auto* wproj = static_cast<const uint8_t*>(full.wproj.w.data);
  const auto* local = static_cast<const uint8_t*>(rank.wproj.w.data);
  const int item = full.wproj.fp8() ? 1 : 2, pitch = H * item;
  for (int s = 0; s < num_segs(full.full); ++s) {
    const int count = tp.segment_rows(s);
    const int first = SEG_BEGIN[s] + tp.tp_rank * count, offset = tp.weight_offset(s);
    INGINE_CHECK(same_rows(local + size_t(offset) * pitch, pitch, wproj + size_t(first) * pitch, pitch, pitch, count));
    if (full.wproj.fp8())
      INGINE_CHECK(same_rows(f32(rank.wproj.scale) + offset, 4, f32(full.wproj.scale) + first, 4, 4, count));
  }
  const int C = tp.q_dim_local * item;
  INGINE_CHECK(same_rows(rank.wo.w.data, C, static_cast<const uint8_t*>(full.wo.w.data) + tp.tp_rank * C,
                         HEADS * HEAD_DIM * item, C, H));
  if (full.wo.fp8()) INGINE_CHECK(same_rows(rank.wo.scale.data, 4, full.wo.scale.data, 4, 4, H));

  const auto& a = full.experts;
  const auto& b = rank.experts;
  const bool nv = a.fmt == ExpertFormat::NVFP4;
  const auto slice = tp.expert_slice(nv);
  auto rows = [&](const Tensor& ref, const Tensor& shard) {
    const int c = ref.ne[0], src = 2 * INTER * c, dst = 2 * slice.packed * c;
    INGINE_CHECK(same_rows(shard.data, dst, static_cast<const uint8_t*>(ref.data) + 2 * slice.begin * c,
                           src, 2 * slice.width * c, EXPERTS));
  };
  auto cols = [&](const Tensor& ref, const Tensor& shard, int block) {
    INGINE_CHECK(same_rows(shard.data, slice.packed / block, static_cast<const uint8_t*>(ref.data) + slice.begin / block,
                           INTER / block, slice.width / block, EXPERTS * H));
  };
  rows(a.w13, b.w13);
  rows(a.w13_scale, b.w13_scale);
  cols(a.w2, b.w2, nv ? 2 : 1);
  cols(a.w2_scale, b.w2_scale, nv ? 16 : 32);
  if (nv) {
    INGINE_CHECK(same_rows(b.w13_factor.data, 4, a.w13_factor.data, 4, 4, EXPERTS));
    INGINE_CHECK(same_rows(b.w2_factor.data, 4, a.w2_factor.data, 4, 4, EXPERTS));
  }
  INGINE_CHECK(tp.activation_offset(SEG_K) % 4 == 0 && tp.projection_stride(full.full) % 4 == 0);
}

struct Attention {
  TpLayout tp;
  int M, rows, stride, element;
  DeviceArray<float> proj, kstate, vstate;
  DeviceArray<__nv_bfloat16> out;
  DeviceArray<uint8_t> k, v, workspace;
  DeviceArray<int64_t> tags, position;

  Attention(TpLayout tp_, int m, bool full, bool fp8, int64_t pos)
      : tp(tp_), M(m), rows(full ? 1088 : WINDOW), stride(tp.kv_dim_local), element(fp8 ? 1 : 2),
        proj(size_t(M) * tp.projection_stride(full)), kstate(3 * stride), vstate(3 * stride),
        out(M * tp.q_dim_local), k(rows * stride * element), v(rows * stride * element),
        workspace(attn::workspace_bytes(M, tp.hkv_local)), tags(WINDOW), position(std::vector<int64_t>{pos}) {
    zero(workspace);
  }

  void seed(int pos) {
    const auto data = values(k.count, 531, .125f);
    auto init = [&](DeviceArray<uint8_t>& cache, int off) {
      std::vector<uint8_t> bytes(cache.count);
      for (size_t i = 0; i < cache.count / element; ++i) {
        if (element == 2) {
          const auto b = float_to_bf16(data[(i + off) % data.size()]);
          std::memcpy(bytes.data() + 2 * i, &b, 2);
        } else {
          const auto code = uint8_t((i * 17 + off) % 112);
          bytes[i] = code;
        }
      }
      INGINE_TEST_CUDA(cudaMemcpy(cache.get(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    };
    init(k, 0);
    init(v, 2001);
    const auto state = values(3 * stride, 64, .15f);
    INGINE_TEST_CUDA(cudaMemcpy(kstate.get(), state.data(), kstate.count * 4, cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaMemcpy(vstate.get(), state.data(), vstate.count * 4, cudaMemcpyHostToDevice));
    std::vector<int64_t> t(WINDOW, -1);
    for (int i = std::max(0, pos - WINDOW); i < pos; ++i) t[i % WINDOW] = i;
    INGINE_TEST_CUDA(cudaMemcpy(tags.get(), t.data(), t.size() * 8, cudaMemcpyHostToDevice));
  }

  void seed_from(const Attention& src) {
    const int i = tp.tp_rank * stride, p = stride * element, sp = src.stride * element;
    copy_rows(k.get(), p, src.k.get() + i * element, sp, p, rows);
    copy_rows(v.get(), p, src.v.get() + i * element, sp, p, rows);
    copy_rows(kstate.get(), stride * 4, src.kstate.get() + i, src.stride * 4, stride * 4, 3);
    copy_rows(vstate.get(), stride * 4, src.vstate.get() + i, src.stride * 4, stride * 4, 3);
    INGINE_TEST_CUDA(cudaMemcpyAsync(tags.get(), src.tags.get(), WINDOW * 8, cudaMemcpyDeviceToDevice, stream));
  }

  void run(const Layer& layer, const __nv_bfloat16* x, bool messages) {
    std::vector<int> seg;
    for (int s = 0; s < num_segs(layer.full); ++s) {
      seg.push_back(tp.weight_offset(s));
      seg.push_back(tp.weight_offset(s + 1));
      seg.push_back(layer.full || messages ? SEG_INPUT[s] : 0);
    }
    DeviceArray<int> table(seg);
    dense::proj(x, M, matrix(layer.wproj), table.get(), num_segs(layer.full), proj.get(), false, stream);
    attn::DecodeArgs a;
    a.proj = proj.get();
    a.M = M;
    a.q_scale = f32(layer.q_scale);
    a.r_scale = f32(layer.r_scale);
    a.rel_proj = f32(layer.rel_proj);
    a.k_conv = f32(layer.full ? layer.k_static : layer.k_conv);
    a.v_conv = f32(layer.full ? layer.v_static : layer.v_conv);
    a.conv_k_state = kstate.get();
    a.conv_v_state = vstate.get();
    a.k_cache = k.get();
    a.v_cache = v.get();
    a.cache_fp8 = element == 1;
    a.cache_rows = rows;
    a.kpos = tags.get();
    a.pos = position.get();
    a.window = layer.full ? 0 : WINDOW;
    a.log_scaling = layer.full;
    a.workspace = workspace.get();
    a.workspace_size = workspace.count;
    a.out = out.get();
    a.hkv = tp.hkv_local;
    attn::attn_decode(a, stream);
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  }

  void check(const Attention& ref, bool full) const {
    for (int s = 0; s < num_segs(full); ++s) {
      const int count = tp.segment_rows(s);
      // The byte comparison begins at the first column of this segment in both row layouts.
      const auto a = proj.download(), b = ref.proj.download();
      for (int m = 0; m < M; ++m)
        INGINE_CHECK(std::memcmp(a.data() + m * tp.projection_stride(full) + tp.activation_offset(s),
                                 b.data() + m * ref.tp.projection_stride(full) + SEG_BEGIN[s] + tp.tp_rank * count,
                                 count * sizeof(float)) == 0);
    }
    const int i = tp.tp_rank * stride, p = stride * element, sp = ref.stride * element;
    INGINE_CHECK(same_rows(k.get(), p, ref.k.get() + i * element, sp, p, rows));
    INGINE_CHECK(same_rows(v.get(), p, ref.v.get() + i * element, sp, p, rows));
    INGINE_CHECK(same_rows(kstate.get(), stride * 4, ref.kstate.get() + i, ref.stride * 4, stride * 4, 3));
    INGINE_CHECK(same_rows(vstate.get(), stride * 4, ref.vstate.get() + i, ref.stride * 4, stride * 4, 3));
    if (!full) INGINE_CHECK(tags.download() == ref.tags.download());
    const int q = tp.q_dim_local;
    INGINE_CHECK(same_rows(out.get(), q * 2, ref.out.get() + tp.tp_rank * q, ref.tp.q_dim_local * 2, q * 2, M));
  }
};

struct Partial {
  DeviceArray<float> f, experts;
  DeviceArray<__nv_bfloat16> b;
  Partial(int m, bool moe = false) : f(m * H), experts(moe ? m * TOP_K * H : 1), b(m * H) {}
};

__global__ void exchange(const float* a, const float* b, const __nv_bfloat16* aq, const __nv_bfloat16* bq,
                          float* f, __nv_bfloat16* q) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  f[i] = a[i] + b[i];
  q[i] = __float2bfloat16_rn(__bfloat162float(aq[i]) + __bfloat162float(bq[i]));
}

__global__ void ordered_exchange(const float* a, const float* b, const float* w, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int ch = i % H, m = i / H;
  float y = 0;
  for (int k = 0; k < TOP_K; ++k) {
    const int slot = m * TOP_K + k, j = slot * H + ch;
    const float dot = __bfloat162float(__float2bfloat16_rn(a[j] + b[j]));
    y = __fadd_rn(y, __fmul_rn(w[slot], dot));
  }
  out[i] = y;
}

void reduce(const Partial& a, const Partial& b, Partial& out, int m) {
  exchange<<<m * H / 128, 128, 0, stream>>>(a.f.get(), b.f.get(), a.b.get(), b.b.get(), out.f.get(), out.b.get());
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  const auto fa = a.f.download(), fb = b.f.download(), f = out.f.download();
  const auto qa = a.b.download(), qb = b.b.download(), q = out.b.download();
  for (int i = 0; i < m * H; ++i) {
    INGINE_CHECK(f[i] == fa[i] + fb[i]);
    INGINE_CHECK(bf16_to_float(q[i]) == round_bf16(bf16_to_float(qa[i]) + bf16_to_float(qb[i])));
  }
}

struct Conv {
  DeviceArray<float> state, hist, out;
  explicit Conv(int m) : state(values(3 * H, 340, .07f)), hist(m * H), out(m * H) {}
  void run(int m, dense::ConvKind kind, const Tensor& w, const float* x, const __nv_bfloat16* b = nullptr) {
    dense::conv_residual_epilogue(x, b, m, kind, f32(w), state.get(), out.get(), hist.get(), false, stream);
  }
  void exact(const Conv& ref) const {
    INGINE_CHECK(same_rows(state.get(), 4, ref.state.get(), 4, 4, state.count));
    INGINE_CHECK(same_rows(hist.get(), 4, ref.hist.get(), 4, 4, hist.count));
    INGINE_CHECK(same_rows(out.get(), 4, ref.out.get(), 4, 4, out.count));
  }
};

float error(const std::vector<float>& a, const std::vector<float>& b) {
  float max = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    INGINE_CHECK(std::isfinite(a[i]) && std::isfinite(b[i]));
    max = std::max(max, std::abs(a[i] - b[i]));
  }
  return max;
}

int report_conv(const char* label, const Conv& ref, const Conv& got) {
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  const auto a = ref.hist.download(), b = got.hist.download();
  int outside_one = 0;
  auto ordered = [](float f) {
    const int bits = int(std::bit_cast<uint32_t>(f) >> 16);
    return bits & 32768 ? 32768 - (bits & 32767) : (32768 + bits);
  };
  for (size_t i = 0; i < a.size(); ++i) outside_one += std::abs(ordered(a[i]) - ordered(b[i])) > 1;
  const float e = error(ref.out.download(), got.out.download()), eh = error(a, b);
  const float es = error(ref.state.download(), got.state.download());
  std::printf(" %s[out=%.6g hist=%.6g state=%.6g >1BF16ulp=%d/%zu]", label, e, eh, es, outside_one, a.size());
  const auto out = ref.out.download();
  const float max_ref = std::max(std::abs(*std::min_element(out.begin(), out.end())), *std::max_element(out.begin(), out.end()));
  INGINE_CHECK(e <= .03f * max_ref + 2e-5f);
  return outside_one;
}

void test_layer(const Layer& layer, const Layer& l0, const Layer& l1, int m, bool fp8, int pass, int pos) {
  Attention all(TpLayout{}, m, layer.full, fp8, pos);
  Attention a(TpLayout(2, 0), m, layer.full, fp8, pos), b(TpLayout(2, 1), m, layer.full, fp8, pos);
  all.seed(pos);
  a.seed_from(all);
  b.seed_from(all);
  const bool messages = pass > 0;
  DeviceArray<__nv_bfloat16> xin(bf16_values(3 * m * H, 339 + pass, 1.3f));
  if (messages) {
    DeviceArray<float> base(values(3 * m * H, 778)), msg(values(3 * m * H, 962));
    dense::message_gates(base.get(), !layer.full, msg.get(), 3, m, bf16(layer.qkv_gate_down),
                         layer.qkv_gate_forget16.data, layer.qkv_gate_gain16.data, false, f32(layer.attn_norm3),
                         xin.get(), nullptr, false, stream);
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  }
  all.run(layer, xin.get(), messages);
  a.run(l0, xin.get(), messages);
  b.run(l1, xin.get(), messages);
  a.check(all, layer.full);
  b.check(all, layer.full);

  DeviceArray<__nv_bfloat16> joined(m * HEADS * HEAD_DIM);
  const int qs = a.tp.q_dim_local;
  copy_rows(joined.get(), 4 * qs, a.out.get(), 2 * qs, 2 * qs, m);
  copy_rows(joined.get() + qs, 4 * qs, b.out.get(), 2 * qs, 2 * qs, m);
  Conv wo(m), wo_oracle(m), wo_split(m), wo_f(m), wo_b(m);
  dense::proj_conv(all.out.get(), m, matrix(layer.wo), f32(layer.attn_conv), wo.state.get(), wo.out.get(),
                   wo.hist.get(), false, stream);
  dense::proj_conv(joined.get(), m, matrix(layer.wo), f32(layer.attn_conv), wo_oracle.state.get(), wo_oracle.out.get(),
                   wo_oracle.hist.get(), false, stream);
  wo_oracle.exact(wo);
  Partial raw(m), pa(m), pb(m), sum(m);
  dense::proj_raw(all.out.get(), m, matrix(layer.wo), raw.f.get(), raw.b.get(), false, stream);
  wo_split.run(m, dense::ConvKind::Attention, layer.attn_conv, raw.f.get());
  wo_split.exact(wo);
  dense::proj_raw(a.out.get(), m, matrix(l0.wo), pa.f.get(), pa.b.get(), false, stream);
  dense::proj_raw(b.out.get(), m, matrix(l1.wo), pb.f.get(), pb.b.get(), false, stream);
  reduce(pa, pb, sum, m);
  wo_f.run(m, dense::ConvKind::Attention, layer.attn_conv, sum.f.get());
  wo_b.run(m, dense::ConvKind::Attention, layer.attn_conv, nullptr, sum.b.get());

  // Use a shared prior residual and an active ablation on loop passes, with the same operand for
  // the isolated expert comparison. Deliberately include extreme expert IDs and uneven reuse.
  DeviceArray<float> base(values(m * H, 73, 1.25f)), abldir(values(H, 17, .02f));
  DeviceArray<float> ablw(std::vector<float>{.75f, 0.f, -std::numeric_limits<float>::max(), 0.f});
  DeviceArray<__nv_bfloat16> xm(m * H);
  dense::MixParams p;
  p.M = m;
  p.base = base.get();
  p.write = wo.out.get();
  p.abl_r = messages ? abldir.get() : nullptr;
  p.abl_w = messages ? ablw.get() : nullptr;
  p.queries = f32(layer.mlp_query);
  p.norm_w = f32(layer.mlp_norm1);
  p.out_bf16 = xm.get();
  dense::mix(p, false, stream);
  std::vector<int> ids(m * TOP_K);
  std::vector<float> wts(m * TOP_K);
  for (int n = 0; n < m; ++n) {
    for (int k = 0; k < TOP_K; ++k) {
      ids[n * TOP_K + k] = k == 0 ? 0 : k == 7 ? 127 : (k % 2 ? k : (8 * n + k));
      wts[n * TOP_K + k] = .5f + (n + k) / 16.f;
    }
  }
  DeviceArray<int> idx(ids);
  DeviceArray<float> wt(wts);
  DeviceArray<__nv_bfloat16> act(m * TOP_K * INTER), aa(m * TOP_K * INTER / 2), ab(m * TOP_K * INTER / 2),
      reco(m * TOP_K * INTER);
  const auto format = static_cast<moe::Format>(layer.experts.fmt);
  auto up = [&](const Layer& l, auto& dst, int width) {
    const auto& e = l.experts;
    moe::moe_up(xm.get(), m, idx.get(), e.w13.data, e.w13_scale.data, f32(e.w13_factor), format, dst.get(), false, stream, width);
  };
  up(layer, act, INTER);
  up(l0, aa, INTER / 2);
  up(l1, ab, INTER / 2);
  const int I = INTER;
  copy_rows(reco.get(), I * 2, aa.get(), I, I, m * TOP_K);
  copy_rows(reco.get() + I / 2, I * 2, ab.get(), I, I, m * TOP_K);
  INGINE_CHECK(same_rows(reco.get(), I * 2, act.get(), I * 2, I * 2, m * TOP_K));

  auto legacy = [&](auto& input, Conv& dst) {
    const auto& e = layer.experts;
    moe::moe_down(input.get(), m, idx.get(), wt.get(), e.w2.data, e.w2_scale.data, f32(e.w2_factor), format,
                  f32(layer.mlp_conv), dst.state.get(), dst.out.get(), dst.hist.get(), false, stream);
  };
  Conv mlp(m), mlp_oracle(m), mlp_split(m), mlp_f(m), mlp_b(m), mlp_e(m);
  legacy(act, mlp);
  legacy(reco, mlp_oracle);
  mlp_oracle.exact(mlp);
  mlp_split.run(m, dense::ConvKind::Mlp, layer.mlp_conv, mlp.hist.get());
  mlp_split.exact(mlp);

  Partial ea(m, true), eb(m, true), esum(m);
  auto down = [&](const Layer& l, const auto& input, Partial& dst, float* workspace = nullptr) {
    const auto& e = l.experts;
    moe::moe_down_partial(input.get(), m, idx.get(), wt.get(), e.w2.data, e.w2_scale.data, f32(e.w2_factor),
                          format, I / 2, dst.b.get(), dst.f.get(), dst.experts.get(), false, stream, workspace);
  };
  down(l0, aa, ea);
  down(l1, ab, eb);
  if (m > 1) {
    Partial split(m, true);
    down(l0, aa, split, split.experts.get());
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    INGINE_CHECK(same_rows(split.f.get(), H * 4, ea.f.get(), H * 4, H * 4, m));
    INGINE_CHECK(same_rows(split.b.get(), H * 2, ea.b.get(), H * 2, H * 2, m));
    INGINE_CHECK(same_rows(split.experts.get(), H * 4, ea.experts.get(), H * 4, H * 4, m * TOP_K));
  }
  reduce(ea, eb, esum, m);
  for (const Partial* rank : {&ea, &eb}) {
    const auto dots = rank->experts.download(), hidden = rank->f.download();
    const auto wire = rank->b.download();
    for (int n = 0; n < m; ++n) {
      for (int c = 0; c < H; ++c) {
        float expected = 0.f;
        for (int k = 0; k < TOP_K; ++k) {
          const int slot = n * TOP_K + k;
          const volatile float weighted = wts[slot] * dots[slot * H + c];
          expected += weighted;
        }
        INGINE_CHECK(hidden[n * H + c] == expected);
        INGINE_CHECK(bf16_to_float(wire[n * H + c]) == round_bf16(expected));
      }
    }
  }
  mlp_f.run(m, dense::ConvKind::Mlp, layer.mlp_conv, esum.f.get());
  mlp_b.run(m, dense::ConvKind::Mlp, layer.mlp_conv, nullptr, esum.b.get());
  ordered_exchange<<<m * H / 128, 128, 0, stream>>>(ea.experts.get(), eb.experts.get(), wt.get(), esum.f.get());
  mlp_e.run(m, dense::ConvKind::Mlp, layer.mlp_conv, esum.f.get());
  std::printf("layer=%d pass=%d M=%d dense=%s KV=%s pos=%d exact=proj,attn,KV,KVconv,W13,reconstructed_writes",
              layer.idx, pass, m, layer.wo.fp8() ? "FP8" : "BF16", fp8 ? "FP8" : "BF16", pos);
  INGINE_CHECK(report_conv("wo-FP32", wo, wo_f) == 0);
  report_conv("wo-BF16", wo, wo_b);
  report_conv("MLP-FP32", mlp, mlp_f);
  report_conv("MLP-BF16", mlp, mlp_b);
  report_conv("MLP-ordered", mlp, mlp_e);
  std::puts("");
}

__global__ void tail_codes(uint8_t* w, int rows, int stride, int tail, uint8_t code) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * stride; i += gridDim.x * blockDim.x)
    w[i] = (i % stride >= stride - tail) ? code : 0;
}

void test_tail(moe::Format fmt) {
  constexpr int I = INTER / 2, M = 8;
  const bool nv = fmt == moe::Format::NVFP4;
  const int codes = nv ? I / 2 : I, scales = nv ? I / 16 : I / 32;
  // Exercise both 32-channel chunks of each 64-channel tail, with a different contribution per chunk/rank.
  DeviceArray<uint8_t> w(TOP_K * H * codes), ws(TOP_K * H * scales);
  DeviceArray<float> factors(std::vector<float>(TOP_K, 1.f)), wts(std::vector<float>(M * TOP_K, .125f));
  tail_codes<<<256, 128, 0, stream>>>(w.get(), TOP_K * H, codes, nv ? 32 : 64, nv ? 0x22 : 0x38);
  INGINE_TEST_CUDA(cudaMemsetAsync(ws.get(), nv ? 0x38 : 127, ws.count, stream));
  std::vector<int> ids(M * TOP_K);
  for (int i = 0; i < int(ids.size()); ++i) ids[i] = i % TOP_K;
  DeviceArray<int> idx(ids);
  DeviceArray<__nv_bfloat16> both(M * TOP_K * INTER);
  for (int rank = 0; rank < 2; ++rank) {
    std::vector<__nv_bfloat16> x(M * TOP_K * I, float_to_bf16(0.f));
    for (int i = 0; i < M * TOP_K; ++i) {
      const float q = (rank + 1) / 32.f;
      std::fill_n(x.data() + i * I + 512, 32, float_to_bf16(q));
      std::fill_n(x.data() + i * I + 544, 32, float_to_bf16(2 * q));
    }
    DeviceArray<__nv_bfloat16> act(x);
    copy_rows(both.get() + rank * I, INTER * 2, act.get(), I * 2, I * 2, M * TOP_K);
    Partial out(M, true);
    moe::moe_down_partial(act.get(), M, idx.get(), wts.get(), w.get(), ws.get(), factors.get(), fmt, I,
                          out.b.get(), out.f.get(), out.experts.get(), false, stream);
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    const float expect = 3.f * (rank + 1);
    for (float v : out.f.download()) INGINE_CHECK(v == expect);
    for (const auto v : out.b.download()) INGINE_CHECK(bf16_to_float(v) == expect);
    for (float v : out.experts.download()) INGINE_CHECK(v == expect);
  }
  DeviceArray<uint8_t> full_w(2 * w.count), full_ws(2 * ws.count);
  for (int r = 0; r < 2; ++r) {
    copy_rows(full_w.get() + r * codes, 2 * codes, w.get(), codes, codes, TOP_K * H);
    copy_rows(full_ws.get() + r * scales, 2 * scales, ws.get(), scales, scales, TOP_K * H);
  }
  Partial full(M, true);
  moe::moe_down_partial(both.get(), M, idx.get(), wts.get(), full_w.get(), full_ws.get(), factors.get(), fmt, INTER,
                        full.b.get(), full.f.get(), full.experts.get(), false, stream);
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  for (float v : full.f.download()) INGINE_CHECK(v == 9.f);
  for (float v : full.experts.download()) INGINE_CHECK(v == 9.f);
  std::printf("tail %s: both 32-channel chunks at 512..575 and 1088..1151, exact\n", nv ? "NVFP4" : "MXFP8");
}

double kl(const float* ref, const float* got, int count) {
  const float rm = *std::max_element(ref, ref + count), gm = *std::max_element(got, got + count);
  double rs = 0., gs = 0., d = 0.;
  for (int i = 0; i < count; ++i) {
    const double p = std::exp(double(ref[i]) - rm);
    rs += p;
    gs += std::exp(double(got[i]) - gm);
    d += p * (double(ref[i]) - rm - double(got[i]) + gm);
  }
  return d / rs + std::log(gs / rs);
}

template <class Launch>
double micros(Launch launch) {
  cudaEvent_t start, end;
  INGINE_TEST_CUDA(cudaEventCreate(&start));
  INGINE_TEST_CUDA(cudaEventCreate(&end));
  for (int i = 0; i < 4; ++i) launch();
  std::array<float, 5> samples{};
  for (float& ms : samples) {
    INGINE_TEST_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < 12; ++i) launch();
    INGINE_TEST_CUDA(cudaEventRecord(end, stream));
    INGINE_TEST_CUDA(cudaEventSynchronize(end));
    INGINE_TEST_CUDA(cudaEventElapsedTime(&ms, start, end));
  }
  INGINE_TEST_CUDA(cudaEventDestroy(start));
  INGINE_TEST_CUDA(cudaEventDestroy(end));
  std::sort(samples.begin(), samples.end());
  return samples[2] * 1000. / 12.;
}

void test_head() {
  const int V = UNPADDED_VOCAB, L = V / 2;
  std::vector<DeviceBuffer> storage;
  const auto bf = load_head(INGINE_MODEL_DIR, false, false, stream, storage);
  const auto fp = load_head(INGINE_MODEL_DIR, true, false, stream, storage);
  const auto nv = load_head(INGINE_MODEL_DIR, true, true, stream, storage).nvfp4;
  const auto a = load_head(INGINE_MODEL_DIR, true, true, stream, storage, TpLayout(2, 0)).nvfp4;
  const auto b = load_head(INGINE_MODEL_DIR, true, true, stream, storage, TpLayout(2, 1)).nvfp4;
  for (const auto& shard : {a, b}) {
    INGINE_CHECK(shard.outer_scale == nv.outer_scale && shard.rows == L);
    INGINE_CHECK(same_rows(shard.codes, H / 2, nv.codes + size_t(shard.v_begin) * H / 2, H / 2, H / 2, L));
    INGINE_CHECK(same_rows(shard.scales, H / 16, nv.scales + size_t(shard.v_begin) * H / 16, H / 16, H / 16, L));
  }
  std::printf("head packing: g=%.9g bytes=%zu both ranks reconstructed bit-exactly\n",
              nv.outer_scale, size_t(V) * (H / 2 + H / 16));

  const auto hidden = bf16_values(8 * H, 445, .6f);
  DeviceArray<__nv_bfloat16> x(hidden), zeros(H);
  zero(zeros);
  int bf_matches = 0, tested = 0;
  for (int M : {1, 2, 4, 8}) {
    DeviceArray<float> out(M * V), baseline(M * V), lo(M * L), hi(M * L);
    DeviceArray<head::Candidate> full(M), left(M), right(M);
    DeviceArray<head::Candidate> scratch(head::nvfp4_candidate_workspace(M, V) / sizeof(head::Candidate));
    dense::head(x.get(), M, matrix(bf.dense), baseline.get(), false, stream);
    head::head_nvfp4(x.get(), M, nv, out.get(), full.get(), scratch.get(), false, stream);
    head::head_nvfp4(x.get(), M, a, lo.get(), left.get(), scratch.get(), false, stream);
    head::head_nvfp4(x.get(), M, b, hi.get(), right.get(), scratch.get(), false, stream);
    INGINE_CHECK(same_rows(lo.get(), L * 4, out.get(), V * 4, L * 4, M));
    INGINE_CHECK(same_rows(hi.get(), L * 4, out.get() + L, V * 4, L * 4, M));
    const auto ref = baseline.download(), got = out.download();
    const auto aa = left.download(), ab = right.download(), all = full.download();
    int top1 = 0;
    double total_kl = 0., max_kl = 0.;
    for (int m = 0; m < M; ++m) {
      const auto* p = ref.data() + size_t(m) * V;
      const auto* q = got.data() + size_t(m) * V;
      const int ref_id = int(std::max_element(p, p + V) - p), got_id = int(std::max_element(q, q + V) - q);
      const auto pick = aa[m].val >= ab[m].val ? aa[m] : ab[m];  // rank 0 owns lower IDs on ties
      INGINE_CHECK(pick.global_id == got_id && pick.val == q[got_id]);
      INGINE_CHECK(all[m].global_id == got_id && all[m].val == pick.val);
      top1 += ref_id == got_id;
      const double d = kl(p, q, V);
      total_kl += d;
      max_kl = std::max(d, max_kl);
    }
    INGINE_CHECK(max_kl < .01);
    bf_matches += top1;
    tested += M;
    const auto fp_time = micros([&] { dense::head(x.get(), M, matrix(fp.dense), out.get(), false, stream); });
    const auto nv_time = micros([&] { head::head_nvfp4(x.get(), M, nv, out.get(), nullptr, nullptr, false, stream); });
    const auto rank_time = micros([&] { head::head_nvfp4(x.get(), M, b, nullptr, right.get(), scratch.get(), false, stream); });
    std::printf("head M=%d exact=logits+argmax BF16-top1=%d/%d KL=%.9g maxKL=%.9g FP8-us=%.3f NVFP4-us=%.3f "
                "rank1-candidate-us=%.3f speedup=%.4fx\n",
                M, top1, M, total_kl / M, max_kl, fp_time, nv_time, rank_time, fp_time / nv_time);
    if (M == 1) {
      head::head_nvfp4(zeros.get(), 1, a, nullptr, left.get(), scratch.get(), false, stream);
      head::head_nvfp4(zeros.get(), 1, b, nullptr, right.get(), scratch.get(), false, stream);
      INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
      INGINE_CHECK(left.download()[0].global_id == 0 && left.download()[0].val == 0.f);
      INGINE_CHECK(right.download()[0].global_id == L && right.download()[0].val == 0.f);
    }
  }
  std::printf("head BF16 top-1 on deterministic synthetic hidden: %d/%d\n", bf_matches, tested);
}

bool same_tensor(const Tensor& a, const Tensor& b) {
  INGINE_CHECK(a.bytes() == b.bytes());
  return same_rows(a.data, 128, b.data, 128, 128, a.bytes() / 128) &&
         (a.bytes() % 128 == 0 ||
          same_rows(static_cast<const char*>(a.data) + a.bytes() / 128 * 128, a.bytes() % 128,
                    static_cast<const char*>(b.data) + b.bytes() / 128 * 128, b.bytes() % 128, a.bytes() % 128, 1));
}

std::vector<float> host_floats(const Tensor& t) {
  std::vector<float> values(t.numel());
  INGINE_TEST_CUDA(cudaMemcpy(values.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
  return values;
}

const char* mode_name(TpReduction m) {
  switch (m) {
    case TpReduction::Reconstruct: return "reconstructed";
    case TpReduction::Fp32: return "FP32";
    case TpReduction::Bf16: return "BF16";
    case TpReduction::ExpertOrdered: return "expert-ordered";
  }
  std::abort();
}

void check_state(const TpSimulation& sim, const State& ref, const Buffers& a, const Buffers& b) {
  constexpr int Local = KV_DIM / 2;
  for (int r = 0; r < 2; ++r) {
    const auto& st = sim.state(r);
    INGINE_CHECK(same_tensor(st.ring_pos, ref.ring_pos));
    INGINE_CHECK(same_tensor(st.conv_attn, ref.conv_attn));
    INGINE_CHECK(same_tensor(st.conv_mlp, ref.conv_mlp));
    auto local = [&](const Tensor& shard, const Tensor& full, int es) {
      return same_rows(shard.data, Local * es, static_cast<const char*>(full.data) + r * Local * es,
                       KV_DIM * es, Local * es, full.numel() / KV_DIM);
    };
    INGINE_CHECK(local(st.k_ring, ref.k_ring, 2));
    INGINE_CHECK(local(st.v_ring, ref.v_ring, 2));
    const int kv_size = ref.kv_fp8 ? 1 : 2;
    INGINE_CHECK(local(st.k_full, ref.k_full, kv_size));
    INGINE_CHECK(local(st.v_full, ref.v_full, kv_size));
    INGINE_CHECK(local(st.conv_k, ref.conv_k, 4));
    INGINE_CHECK(local(st.conv_v, ref.conv_v, 4));
  }
  for (auto [x, y] : {std::pair{&a.logits, &b.logits}, {&a.taps, &b.taps}, {&a.xf, &b.xf},
                      {&a.hist_k, &b.hist_k}, {&a.hist_v, &b.hist_v}, {&a.hist_attn, &b.hist_attn},
                      {&a.hist_mlp, &b.hist_mlp}, {&a.idx, &b.idx}, {&a.wts, &b.wts}})
    INGINE_CHECK(same_tensor(*x, *y));
}

void test_simulation(bool dense_fp8, bool kv_fp8, bool nvfp4, bool ablated) {
  constexpr int C = 64;
  auto weights = load_weights(INGINE_MODEL_DIR, dense_fp8, stream, nvfp4);
  const auto ablation = load_ablation(ablated ? std::filesystem::path(INGINE_MODEL_DIR) /
                                                  "inkling-2-mini-abliterated.safetensors"
                                           : std::filesystem::path{});
  INGINE_CHECK(ablation.active == ablated);
  auto state = make_state(C, kv_fp8);
  TpSimulation sim(INGINE_MODEL_DIR, C, dense_fp8, kv_fp8, nvfp4, stream);
  Decoder reference(weights, ablation, state, false), parallel(weights, ablation, state, false, &sim);
  std::vector<DeviceBuffer> head_storage;
  HeadWeights bf_head, fp_head;
  if (nvfp4) {
    bf_head = load_head(INGINE_MODEL_DIR, false, false, stream, head_storage);
    fp_head = load_head(INGINE_MODEL_DIR, true, false, stream, head_storage);
  }
  std::array<Buffers, 4> bases{make_buffers(1), make_buffers(2), make_buffers(4), make_buffers(8)};
  std::array<Buffers, 4> locals{make_buffers(1), make_buffers(2), make_buffers(4), make_buffers(8)};
  const std::array<int64_t, 15> tokens{0, 645, 358, 1097, 37, 318, 1, 722, 242, 19, 1356, 47, 83, 637, 920};
  DeviceArray<int64_t> pos(1);
  for (auto mode : {TpReduction::Reconstruct, TpReduction::Fp32, TpReduction::Bf16, TpReduction::ExpertOrdered}) {
    int64_t p = 0;
    reset_state(state, stream);
    sim.reset(stream);
    double max_kl = 0, total_kl = 0;
    float max_logit = 0, max_attn = 0, max_mlp = 0;
    int top1 = 0, first_drift = NUM_VISITS;
    for (int j = 0; j < 4; ++j) {
      auto& a = bases[j];
      auto& b = locals[j];
      INGINE_TEST_CUDA(cudaMemcpyAsync(a.token.data, tokens.data() + p, a.m * 8, cudaMemcpyHostToDevice, stream));
      INGINE_TEST_CUDA(cudaMemcpyAsync(b.token.data, tokens.data() + p, b.m * 8, cudaMemcpyHostToDevice, stream));
      INGINE_TEST_CUDA(cudaMemcpyAsync(pos.get(), &p, 8, cudaMemcpyHostToDevice, stream));
      reference.decode(a, pos.get(), true, stream);
      parallel.decode(b, pos.get(), true, stream, mode);
      INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
      if (mode == TpReduction::Reconstruct) {
        check_state(sim, state, a, b);
        if (nvfp4) {
          const int V = UNPADDED_VOCAB;
          DeviceArray<float> bf(a.m * V), fp(a.m * V);
          dense::head(bf16(a.xf), a.m, matrix(bf_head.dense), bf.get(), false, stream);
          dense::head(bf16(a.xf), a.m, matrix(fp_head.dense), fp.get(), false, stream);
          INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
          const auto exact = bf.download(), f8 = fp.download(), nv = host_floats(a.logits);
          int same = 0, stable = 0;
          double nvkl = 0, fpkl = 0;
          for (int row = 0; row < a.m; ++row) {
            const auto* x = exact.data() + row * V;
            const auto* y = nv.data() + row * V;
            const int top = int(std::max_element(x, x + V) - x), want = int(std::max_element(y, y + V) - y);
            float second = -std::numeric_limits<float>::infinity(), e = 0.f;
            for (int c = 0; c < V; ++c) {
              if (c != top) second = std::max(second, x[c]);
              e = std::max(e, std::abs(x[c] - y[c]));
            }
            same += top == want;
            if (x[top] - second > 2 * e) {
              ++stable;
              INGINE_CHECK(top == want);
            }
            nvkl += kl(x, y, V);
            fpkl += kl(x, f8.data() + row * V, V);
          }
          std::printf("model hidden M=%d NVFP4-vs-BF16 KL=%.9g FP8-vs-BF16 KL=%.9g top1=%d/%d wide-margins=%d\n",
                      a.m, nvkl / a.m, fpkl / a.m, same, a.m, stable);
        }
        if (j == 3) {
          // Eager warmup above installed the block pointer tables. The same rank-local launches
          // and stream-ordered exchanges must also record and replay without host work.
          cudaGraph_t graph;
          cudaGraphExec_t exec;
          INGINE_TEST_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
          parallel.decode(b, pos.get(), true, stream, mode);
          INGINE_TEST_CUDA(cudaStreamEndCapture(stream, &graph));
          INGINE_TEST_CUDA(cudaGraphInstantiate(&exec, graph, 0));
          p += a.m;
          INGINE_TEST_CUDA(cudaMemcpyAsync(pos.get(), &p, 8, cudaMemcpyHostToDevice, stream));
          reference.decode(a, pos.get(), true, stream);
          INGINE_TEST_CUDA(cudaGraphLaunch(exec, stream));
          INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
          check_state(sim, state, a, b);
          INGINE_TEST_CUDA(cudaGraphExecDestroy(exec));
          INGINE_TEST_CUDA(cudaGraphDestroy(graph));
        }
      } else {
        for (auto [u, v] : {std::pair{&sim.state(0).conv_attn, &sim.state(1).conv_attn},
                             {&sim.state(0).conv_mlp, &sim.state(1).conv_mlp},
                             {&sim.state(0).ring_pos, &sim.state(1).ring_pos}})
          INGINE_CHECK(same_tensor(*u, *v));
        const auto ha = host_floats(a.hist_attn), hb = host_floats(b.hist_attn),
                   ma = host_floats(a.hist_mlp), mb = host_floats(b.hist_mlp);
        max_attn = std::max(max_attn, error(ha, hb));
        max_mlp = std::max(max_mlp, error(ma, mb));
        for (size_t k = 0; k < ha.size(); ++k)
          if (ha[k] != hb[k] || ma[k] != mb[k]) {
            first_drift = std::min(first_drift, int(k) / (a.m * H));
            break;
          }
      }
      const auto la = host_floats(a.logits), lb = host_floats(b.logits);
      max_logit = std::max(max_logit, error(la, lb));
      for (int row = 0; row < a.m; ++row) {
        const float* x = la.data() + row * UNPADDED_VOCAB;
        const float* y = lb.data() + row * UNPADDED_VOCAB;
        const double d = kl(x, y, UNPADDED_VOCAB);
        max_kl = std::max(max_kl, d);
        total_kl += d;
        top1 += std::max_element(x, x + UNPADDED_VOCAB) - x == std::max_element(y, y + UNPADDED_VOCAB) - y;
      }
      p += a.m;
    }
    std::printf("48-visit dense=%s KV=%s head=%s ablation=%s %s: logit-max=%.6g mean-KL=%.9g max-KL=%.9g top1=%d/15 "
                "attn-history-max=%.6g mlp-history-max=%.6g first-drift-visit=%d\n",
                dense_fp8 ? "FP8" : "BF16", kv_fp8 ? "FP8" : "BF16", nvfp4 ? "NVFP4" : dense_fp8 ? "FP8" : "BF16",
                ablated ? "on" : "off", mode_name(mode), max_logit, total_kl / tokens.size(), max_kl, top1, max_attn, max_mlp,
                first_drift == NUM_VISITS ? -1 : first_drift);
    INGINE_CHECK(max_kl <= .25);
  }
}

void layer_cases() {
  for (const int l : {0, 6, 11, 20, 23}) {
    std::vector<DeviceBuffer> storage;
    const bool fp8 = l == 6 || l == 11 || l == 23;
    const auto ref = load_layer(INGINE_MODEL_DIR, l, fp8, stream, storage);
    const auto a = load_layer(INGINE_MODEL_DIR, l, fp8, stream, storage, TpLayout(2, 0));
    const auto b = load_layer(INGINE_MODEL_DIR, l, fp8, stream, storage, TpLayout(2, 1));
    check_packed(ref, a, TpLayout(2, 0));
    check_packed(ref, b, TpLayout(2, 1));
    if (l == 0 || l == 23) {
      for (const int size : {4, 8}) {
        int covered = 0;
        for (int rank = 0; rank < size; ++rank) {
          std::vector<DeviceBuffer> local_storage;
          const TpLayout tp(size, rank);
          const auto slice = tp.expert_slice(l < 20);
          INGINE_CHECK(slice.begin == covered);
          covered += slice.width;
          const auto local = load_layer(INGINE_MODEL_DIR, l, fp8, stream, local_storage, tp);
          check_packed(ref, local, tp);
        }
        INGINE_CHECK(covered == INTER);
        std::printf("checkpoint packing TP=%d layer=%d all %d ranks exact\n", size, l, size);
      }
    }
    for (int m : {1, 2, 4, 8}) {
      const int pass = ref.loop ? (m <= 2 ? 1 : 2) : 0;
      const int pos = l == 0 && m == 1 ? 0 : 1022;
      test_layer(ref, a, b, m, false, pass, pos);
      if (ref.full) test_layer(ref, a, b, m, true, pass, pos);
    }
  }
}

// Optional exact replay of the existing Phase 0 artifacts on rank0. The default CTest remains
// self-contained apart from the checkpoint, since those telemetry artifacts are external.
void phase0_head_replay(const std::filesystem::path& corpus, const std::filesystem::path& recorded, int cap) {
  std::ifstream rows(corpus), bf_file(recorded / "bf16_ref.f32", std::ios::binary),
      sim_file(recorded / "sim_head_nvfp4.f32", std::ios::binary);
  INGINE_CHECK(rows && bf_file && sim_file);
  RuntimeOptions options;
  options.model_dir = INGINE_MODEL_DIR;
  options.dense_fp8 = false;
  options.kv_fp8 = false;
  options.head_nvfp4 = true;
  options.max_ctx = cap;
  Runtime runtime(options);
  auto s = runtime.stream();
  Prefiller prefill(runtime.weights(), runtime.ablation(), options.prefill_chunk, s);
  DeviceArray<int32_t> in(options.prefill_chunk);
  DeviceArray<float> logits(UNPADDED_VOCAB);
  std::vector<float> bf(UNPADDED_VOCAB), sim(UNPADDED_VOCAB), got(UNPADDED_VOCAB);
  int n = 0, top_bf = 0, top_sim = 0, prompt_len = 0;
  double bf_total = 0., sim_total = 0., bf_max = 0., sim_max = 0.;
  float logit_max = 0.f;
  auto compare = [&](const float* device) {
    const int bytes = UNPADDED_VOCAB * sizeof(float);
    INGINE_CHECK(bool(bf_file.read(reinterpret_cast<char*>(bf.data()), bytes)));
    INGINE_CHECK(bool(sim_file.read(reinterpret_cast<char*>(sim.data()), bytes)));
    INGINE_TEST_CUDA(cudaMemcpyAsync(got.data(), device, bytes, cudaMemcpyDeviceToHost, s));
    INGINE_TEST_CUDA(cudaStreamSynchronize(s));
    const double db = kl(bf.data(), got.data(), UNPADDED_VOCAB);
    const double ds = kl(sim.data(), got.data(), UNPADDED_VOCAB);
    bf_total += db;
    sim_total += ds;
    bf_max = std::max(db, bf_max);
    sim_max = std::max(ds, sim_max);
    logit_max = std::max(logit_max, error(sim, got));
    auto top = [](const auto& v) { return std::max_element(v.begin(), v.end()) - v.begin(); };
    top_bf += top(bf) == top(got);
    top_sim += top(sim) == top(got);
    ++n;
  };
  for (std::string line; std::getline(rows, line);) {
    if (line.empty()) continue;
    const auto item = nlohmann::json::parse(line);
    const auto prompt = item.at("prompt").get<std::vector<int32_t>>();
    const auto output = item.at("output").get<std::vector<int32_t>>();
    prompt_len = prompt.size();
    reset_state(runtime.state(), s);
    for (int p = 0; p < prompt_len; p += options.prefill_chunk) {
      const int count = std::min(options.prefill_chunk, prompt_len - p);
      INGINE_TEST_CUDA(cudaMemcpyAsync(in.get(), prompt.data() + p, count * 4, cudaMemcpyHostToDevice, s));
      prefill.run(runtime.state(), in.get(), count, p, logits.get(), nullptr);
    }
    compare(logits.get());
    const int count = std::min<int>(16, output.size());
    for (int p = 0, sample = 0; sample < count; ++p) {
      const float* row = runtime.decode_logits(output[p], prompt_len + p);
      const int target = count == 1 ? 0 : sample * (int(output.size()) - 1) / (count - 1);
      if (p == target) {
        compare(row);
        ++sample;
      }
    }
  }
  INGINE_CHECK(n && bf_file.peek() == EOF && sim_file.peek() == EOF);
  std::printf("Phase 0 replay prompt=%d rows=%d BF16-mean-KL=%.9g BF16-max-KL=%.9g BF16-top1=%d/%d "
              "sim-mean-KL=%.9g sim-max-KL=%.9g sim-top1=%d/%d sim-logit-max=%.9g\n",
              prompt_len, n, bf_total / n, bf_max, top_bf, n, sim_total / n, sim_max, top_sim, n, logit_max);
}

// Exercise the local KV stride and the replicated convolution stride across a sliding-ring wrap.
void test_sharded_verify_rollback() {
  constexpr int width = 8, commit = 3, start = WINDOW - 2;
  const TpLayout layout(2);
  const int kv = layout.kv_dim_local;
  State state = make_state(start + width, true, layout);
  StatePtrs st{{ptr<float>(state.conv_k), ptr<float>(state.conv_v), ptr<float>(state.conv_attn),
                ptr<float>(state.conv_mlp)}, ptr<__nv_bfloat16>(state.k_ring), ptr<__nv_bfloat16>(state.v_ring),
               ptr<std::int64_t>(state.ring_pos), kv};
  DeviceArray<__nv_bfloat16> saved_k(NUM_SLIDING_VISITS * width * kv), saved_v(NUM_SLIDING_VISITS * width * kv);
  DeviceArray<std::int64_t> saved_pos(NUM_SLIDING_VISITS * width);
  VerifyBackup backup{{}, saved_k.get(), saved_v.get(), saved_pos.get()};
  std::vector<DeviceArray<float>> saved_conv, history;
  const float* hist[4];
  for (int i = 0; i < 4; ++i) {
    const int cols = i < 2 ? kv : HIDDEN;
    saved_conv.emplace_back(NUM_VISITS * 3 * cols);
    history.emplace_back(std::vector<float>(NUM_VISITS * width * cols, float(i + 1)));
    backup.conv[i] = saved_conv.back().get();
    hist[i] = history.back().get();
  }
  verify_save(st, backup, nullptr, start, width, stream);
  for (int visit = 0; visit < NUM_SLIDING_VISITS; ++visit) {
    for (int row = 0; row < width; ++row) {
      const int slot = (start + row) % WINDOW;
      const std::size_t at = (std::size_t(visit) * WINDOW + slot) * kv;
      const std::int64_t pos = start + row;
      for (auto* ring : {st.k_ring, st.v_ring})
        INGINE_TEST_CUDA(cudaMemsetAsync(ring + at, 0x3f, kv * sizeof(__nv_bfloat16), stream));
      INGINE_TEST_CUDA(cudaMemcpyAsync(state.ring_pos_at(visit) + slot, &pos, sizeof(pos), cudaMemcpyHostToDevice,
                                      stream));
    }
  }
  verify_rollback(st, backup, hist, nullptr, start, nullptr, commit, width, stream);
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  for (int visit : {0, NUM_SLIDING_VISITS - 1}) {
    for (int row = 0; row < width; ++row) {
      const int slot = (start + row) % WINDOW;
      const std::size_t at = (std::size_t(visit) * WINDOW + slot) * kv;
      std::int64_t pos;
      std::uint16_t bits;
      INGINE_TEST_CUDA(cudaMemcpy(&pos, state.ring_pos_at(visit) + slot, sizeof(pos), cudaMemcpyDeviceToHost));
      INGINE_CHECK(pos == (row < commit ? start + row : -1));
      for (auto* ring : {st.k_ring, st.v_ring}) {
        INGINE_TEST_CUDA(cudaMemcpy(&bits, ring + at + kv - 1, sizeof(bits), cudaMemcpyDeviceToHost));
        INGINE_CHECK(bits == (row < commit ? 0x3f3f : 0));
      }
    }
  }
  for (int i = 0; i < 4; ++i) {
    const int cols = i < 2 ? kv : HIDDEN;
    std::vector<float> host(NUM_VISITS * 3 * cols);
    INGINE_TEST_CUDA(cudaMemcpy(host.data(), st.conv[i], host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    INGINE_CHECK(std::all_of(host.begin(), host.end(), [i](float v) { return v == float(i + 1); }));
  }
  std::printf("PASS TP=2 verify rollback across sliding-ring wrap\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (skip_without_gpu()) return kSkip;
  if (!std::filesystem::exists(std::filesystem::path(INGINE_MODEL_DIR) / "model.safetensors.index.json")) {
    std::printf("skip: checkpoint missing at %s\n", INGINE_MODEL_DIR);
    return kSkip;
  }
  INGINE_TEST_CUDA(cudaSetDevice(0));
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const std::string_view mode = argc > 1 ? argv[1] : "";
  if (mode == "--phase0-head") {
    INGINE_CHECK(argc == 5);
    phase0_head_replay(argv[2], argv[3], std::stoi(argv[4]));
  }
  if (mode.empty() || mode == "--layers-only") {
    layer_cases();
    test_tail(moe::Format::NVFP4);
    test_tail(moe::Format::MXFP8);
  }
  if (mode.empty() || mode == "--head-only") test_head();
  if (mode.empty() || mode == "--rollback-only") test_sharded_verify_rollback();
  if (mode.empty() || mode == "--simulation-only") {
    test_simulation(true, true, true, false);
    test_simulation(false, false, false, true);
  }
  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
}
