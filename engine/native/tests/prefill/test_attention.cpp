// Prefill attention against an FP64 oracle of model.Model._attend (+ flex's sink factor and the output gate):
// full-attention caches (BF16 and FP8 E4M3, keys past the relative extent), sliding rings (partially filled,
// wrapped, a stale slot, chunks longer than the window), ring_gather / key_tile_bounds / ring_update.
#include "kernels/common/kv12.cuh"
#include "kernels/prefill/prefill.h"
#include "model/tp_layout.h"
#include "prefill/prefill_test.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace ingine::test;
using namespace ingine::test::prefill;
namespace kv12 = ingine::kernels::kv12;
namespace pk = ingine::kernels::prefill;

namespace {

constexpr int HQ = 24, HK = 8, D = 128, REL = 1024, W = 1024;
constexpr int PROJ_LD = 8664, GATE = 3072, SINK = 6528;

struct Inputs {
  int T = 0, pos0 = 0;
  std::vector<uint16_t> q, rel, proj;  // [T,24,128], [T,24,1024], [T,PROJ_LD]
};

Inputs make_inputs(Rng& rng, int T, int pos0) {
  Inputs in;
  in.T = T, in.pos0 = pos0;
  in.q.resize(size_t(T) * HQ * D);
  for (auto& x : in.q) x = bf16_bits(float(rng.normal() * 1.2));
  in.rel.resize(size_t(T) * HQ * REL);
  for (auto& x : in.rel) x = bf16_bits(float(rng.normal() * 1.5));
  in.proj.assign(size_t(T) * PROJ_LD, 0);
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < HQ * D; ++c) in.proj[size_t(t) * PROJ_LD + GATE + c] = bf16_bits(float(rng.normal()));
    for (int h = 0; h < HQ; ++h) in.proj[size_t(t) * PROJ_LD + SINK + h] = bf16_bits(float(rng.normal() * 2.0));
  }
  return in;
}

// One key/value row [8, 128]: keys of norm <= ~22, values RMS-normalised per head.
void make_kv_row(Rng& rng, float* k, float* v) {
  for (int h = 0; h < HK; ++h) {
    double ss = 0;
    for (int d = 0; d < D; ++d) {
      k[h * D + d] = rbf(rng.normal() * 1.4);
      const double x = rng.normal();
      v[h * D + d] = float(x);
      ss += x * x;
    }
    const double r = 1.0 / std::sqrt(ss / D + 1e-6);
    for (int d = 0; d < D; ++d) v[h * D + d] = rbf(v[h * D + d] * r);
  }
}

// Reference output (BF16 roundings of model._visit emulated) for rows (t, h) with t % stride == 0.
std::vector<double> oracle(const Inputs& in, const std::vector<float>& K, const std::vector<float>& V,
                           const std::vector<int>& kpos, int window, int stride, std::vector<size_t>& index) {
  std::vector<double> out;
  const int L = int(kpos.size());
  std::vector<double> s(L);
  for (int t = 0; t < in.T; t += stride) {
    const int qp = in.pos0 + t;
    for (int h = 0; h < HQ; ++h) {
      const int g = h / 3;
      const uint16_t* q = &in.q[(size_t(t) * HQ + h) * D];
      double mx = -INFINITY;
      for (int j = 0; j < L; ++j) {
        const int d = qp - kpos[j];
        const bool ok = kpos[j] >= 0 && d >= 0 && (window == 0 || d < window);
        if (!ok) {
          s[j] = -INFINITY;
          continue;
        }
        double dot = 0;
        const float* k = &K[(size_t(j) * HK + g) * D];
        for (int e = 0; e < D; ++e) dot += double(bf16_val(q[e])) * k[e];
        s[j] = dot / std::sqrt(128.0) + (d < REL ? double(bf16_val(in.rel[(size_t(t) * HQ + h) * REL + d])) : 0.0);
        mx = std::max(mx, s[j]);
      }
      const double sink = bf16_val(in.proj[size_t(t) * PROJ_LD + SINK + h]);
      double z = 0;
      std::vector<double> o(D, 0.0);
      for (int j = 0; j < L; ++j) {
        if (s[j] == -INFINITY) continue;
        const double p = std::exp(s[j] - mx);
        z += p;
        const float* v = &V[(size_t(j) * HK + g) * D];
        for (int e = 0; e < D; ++e) o[e] += p * v[e];
      }
      const double lse = mx + std::log(z);
      const double factor = 1.0 / (1.0 + std::exp(sink - lse));
      for (int e = 0; e < D; ++e) {
        const double a = rbf(o[e] / z * factor);
        const double gate = bf16_val(in.proj[size_t(t) * PROJ_LD + GATE + h * D + e]);
        out.push_back(a / (1.0 + std::exp(-gate)));
        index.push_back((size_t(t) * HQ + h) * D + e);
      }
    }
  }
  return out;
}

bool run_attention(const char* name, const Inputs& in, const void* k_dev, const void* v_dev, bool fp8,
                   const int* kpos_dev, const int4* bounds_dev, int L, int window, const std::vector<float>& K,
                   const std::vector<float>& V, const std::vector<int>& kpos, int stride) {
  DeviceArray<uint16_t> q(in.q), rel(in.rel), proj(in.proj), out(size_t(in.T) * HQ * D);
  pk::AttnParams p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.get());
  p.k = k_dev, p.v = v_dev, p.kv_fp8 = fp8;
  p.kpos = kpos_dev, p.bounds = bounds_dev, p.L = L;
  p.T = in.T, p.pos0 = in.pos0, p.window = window;
  p.rel = reinterpret_cast<const __nv_bfloat16*>(rel.get());
  p.proj = reinterpret_cast<const __nv_bfloat16*>(proj.get()), p.ld = PROJ_LD;
  p.out = reinterpret_cast<__nv_bfloat16*>(out.get());
  pk::attention(p, nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto got_bits = out.download();
  std::vector<size_t> index;
  const auto ref = oracle(in, K, V, kpos, window, stride, index);
  std::vector<float> got(index.size());
  for (size_t i = 0; i < index.size(); ++i) got[i] = bf16_val(got_bits[index[i]]);
  // one BF16 rounding of the output (plus a rare flip of the rounded pre-gate value) and FP16/FP8 probabilities
  const char* fp8_env = std::getenv("INGINE_PREFILL_FP8_ATTN");
  const bool fp8_mma = fp8 && window == 0 && !kpos_dev && (!fp8_env || !*fp8_env || std::atoi(fp8_env) != 0);
  return check_close(name, got, ref, fp8_mma ? 2.5e-2 : 1.2e-2, fp8_mma ? 2.0e-3 : 2e-4);
}

bool full_case(Rng& rng, bool fp8, int pos0, int T) {
  const int L = pos0 + T, rows = L + 37;  // extra rows past L: never read
  std::vector<float> K(size_t(rows) * HK * D), V(K.size());
  for (int j = 0; j < rows; ++j) make_kv_row(rng, &K[size_t(j) * HK * D], &V[size_t(j) * HK * D]);
  DeviceArray<uint8_t> kd(K.size() * (fp8 ? 1 : 2)), vd(K.size() * (fp8 ? 1 : 2));
  if (fp8) {
    std::vector<uint8_t> kb(K.size()), vb(K.size());
    for (size_t i = 0; i < K.size(); ++i) {
      kb[i] = e4m3_bits(K[i]), vb[i] = e4m3_bits(V[i]);
      K[i] = e4m3_to_float(kb[i]), V[i] = e4m3_to_float(vb[i]);
    }
    kd.upload(kb), vd.upload(vb);
  } else {
    std::vector<uint16_t> kb(K.size()), vb(K.size());
    for (size_t i = 0; i < K.size(); ++i) kb[i] = bf16_bits(K[i]), vb[i] = bf16_bits(V[i]);
    std::vector<uint8_t> kbytes(K.size() * 2), vbytes(K.size() * 2);
    std::memcpy(kbytes.data(), kb.data(), kbytes.size());
    std::memcpy(vbytes.data(), vb.data(), vbytes.size());
    kd.upload(kbytes), vd.upload(vbytes);
  }
  std::vector<int> kpos(L);
  for (int j = 0; j < L; ++j) kpos[j] = j;
  const Inputs in = make_inputs(rng, T, pos0);
  char name[96];
  std::snprintf(name, sizeof name, "full %s pos0=%d T=%d", fp8 ? "fp8" : "bf16", pos0, T);
  return run_attention(name, in, kd.get(), vd.get(), fp8, nullptr, nullptr, L, 0, K, V, kpos, 1);
}

void kv12_case(Rng& rng, int hkv) {
  using namespace ingine::model;
  const TpLayout tp(HK / hkv);
  constexpr int prefix = 37, T = 71, L = prefix + T;
  const int ld = tp.projection_rows(true), q_dim = tp.q_dim_local, kv_dim = tp.kv_dim_local;
  const int k_base = tp.weight_offset(SEG_K), v_base = tp.weight_offset(SEG_V);
  std::vector<std::uint16_t> input(std::size_t(L) * ld);
  for (int t = 0; t < L; ++t) {
    auto* row = input.data() + std::size_t(t) * ld;
    for (int c = 0; c < 2 * q_dim; ++c) row[c] = bf16_bits(float(rng.uniform(-1.0, 1.0)));
    for (int c = 0; c < kv_dim; ++c) {
      // Alternate heads that can pack inline with heads that must use overflow.
      const float value = t % 2 && c % D < 32 ? 0x1p-30f : 1.f;
      row[k_base + c] = bf16_bits(value);
      row[v_base + c] = bf16_bits(c % 2 ? value : -value);
    }
  }
  DeviceArray<std::uint16_t> proj(input), q(L * q_dim), r(L * tp.hq_local * D_REL);
  DeviceArray<std::uint16_t> rel(T * tp.hq_local * REL), ref(T * q_dim), out(T * q_dim);
  DeviceArray<std::uint16_t> k_bf16(L * kv_dim), v_bf16(L * kv_dim);
  DeviceArray<kv12::Head> k(L * hkv), v(L * hkv);
  DeviceArray<kv12::Spill> ks(L * hkv), vs(L * hkv);
  DeviceArray<float> scale(std::vector<float>(q_dim, 1.f)), weights(4 * kv_dim), state(3 * kv_dim);
  pk::AttnPrepParams p;
  p.full = true, p.hkv = hkv, p.ld = ld;
  p.q_scale = p.r_scale = scale.get();
  p.k_w = p.v_w = weights.get();
  p.conv_k = p.conv_v = state.get();
  for (int start : {0, prefix}) {
    p.T = start == 0 ? prefix : T;
    p.pos0 = start;
    p.proj = reinterpret_cast<const __nv_bfloat16*>(proj.get() + start * ld);
    p.q = reinterpret_cast<__nv_bfloat16*>(q.get() + start * q_dim);
    p.rel = reinterpret_cast<__nv_bfloat16*>(r.get() + start * tp.hq_local * D_REL);
    p.out_kv12 = false;
    p.k_out = k_bf16.get() + start * kv_dim;
    p.v_out = v_bf16.get() + start * kv_dim;
    pk::attn_prep(p, nullptr);
    p.out_kv12 = true;
    p.k_out = k.get() + start * hkv;
    p.v_out = v.get() + start * hkv;
    p.k_scale_out = ks.get() + start * hkv;
    p.v_scale_out = vs.get() + start * hkv;
    pk::attn_prep(p, nullptr);
  }
  pk::AttnParams at;
  at.q = p.q, at.proj = p.proj, at.ld = ld, at.hkv = hkv, at.T = T, at.L = L, at.pos0 = prefix;
  at.rel = reinterpret_cast<const __nv_bfloat16*>(rel.get());
  at.k = k_bf16.get(), at.v = v_bf16.get(), at.out = reinterpret_cast<__nv_bfloat16*>(ref.get());
  pk::attention(at, nullptr);
  at.k = k.get(), at.v = v.get(), at.k_scale = ks.get(), at.v_scale = vs.get(), at.kv_kv12 = true;
  at.out = reinterpret_cast<__nv_bfloat16*>(out.get());
  pk::attention(at, nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  for (bool key : {false, true}) {
    const auto bf16 = key ? k_bf16.download() : v_bf16.download();
    const auto heads = key ? k.download() : v.download();
    const auto spills = key ? ks.download() : vs.download();
    for (std::size_t i = 0; i < heads.size(); ++i) {
      INGINE_CHECK((reinterpret_cast<const uint8_t*>(&heads[i])[207] == 0) == (i / hkv % 2 != 0));
      for (int c = 0; c < D / 8; ++c) {
        std::uint32_t restored[4];
        kv12::unpack8(heads[i], spills[i], c, restored);
        INGINE_CHECK(std::memcmp(restored, bf16.data() + i * D + c * 8, sizeof(restored)) == 0);
      }
    }
  }
  INGINE_CHECK(out.download() == ref.download());
}

// Sliding visit: ring state before the chunk (ring_pos per slot), chunk keys, gathered rows.
bool sliding_case(Rng& rng, int pos0, int T, bool stale, int stride) {
  std::vector<float> ringK(size_t(W) * HK * D), ringV(ringK.size());
  std::vector<int64_t> ring_pos(W, -1);
  for (int s = 0; s < W; ++s) make_kv_row(rng, &ringK[size_t(s) * HK * D], &ringV[size_t(s) * HK * D]);
  for (int p = std::max(0, pos0 - W); p < pos0; ++p) ring_pos[p % W] = p;
  if (stale) ring_pos[(pos0 + 5) % W] = pos0 + 5;  // a slot holding a position at/after the chunk start
  std::vector<uint16_t> rk(ringK.size()), rv(ringK.size());
  for (size_t i = 0; i < rk.size(); ++i) rk[i] = bf16_bits(ringK[i]), rv[i] = bf16_bits(ringV[i]);

  const int L = W + T, kpos_rows = (L + 63) / 64 * 64;
  std::vector<float> K(size_t(L) * HK * D), V(K.size());
  std::vector<int> kpos(L);
  for (int i = 0; i < W; ++i) {
    const int slot = (pos0 + i) % W;
    std::copy_n(&ringK[size_t(slot) * HK * D], HK * D, &K[size_t(i) * HK * D]);
    std::copy_n(&ringV[size_t(slot) * HK * D], HK * D, &V[size_t(i) * HK * D]);
    kpos[i] = int(ring_pos[slot]);
  }
  std::vector<uint16_t> rows_k(size_t(L) * HK * D, 0), rows_v(rows_k.size(), 0);
  for (int t = 0; t < T; ++t) {
    float* k = &K[size_t(W + t) * HK * D];
    float* v = &V[size_t(W + t) * HK * D];
    make_kv_row(rng, k, v);
    for (int c = 0; c < HK * D; ++c)
      rows_k[size_t(W + t) * HK * D + c] = bf16_bits(k[c]), rows_v[size_t(W + t) * HK * D + c] = bf16_bits(v[c]);
    kpos[W + t] = pos0 + t;
  }
  DeviceArray<uint16_t> ring_k(rk), ring_v(rv), kr(rows_k), vr(rows_v);
  DeviceArray<int64_t> rpos(ring_pos);
  DeviceArray<int> kp(kpos_rows);
  DeviceArray<int4> bounds(kpos_rows / 64);
  pk::ring_gather(reinterpret_cast<const __nv_bfloat16*>(ring_k.get()),
                  reinterpret_cast<const __nv_bfloat16*>(ring_v.get()), rpos.get(), pos0, T,
                  reinterpret_cast<__nv_bfloat16*>(kr.get()), reinterpret_cast<__nv_bfloat16*>(vr.get()), kp.get(),
                  kpos_rows, nullptr);
  pk::key_tile_bounds(kp.get(), L, bounds.get(), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  bool ok = true;
  {
    const auto gk = kp.download();
    const auto gr = kr.download();
    size_t bad = 0;
    for (int i = 0; i < kpos_rows; ++i) bad += gk[i] != (i < L ? kpos[i] : -1);
    for (size_t i = 0; i < size_t(W) * HK * D; ++i) bad += gr[i] != bf16_bits(K[i]);
    std::printf("%s ring_gather rows/positions: %zu mismatches\n", bad ? "FAIL" : "PASS", bad);
    ok &= bad == 0;
  }
  const Inputs in = make_inputs(rng, T, pos0);
  char name[96];
  std::snprintf(name, sizeof name, "sliding pos0=%d T=%d%s", pos0, T, stale ? " stale" : "");
  ok &= run_attention(name, in, kr.get(), vr.get(), false, kp.get(), bounds.get(), L, W, K, V, kpos, stride);

  pk::ring_update(reinterpret_cast<const __nv_bfloat16*>(kr.get()), reinterpret_cast<const __nv_bfloat16*>(vr.get()),
                  pos0, T, reinterpret_cast<__nv_bfloat16*>(ring_k.get()),
                  reinterpret_cast<__nv_bfloat16*>(ring_v.get()), rpos.get(), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  {
    auto exp_pos = ring_pos;
    auto exp_k = rk;
    const int n = std::min(T, W);
    for (int c = T - n; c < T; ++c) {
      const int slot = (pos0 + c) % W;
      exp_pos[slot] = pos0 + c;
      std::copy_n(&rows_k[size_t(W + c) * HK * D], HK * D, &exp_k[size_t(slot) * HK * D]);
    }
    const auto gp = rpos.download();
    const auto gk = ring_k.download();
    size_t bad = 0;
    for (int s = 0; s < W; ++s) bad += gp[s] != exp_pos[s];
    for (size_t i = 0; i < exp_k.size(); ++i) bad += gk[i] != exp_k[i];
    std::printf("%s ring_update: %zu mismatches\n", bad ? "FAIL" : "PASS", bad);
    ok &= bad == 0;
  }
  return ok;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  Rng rng(1234);
  bool ok = true;
  ok &= full_case(rng, false, 0, 45);
  ok &= full_case(rng, false, 1200, 77);
  ok &= full_case(rng, true, 1200, 77);
  ok &= full_case(rng, true, 3, 130);
  for (int hkv : {HK, HK / 2}) kv12_case(rng, hkv);
  ok &= sliding_case(rng, 0, 50, false, 1);
  ok &= sliding_case(rng, 700, 300, false, 3);
  ok &= sliding_case(rng, 5000, 1100, true, 7);
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
