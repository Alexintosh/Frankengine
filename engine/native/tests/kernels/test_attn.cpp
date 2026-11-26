// Attention core (src/kernels/attn) against an FP64 oracle of KERNELS.md area `attn` (adapted from the PyTorch
// engine's test_attn.py): sliding layers with an empty, partially filled, wrapped, and wrapped ring
// holding stale speculative entries, M = 1..8; full layers at p0 = 0, 3000 and 9000 (log scaling active)
// with BF16 and FP8 (E4M3) caches, plus every FP8 verify width M = 2..8 at p0 = 4637 (a partial last 64-key
// tile). Checks the output, the new cache rows, ring positions and conv states; full layers also rerun with
// random bytes in the rows at and after p0 + M (stale speculative rows) and require identical output bits.
// Synthetic layer parameters; caches of 10016 rows (about 40 MB).
#include "kernels/oracle.h"

#include "kernels/attn/attn.h"

#include <cuda_runtime.h>

#include <cstring>
#include <string>

using namespace ingine::test;
namespace attn = ingine::kernels::attn;

namespace {

constexpr int HQ = 24, HKV = 8, D = 128, DREL = 16, EXT = 1024, WIN = 1024, C = HKV * D;
constexpr int OFF_GATE = 3072, OFF_REL = 6144, OFF_SINK = 6528, OFF_K = 6552, OFF_V = 7576, OFF_KDYN = 8600,
              OFF_VDYN = 8632;
constexpr int CAP = 10016;
constexpr double QK_CLIP = 22.627416997969522, REL_CLIP = 8.0;
cudaStream_t g_stream = nullptr;

void clip_rows(std::vector<double>& x, int rows, int n, double t) {
  for (int r = 0; r < rows; ++r) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += x[size_t(r) * n + i] * x[size_t(r) * n + i];
    const double f = std::min(1.0, t / (std::sqrt(ss) + 1e-6));
    for (int i = 0; i < n; ++i) x[size_t(r) * n + i] *= f;
  }
}

// cache entry values (BF16 or E4M3) as the cache stores them, and their bytes
struct Cache {
  std::vector<float> k, v;  // [rows, 8, 128] values
  std::optional<DeviceArray<uint8_t>> dk, dv;
};

float stored(float x, bool fp8) { return fp8 ? e4m3_to_float(float_to_e4m3(x)) : x; }

Cache make_cache(Rng& rng, int rows, bool fp8) {
  Cache c;
  c.k.resize(size_t(rows) * C);
  c.v.resize(c.k.size());
  for (int r = 0; r < rows; ++r)
    for (int g = 0; g < HKV; ++g) {
      std::vector<double> k(D), v(D);
      const double scale = rng.uniform(0.5, 3.5);
      double ss = 0, sv = 0;
      for (int d = 0; d < D; ++d) k[d] = rng.normal() * scale, ss += k[d] * k[d];
      const double f = std::min(1.0, QK_CLIP / (std::sqrt(ss) + 1e-6));
      for (int d = 0; d < D; ++d) v[d] = rng.normal(), sv += v[d] * v[d];
      const double rv = 1.0 / std::sqrt(sv / D + 1e-6);
      for (int d = 0; d < D; ++d) {
        c.k[(size_t(r) * HKV + g) * D + d] = stored(round_bf16(float(k[d] * f)), fp8);
        c.v[(size_t(r) * HKV + g) * D + d] = stored(round_bf16(float(v[d] * rv)), fp8);
      }
    }
  auto bytes = [&](const std::vector<float>& x) {
    std::vector<uint8_t> b(x.size() * (fp8 ? 1 : 2));
    for (size_t i = 0; i < x.size(); ++i) {
      if (fp8) {
        b[i] = float_to_e4m3(x[i]);
      } else {
        const __nv_bfloat16 h = float_to_bf16(x[i]);
        std::memcpy(&b[2 * i], &h, 2);
      }
    }
    return b;
  };
  c.dk.emplace(bytes(c.k));
  c.dv.emplace(bytes(c.v));
  return c;
}

std::vector<float> cache_values(const DeviceArray<uint8_t>& d, bool fp8) {
  const auto b = d.download();
  std::vector<float> o(b.size() / (fp8 ? 1 : 2));
  for (size_t i = 0; i < o.size(); ++i) {
    if (fp8) {
      o[i] = e4m3_to_float(b[i]);
    } else {
      __nv_bfloat16 h;
      std::memcpy(&h, &b[2 * i], 2);
      o[i] = bf16_to_float(h);
    }
  }
  return o;
}

struct Case {
  std::string name;
  bool full, fp8;
  int M, p0;
  std::vector<int64_t> kpos;  // sliding: ring positions before the call
};

bool run(Rng& rng, const Case& cs) {
  const int M = cs.M, p0 = cs.p0;
  const bool full = cs.full, fp8 = cs.fp8;
  const int R = full ? 8664 : 8600, rows = full ? CAP : WIN;
  // ---- inputs
  std::vector<float> proj(size_t(M) * R);
  for (int m = 0; m < M; ++m)
    for (int r = 0; r < R; ++r) {
      double s = 1.0;
      if (r >= OFF_REL && r < OFF_SINK) s = 3.0;
      else if (r >= OFF_SINK && r < OFF_K) s = 2.0;
      else if (r >= OFF_KDYN) s = 0.3;
      proj[size_t(m) * R + r] = round_bf16(float(rng.normal() * s));
    }
  // layer parameters hold BF16 checkpoint values (stored as FP32), as in the real model
  std::vector<float> q_scale = randn(rng, HQ * D, 0.3), r_scale = randn(rng, HQ * DREL, 0.3);
  for (auto& v : q_scale) v = round_bf16(v + 1.f);
  for (auto& v : r_scale) v = round_bf16(v + 1.f);
  const std::vector<float> rel_proj = randn_bf16(rng, DREL * EXT, 0.5);
  const std::vector<float> kconv = randn_bf16(rng, 4 * C, 0.3), vconv = randn_bf16(rng, 4 * C, 0.3);  // [C,4] | [4,C]
  const std::vector<float> ck0 = randn_bf16(rng, 3 * C, 1.0), cv0 = randn_bf16(rng, 3 * C, 1.0);
  Cache cache = make_cache(rng, rows, fp8);
  DeviceArray<float> dproj(proj), dqs(q_scale), drs(r_scale), drp(rel_proj), dkc(kconv), dvc(vconv), dck(ck0),
      dcv(cv0);
  std::vector<int64_t> kpos = cs.kpos;
  if (full) kpos.assign(1, 0);
  DeviceArray<int64_t> dkpos(kpos), dpos(std::vector<int64_t>{p0});
  DeviceArray<uint8_t> ws(attn::workspace_bytes(8));
  DeviceArray<__nv_bfloat16> out(size_t(M) * HQ * D);
  attn::DecodeArgs a;
  a.proj = dproj.get();
  a.M = M;
  a.q_scale = dqs.get();
  a.r_scale = drs.get();
  a.rel_proj = drp.get();
  a.k_conv = dkc.get();
  a.v_conv = dvc.get();
  a.conv_k_state = dck.get();
  a.conv_v_state = dcv.get();
  a.k_cache = cache.dk->get();
  a.v_cache = cache.dv->get();
  a.cache_fp8 = fp8;
  a.cache_rows = rows;
  a.kpos = full ? nullptr : dkpos.get();
  a.pos = dpos.get();
  a.window = full ? 0 : WIN;
  a.log_scaling = full;
  a.workspace = ws.get();
  a.workspace_size = ws.count;
  a.out = out.get();
  attn::attn_decode(a, g_stream);
  INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream));
  const std::vector<__nv_bfloat16> out_first = out.download();

  // ---- oracle: q / rel post-processing
  std::vector<double> q(size_t(M) * HQ * D), rel(size_t(M) * HQ * DREL), rl(size_t(M) * HQ * EXT);
  for (int m = 0; m < M; ++m) {
    const int p = p0 + m;
    const double tau = 1.0 + 0.1 * std::log(std::max((p + 1) / 8192.0, 1.0));
    std::vector<double> qm(HQ * D), rm(HQ * DREL);
    for (int i = 0; i < HQ * D; ++i) qm[i] = float(proj[size_t(m) * R + i] * q_scale[i]);
    for (int i = 0; i < HQ * DREL; ++i) rm[i] = float(proj[size_t(m) * R + OFF_REL + i] * r_scale[i]);
    clip_rows(qm, HQ, D, QK_CLIP);
    clip_rows(rm, HQ, DREL, REL_CLIP);
    for (auto& v : qm) v = full ? rbf(rbf(v) * tau) : rbf(v);
    for (auto& v : rm) v = full ? rbf(rbf(v) * tau) : rbf(v);
    std::copy(qm.begin(), qm.end(), q.begin() + std::ptrdiff_t(m) * HQ * D);
    std::copy(rm.begin(), rm.end(), rel.begin() + std::ptrdiff_t(m) * HQ * DREL);
    for (int h = 0; h < HQ; ++h)
      for (int d = 0; d < EXT; ++d) {
        double s = 0;
        for (int j = 0; j < DREL; ++j) s += rm[h * DREL + j] * rbf(rel_proj[size_t(j) * EXT + d]);
        rl[(size_t(m) * HQ + h) * EXT + d] = rbf(s);
      }
  }
  // ---- oracle: key/value convs, clip / norm, cache rounding
  std::vector<double> nk(size_t(M) * C), nv(size_t(M) * C);
  std::vector<float> want_ck(3 * C), want_cv(3 * C);
  for (int kv = 0; kv < 2; ++kv) {
    const std::vector<float>& st = kv ? cv0 : ck0;
    const std::vector<float>& w = kv ? vconv : kconv;
    const int off = kv ? OFF_V : OFF_K, offd = kv ? OFF_VDYN : OFF_KDYN;
    std::vector<double>& o = kv ? nv : nk;
    for (int c = 0; c < C; ++c) {
      std::vector<double> ext(M + 3);
      for (int j = 0; j < 3; ++j) ext[j] = st[size_t(j) * C + c];
      for (int m = 0; m < M; ++m) ext[3 + m] = proj[size_t(m) * R + off + c];
      for (int m = 0; m < M; ++m) {
        double s = 0;
        if (full) {
          for (int lag = 0; lag < 4; ++lag)
            s += (double(w[size_t(lag) * C + c]) + proj[size_t(m) * R + offd + (c / D) * 4 + lag]) * ext[m + 3 - lag];
          o[size_t(m) * C + c] = rbf((s + ext[m + 3]) / 3.0);
        } else {
          for (int j = 0; j < 4; ++j) s += double(w[size_t(c) * 4 + j]) * ext[m + j];
          o[size_t(m) * C + c] = rbf((s + ext[m + 3]) / std::sqrt(5.0));
        }
      }
      for (int j = 0; j < 3; ++j) (kv ? want_cv : want_ck)[size_t(j) * C + c] = float(ext[M + j]);
    }
  }
  clip_rows(nk, M * HKV, D, QK_CLIP);
  for (int r = 0; r < M * HKV; ++r) {
    double ss = 0;
    for (int d = 0; d < D; ++d) ss += nv[size_t(r) * D + d] * nv[size_t(r) * D + d];
    const double rv = 1.0 / std::sqrt(ss / D + 1e-6);
    for (int d = 0; d < D; ++d) nv[size_t(r) * D + d] *= rv;
  }
  for (auto& v : nk) v = stored(float(rbf(v)), fp8);
  for (auto& v : nv) v = stored(float(rbf(v)), fp8);

  // ---- oracle: attention over the old entries valid for each token + the call's keys up to it
  std::vector<double> exact(size_t(M) * HQ * D), spec(exact.size());
  for (int m = 0; m < M; ++m) {
    const int p = p0 + m;
    std::vector<const float*> ok_k, ok_v;
    std::vector<int> dist;
    for (int r = 0; r < rows; ++r) {
      const int64_t kp = full ? r : kpos[r];
      if (full ? kp >= p0 : (kp < 0 || kp >= p0 || p - kp >= WIN)) continue;
      ok_k.push_back(&cache.k[size_t(r) * C]);
      ok_v.push_back(&cache.v[size_t(r) * C]);
      dist.push_back(int(p - kp));
    }
    std::vector<float> nkf(size_t(M) * C), nvf(size_t(M) * C);
    for (size_t i = 0; i < nkf.size(); ++i) nkf[i] = float(nk[i]), nvf[i] = float(nv[i]);
    for (int i = 0; i <= m; ++i) {
      ok_k.push_back(&nkf[size_t(i) * C]);
      ok_v.push_back(&nvf[size_t(i) * C]);
      dist.push_back(m - i);
    }
    for (int h = 0; h < HQ; ++h) {
      const int g = h / 3;
      const double* qh = &q[(size_t(m) * HQ + h) * D];
      std::vector<double> s(ok_k.size());
      double mx = proj[size_t(m) * R + OFF_SINK + h];
      for (size_t j = 0; j < ok_k.size(); ++j) {
        double d = 0;
        for (int e = 0; e < D; ++e) d += qh[e] * ok_k[j][g * D + e];
        s[j] = d / std::sqrt(128.0) + (dist[j] < EXT ? rl[(size_t(m) * HQ + h) * EXT + dist[j]] : 0.0);
        mx = std::max(mx, s[j]);
      }
      double den = std::exp(proj[size_t(m) * R + OFF_SINK + h] - mx);
      std::vector<double> o(D, 0.0);
      for (size_t j = 0; j < ok_k.size(); ++j) {
        const double e = std::exp(s[j] - mx);
        den += e;
        for (int d = 0; d < D; ++d) o[d] += e * ok_v[j][g * D + d];
      }
      for (int d = 0; d < D; ++d) {
        const size_t i = (size_t(m) * HQ + h) * D + d;
        const double gate = 1.0 / (1.0 + std::exp(-double(proj[size_t(m) * R + OFF_GATE + h * D + d])));
        exact[i] = o[d] / den * gate;
        spec[i] = rbf(rbf(o[d] / den) * gate);
      }
    }
  }
  const std::string n = cs.name;
  bool ok = Check{(n + " out").c_str(), 1.0 / 128, 2e-3, 0.02}(bf16_download(out), exact, &spec);
  // new cache rows (rare one-ulp flips of the BF16 rounding allowed)
  const auto gk = cache_values(*cache.dk, fp8), gv = cache_values(*cache.dv, fp8);
  std::vector<float> rk, rv;
  std::vector<double> wk, wv;
  for (int m = 0; m < M; ++m) {
    const int row = full ? p0 + m : (p0 + m) % WIN;
    for (int c = 0; c < C; ++c) {
      rk.push_back(gk[size_t(row) * C + c]), wk.push_back(nk[size_t(m) * C + c]);
      rv.push_back(gv[size_t(row) * C + c]), wv.push_back(nv[size_t(m) * C + c]);
    }
  }
  const double ulp = fp8 ? 1.0 / 8 : 1.0 / 128;
  ok &= Check{(n + " new keys").c_str(), ulp, 1e-6, 2e-3}(rk, wk, &wk);
  ok &= Check{(n + " new values").c_str(), ulp, 1e-6, 2e-3}(rv, wv, &wv);
  std::vector<double> sk(want_ck.begin(), want_ck.end()), sv(want_cv.begin(), want_cv.end());
  ok &= Check{(n + " conv_k state").c_str(), 0, 0, 0}(dck.download(), sk);
  ok &= Check{(n + " conv_v state").c_str(), 0, 0, 0}(dcv.download(), sv);
  if (!full) {
    const auto got = dkpos.download();
    size_t bad = 0;
    for (int m = 0; m < M; ++m) bad += got[size_t(p0 + m) % WIN] != p0 + m;
    std::printf("%s %s kpos: %zu wrong\n", bad ? "FAIL" : "PASS", n.c_str(), bad);
    ok &= bad == 0;
  }
  if (full) {
    // Rows at and after p0 hold whatever earlier (rejected speculative) calls left there; the call writes rows
    // p0 .. p0 + M - 1 itself. Re-run twice with fresh random bytes in all of them (NaN / Inf patterns
    // included) and the conv states reset: the output must be bit-identical.
    const size_t rb = size_t(C) * (fp8 ? 1 : 2);
    size_t diff = 0;
    for (int rep = 0; rep < 2; ++rep) {
      std::vector<uint8_t> kb = cache.dk->download(), vb = cache.dv->download();
      for (size_t i = size_t(p0) * rb; i < kb.size(); ++i) kb[i] = uint8_t(rng.next()), vb[i] = uint8_t(rng.next());
      cache.dk->upload(kb);
      cache.dv->upload(vb);
      dck.upload(ck0);
      dcv.upload(cv0);
      attn::attn_decode(a, g_stream);
      INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream));
      const std::vector<__nv_bfloat16> again = out.download();
      for (size_t i = 0; i < again.size(); ++i) diff += std::memcmp(&again[i], &out_first[i], 2) != 0;
    }
    std::printf("%s %s stale rows after p0 + M: %zu output values differ\n", diff ? "FAIL" : "PASS", n.c_str(), diff);
    ok &= diff == 0;
  }
  return ok;
}

std::vector<int64_t> ring(const std::string& kind, int& p0) {
  std::vector<int64_t> kpos(WIN, -1);
  if (kind == "empty") {
    p0 = 0;
  } else if (kind == "partial") {
    p0 = 500;
    for (int s = 0; s < p0; ++s) kpos[s] = s;
  } else {
    p0 = 50000 + 37;
    for (int s = 0; s < WIN; ++s) kpos[s] = p0 - 1 - ((p0 - 1 - s) % WIN + WIN) % WIN;
    if (kind == "stale")  // tokens p0..p0+5 of a rejected speculative call are still in the ring
      for (int i = 0; i < 6; ++i) kpos[(p0 + i) % WIN] = p0 + i;
  }
  return kpos;
}


// Regression test: dynamic batch size growth (batch=2 followed by batch=6 with nact > 1)
// must never overwrite zero-initialized split-K counters of higher sequence indices z >= batch.
bool test_dynamic_batch_workspace(Rng& rng) {
  constexpr int max_batch = 8, max_m = 5, hkv = 4, hq = hkv * 3, c_local = hkv * D;
  constexpr int R = 8664 * hkv / 8;
  constexpr int p0 = 600, ctx = 1024;
  DeviceArray<uint8_t> ws(attn::workspace_bytes_batched(max_m, max_batch, hkv));
  INGINE_TEST_CUDA(cudaMemset(ws.get(), 0, ws.count));

  std::vector<float> q_scale(hq * D, 1.0f), r_scale(hq * DREL, 1.0f);
  std::vector<float> rel_proj(DREL * EXT, 0.0f), kconv(4 * c_local, 0.0f), vconv(4 * c_local, 0.0f);
  std::vector<__nv_bfloat16> proj(size_t(max_batch) * max_m * R);
  for (auto& x : proj) x = __float2bfloat16_rn(float(rng.normal() * 0.25));
  std::vector<__nv_bfloat16> kc(size_t(max_batch) * ctx * c_local), vc(size_t(max_batch) * ctx * c_local);
  for (size_t i = 0; i < kc.size(); ++i) {
    kc[i] = __float2bfloat16_rn(float(rng.normal() * 0.1));
    vc[i] = __float2bfloat16_rn(float(rng.normal() * 0.1));
  }
  std::vector<float> ck(size_t(max_batch) * 3 * c_local, 0.0f), cv(size_t(max_batch) * 3 * c_local, 0.0f);
  std::vector<int64_t> pos(max_batch, p0);
  std::vector<int> slots(max_batch);
  for (int i = 0; i < max_batch; ++i) slots[i] = i;

  DeviceArray<float> d_qs(q_scale), d_rs(r_scale), d_rp(rel_proj), d_kw(kconv), d_vw(vconv);
  DeviceArray<float> d_ck(ck), d_cv(cv);
  DeviceArray<__nv_bfloat16> d_proj(proj), d_kc(kc), d_vc(vc);
  DeviceArray<__nv_bfloat16> d_out_fresh(size_t(max_batch) * max_m * hq * D);
  DeviceArray<__nv_bfloat16> d_out_after(size_t(max_batch) * max_m * hq * D);
  DeviceArray<int64_t> d_pos(pos);
  DeviceArray<int> d_slots(slots);

  auto make_args = [&](int batch, __nv_bfloat16* out_ptr) {
    attn::DecodeArgs a{};
    a.proj16 = d_proj.get();
    a.conv_write = false;
    a.M = max_m;
    a.hkv = hkv;
    a.batch = batch;
    a.q_scale = d_qs.get(); a.r_scale = d_rs.get(); a.rel_proj = d_rp.get();
    a.k_conv = d_kw.get(); a.v_conv = d_vw.get();
    a.conv_k_state = d_ck.get(); a.conv_v_state = d_cv.get();
    a.k_cache = d_kc.get(); a.v_cache = d_vc.get();
    a.cache_fp8 = false; a.cache_rows = ctx; a.window = 0; a.log_scaling = true;
    a.pos = d_pos.get(); a.slots = d_slots.get();
    a.cache_stride = int64_t(ctx) * c_local;
    a.conv_stride = int64_t(3) * c_local;
    a.workspace = ws.get(); a.workspace_size = ws.count;
    a.out = out_ptr;
    return a;
  };

  // Reference: batch=6 on a pristine zeroed workspace
  INGINE_TEST_CUDA(cudaMemset(d_out_fresh.get(), 0x5a, d_out_fresh.count * sizeof(__nv_bfloat16)));
  attn::attn_decode(make_args(6, d_out_fresh.get()), g_stream);
  INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream));
  const auto ref = d_out_fresh.download();

  // Trigger: run batch=2 (nact > 1) on same workspace, then batch=6 into sentinel-filled output
  attn::attn_decode(make_args(2, d_out_after.get()), g_stream);
  INGINE_TEST_CUDA(cudaMemset(d_out_after.get(), 0x5a, d_out_after.count * sizeof(__nv_bfloat16)));
  attn::attn_decode(make_args(6, d_out_after.get()), g_stream);
  INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream));
  const auto got = d_out_after.download();

  int bad = 0;
  for (size_t i = 0; i < size_t(6) * max_m * hq * D; ++i) {
    if (__bfloat16_as_ushort(got[i]) != __bfloat16_as_ushort(ref[i])) ++bad;
  }
  std::printf("%s dynamic-batch workspace (batch=2 -> batch=6, hkv=4, p0=600): mismatched=%d\n",
              bad == 0 ? "PASS" : "FAIL", bad);
  return bad == 0;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  Rng rng(4242);
  bool ok = true;
  for (const char* kind : {"empty", "partial", "wrapped", "stale"})
    for (int M : {1, 3, 8}) {
      Case c;
      c.full = false, c.fp8 = false, c.M = M;
      c.kpos = ring(kind, c.p0);
      c.name = std::string("sliding ") + kind + " M=" + std::to_string(M);
      ok &= run(rng, c);
    }
  for (int M : {2, 4, 5, 6, 7}) {
    Case c;
    c.full = false, c.fp8 = false, c.M = M;
    c.kpos = ring("stale", c.p0);
    c.name = "sliding stale M=" + std::to_string(M);
    ok &= run(rng, c);
  }
  for (bool fp8 : {false, true})
    for (int p0 : {0, 3000, 9000})
      for (int M : {1, 3, 8}) {
        Case c;
        c.full = true, c.fp8 = fp8, c.M = M, c.p0 = p0;
        c.name = std::string("full ") + (fp8 ? "fp8" : "bf16") + " p0=" + std::to_string(p0) + " M=" + std::to_string(M);
        ok &= run(rng, c);
      }
  for (int M = 2; M <= 8; ++M) {  // every verify width, p0 not a multiple of the 64-key E4M3 tile
    Case c;
    c.full = true, c.fp8 = true, c.M = M, c.p0 = 4637;
    c.name = "full fp8 p0=4637 M=" + std::to_string(M);
    ok &= run(rng, c);
  }
  ok &= test_dynamic_batch_workspace(rng);
  INGINE_TEST_CUDA(cudaStreamDestroy(g_stream));
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
