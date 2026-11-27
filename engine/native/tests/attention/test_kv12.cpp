// The host-only codec suite also builds with a plain C++ compiler.
// CTest additionally exercises warp packing and the production decode launchers.
#include "kernels/common/kv12.cuh"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifdef __CUDACC__
#include "common/test_util.h"
#include "kernels/attn/attn.h"
#endif

namespace {
namespace k = ingine::kernels::kv12;
using Row = std::array<uint16_t, k::dims>;

void check(bool ok, const char* label, size_t row, int dim = -1) {
  if (!ok) {
    std::fprintf(stderr, "FAIL %s at row %zu, channel %d\n", label, row, dim);
    std::exit(1);
  }
}

void compare_row(const k::Head& h, const k::Spill& s, const uint16_t* want, size_t row) {
  for (int c = 0; c < 16; ++c) {
    uint32_t pairs[4];
    k::unpack8(h, s, c, pairs);
    for (int i = 0; i < 8; ++i) {
      const int d = 8 * c + i;
      check(uint16_t(pairs[i / 2] >> (16 * (i % 2))) == want[d], "BF16 bits", row, d);
    }
  }
}

std::vector<Row> fixtures() {
  std::vector<Row> rows;
  // Every 16-bit encoding in a compressible row, including signed zeros,
  // all subnormals, infinities and every signaling/quiet NaN payload.
  for (int high = 0; high < 512; ++high) {
    Row r;
    for (int d = 0; d < k::dims; ++d) r[d] = uint16_t(high * k::dims + d);
    rows.push_back(r);
  }
  // Every possible exception count, spread across lane and nibble boundaries.
  for (int n = 0; n < k::dims; ++n) {
    Row r;
    for (int i = 0; i < k::dims; ++i) {
      int exp = i < n ? 1 + (i % 100) : 128 - (i % 14);
      if (i == 127) exp = 128;
      r[(i * 73) % k::dims] = uint16_t(((i & 1) << 15) | (exp << 7) | i);
    }
    rows.push_back(r);
  }
  std::mt19937 rng(0x1264);
  for (int i = 0; i < 1024; ++i) {
    Row r;
    for (auto& x : r) x = uint16_t(rng());
    rows.push_back(r);
  }
  // Exactly the first exponent delta outside the window versus the last inside.
  Row edge;
  edge.fill(0x3f80);
  edge[0] = uint16_t((127 - 14) << 7);
  edge[1] = uint16_t((127 - 13) << 7);
  rows.push_back(edge);
  return rows;
}

void host_codec(const std::vector<Row>& rows) {
  k::Head h;
  k::Spill spill;
  for (size_t i = 0; i < rows.size(); ++i) {
    std::memset(&h, 0xcc, sizeof(h));
    std::memset(&spill, 0xa5, sizeof(spill));
    k::pack(rows[i].data(), h, spill);
    compare_row(h, spill, rows[i].data(), i);
    const bool raw = (h.words[51] >> 24) == 0;
    if (i < 512) check(!raw, "constant biased exponent", i);
    if (i >= 512 && i < 640) check(raw == (i - 512 > 15), "inline/overflow boundary", i);
    if (!raw) for (auto w : spill.words) check(w == 0xa5a5a5a5, "unexpected spill write", i);
  }
  check((h.words[32] & 255) == 0xef, "deltas 14 and 13", rows.size() - 1);
  std::printf("PASS host codec (%zu heads)\n", rows.size());
}

#ifdef __CUDACC__
using namespace ingine::test;
namespace attn = ingine::kernels::attn;

__global__ void encode_rows(const uint2* bits, k::Head* heads, k::Spill* tails, int n) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (warp < n) k::pack4(bits[warp * 32 + (threadIdx.x & 31)], heads[warp], tails[warp]);
}

__global__ void decode_rows(const k::Head* heads, const k::Spill* tails, uint4* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n * 16) out[i] = k::load8(heads, tails, i / 16, i % 16);
}

void device_codec(const std::vector<Row>& rows) {
  const int n = int(rows.size());
  DeviceArray<Row> src(rows), decoded(n);
  DeviceArray<k::Head> heads(n);
  DeviceArray<k::Spill> tails(n);
  INGINE_TEST_CUDA(cudaMemset(tails.get(), 0xa5, n * sizeof(k::Spill)));
  encode_rows<<<(n + 3) / 4, 128>>>(reinterpret_cast<const uint2*>(src.get()), heads.get(), tails.get(), n);
  INGINE_TEST_CUDA(cudaGetLastError());
  decode_rows<<<(n * 16 + 127) / 128, 128>>>(heads.get(), tails.get(), reinterpret_cast<uint4*>(decoded.get()), n);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto h = heads.download();
  const auto s = tails.download();
  const auto got = decoded.download();
  for (int i = 0; i < n; ++i) {
    k::Head ref;
    k::Spill spill;
    std::memset(&spill, 0xa5, sizeof(spill));
    k::pack(rows[i].data(), ref, spill);
    check(!std::memcmp(&h[i], &ref, sizeof(ref)), "warp encoder bytes", i);
    check(!std::memcmp(&s[i], &spill, sizeof(spill)), "warp spill bytes", i);
    check(got[i] == rows[i], "device round-trip", i);
  }
}

struct Cache {
  std::vector<Row> original;
  std::vector<k::Head> heads;
  std::vector<k::Spill> spills;
  DeviceArray<Row> bf;
  DeviceArray<k::Head> packed;
  DeviceArray<k::Spill> tail;

  Cache(int n, std::mt19937& rng, bool tiny = false)
      : original(n), heads(n), spills(n), bf(n), packed(n), tail(n) {
    for (int i = 0; i < n; ++i) {
      for (int d = 0; d < 128; ++d) {
        int e = 123 + rng() % 6;
        if (d < i % 23) e = 90 + d % 12;
        original[i][d] = tiny ? uint16_t(99 << 7) : uint16_t((rng() & 0x807f) | (e << 7));
      }
      k::pack(original[i].data(), heads[i], spills[i]);
    }
    bf.upload(original);
    packed.upload(heads);
    tail.upload(spills);
  }

  void compare(const std::vector<bool>& written) const {
    const auto ref = bf.download();
    const auto got = packed.download();
    const auto aux = tail.download();
    for (size_t i = 0; i < ref.size(); ++i) {
      compare_row(got[i], aux[i], ref[i].data(), i);
      if (!written[i]) {
        check(ref[i] == original[i], "BF16 cache guard", i);
        check(!std::memcmp(&got[i], &heads[i], sizeof(k::Head)), "packed cache guard", i);
        check(!std::memcmp(&aux[i], &spills[i], sizeof(k::Spill)), "spill guard", i);
      }
    }
  }
};

enum class Path { cluster, batch, slots };

void attention_parity(int m, int hkv, int pos, Path path, bool probe_tiny = false) {
  const bool batched = path == Path::batch, table = path == Path::slots;
  const int batch = path == Path::cluster ? 1 : 2;
  const int ctx = pos + m + 4, channels = hkv * 128, stride = 8664 * hkv / 8;
  // One extra row per physical slot is padding. Swap active slots and leave
  // one hole to catch slot/sequence indexing or byte-stride mixups.
  const int slot_heads = (ctx + 1) * hkv, physical = batch == 1 ? 1 : 3;
  std::mt19937 rng(5643 + m + pos);
  Cache keys(physical * slot_heads, rng), values(physical * slot_heads, rng, probe_tiny);
  auto floats = [&](size_t n) {
    std::vector<float> f(n);
    for (auto& x : f) x = (int(rng() % 2001) - 1000) * 0.001f;
    return f;
  };
  const int short_m = std::max(1, m - 2);
  const int tokens = batched ? m + short_m : m * batch;
  auto projection = floats(tokens * stride);
  if (probe_tiny) {
    // With zero dynamic/static taps, new K is proj(K)/3. Force exactly
    // 16 exceptions in every new K, and zero new V. Historical V = 2^-28:
    // prematurely converting scalar M=1 values to FP16 would erase all output.
    const int off_k = 2 * 3 * channels + 3 * hkv * 17;
    for (int t = 0; t < tokens; ++t) {
      auto* krow = projection.data() + t * stride + off_k;
      std::fill(krow + channels, projection.data() + (t + 1) * stride, 0.f);
      for (int d = 0; d < channels; ++d) krow[d] = d % 128 < 16 ? 0x1p-25f : 1.f;
    }
  }
  std::vector<__nv_bfloat16> projection16(projection.size());
  std::transform(projection.begin(), projection.end(), projection16.begin(), float_to_bf16);
  DeviceArray<__nv_bfloat16> proj16(projection16);
  DeviceArray<float> proj(projection), q(floats(channels * 3)), r(floats(hkv * 3 * 16));
  DeviceArray<float> rp(floats(16 * 1024)), kw(floats(channels * 4)), vw(floats(channels * 4));
  if (probe_tiny) {
    INGINE_TEST_CUDA(cudaMemset(kw.get(), 0, kw.count * sizeof(float)));
    INGINE_TEST_CUDA(cudaMemset(vw.get(), 0, vw.count * sizeof(float)));
  }
  const auto ck = floats(physical * 3 * channels), cv = floats(physical * 3 * channels);
  DeviceArray<float> convk(ck), convv(cv);
  // pos is indexed by packed sequence, cache and convolution state by physical slot.
  const std::vector<int32_t> slots = batch == 1 ? std::vector<int32_t>{0} : std::vector<int32_t>{2, 0};
  const std::vector<int64_t> positions = batch == 1 ? std::vector<int64_t>{pos} : std::vector<int64_t>{pos, std::max(0, pos - 3)};
  DeviceArray<int32_t> ds(slots), rows(std::vector<int32_t>{0, m, m + short_m});
  DeviceArray<int64_t> dp(positions);
  const auto nbytes = batched ? attn::workspace_bytes_batched(m, batch, hkv) : attn::workspace_bytes(m, hkv) * batch;
  DeviceArray<uint8_t> ws(nbytes);
  DeviceArray<__nv_bfloat16> out(tokens * 3 * channels);
  auto run = [&](bool packed) {
    convk.upload(ck);
    convv.upload(cv);
    INGINE_TEST_CUDA(cudaMemset(ws.get(), 0, nbytes));
    INGINE_TEST_CUDA(cudaMemset(out.get(), packed ? 0x3a : 0x7b, out.count * 2));
    attn::DecodeArgs a;
    a.proj = proj.get(); a.hkv = hkv; a.M = m; a.cache_rows = ctx;
    a.q_scale = q.get(); a.r_scale = r.get(); a.rel_proj = rp.get();
    a.k_conv = kw.get(); a.v_conv = vw.get();
    a.conv_k_state = convk.get(); a.conv_v_state = convv.get();
    a.cache_kv12 = packed;
    a.k_cache = packed ? static_cast<void*>(keys.packed.get()) : keys.bf.get();
    a.v_cache = packed ? static_cast<void*>(values.packed.get()) : values.bf.get();
    a.k_scale = packed ? keys.tail.get() : nullptr;
    a.v_scale = packed ? values.tail.get() : nullptr;
    a.cache_stride = slot_heads * (packed ? int(sizeof(k::Head)) : 128);
    a.scale_stride = slot_heads * sizeof(k::Spill);
    a.conv_stride = 3 * channels;
    a.pos = dp.get(); a.log_scaling = true;
    a.workspace = ws.get(); a.workspace_size = nbytes; a.out = out.get();
    if (table) {
      std::vector<attn::BatchAttnSlot> ss(batch);
      for (int z = 0; z < batch; ++z) {
        const int h = slots[z] * slot_heads;
        auto& entry = ss[z];
        entry.k_cache = packed ? static_cast<void*>(keys.packed.get() + h) : keys.bf.get() + h;
        entry.v_cache = packed ? static_cast<void*>(values.packed.get() + h) : values.bf.get() + h;
        entry.k_scale = packed ? keys.tail.get() + h : nullptr;
        entry.v_scale = packed ? values.tail.get() + h : nullptr;
        entry.conv_k_state = convk.get() + slots[z] * a.conv_stride;
        entry.conv_v_state = convv.get() + slots[z] * a.conv_stride;
        entry.position = positions[z];
      }
      DeviceArray<attn::BatchAttnSlot> device_slots(ss);
      attn::attn_decode_batch(a, device_slots.get(), batch, nullptr);
      INGINE_TEST_CUDA(cudaDeviceSynchronize());
    } else {
      if (batched) {
        a.proj = nullptr; a.proj16 = proj16.get(); a.conv_write = (m == 1);
        a.batch = batch; a.rows = rows.get(); a.slots = ds.get();
      }
      attn::attn_decode(a, nullptr);
      INGINE_TEST_CUDA(cudaDeviceSynchronize());
    }
    return out.download();
  };
  const auto expected = run(false);
  const auto expected_ck = convk.download(), expected_cv = convv.download();
  const auto got = run(true);
  check(convk.download() == expected_ck && convv.download() == expected_cv, "convolution state parity", pos);
  if (batched && m > 1) check(expected_ck == ck && expected_cv == cv, "deferred convolution state", pos);
  for (size_t i = 0; i < got.size(); ++i)
    check(!std::memcmp(&got[i], &expected[i], 2), "production attention parity", i / 128, int(i % 128));
  std::vector<bool> written(physical * slot_heads);
  for (int z = 0; z < batch; ++z) {
    const int begin = slots[z] * slot_heads + int(positions[z]) * hkv;
    std::fill_n(written.begin() + begin, (batched && z == 1 ? short_m : m) * hkv, true);
  }
  if (probe_tiny) {
    const auto heads = keys.packed.download();
    check((heads[pos * hkv].words[51] >> 24) == 0, "new K must use overflow", pos);
    check(std::any_of(expected.begin(), expected.end(), [](__nv_bfloat16 x) {
      return (__bfloat16_as_ushort(x) & 0x7fff) != 0;
    }), "small V reference must survive", pos);
  }
  keys.compare(written);
  values.compare(written);
}
#endif
}  // namespace

int main(int argc, char** argv) {
  const auto rows = fixtures();
  host_codec(rows);
#ifdef __CUDACC__
  if (argc > 1 && std::strcmp(argv[1], "--host") == 0) return 0;
  if (skip_without_gpu()) return kSkip;
  device_codec(rows);
  for (int hkv : {4, 8}) {
    attention_parity(1, hkv, 0, Path::cluster);
    for (int m = 1; m <= 8; ++m) {
      attention_parity(m, hkv, 545, Path::cluster);
      attention_parity(m, hkv, 273, Path::batch);
    }
    attention_parity(1, hkv, 65, Path::slots);
    attention_parity(1, hkv, 33, Path::cluster, true);
  }
  std::puts("PASS warp codec and bit-exact decode attention");
#else
  (void)argc;
  (void)argv;
#endif
}
