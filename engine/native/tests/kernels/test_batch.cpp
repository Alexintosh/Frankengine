#include "kernels/oracle.h"

#include "kernels/attn/attn.h"
#include "kernels/dense/dense.h"

#include <cstring>

using namespace ingine::test;
namespace attn = ingine::kernels::attn;
namespace dense = ingine::kernels::dense;

namespace {

template <typename T>
bool bit_equal(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

template <typename Launch>
void replay(cudaStream_t stream, Launch launch) {
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  INGINE_TEST_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  launch();
  INGINE_TEST_CUDA(cudaStreamEndCapture(stream, &graph));
  INGINE_TEST_CUDA(cudaGraphInstantiate(&exec, graph, 0));
  INGINE_TEST_CUDA(cudaGraphLaunch(exec, stream));
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  INGINE_TEST_CUDA(cudaGraphExecDestroy(exec));
  INGINE_TEST_CUDA(cudaGraphDestroy(graph));
}

bool attention(int batch, int hkv, bool full, bool fp8, cudaStream_t stream, Rng& rng) {
  const int channels = hkv * 128;
  const int stride = (full ? 8664 : 8600) * hkv / 8;
  const int rows = full ? 2048 : 1024;
  const int cache_stride = rows * channels * (fp8 ? 1 : 2);
  DeviceArray<float> proj(randn_bf16(rng, batch * stride, .5));
  DeviceArray<float> qs(std::vector<float>(3 * channels, 1.f));
  DeviceArray<float> rs(std::vector<float>(3 * hkv * 16, 1.f));
  DeviceArray<float> rel(randn_bf16(rng, 16 * 1024, .5));
  DeviceArray<float> kw(randn_bf16(rng, 4 * channels, .25));
  DeviceArray<float> vw(randn_bf16(rng, 4 * channels, .25));
  const auto init_k = randn_bf16(rng, batch * 3 * channels, .5);
  const auto init_v = randn_bf16(rng, batch * 3 * channels, .5);
  DeviceArray<float> bk(init_k), bv(init_v), sk(init_k), sv(init_v);
  DeviceArray<uint8_t> bkc(batch * cache_stride), bvc(batch * cache_stride);
  DeviceArray<uint8_t> skc(batch * cache_stride), svc(batch * cache_stride);
  const std::vector<int64_t> all_positions{0, 17, 127, 1023, 1024, 1201, 1519, 1999};
  std::vector<int64_t> tags(batch * rows, -1);
  for (int r = 0; !full && r < batch; ++r) {
    for (int64_t p = std::max<int64_t>(0, all_positions[r] - rows); p < all_positions[r]; ++p)
      tags[r * rows + p % rows] = p;
  }
  DeviceArray<int64_t> bp(tags), sp(tags), positions(all_positions);
  std::vector<attn::BatchAttnSlot> slots;
  for (int r = 0; r < batch; ++r) {
    slots.push_back({bkc.get() + r * cache_stride, bvc.get() + r * cache_stride,
                     bk.get() + r * 3 * channels, bv.get() + r * 3 * channels,
                     full ? nullptr : bp.get() + r * rows, all_positions[r]});
  }
  DeviceArray<attn::BatchAttnSlot> table(slots);
  DeviceArray<uint8_t> ws(batch * attn::workspace_bytes(1, hkv)), single_ws(attn::workspace_bytes(1, hkv));
  DeviceArray<__nv_bfloat16> bo(batch * 3 * channels), so(batch * 3 * channels);
  attn::DecodeArgs args;
  args.proj = proj.get();
  args.q_scale = qs.get();
  args.r_scale = rs.get();
  args.rel_proj = rel.get();
  args.k_conv = kw.get();
  args.v_conv = vw.get();
  args.cache_fp8 = fp8;
  args.cache_rows = rows;
  args.window = full ? 0 : rows;
  args.log_scaling = full;
  args.hkv = hkv;
  args.workspace = ws.get();
  args.workspace_size = ws.count;
  args.out = bo.get();

  // Configure the kernel on the current thread before capturing it. Reset the fixture after warmup.
  attn::attn_decode_batch(args, table.get(), batch, stream);
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  bk.upload(init_k);
  bv.upload(init_v);
  bp.upload(tags);
  INGINE_TEST_CUDA(cudaMemsetAsync(bkc.get(), 0, bkc.count, stream));
  INGINE_TEST_CUDA(cudaMemsetAsync(bvc.get(), 0, bvc.count, stream));
  replay(stream, [&] { attn::attn_decode_batch(args, table.get(), batch, stream); });

  args.workspace = single_ws.get();
  args.workspace_size = single_ws.count;
  for (int r = 0; r < batch; ++r) {
    args.proj = proj.get() + r * stride;
    args.k_cache = skc.get() + r * cache_stride;
    args.v_cache = svc.get() + r * cache_stride;
    args.conv_k_state = sk.get() + r * 3 * channels;
    args.conv_v_state = sv.get() + r * 3 * channels;
    args.kpos = full ? nullptr : sp.get() + r * rows;
    args.pos = positions.get() + r;
    args.out = so.get() + r * 3 * channels;
    attn::attn_decode(args, stream);
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  return bit_equal(bo.download(), so.download()) &&
         bit_equal(bk.download(), sk.download()) && bit_equal(bv.download(), sv.download()) &&
         bit_equal(bkc.download(), skc.download()) && bit_equal(bvc.download(), svc.download()) &&
         bit_equal(bp.download(), sp.download());
}

bool convolution(int batch, dense::ConvKind kind, bool bf16, cudaStream_t stream, Rng& rng) {
  constexpr int H = 2304;
  const auto input = randn(rng, batch * H, .5);
  std::vector<__nv_bfloat16> rounded;
  for (float x : input) rounded.push_back(float_to_bf16(x));
  DeviceArray<float> x(input), w(randn(rng, 4 * H, .5));
  DeviceArray<__nv_bfloat16> x16(rounded);
  const auto initial = randn_bf16(rng, batch * 3 * H, .5);
  DeviceArray<float> bs(initial), ss(initial), bo(batch * H), so(batch * H);
  std::vector<float*> slots;
  for (int r = 0; r < batch; ++r) slots.push_back(bs.get() + r * 3 * H);
  DeviceArray<float*> table(slots);
  replay(stream, [&] {
    dense::conv_residual_epilogue_batch(bf16 ? nullptr : x.get(), bf16 ? x16.get() : nullptr, batch,
                                        kind, w.get(), table.get(), bo.get(), true, stream);
  });
  for (int r = 0; r < batch; ++r) {
    dense::conv_residual_epilogue(bf16 ? nullptr : x.get() + r * H, bf16 ? x16.get() + r * H : nullptr, 1,
                                  kind, w.get(), ss.get() + r * 3 * H, so.get() + r * H, nullptr, true, stream);
  }
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  return bit_equal(bo.download(), so.download()) && bit_equal(bs.download(), ss.download());
}

}  // namespace

int main() {
  Rng rng(181);
  cudaStream_t stream;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  bool ok = true;
  for (int batch : {2, 4, 8}) {
    for (int hkv : {4, 8}) {
      for (int mode = 0; mode < 3; ++mode) {
        if (!attention(batch, hkv, mode != 0, mode == 2, stream, rng)) {
          std::fprintf(stderr, "attention mismatch batch=%d hkv=%d mode=%d\n", batch, hkv, mode);
          ok = false;
        }
      }
    }
    for (auto kind : {dense::ConvKind::Attention, dense::ConvKind::Mlp}) {
      for (bool bf16 : {false, true}) {
        if (!convolution(batch, kind, bf16, stream, rng)) {
          std::fprintf(stderr, "residual conv mismatch batch=%d bf16=%d\n", batch, bf16);
          ok = false;
        }
      }
    }
  }
  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
  return ok ? 0 : 1;
}
