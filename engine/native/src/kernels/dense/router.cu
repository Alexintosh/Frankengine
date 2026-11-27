// router: FP32 logits, sigmoid, biased top-8, normalised x8 weights, for M = 1..8 tokens.
//
// Design. The router matrix is small (128 x 2304: 1.2 MB FP32 / 0.6 MB BF16) but top-8 needs all 128
// logits of a token, so the op is one cluster of CL = 16 CTAs (8 if the device cannot co-schedule 16):
// * on the scalar path, CTA r loads its rows [r * 128/CL, +128/CL) into registers and copies the M BF16
//   activation rows into shared memory. One warp per expert streams 16-byte chunks (18 FP32 or 9 BF16
//   chunks per lane). Spreading the 1.2 MB FP32 matrix over 16 SMs keeps per-SM traffic at ~74 KB.
// * the M logits of a row are reduce-scattered over the warp (lane m holds token m), gathered per token
//   and sent to CTA m, which owns token m's selection, as 16-byte st.async stores (4 rows each) completing
//   on CTA m's mbarrier (dsmem.cuh; a relaxed cluster arrive at kernel start + wait before the remote
//   stores guarantees every CTA is resident and its barrier initialised). CTAs that own no token exit
//   after sending.
// * selection in CTA m: thread e computes v_e = sigmoid(logit_e) + bias_e and its rank among the 128
//   values (#greater + #equal with a lower index: ties go to the lower expert) as the number of greater
//   unique keys (order-preserving bits of v_e, then 127 - e), counted with broadcast shared reads by all
//   threads of the CTA (each a part of the 128 keys), so the 8 winners land in rank order (descending
//   score) without a serial argmax chain; lanes 0..7 then write idx and wts = s / (sum s + 1e-20) * 8.
// * BF16 M >= 2: one 16-expert MMA tile per CTA (two K partitions with CL = 16), reduced and sent
//   to the same selection CTAs.
// * pdl=True: weights and bias are loaded before griddepcontrol.wait (pdl_wait_fenced: ptxas would sink
//   them below a plain wait), the activations after it.
// Ported from engine/csrc/dense/router.cu (torch launchers -> raw pointers; mbarrier exchange).
#include <cooperative_groups.h>

#include "kernels/common/l2.cuh"
#include "kernels/common/nvfp4.cuh"
#include "kernels/dense/cluster.cuh"
#include "kernels/dense/dense.cuh"
#include "kernels/dense/dense.h"
#include "kernels/dense/dsmem.cuh"

#include <cstdlib>
#include <type_traits>

namespace cg = cooperative_groups;

namespace ingine::kernels::dense {
namespace {

constexpr int EXPERTS = 128, TOPK = 8;
// L2 class of the weight loads (l2.cuh): none (see gates.cu: evict-last residency made the step slower)
constexpr l2::Evict WEIGHT_L2 = l2::Evict::none;

template <int NT, int KPARTS = 1>
__device__ __forceinline__ void select_top8(int m, const float* logits, float b, int* idx, float* wts) {
  // selection: unique 39-bit keys (order-preserving bits of v, then 127 - e: ties go to the lower expert),
  // rank = number of greater keys. Count the greater keys in NT / 128 partitions of the key range.
  const int t = threadIdx.x;
  constexpr int PARTS = NT / EXPERTS, W = EXPERTS / PARTS;
  __shared__ unsigned long long keys[EXPERTS];
  __shared__ int cnt[PARTS * EXPERTS];
  __shared__ float sel_s[TOPK];
  __shared__ int sel_e[TOPK];
  float s = 0.f;
  if (t < EXPERTS) {
    float logit = logits[t];
    if constexpr (KPARTS == 2) logit += logits[EXPERTS + t];
    s = 1.f / (1.f + expf(-logit));
    const uint32_t u = __float_as_uint(s + b + 0.f);  // + 0: -0 -> +0, which compares equal
    const uint32_t o = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    keys[t] = (static_cast<unsigned long long>(o) << 7) | static_cast<unsigned>(EXPERTS - 1 - t);
  }
  __syncthreads();
  if (t < PARTS * EXPERTS) {
    const int e = t % EXPERTS, part = t / EXPERTS;
    const unsigned long long mine = keys[e];
    int c[4] = {0, 0, 0, 0};
#pragma unroll
    for (int j = 0; j < W; j += 4)
#pragma unroll
      for (int k = 0; k < 4; ++k) c[k] += keys[part * W + j + k] > mine;
    cnt[t] = c[0] + c[1] + c[2] + c[3];
  }
  __syncthreads();
  if (t < EXPERTS) {
    int r = 0;
#pragma unroll
    for (int p = 0; p < PARTS; ++p) r += cnt[p * EXPERTS + t];
    if (r < TOPK) {
      sel_e[r] = t;
      sel_s[r] = s;
    }
  }
  __syncthreads();
  if (t < TOPK) {
    const float my_s = sel_s[t];
    float sum = my_s;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffu, sum, o);
    idx[m * TOPK + t] = sel_e[t];
    wts[m * TOPK + t] = my_s / (sum + 1e-20f) * 8.f;
  }
}

template <int CL, bool WF32, int M>
__global__ void __launch_bounds__(32 * EXPERTS / CL) router_kernel(const __nv_bfloat16* __restrict__ x,
                                                                   const void* __restrict__ w,
                                                                   const float* __restrict__ bias,
                                                                   int* __restrict__ idx, float* __restrict__ wts,
                                                                   uint32_t zero) {
  constexpr int RPC = EXPERTS / CL, NT = 32 * RPC;
  constexpr int EPC = WF32 ? 4 : 8;  // weights per 16-byte chunk
  constexpr int CH = H / (EPC * 32);
  __shared__ __align__(16) __nv_bfloat16 xs[M][H];
  __shared__ __align__(16) float logits[EXPERTS];  // token `rank`'s logits (CTAs rank < M)
  __shared__ __align__(8) uint64_t mbar;           // completes when the 128 logits arrived
  __shared__ __align__(16) float part[M][RPC];     // this CTA's logits

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const uint32_t logits_u = smem_u32(logits), mb = smem_u32(&mbar);
  if (threadIdx.x == 0) {
    mb_init(mb);
    fence_mbarrier_init();
    if (rank < M) mb_expect(mb, 4u * EXPERTS);
  }
  cluster_arrive_relaxed();
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int row = rank * RPC + warp;

  // weights and bias (constants) before the PDL wait, activations after it
  const uint8_t* wr = reinterpret_cast<const uint8_t*>(w) + size_t(row) * H * (WF32 ? 4 : 2);
  uint4 wv[CH];
  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
#pragma unroll
  for (int i = 0; i < CH; ++i) wv[i] = l2::ld_stream<WEIGHT_L2>(wr + (size_t(i) * 32 + lane) * 16, wpol);
  const float b = t < EXPERTS ? __ldg(bias + t) : 0.f;  // for the selection phase

  pdl_wait_fenced(zero);  // keeps the weight loads above ahead of the wait
  pdl_trigger();
  for (int k = t; k < M * H / 8; k += NT) cp_async16(&xs[0][0] + k * 8, x + size_t(k) * 8);
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();

  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    const int col = (i * 32 + lane) * EPC;
    float wf[EPC], xf[EPC];
    if constexpr (WF32) {
      wf[0] = __uint_as_float(wv[i].x); wf[1] = __uint_as_float(wv[i].y);
      wf[2] = __uint_as_float(wv[i].z); wf[3] = __uint_as_float(wv[i].w);
    } else {
      unpack_bf16x8(wv[i], wf);
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      if constexpr (WF32) {
        const uint2 u = *reinterpret_cast<const uint2*>(&xs[m][col]);
        xf[0] = bf_lo(u.x); xf[1] = bf_hi(u.x); xf[2] = bf_lo(u.y); xf[3] = bf_hi(u.y);
      } else {
        unpack_bf16x8(*reinterpret_cast<const uint4*>(&xs[m][col]), xf);
      }
      float a = acc[m];
#pragma unroll
      for (int j = 0; j < EPC; ++j) a = fmaf(wf[j], xf[j], a);
      acc[m] = a;
    }
  }
  const float logit = reduce_scatter<32, M>(acc);  // lane m: token m

  // gather the CTA's RPC logits of each token, then one 16-byte store per (token, 4 rows)
  if (lane < M) part[lane][warp] = logit;
  __syncthreads();
  cluster_wait();  // every CTA has started and initialised its barrier
  if (t < M * (RPC / 4)) {
    const int tok = t % M, grp = t / M;
    st_async4(remote(logits_u + 4u * (rank * RPC + grp * 4), tok),
              *reinterpret_cast<const float4*>(&part[tok][grp * 4]), remote(mb, tok));
  }
  if (rank >= M) return;
  mb_wait(mb);

  select_top8<NT>(rank, logits, b, idx, wts);
}

template <int CL>
constexpr int router_mma_threads = 32 * (CL == 8 ? 9 : 8);

template <int CL>
__global__ void __launch_bounds__(router_mma_threads<CL>) router_mma_kernel(
    const __nv_bfloat16* __restrict__ x, int M, const __nv_bfloat16* __restrict__ w,
    const float* __restrict__ bias, int* __restrict__ idx, float* __restrict__ wts, uint32_t zero) {
  constexpr int NT = router_mma_threads<CL>, NW = NT / 32, KPARTS = CL / 8;
  constexpr int KCTA = H / KPARTS, KW = KCTA / NW;
  constexpr int CHUNKS = KW / 32;  // each pair of uint4 weights covers two MMA steps
  constexpr bool TAIL = KW % 32 != 0;
  static_assert(KCTA % NW == 0 && KW % 16 == 0, "uneven K split");
  __shared__ __align__(16) __nv_bfloat16 xs[8][KCTA];
  __shared__ __align__(16) float red[NW][8][16];
  __shared__ __align__(16) float logits[KPARTS][EXPERTS];
  __shared__ __align__(8) uint64_t mbar;

  const int rank = cg::this_cluster().block_rank(), tile = rank / KPARTS, part = rank % KPARTS;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int row = lane >> 2, pair = lane & 3;
  const uint32_t mb = smem_u32(&mbar);
  if (t == 0) {
    mb_init(mb);
    fence_mbarrier_init();
    if (rank < M) mb_expect(mb, sizeof(logits));
  }
  cluster_arrive_relaxed();

  const __nv_bfloat16* wa = w + (tile * 16 + row) * H + part * KCTA + warp * KW;
  const __nv_bfloat16* wb = wa + 8 * H;
  uint4 a[CHUNKS], b[CHUNKS];
  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
#pragma unroll
  for (int i = 0; i < CHUNKS; ++i) {
    const int k = i * 32 + pair * 8;
    a[i] = l2::ld_stream<WEIGHT_L2>(wa + k, wpol);
    b[i] = l2::ld_stream<WEIGHT_L2>(wb + k, wpol);
  }
  uint2 at, bt;
  if constexpr (TAIL) {
    const int k = CHUNKS * 32 + pair * 4;
    at = __ldg(reinterpret_cast<const uint2*>(wa + k));
    bt = __ldg(reinterpret_cast<const uint2*>(wb + k));
  }
  const float bias_e = t < EXPERTS ? __ldg(bias + t) : 0.f;

  pdl_wait_fenced(zero);
  pdl_trigger();
  for (int i = t; i < M * (KCTA / 8); i += NT) {
    const int m = i / (KCTA / 8), k = i % (KCTA / 8) * 8;
    cp_async16(&xs[m][k], x + size_t(m) * H + part * KCTA + k);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();

  // A and B use the same permutation of K pairs, letting each lane load consecutive BF16 values.
  // The unused N columns can read token zero because their accumulators are discarded.
  const __nv_bfloat16* xb = xs[row < M ? row : 0] + warp * KW;
  float acc[4] = {};
#pragma unroll
  for (int i = 0; i < CHUNKS; ++i) {
    const uint4 v = *reinterpret_cast<const uint4*>(xb + i * 32 + pair * 8);
    nvfp4::mma_bf16(acc, a[i].x, b[i].x, a[i].y, b[i].y, v.x, v.y);
    nvfp4::mma_bf16(acc, a[i].z, b[i].z, a[i].w, b[i].w, v.z, v.w);
  }
  if constexpr (TAIL) {
    const uint2 v = *reinterpret_cast<const uint2*>(xb + CHUNKS * 32 + pair * 4);
    nvfp4::mma_bf16(acc, at.x, bt.x, at.y, bt.y, v.x, v.y);
  }
  red[warp][2 * pair][row] = acc[0];
  red[warp][2 * pair + 1][row] = acc[1];
  red[warp][2 * pair][row + 8] = acc[2];
  red[warp][2 * pair + 1][row + 8] = acc[3];
  __syncthreads();

  cluster_wait();
  if (t < M * 4) {
    const int m = t / 4, r = (t % 4) * 4;
    float4 v = *reinterpret_cast<const float4*>(&red[0][m][r]);
#pragma unroll
    for (int i = 1; i < NW; ++i) {
      const float4 u = *reinterpret_cast<const float4*>(&red[i][m][r]);
      v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w;
    }
    st_async4(remote(smem_u32(&logits[part][tile * 16 + r]), m), v, remote(mb, m));
  }
  if (rank >= M) return;
  mb_wait(mb);
  select_top8<NT, KPARTS>(rank, &logits[0][0], bias_e, idx, wts);
}

template <int CL, bool WF32>
void launch_router(int M, const __nv_bfloat16* x, const void* w, const float* bias, int* idx, float* wts, bool pdl,
                   cudaStream_t stream) {
  auto go = [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    launch_ex(router_kernel<CL, WF32, MM>, dim3(CL), 32 * EXPERTS / CL, 0, CL, pdl, stream, x, w, bias, idx, wts,
              0u);
  };
  switch (M) {
    case 1: go(std::integral_constant<int, 1>{}); break;
    case 2: go(std::integral_constant<int, 2>{}); break;
    case 3: go(std::integral_constant<int, 3>{}); break;
    case 4: go(std::integral_constant<int, 4>{}); break;
    case 5: go(std::integral_constant<int, 5>{}); break;
    case 6: go(std::integral_constant<int, 6>{}); break;
    case 7: go(std::integral_constant<int, 7>{}); break;
    case 8: go(std::integral_constant<int, 8>{}); break;
    default: require(false, "router: M must be in 1..8");
  }
}

// 16-CTA clusters need the non-portable opt-in and a GPC with 16 free SMs; fall back to 8 otherwise.
int router_cluster_size() {
  static const int cl = [] {
    auto k = router_kernel<16, true, 8>;
    if (cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) != cudaSuccess) {
      (void)cudaGetLastError();
      return 8;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(16);
    cfg.blockDim = dim3(32 * EXPERTS / 16);
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 16; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    int n = 0;
    if (cudaOccupancyMaxActiveClusters(&n, k, &cfg) != cudaSuccess) { (void)cudaGetLastError(); n = 0; }
    return n > 0 ? 16 : 8;
  }();
  return cl;
}

}  // namespace

void router(const __nv_bfloat16* x, int M, const void* w, bool w_fp32, const float* bias, int* idx, float* wts,
            bool pdl, cudaStream_t stream) {
  require(x && w && bias && idx && wts, "router: null argument");
  require(M >= 1 && M <= 8, "router: M must be in 1..8");
  const bool c16 = router_cluster_size() == 16;
  const char* mma = (!w_fp32 && M >= 2) ? std::getenv("INGINE_ROUTER_MMA") : nullptr;
  const bool use_mma = !w_fp32 && M >= 2 && (mma ? mma[0] != '0' : c16);
  if (use_mma) {
    const auto* wb = static_cast<const __nv_bfloat16*>(w);
    if (c16) launch_ex(router_mma_kernel<16>, dim3(16), router_mma_threads<16>, 0, 16, pdl, stream,
                       x, M, wb, bias, idx, wts, 0u);
    else launch_ex(router_mma_kernel<8>, dim3(8), router_mma_threads<8>, 0, 8, pdl, stream,
                   x, M, wb, bias, idx, wts, 0u);
  } else if (w_fp32) {
    if (c16) launch_router<16, true>(M, x, w, bias, idx, wts, pdl, stream);
    else launch_router<8, true>(M, x, w, bias, idx, wts, pdl, stream);
  } else {
    if (c16) launch_router<16, false>(M, x, w, bias, idx, wts, pdl, stream);
    else launch_router<8, false>(M, x, w, bias, idx, wts, pdl, stream);
  }
}

}  // namespace ingine::kernels::dense
