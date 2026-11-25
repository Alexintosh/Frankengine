// message_gates: loop-pass message gates (rank-64 down projection, tanh forget/gain, gated mix, RMS norm).
//
// One cluster per gate, with distributed shared memory exchanges for the down projection and output RMS
// norm. Each CTA computes the input norms, stages normalized BF16 inputs, and forms its down rows with
// two warps per row (one per input half). With eight CTAs, each warp pair handles two rows. The rounded
// down outputs are broadcast as float4 stores to every CTA's copy of g.
// Each CTA applies forget and gain to its own channels, broadcasts its sums of squares, and writes the
// normalized result. With eight CTAs, it stages forget/gain weights in channel tiles to fit shared memory.
// PDL: prefetch the first forget/gain tile and all local down weights before waiting for h and msg.
// Ported from engine/csrc/dense/gates.cu (torch launchers -> raw pointers).
#include <cooperative_groups.h>

#include "kernels/common/l2.cuh"
#include "kernels/dense/cluster.cuh"
#include "kernels/dense/dense.cuh"
#include "kernels/dense/dense.h"
#include "kernels/dense/dsmem.cuh"
#include "kernels/dense/reduce.cuh"

#include <type_traits>

namespace cg = cooperative_groups;

namespace ingine::kernels::dense {
namespace {

constexpr int RANK = 64;
// L2 class of the weight loads (l2.cuh): none. Evict::last does keep the gate weights (56 MB over the loop
// layers) and the router matrices in the L2 through a token's weight streams, but the decode step was ~2%
// slower with it: their fetch already overlaps the previous kernel, and the L2 (~96 MB) then has ~25 MB
// left for everything else.
constexpr l2::Evict WEIGHT_L2 = l2::Evict::none;

struct GateArgs {
  const float* h;    // [G, M, H], or [M, H] read by every gate (h_shared)
  bool h_shared;
  const float* msg;  // [G, M, H]
  const __nv_bfloat16* w_down;  // [G, 64, 2H]
  const void* w_forget;         // [G, H, 64] fp32 or bf16
  const void* w_gain;
  const float* norm_w;  // [G, H]
  __nv_bfloat16* out_bf16;  // [G, M, H]
  float* out_f32;           // [G, M, H] or null
  uint32_t zero;            // 0 (opaque to the compiler: pdl_wait_fenced)
};

template <int CL, bool FG32>
struct GateCfg {
  static constexpr int RPC = RANK / CL;           // down rows per CTA
  static constexpr int NW = 8;                    // four warp pairs: (row, half)
  static constexpr int NT = 32 * NW;
  static constexpr int ROW_GROUPS = RPC / (NW / 2);
  static constexpr int CPC = H / CL;              // channels per CTA
  static constexpr int TILE_C = CL == 8 && FG32 ? CPC / 4 : 144;
  static constexpr int PASSES = CPC / TILE_C;
  static constexpr int FGB = FG32 ? 4 : 2;        // bytes per forget/gain weight
  static constexpr int ROW = RANK * FGB;          // bytes of one forget/gain row
  static constexpr int RS = ROW + 16;             // padded row stride in shared memory
  static constexpr int SLICE = TILE_C * RS;       // bytes of one padded matrix tile
  static_assert(CPC % TILE_C == 0 && TILE_C <= NT && ROW % 16 == 0 && RPC % (NW / 2) == 0, "uneven split");
  static constexpr int SMEM = 2 * SLICE;
  static constexpr int XMAX = FG32 ? 1 : 4;       // tokens staged per barrier
};

template <int CL, bool FG32, int M>
__global__ void __launch_bounds__(GateCfg<CL, FG32>::NT) gates_kernel(GateArgs a) {
  using C = GateCfg<CL, FG32>;
  constexpr int DCH = H / (8 * 32);  // 9 chunks of 8 down weights per lane
  constexpr int XPT = H / C::NT;     // inputs per thread per half when staging x
  constexpr int MQ = (M + 3) / 4;    // 16-byte stores holding M values
  constexpr int XG = M < C::XMAX ? M : C::XMAX;  // tokens per staging group
  static_assert(H % C::NT == 0, "x staging split");
  extern __shared__ __align__(16) uint8_t fg[];  // forget tile, then gain tile (padded rows)
  __shared__ __align__(16) float gall[M][RANK];  // full down output (BF16 values), received
  // bf16([n(h), n(msg)]) of the tokens of a group (M = 1 with FP16 slices: two, the shared-memory size the
  // plain step was tuned with)
  constexpr int XBUFFERS = M == 1 && C::XMAX > 1 ? 2 : XG;
  __shared__ __align__(16) __nv_bfloat16 xs[XBUFFERS][2 * H];
  __shared__ __align__(16) float part[M][2][C::RPC];  // per-half partial dots
  __shared__ __align__(16) float ssall[CL][4 * MQ];   // per-CTA sums of squares of the gated output, received
  __shared__ float red[C::NW * 32], red2[C::NW * 32];
  __shared__ __align__(8) uint64_t mbar[2];  // g, sums of squares
  static_assert(CL != 8 || C::SMEM + sizeof(gall) + sizeof(xs) + sizeof(part) + sizeof(ssall) + sizeof(red) +
                              sizeof(red2) + sizeof(mbar) <= 101376,
                "message_gates must fit in GB10 shared memory");

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int g = blockIdx.x / CL;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int rl = warp >> 1, half = warp & 1;  // down projection: warp = (row_local, half)
  const float* hg = a.h + (a.h_shared ? size_t(0) : size_t(g) * M * H);
  const float* mg = a.msg + size_t(g) * M * H;
  const uint32_t mb_g = smem_u32(&mbar[0]), mb_s = smem_u32(&mbar[1]);
  if (t == 0) {
    mb_init(mb_g);
    mb_init(mb_s);
    fence_mbarrier_init();
    mb_expect(mb_g, 16u * CL * M * C::ROW_GROUPS);
    mb_expect(mb_s, 16u * CL * MQ);
  }
  cluster_arrive_relaxed();  // waited for before the first remote store

  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
  const int c0 = rank * C::CPC;
  auto stage_fg = [&](int pass) {
    const size_t off = (size_t(g) * H + c0 + pass * C::TILE_C) * C::ROW;
    const uint8_t* fsrc = reinterpret_cast<const uint8_t*>(a.w_forget) + off;
    const uint8_t* gsrc = reinterpret_cast<const uint8_t*>(a.w_gain) + off;
    constexpr int RCH = C::ROW / 16;
    for (int k = t; k < C::TILE_C * RCH; k += C::NT) {
      const int dst = (k / RCH) * C::RS + (k % RCH) * 16;
      l2::cp_async16<WEIGHT_L2>(fg + dst, fsrc + size_t(k) * 16, wpol);
      l2::cp_async16<WEIGHT_L2>(fg + C::SLICE + dst, gsrc + size_t(k) * 16, wpol);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  stage_fg(0);

  // Down projection weights -> registers.
  uint4 wv[C::ROW_GROUPS][DCH];
#pragma unroll
  for (int r = 0; r < C::ROW_GROUPS; ++r) {
    const int row = rank * C::RPC + r * (C::NW / 2) + rl;
    const __nv_bfloat16* wr = a.w_down + (size_t(g) * RANK + row) * 2 * H + half * H;
#pragma unroll
    for (int i = 0; i < DCH; ++i) wv[r][i] = l2::ld_stream<WEIGHT_L2>(wr + (i * 32 + lane) * 8, wpol);
  }

  pdl_wait_fenced(a.zero);
  pdl_trigger();

  // row norms and down projection, XG tokens at a time (see above)
#pragma unroll
  for (int m0 = 0; m0 < M; m0 += XG) {
    if (m0 > 0) __syncthreads();  // the previous group's dots are done with the staging buffers
    // inputs: XPT of h and of msg per thread and token, coalesced; thread partials of the sums of squares
    float xh[XG][XPT], xm[XG][XPT], ss[2 * XG];
#pragma unroll
    for (int j = 0; j < XG; ++j) {
      const int m = m0 + j;
      float sh = 0.f, sm = 0.f;
      if (m < M) {
#pragma unroll
        for (int k = 0; k < XPT; ++k) {
          xh[j][k] = hg[size_t(m) * H + k * C::NT + t];
          xm[j][k] = mg[size_t(m) * H + k * C::NT + t];
          sh = fmaf(xh[j][k], xh[j][k], sh);
          sm = fmaf(xm[j][k], xm[j][k], sm);
        }
      }
      ss[2 * j] = sh;
      ss[2 * j + 1] = sm;
    }
    block_sum_bcast<2 * XG, C::NW>(ss, red2);
#pragma unroll
    for (int j = 0; j < XG; ++j) {
      if (m0 + j < M) {
        const float rh = rsqrtf(ss[2 * j] / H + EPS), rm = rsqrtf(ss[2 * j + 1] / H + EPS);
        __nv_bfloat16* xb = xs[j];
#pragma unroll
        for (int k = 0; k < XPT; ++k) {
          xb[k * C::NT + t] = __float2bfloat16_rn(xh[j][k] * rh);
          xb[H + k * C::NT + t] = __float2bfloat16_rn(xm[j][k] * rm);
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < XG; ++j) {
      const int m = m0 + j;
      if (m < M) {
        const __nv_bfloat16* xb = xs[j];
#pragma unroll
        for (int r = 0; r < C::ROW_GROUPS; ++r) {
          float acc = 0.f;
#pragma unroll
          for (int i = 0; i < DCH; ++i) {
            float wf[8], xf[8];
            unpack_bf16x8(wv[r][i], wf);
            unpack_bf16x8(*reinterpret_cast<const uint4*>(&xb[half * H + (i * 32 + lane) * 8]), xf);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc = fmaf(wf[e], xf[e], acc);
          }
          acc = group_sum<32>(acc);
          if (lane == 0) part[m][half][r * (C::NW / 2) + rl] = acc;
        }
      }
    }
  }
  __syncthreads();

  cluster_wait();  // all CTAs resident, barriers initialised: remote stores may start
  if (t < M * CL * C::ROW_GROUPS) {
    const int m = t % M, d = (t / M) % CL, r = t / (M * CL);
    const int row = r * (C::NW / 2);
    const float4 p0 = *reinterpret_cast<const float4*>(&part[m][0][row]);
    const float4 p1 = *reinterpret_cast<const float4*>(&part[m][1][row]);
    const float4 v = make_float4(round_bf16(p0.x + p1.x), round_bf16(p0.y + p1.y), round_bf16(p0.z + p1.z),
                                 round_bf16(p0.w + p1.w));
    st_async4(remote(smem_u32(&gall[m][rank * C::RPC + row]), d), v, remote(mb_g, d));
  }

  const bool own = t < C::TILE_C;
  float gated[C::PASSES][M], nw[C::PASSES], sq[M] = {};
#pragma unroll
  for (int pass = 0; pass < C::PASSES; ++pass) {
    if (pass > 0) {
      __syncthreads();  // all threads have finished reading the previous tile
      stage_fg(pass);
    }
    const int c = c0 + pass * C::TILE_C + t;
    float hv[M], mv[M];
    if (own) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        hv[m] = hg[size_t(m) * H + c];
        mv[m] = mg[size_t(m) * H + c];
      }
      nw[pass] = a.norm_w[size_t(g) * H + c];
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();  // forget/gain weights from all threads' copies are visible
    if (pass == 0) mb_wait(mb_g);

    if (own) {
      constexpr int EPC = 16 / C::FGB;  // weights per chunk
      const uint8_t* fr = fg + t * C::RS;
      const uint8_t* gr = fg + C::SLICE + t * C::RS;
      float df[M], dg[M];
#pragma unroll
      for (int m = 0; m < M; ++m) df[m] = dg[m] = 0.f;
#pragma unroll 4
      for (int k = 0; k < RANK / EPC; ++k) {
        const uint4 uf = *reinterpret_cast<const uint4*>(fr + k * 16);
        const uint4 ug = *reinterpret_cast<const uint4*>(gr + k * 16);
        float wf[EPC], wg[EPC];
        if constexpr (FG32) {
          wf[0] = __uint_as_float(uf.x); wf[1] = __uint_as_float(uf.y); wf[2] = __uint_as_float(uf.z); wf[3] = __uint_as_float(uf.w);
          wg[0] = __uint_as_float(ug.x); wg[1] = __uint_as_float(ug.y); wg[2] = __uint_as_float(ug.z); wg[3] = __uint_as_float(ug.w);
        } else {
          unpack_bf16x8(uf, wf);
          unpack_bf16x8(ug, wg);
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
#pragma unroll
          for (int e = 0; e < EPC; e += 4) {
            const float4 gv = *reinterpret_cast<const float4*>(&gall[m][k * EPC + e]);
            df[m] = fmaf(wf[e], gv.x, fmaf(wf[e + 1], gv.y, fmaf(wf[e + 2], gv.z, fmaf(wf[e + 3], gv.w, df[m]))));
            dg[m] = fmaf(wg[e], gv.x, fmaf(wg[e + 1], gv.y, fmaf(wg[e + 2], gv.z, fmaf(wg[e + 3], gv.w, dg[m]))));
          }
        }
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float v = (1.f - tanhf(df[m])) * hv[m] + tanhf(dg[m]) * mv[m];
        gated[pass][m] = v;
        sq[m] = pass == 0 ? v * v : fmaf(v, v, sq[m]);
        if (a.out_f32) a.out_f32[(size_t(g) * M + m) * H + c] = v;
      }
    }
  }

  // per-token sums of squares of this CTA's channels, sent to every CTA
  block_sum_bcast<M, C::NW>(sq, red);
  if (t < CL) {  // to CTA t: the M sums as 16-byte stores
    float r[4 * MQ] = {};
#pragma unroll
    for (int m = 0; m < M; ++m) r[m] = sq[m];
#pragma unroll
    for (int q = 0; q < MQ; ++q)
      st_async4(remote(smem_u32(&ssall[rank][4 * q]), t), make_float4(r[4 * q], r[4 * q + 1], r[4 * q + 2], r[4 * q + 3]),
                remote(mb_s, t));
  }
  mb_wait(mb_s);

  if (own) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float ss = 0.f;
#pragma unroll
      for (int d = 0; d < CL; ++d) ss += ssall[d][m];
      const float rinv = rsqrtf(ss / H + EPS);
#pragma unroll
      for (int pass = 0; pass < C::PASSES; ++pass) {
        const int c = c0 + pass * C::TILE_C + t;
        a.out_bf16[(size_t(g) * M + m) * H + c] = __float2bfloat16_rn(gated[pass][m] * rinv * nw[pass]);
      }
    }
  }
}

template <int CL, bool FG32>
void launch_gates(int G, int M, const GateArgs& a, bool pdl, cudaStream_t stream) {
  using C = GateCfg<CL, FG32>;
  auto go = [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    launch_ex(gates_kernel<CL, FG32, MM>, dim3(G * CL), C::NT, C::SMEM, CL, pdl, stream, a);
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
    default: require(false, "message_gates: M must be in 1..8");
  }
}

// Probe the largest 16-CTA shared-memory footprint, once per current device on this thread.
int gates_cluster_size() {
  struct Selection {
    int device = -1;
    int cl = 8;
  };
  static thread_local Selection selected;
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  if (selected.device != device) {
    using C = GateCfg<16, true>;
    auto kernel = gates_kernel<16, true, 8>;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {16, 1, 1};
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(16);
    cfg.blockDim = dim3(C::NT);
    cfg.dynamicSmemBytes = C::SMEM;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    int active = 0;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) != cudaSuccess ||
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) != cudaSuccess ||
        cudaOccupancyMaxActiveClusters(&active, kernel, &cfg) != cudaSuccess) {
      (void)cudaGetLastError();
      active = 0;
    }
    selected = {device, active > 0 ? 16 : 8};
  }
  return selected.cl;
}

}  // namespace

void message_gates(const float* h, bool h_shared, const float* msg, int G, int M, const __nv_bfloat16* w_down,
                   const void* w_forget, const void* w_gain, bool fg_fp32, const float* norm_w,
                   __nv_bfloat16* out_bf16, float* out_f32, bool pdl, cudaStream_t stream) {
  require(G >= 1 && G <= 3 && M >= 1 && M <= 8, "message_gates: G in 1..3, M in 1..8");
  require(h && msg && w_down && w_forget && w_gain && norm_w && out_bf16, "message_gates: null argument");
  GateArgs a;
  a.h = h;
  a.h_shared = h_shared;
  a.msg = msg;
  a.w_down = w_down;
  a.w_forget = w_forget;
  a.w_gain = w_gain;
  a.norm_w = norm_w;
  a.out_bf16 = out_bf16;
  a.out_f32 = out_f32;
  a.zero = 0;
  if (gates_cluster_size() == 16) {
    if (fg_fp32) launch_gates<16, true>(G, M, a, pdl, stream);
    else launch_gates<16, false>(G, M, a, pdl, stream);
  } else {
    if (fg_fp32) launch_gates<8, true>(G, M, a, pdl, stream);
    else launch_gates<8, false>(G, M, a, pdl, stream);
  }
}

}  // namespace ingine::kernels::dense
