// message_gates: loop-pass message gates (rank-64 down projection, tanh forget/gain, gated mix, RMS norm).
//
// One cluster per gate, with distributed shared memory exchanges for the down projection and output RMS
// norm. Each CTA computes the input norms, stages normalized BF16 inputs, and forms its down rows with
// two warps per row (one per input half). With eight CTAs, each warp pair handles two rows. The rounded
// down outputs are broadcast as float4 stores to every CTA's copy of g.
// Each CTA applies forget and gain to its own channels, broadcasts its sums of squares, and writes the
// normalized result. With eight CTAs, it stages forget/gain weights in channel tiles to fit shared memory.
// BF16 M >= 2 uses MMA for both projections. CTAs share the work of a 16-row down tile across K
// partitions. Each token owner reduces the partials and broadcasts the rounded down outputs.
// PDL: prefetch the first forget/gain tile and all local down weights before waiting for h and msg.
// Ported from engine/csrc/dense/gates.cu (torch launchers -> raw pointers).
#include <cooperative_groups.h>

#include "kernels/common/l2.cuh"
#include "kernels/common/nvfp4.cuh"
#include "kernels/dense/cluster.cuh"
#include "kernels/dense/dense.cuh"
#include "kernels/dense/dense.h"
#include "kernels/dense/dsmem.cuh"
#include "kernels/dense/reduce.cuh"

#include <cstdlib>
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

template <class C, int NT>
__device__ __forceinline__ void stage_fg(const GateArgs& a, size_t channel, uint8_t* fg, std::uint64_t wpol) {
  const auto* fsrc = static_cast<const uint8_t*>(a.w_forget) + channel * C::ROW;
  const auto* gsrc = static_cast<const uint8_t*>(a.w_gain) + channel * C::ROW;
  constexpr int RCH = C::ROW / 16;
  for (int k = threadIdx.x; k < C::TILE_C * RCH; k += NT) {
    const int dst = (k / RCH) * C::RS + (k % RCH) * 16;
    l2::cp_async16<WEIGHT_L2>(fg + dst, fsrc + k * 16, wpol);
    l2::cp_async16<WEIGHT_L2>(fg + C::SLICE + dst, gsrc + k * 16, wpol);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int CL, int M, int NW>
__device__ __forceinline__ void exchange_squares(float (&sq)[M], float* red, float (&ssall)[CL][4 * ((M + 3) / 4)],
                                                  int rank, uint32_t mb) {
  constexpr int MQ = (M + 3) / 4;
  block_sum_bcast<M, NW>(sq, red);
  const int t = threadIdx.x;
  if (t < CL) {  // send to CTA t
    float r[4 * MQ] = {};
#pragma unroll
    for (int m = 0; m < M; ++m) r[m] = sq[m];
#pragma unroll
    for (int q = 0; q < MQ; ++q)
      st_async4(remote(smem_u32(&ssall[rank][4 * q]), t),
                make_float4(r[4 * q], r[4 * q + 1], r[4 * q + 2], r[4 * q + 3]), remote(mb, t));
  }
  mb_wait(mb);
}

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
  const size_t gate_channel = size_t(g) * H + c0;
  stage_fg<C, C::NT>(a, gate_channel, fg, wpol);

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
      stage_fg<C, C::NT>(a, gate_channel + pass * C::TILE_C, fg, wpol);
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
  exchange_squares<CL, M, C::NW>(sq, red, ssall, rank, mb_s);

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

constexpr int GATE_MMA_WARPS = 9, GATE_MMA_THREADS = 32 * GATE_MMA_WARPS;

template <int CL, int M>
__global__ void __launch_bounds__(GATE_MMA_THREADS) gates_mma_kernel(GateArgs a) {
  using C = GateCfg<CL, false>;
  constexpr int NW = GATE_MMA_WARPS, NT = GATE_MMA_THREADS;
  constexpr int KPARTS = CL / (RANK / 16), KCTA = 2 * H / KPARTS, KW = KCTA / NW;
  constexpr int CHUNKS = KW / 32;  // two MMA steps per uint4 pair
  constexpr int MQ = (M + 3) / 4;
  static_assert(H % KCTA == 0 && KCTA % NW == 0 && KW % 32 == 0, "uneven down projection split");
  static_assert(C::TILE_C == NW * 16, "one up projection tile per warp");
  extern __shared__ __align__(16) uint8_t fg[];
  auto* gated = reinterpret_cast<float (*)[C::CPC]>(fg + C::SMEM);
  __shared__ __align__(16) __nv_bfloat16 xs[M][KCTA];
  __shared__ __align__(16) float down[NW][M][16];
  __shared__ __align__(16) float partial[KPARTS][RANK];  // received by the owner of token rank
  __shared__ __align__(16) uint32_t grow[RANK / 2];       // owner's packed BF16 down row
  __shared__ __align__(16) __nv_bfloat16 gall[M][RANK];   // broadcast down outputs
  __shared__ __align__(16) float ssall[CL][4 * MQ];
  __shared__ float red[NW * 32];
  __shared__ __align__(8) uint64_t mbar[3];
  constexpr size_t STATIC_SMEM =
      sizeof(xs) + sizeof(down) + sizeof(partial) + sizeof(grow) + sizeof(gall) + sizeof(ssall) + sizeof(red) + sizeof(mbar);
  static_assert(STATIC_SMEM <= 48 * 1024, "message_gates MMA must fit the static shared memory limit");
  static_assert(STATIC_SMEM + C::SMEM + sizeof(float[M][C::CPC]) <= 101376,
                "message_gates MMA must fit in shared memory");

  const int rank = cg::this_cluster().block_rank(), g = blockIdx.x / CL;
  const int tile = rank / KPARTS, part = rank % KPARTS;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int row = lane >> 2, pair = lane & 3;
  const float* hg = a.h + (a.h_shared ? size_t(0) : size_t(g) * M * H);
  const float* mg = a.msg + size_t(g) * M * H;
  const int c0 = rank * C::CPC;
  const uint32_t mb_d = smem_u32(&mbar[0]), mb_g = smem_u32(&mbar[1]), mb_s = smem_u32(&mbar[2]);
  if (t == 0) {
    mb_init(mb_d);
    mb_init(mb_g);
    mb_init(mb_s);
    fence_mbarrier_init();
    if (rank < M) mb_expect(mb_d, sizeof(partial));
    mb_expect(mb_g, sizeof(gall));
    mb_expect(mb_s, sizeof(ssall));
  }
  cluster_arrive_relaxed();

  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
  const size_t gate_channel = size_t(g) * H + c0;
  stage_fg<C, NT>(a, gate_channel, fg, wpol);

  const __nv_bfloat16* wa = a.w_down + (size_t(g) * RANK + tile * 16 + row) * 2 * H + part * KCTA + warp * KW;
  const __nv_bfloat16* wb = wa + 8 * 2 * H;
  uint4 aw[CHUNKS], bw[CHUNKS];
#pragma unroll
  for (int i = 0; i < CHUNKS; ++i) {
    const int k = i * 32 + pair * 8;
    aw[i] = l2::ld_stream<WEIGHT_L2>(wa + k, wpol);
    bw[i] = l2::ld_stream<WEIGHT_L2>(wb + k, wpol);
  }

  pdl_wait_fenced(a.zero);
  pdl_trigger();

  // The K partition is wholly in h or msg. Keep the scalar path's full-half norm and its 256-thread
  // reduction order. The ninth warp contributes zero and is used by the projections.
  const bool msg_half = part >= KPARTS / 2;
  const float* input = msg_half ? mg : hg;
  const int k0 = part * KCTA % H;
  constexpr int XNT = C::NT, XPT = H / XNT, XG = 2;
#pragma unroll 1
  for (int m0 = 0; m0 < M; m0 += XG) {
    float values[XG][XPT], ss[XG] = {};
#pragma unroll
    for (int j = 0; j < XG; ++j) {
      if (t < XNT && m0 + j < M) {
        const float* x = input + (m0 + j) * H + t;
#pragma unroll
        for (int k = 0; k < XPT; ++k) {
          const float v = x[k * XNT];
          values[j][k] = v;
          ss[j] = fmaf(v, v, ss[j]);
        }
      }
    }
    block_sum_bcast<XG, NW>(ss, red);
#pragma unroll
    for (int j = 0; j < XG; ++j) {
      if (t < XNT && m0 + j < M) {
        const float rinv = rsqrtf(ss[j] / H + EPS);
#pragma unroll
        for (int k = 0; k < XPT; ++k) {
          const int local = k * XNT + t - k0;
          if (unsigned(local) < KCTA) xs[m0 + j][local] = __float2bfloat16_rn(values[j][k] * rinv);
        }
      }
    }
    __syncthreads();
  }

  const __nv_bfloat16* xb = xs[row < M ? row : 0] + warp * KW;
  float dw[4] = {};
#pragma unroll
  for (int i = 0; i < CHUNKS; ++i) {
    const uint4 v = *reinterpret_cast<const uint4*>(xb + i * 32 + pair * 8);
    nvfp4::mma_bf16(dw, aw[i].x, bw[i].x, aw[i].y, bw[i].y, v.x, v.y);
    nvfp4::mma_bf16(dw, aw[i].z, bw[i].z, aw[i].w, bw[i].w, v.z, v.w);
  }
  if (2 * pair < M) {
    down[warp][2 * pair][row] = dw[0];
    down[warp][2 * pair][row + 8] = dw[2];
  }
  if (2 * pair + 1 < M) {
    down[warp][2 * pair + 1][row] = dw[1];
    down[warp][2 * pair + 1][row + 8] = dw[3];
  }
  __syncthreads();

  cluster_wait();
  if (t < M * 4) {
    const int m = t / 4, r = (t % 4) * 4;
    float4 v = *reinterpret_cast<const float4*>(&down[0][m][r]);
#pragma unroll
    for (int i = 1; i < NW; ++i) {
      const float4 u = *reinterpret_cast<const float4*>(&down[i][m][r]);
      v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w;
    }
    st_async4(remote(smem_u32(&partial[part][tile * 16 + r]), m), v, remote(mb_d, m));
  }
  if (rank < M) {
    if (t < RANK / 2) {
      mb_wait(mb_d);
      const int r = t * 2;
      float lo, hi;
      if constexpr (KPARTS == 2) {
        lo = partial[0][r] + partial[1][r];
        hi = partial[0][r + 1] + partial[1][r + 1];
      } else {
        lo = (partial[0][r] + partial[1][r]) + (partial[2][r] + partial[3][r]);
        hi = (partial[0][r + 1] + partial[1][r + 1]) + (partial[2][r + 1] + partial[3][r + 1]);
      }
      grow[t] = nvfp4::pack_bf16(lo, hi);
    }
    __syncthreads();
    if (t < CL * (RANK / 8)) {
      const int d = t % CL, q = t / CL;
      const uint4 u = *reinterpret_cast<const uint4*>(&grow[q * 4]);
      const float4 v = make_float4(__uint_as_float(u.x), __uint_as_float(u.y),
                                   __uint_as_float(u.z), __uint_as_float(u.w));
      st_async4(remote(smem_u32(&gall[rank][q * 8]), d), v, remote(mb_g, d));
    }
  }

#pragma unroll
  for (int pass = 0; pass < C::PASSES; ++pass) {
    if (pass > 0) {
      __syncthreads();  // all warps finished reading the previous forget/gain tile
      stage_fg<C, NT>(a, gate_channel + pass * C::TILE_C, fg, wpol);
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    if (pass == 0) mb_wait(mb_g);

    const int up_row = warp * 16 + row;
    const uint8_t* fa = fg + up_row * C::RS;
    const uint8_t* fb = fa + 8 * C::RS;
    const __nv_bfloat16* gv = gall[row < M ? row : 0];
    float df[4] = {}, dg[4] = {};
#pragma unroll
    for (int k = 0; k < RANK; k += 32) {
      const int col = k + pair * 8;
      const uint4 v = *reinterpret_cast<const uint4*>(gv + col);
      const uint4 f0 = *reinterpret_cast<const uint4*>(fa + 2 * col);
      const uint4 f1 = *reinterpret_cast<const uint4*>(fb + 2 * col);
      const uint4 g0 = *reinterpret_cast<const uint4*>(fa + C::SLICE + 2 * col);
      const uint4 g1 = *reinterpret_cast<const uint4*>(fb + C::SLICE + 2 * col);
      nvfp4::mma_bf16(df, f0.x, f1.x, f0.y, f1.y, v.x, v.y);
      nvfp4::mma_bf16(df, f0.z, f1.z, f0.w, f1.w, v.z, v.w);
      nvfp4::mma_bf16(dg, g0.x, g1.x, g0.y, g1.y, v.x, v.y);
      nvfp4::mma_bf16(dg, g0.z, g1.z, g0.w, g1.w, v.z, v.w);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int m = 2 * pair + (i & 1);
      const int c = pass * C::TILE_C + up_row + (i / 2) * 8;
      if (m < M) {
        const int j = m * H + c0 + c;
        gated[m][c] = (1.f - tanhf(df[i])) * hg[j] + tanhf(dg[i]) * mg[j];
      }
    }
  }
  __syncthreads();

  // Transpose the warp fragments back to one thread per channel for the same RMS exchange as the scalar
  // kernel. The optional FP32 output is the gated mix before normalization.
  const bool own = t < C::TILE_C;
  float values[C::PASSES][M], norm[C::PASSES], sq[M] = {};
  if (own) {
#pragma unroll
    for (int pass = 0; pass < C::PASSES; ++pass) {
      const int c = pass * C::TILE_C + t, channel = c0 + c;
      norm[pass] = a.norm_w[size_t(g) * H + channel];
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float v = gated[m][c];
        values[pass][m] = v;
        sq[m] = pass == 0 ? v * v : fmaf(v, v, sq[m]);
        if (a.out_f32) a.out_f32[(size_t(g) * M + m) * H + channel] = v;
      }
    }
  }
  exchange_squares<CL, M, NW>(sq, red, ssall, rank, mb_s);
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
        a.out_bf16[(size_t(g) * M + m) * H + c] = __float2bfloat16_rn(values[pass][m] * rinv * norm[pass]);
      }
    }
  }
}

template <int CL, bool FG32, bool MMA = false>
void launch_gates(int G, int M, const GateArgs& a, bool pdl, cudaStream_t stream) {
  using C = GateCfg<CL, FG32>;
  auto go = [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    if constexpr (MMA && MM >= 2) {
      static_assert(!MMA || !FG32);
      constexpr int SMEM = C::SMEM + sizeof(float[MM][C::CPC]);
      launch_ex(gates_mma_kernel<CL, MM>, dim3(G * CL), GATE_MMA_THREADS, SMEM, CL, pdl, stream, a);
    } else {
      launch_ex(gates_kernel<CL, FG32, MM>, dim3(G * CL), C::NT, C::SMEM, CL, pdl, stream, a);
    }
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
  const bool c16 = gates_cluster_size() == 16;
  const char* mma = (!fg_fp32 && M >= 2) ? std::getenv("INGINE_GATES_MMA") : nullptr;
  const bool use_mma = !fg_fp32 && M >= 2 && (mma ? mma[0] != '0' : c16);
  if (c16) {
    if (use_mma) launch_gates<16, false, true>(G, M, a, pdl, stream);
    else if (fg_fp32) launch_gates<16, true>(G, M, a, pdl, stream);
    else launch_gates<16, false>(G, M, a, pdl, stream);
  } else {
    if (use_mma) launch_gates<8, false, true>(G, M, a, pdl, stream);
    else if (fg_fp32) launch_gates<8, true>(G, M, a, pdl, stream);
    else launch_gates<8, false>(G, M, a, pdl, stream);
  }
}

}  // namespace ingine::kernels::dense
