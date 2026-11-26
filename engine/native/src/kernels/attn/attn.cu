// Attention core of one Inkling-2-Mini layer visit for M = 1..8 consecutive decode tokens, in ONE kernel.
// Contract: native/KERNELS.md, area `attn`.
//
// Design
// ------
// * Work: one thread-block cluster of CS blocks (16 when supported, otherwise 8) per KV head. The cached
//   keys [0, L_old) (full layer: positions 0..p0-1 of the linear cache; sliding layer: the 1024 ring slots
//   with their kpos) are dealt to the CS blocks in tiles, round robin, so every block gets an equal share
//   of the expensive relative-bias tail (distance < 1024). Block 0 (the "owner") also takes the call's M
//   new keys. Everything is sized from `pos` on the device, so the launch shape is fixed for a CUDA graph.
// * At short contexts the kernel is bound by dependent round trips, not bytes, so it has few of them:
//   (1) every input that does not depend on `pos` (q / rel / gate / sink rows of `proj` and their scales,
//   the owner's conv inputs and rel_proj[:, 0..7]) is fetched with cp.async into shared memory: one round
//   trip. M = 1 issues the K/V stream (first two buffers, all of a sliding layer's keys) before the PDL
//   wait, so it is in flight during proj's tail and the staged round trip is not queued behind it (K/V
//   issued after the wait but ahead of the staged inputs delays them by ~2 us; after them it costs a
//   round trip, 0.5 us per sliding visit in the decode step); M > 1 issues its first tile after the staged
//   inputs land. (2) The rel_proj column of each key is fetched together with its K/V for the distance a
//   full, in-order ring (or the linear cache) implies, and re-read only if the actual kpos disagrees.
//   (3) q / rel post-processing (scale, L2 clip, BF16, log scaling) is done once per block by a few
//   warps, branch-free, and shared through shared memory. (4) Split partials (max, sum, unnormalised o)
//   go to the workspace (L2); one cluster barrier;
//   block r then produces D / CS output dims of every row (sink logit, sigmoid gate, BF16) from the CS
//   partials. (Distributed shared memory was measured slower here: ~1 GB/s per SM on this GPU.)
//   (5) Programmatic dependent launch: weights are staged before griddepcontrol.wait, and dependents are
//   released as soon as the block's first K/V loads are issued (all blocks are resident by then, so
//   the next kernel, proj_conv, only takes free SM resources and streams its weights while this kernel
//   is latency-bound, its requests queued behind ours; it still waits for this grid's completion before
//   reading our output). What the kernel right before (proj) writes, and what this kernel writes for its
//   own tokens (conv states, cache rows >= p0, workspace), is read after the wait only: cp.async /
//   coherent `asm volatile` loads with memory clobbers, which the compiler cannot hoist. `pos` (any M),
//   kpos and the cache rows below p0 (M = 1) are read before it: their writers ran two or more kernels
//   back, and those are complete once this grid runs, because it starts only after every proj block has
//   passed its own griddepcontrol.wait (gemv triggers after it), which returns once the kernel before
//   proj, and hence every earlier kernel, has completed. This holds as long as attention follows a kernel
//   that triggers only after its own wait. The wait is fenced (a loop over p.zero): ptxas sinks loads
//   below a plain one.
// * Main loop, M = 1: warp = key slice (8 warps). A half-warp owns one key; lane c of the half holds dims
//   8c..8c+7, so each K and V row (256 B) is read with 16-byte vector loads (coherent, no L1
//   allocation). Two register buffers of 2 key pairs per warp alternate (one in flight while the other
//   is consumed): 16-32 KB in flight per block, enough for DRAM bandwidth with 128 blocks. Scores are
//   reduced over the 16 lanes with xor shuffles; the bias rel . rel_proj[:, d] is reduced the same way
//   and rounded to BF16, and skipped warp-uniformly when no key of the warp is within 1024. Online
//   softmax in the log2 domain (q is pre-scaled by log2(e)/sqrt(128)), FP32 accumulators.
// * Main loop, M > 1, on tensor cores: the block loads each tile of keys once with cp.async into shared
//   memory (own region, the first tiles issued before q / rel processing). Rows r = 3 token + head
//   (8-padded). S^T = K Q^T with mma.m16n8k16 bf16 (exact bf16 q, k; fp32 sums; 1/sqrt(128) applied to the
//   sums), (16 keys, 8 rows) items dealt over the warps; mask and relative bias per element; per-item row
//   max and P = exp2(S - max) stored as fp16 hi + lo (P - hi). Then O^T += V^T P^T with mma.m16n8k16 fp16
//   (V in fp16, exact for RMS-normed values), warp w owning dims 16w..16w+15, each item rescaled to the
//   running row max in registers. The owner runs the call's own keys first as a one-m-tile tile.
//   BF16 caches: 32-key tiles, double-buffered, with kpos and the rel_proj window; V converted in place.
//   E4M3 caches: 64-key tiles, two raw E4M3 tiles in flight; K mma fragments are read straight from the
//   raw rows (4 bytes per row and k-step, widened in registers, the k order inside an mma permuted and q
//   read with the same permutation), only V is widened to fp16 (a block pass into the staging region),
//   and the next tile is issued as soon as the S items are done. Measured on the full-layer verify
//   shapes (131K keys): M = 2 227 -> 169 us, M = 4 274 -> 191 us, M = 8 319 -> 286 us (32-key tiles,
//   one in flight, K and V widened, before).
// * The owner runs the key/value short convolutions of its head (+ key clip / value RMS norm) into shared
//   memory; it is the only reader of those conv-state channels. Its global writes (conv states, new
//   cache rows) come at the end. Sliding layers, M = 1: the ring slot and kpos it overwrites hold the
//   entry at distance 1024 (invalid for this token either way; a kpos >= p0 is ignored too), so the
//   other heads may read either value. M > 1: the overwritten slots still hold keys earlier tokens of
//   the call must see: the owner writes them after the cluster barrier (its head is done reading) and
//   the last head to get there (global counter in the workspace, reset by itself) writes kpos. A head
//   takes its turn right after its barrier, on a thread without output work, so the counter's round
//   trip overlaps the output phase instead of ending the kernel (-0.5 us per call).
// * Ring semantics: a cached ring entry is used iff 0 <= kpos < p0 and p - kpos < 1024; entries at
//   kpos >= p0 (left over from rejected speculative tokens) are ignored, as if overwritten first.
//   Values of masked keys are never multiplied in, so unused cache slots may hold anything.
// * FP8 full-layer caches (template F8; sliding rings are BF16 only): E4M3 without scale. The owner rounds
//   its new keys / values to BF16 and then to E4M3 (round to nearest even, saturated to +-448) in shared
//   memory, so the call's own keys are attended with exactly the values the cache will hold, and writes the
//   E4M3 bytes at the end. M = 1 reads 8-byte vectors with the same lanes, dims and buffer depth (half the
//   bytes in flight: the M = 1 kernels already use 128 registers). M > 1: see the main loop above.
// * Capturable: no allocation, no host sync, current stream; positions come from `pos` on the device.
// * Positions are int32 on the device (row offsets 64-bit); the host caps full caches at 2^30 rows.
// Ported from engine/csrc/attn/attn.cu (kernel bodies unchanged; torch launcher -> raw pointers).
#include "kernels/attn/attn.h"
#include "kernels/attn/attn_batch.h"

#include "kernels/common/l2.cuh"

#include "kernels/attn/attn_impl.cuh"
#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include "model/tp_layout.h"

#include <stdexcept>
#include <string>
#include <type_traits>

namespace ingine::kernels::attn {
namespace {

constexpr int D = model::HEAD_DIM, DREL = model::D_REL, EXT = 1024, WIN = 1024, GQA = 3;
constexpr int MAXM = 8;
constexpr int U = 2;  // key pairs per warp per register buffer
constexpr int MAXWARPS = 8;
constexpr float CLIP_QK = 22.627416997969522f, CLIP_REL = 8.f;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float QSCALE = 0.08838834764831845f * LOG2E;  // 1/sqrt(128) * log2(e)
constexpr float NEG_INIT = -1e30f;
constexpr float INV_SQRT5 = 1.f / 2.2360679774997896f;  // torch: x / scalar == x * (1 / float(scalar))
constexpr float INV_3 = 1.f / 3.f;
// workspace: int32 counter (kpos ordering for M > 1, zeroed once at allocation) then the split partials
// fp32 [HKV][CS][3M][PSTR]: o (D) then (max, sum)
constexpr size_t WS_PART = 256;
constexpr int PSTR = D + 4;
constexpr size_t ws_bytes(int M, int cs, int hkv) { return WS_PART + size_t(hkv) * cs * GQA * M * PSTR * 4; }

constexpr int NT = 256;  // threads per block (8 warps) for every M

// dynamic shared memory layout (floats), per M
template <int CS, int M, bool F8, bool Q4 = false>
struct Smem {
    static constexpr int OD = D / CS;
    static constexpr int ROWS = GQA * M;
    // staged inputs (dead once the main loop starts)
    static constexpr int Q = 0;                          // proj q rows [M][3][D]
    static constexpr int QS = Q + M * GQA * D;           // q_scale [3][D]
    static constexpr int REL = QS + GQA * D;             // proj rel [M][3][16]
    static constexpr int RS = REL + M * GQA * DREL;      // r_scale [3][16]
    static constexpr int ST = RS + GQA * DREL;           // conv state [3][2][D]
    static constexpr int X = ST + 3 * 2 * D;             // k/v projections [M][2][D]
    static constexpr int W = X + M * 2 * D;              // conv weights [2][D][4] (sliding) | [2][4][D] (full)
    static constexpr int DYN = W + 2 * D * 4;            // dynamic taps [M][2][4]
    static constexpr int NKF = DYN + M * 2 * 4;          // conv outputs [2][M][D]
    static constexpr int STAGE_END = NKF + 2 * M * D;
    // block merge, aliases the staged inputs
    static constexpr int RED_O = 0;                      // [warps][3][D]
    static constexpr int RED_ML = RED_O + MAXWARPS * GQA * D;
    static constexpr int MERGE_END = RED_ML + MAXWARPS * GQA * 2;
    static constexpr int UNION_END = STAGE_END > MERGE_END ? STAGE_END : MERGE_END;
    // M > 1: K/V tiles loaded by the whole block (own region, so the first tiles are in flight while q /
    // rel and the owner's convolutions are computed); 16-byte chunks are XOR-swizzled by key.
    // BF16 caches: double-buffered 16-bit stages [2][KT][2][D], K rows bf16, V rows converted to fp16
    // in place. E4M3 caches (full layers): two raw tiles [2][K: KT][D] + [V: KT][D] bytes (K chunks
    // swizzled; K is read straight from them), V widened to fp16 in the staging region.
    static constexpr int KT = F8 ? 64 : 32;              // keys per tile
    static constexpr int KVS = (UNION_END + 3) & ~3;     // 32 KB either way
    static constexpr int KPS = KVS + (M > 1 ? 2 * KT * 2 * D * (F8 ? 1 : 2) / 4 : 0);  // kpos (int) [2][KT]
    static constexpr int SRP = KPS + (M > 1 && !F8 ? 2 * KT : 0);      // rel_proj window [2][DREL][KT + MAXM]
    static constexpr int TILE_END = SRP + (M > 1 && !F8 ? 2 * DREL * (KT + MAXM) : 0);
    // dedicated regions
    static constexpr int GATE = (TILE_END + 3) & ~3;     // this block's output gates [ROWS][OD]
    static constexpr int SINK = GATE + ROWS * OD;        // [ROWS]
    static constexpr int NKV = (SINK + ROWS + 3) & ~3;   // new k/v bf16 [2][M][D] (as floats: M * D)
    static constexpr int RELP0 = NKV + M * D;            // rel_proj[:, 0..7] [16][8]
    static constexpr int NST = RELP0 + DREL * 8;         // advanced conv state [3][2][D]
    // M > 1 tensor-core path: query rows r = 3 token + head, padded to NRP (multiple of 8)
    static constexpr int NRP = M > 1 ? (ROWS + 7) / 8 * 8 : 0;
    static constexpr int QB = NST + 3 * 2 * D;           // q bf16 [NRP][D] (swizzled; as floats NRP D / 2)
    static constexpr int RELD = QB + NRP * D / 2;        // rel fp32 [NRP][16]
    static constexpr int PSTRIDE = KT + 8;               // fp16 per P row (80 B: conflict-free ldmatrix)
    static constexpr int PS = RELD + NRP * DREL;         // P as fp16 hi, lo [2][NRP][PSTRIDE]
    static constexpr int ITM = PS + NRP * PSTRIDE;       // per key m-tile row (max, sum) float2 [KT/16][NRP]
    static constexpr int END = ITM + 2 * (KT / 16) * NRP;
    static constexpr int Q4_PACKED = END * 4;
    static constexpr int Q4_SCALE = Q4_PACKED + 2 * M * q4_g64::head_bytes;
    static constexpr size_t BYTES = Q4 ? Q4_SCALE + 2 * M * q4_g64::groups * sizeof(__half) : size_t(END) * 4;
    static_assert(CS != 8 || BYTES <= 101376, "attention must fit in GB10 shared memory");
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return unsigned(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldsm_x2(uint32_t addr, uint32_t (&r)[2]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

// D += A (16x16, row) * B (16x8, col), fp32 accumulators
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

struct Params {
    const float* proj;
    int R;
    long long cap;  // cache rows (full layers)
    int log_scaling;
    const float* q_scale;
    const float* r_scale;
    const float* rel_proj;
    const float* k_conv;
    const float* v_conv;
    float* conv_k;
    float* conv_v;
    void* k_cache;  // __nv_bfloat16, or E4M3 bytes (F8)
    void* v_cache;
    void* k_scale;
    void* v_scale;
    long long* kpos;
    const long long* pos;
    int* counter;
    float* part;
    __nv_bfloat16* out;
    int zero;  // 0 at run time, opaque to the compiler (fenced griddepcontrol.wait)
};

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float hsum16(float x) {
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ float wsum(float x) {
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ void unpack8(const uint4 u, float f[8]) {
    f[0] = __uint_as_float(u.x << 16), f[1] = __uint_as_float(u.x & 0xffff0000u);
    f[2] = __uint_as_float(u.y << 16), f[3] = __uint_as_float(u.y & 0xffff0000u);
    f[4] = __uint_as_float(u.z << 16), f[5] = __uint_as_float(u.z & 0xffff0000u);
    f[6] = __uint_as_float(u.w << 16), f[7] = __uint_as_float(u.w & 0xffff0000u);
}

// E4M3 pair (low byte = lower dim) to floats, exact (cvt.f16x2.e4m3x2, then FP16 -> FP32)
__device__ __forceinline__ float2 e4m3x2_f2(uint32_t pair) {
    return __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(pair), __NV_E4M3)));
}

// 8 E4M3 values (dims 8c..8c+7, byte 0 = lowest dim)
__device__ __forceinline__ void unpack8(const uint2 u, float f[8]) {
    float2 a = e4m3x2_f2(u.x & 0xffffu), b = e4m3x2_f2(u.x >> 16);
    f[0] = a.x, f[1] = a.y, f[2] = b.x, f[3] = b.y;
    a = e4m3x2_f2(u.y & 0xffffu), b = e4m3x2_f2(u.y >> 16);
    f[4] = a.x, f[5] = a.y, f[6] = b.x, f[7] = b.y;
}

// x rounded to E4M3: nearest even, saturated to +-448 (torch: x.clamp(-448, 448).to(float8_e4m3fn))
__device__ __forceinline__ float e4m3r(float x) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3), __NV_E4M3)));
}

// two floats holding E4M3 values to their bytes (a in the low byte)
__device__ __forceinline__ uint32_t pack_e4m3x2(float a, float b) {
    return __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
}

// Loads of anything an earlier kernel writes (caches, kpos, pos) are coherent `asm volatile` with a memory
// clobber, so the compiler cannot hoist them above griddepcontrol.wait (programmatic dependent launch).
// Cached K/V rows are read once per step (the next read of a ring row is a full step, ~1.5 GB of streams,
// later): their L2 lines are filled evict-first (kernels/common/l2.cuh).
constexpr l2::Evict KV_L2 = l2::Evict::first;

__device__ __forceinline__ uint4 ld_kv(const void* ptr, std::uint64_t pol) {
    uint4 v;
    if constexpr (l2::hinted<KV_L2>)
        asm volatile("ld.global.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(ptr), "l"(pol)
                     : "memory");
    else
        asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(ptr)
                     : "memory");
    return v;
}

__device__ __forceinline__ uint2 ld_kv8(const void* ptr, std::uint64_t pol) {
    uint2 v;
    if constexpr (l2::hinted<KV_L2>)
        asm volatile("ld.global.L1::no_allocate.L2::cache_hint.v2.u32 {%0,%1}, [%2], %3;"
                     : "=r"(v.x), "=r"(v.y)
                     : "l"(ptr), "l"(pol)
                     : "memory");
    else
        asm volatile("ld.global.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(ptr) : "memory");
    return v;
}

__device__ __forceinline__ long long ld_s64(const long long* ptr) {
    long long v;
    asm volatile("ld.global.s64 %0, [%1];" : "=l"(v) : "l"(ptr) : "memory");
    return v;
}

__device__ __forceinline__ void cp16(float* dst, const float* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(unsigned(__cvta_generic_to_shared(dst))),
                 "l"(src)
                 : "memory");
}

__device__ __forceinline__ void cp4(float* dst, const float* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(unsigned(__cvta_generic_to_shared(dst))), "l"(src)
                 : "memory");
}

// zero-filling variants: nothing is read when !ok (cp16z: cached K/V rows, KV_L2 lines)
__device__ __forceinline__ void cp16z(float* dst, const float* src, bool ok, std::uint64_t pol) {
    if constexpr (l2::hinted<KV_L2>)
        asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;" ::"r"(
                         unsigned(__cvta_generic_to_shared(dst))),
                     "l"(src), "r"(ok ? 16 : 0), "l"(pol)
                     : "memory");
    else
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(unsigned(__cvta_generic_to_shared(dst))),
                     "l"(src), "r"(ok ? 16 : 0)
                     : "memory");
}

__device__ __forceinline__ void cp4z(float* dst, const float* src, bool ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(unsigned(__cvta_generic_to_shared(dst))), "l"(src),
                 "r"(ok ? 4 : 0)
                 : "memory");
}

__device__ __forceinline__ uint32_t pack2(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// clip factor min(1, t / (sqrt(ss) + 1e-6)) without the IEEE sqrt / div slow-path branches (which
// serialise the six reductions of the q / rel prologue); differs from torch by a couple of FP32 ulp
__device__ __forceinline__ float clip_factor(float ss, float t) {
    const float norm = ss > 0.f ? ss * rsqrtf(ss) : 0.f;
    return fminf(1.f, __fdividef(t, norm + 1e-6f));
}

template <typename KV>  // 8 dims of a cache row: uint4 = BF16, uint2 = E4M3
struct Buf {
    KV k[U], v[U];
    int kp[U];    // key position, -1 = no key
    int dp[U];    // distance the rel_proj prefetch was made for
    float rp[U];  // rel_proj[c][dp] (0 if dp >= 1024)
};

struct BatchParams {
    Params common;
    const BatchAttnSlot* slots;
    std::size_t workspace_stride;

    template <int HKV>
    __device__ Params row() const {
        const int index = blockIdx.z;
        const auto& slot = slots[index];
        Params p = common;
        p.proj += std::size_t(index) * p.R;
        p.out += std::size_t(index) * 3 * HKV * D;
        p.k_cache = slot.k_cache;
        p.v_cache = slot.v_cache;
        p.k_scale = slot.k_scale;
        p.v_scale = slot.v_scale;
        p.conv_k = slot.conv_k_state;
        p.conv_v = slot.conv_v_state;
        p.kpos = reinterpret_cast<long long*>(slot.kpos);
        p.pos = reinterpret_cast<const long long*>(&slot.position);
        auto* workspace = reinterpret_cast<char*>(p.counter) + index * workspace_stride;
        p.counter = reinterpret_cast<int*>(workspace);
        p.part = reinterpret_cast<float*>(workspace + WS_PART);
        return p;
    }
};

template <int CS, int M, bool FULL, bool F8, int HKV = 8, bool Q4 = false>
__global__ void __launch_bounds__(MAXWARPS * 32, 2) attn_decode_kernel(const Params p) {
#include "decode_body.cuh"
}

template <int CS, bool FULL, bool F8, int HKV, bool Q4>
__global__ void __launch_bounds__(MAXWARPS * 32, 2) attn_decode_batch_kernel(const BatchParams input) {
    constexpr int M = 1;
    const Params p = input.row<HKV>();
#include "decode_body.cuh"

}

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("attn: ") + what);
}

using KernelFn = void (*)(Params);

struct Kernel {
    KernelFn fn;
    int cs;
    size_t smem;
};

template <int CS, int M, bool FULL, bool F8, int HKV = 8, bool Q4 = false>
constexpr Kernel kernel_of() {
    return {attn_decode_kernel<CS, M, FULL, F8, HKV, Q4>, CS, Smem<CS, M, F8, Q4>::BYTES};
}

template <int CS, int M, bool FULL, bool F8, int HKV = 8, bool Q4 = false>
Kernel configured_kernel(int device) {
    constexpr Kernel k = kernel_of<CS, M, FULL, F8, HKV, Q4>();
    static thread_local int configured_device = -1;
    if (configured_device != device) {
        if constexpr (CS > 8) CUDA_CHECK(cudaFuncSetAttribute(k.fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        CUDA_CHECK(cudaFuncSetAttribute(k.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(k.smem)));
        configured_device = device;
    }
    return k;
}

template <int CS, bool FULL, bool F8, int HKV, bool Q4>
Kernel select(int M, int device) {
    switch (M) {
        case 1: return configured_kernel<CS, 1, FULL, F8, HKV, Q4>(device);
        case 2: return configured_kernel<CS, 2, FULL, F8, HKV, Q4>(device);
        case 3: return configured_kernel<CS, 3, FULL, F8, HKV, Q4>(device);
        case 4: return configured_kernel<CS, 4, FULL, F8, HKV, Q4>(device);
        case 5: return configured_kernel<CS, 5, FULL, F8, HKV, Q4>(device);
        case 6: return configured_kernel<CS, 6, FULL, F8, HKV, Q4>(device);
        case 7: return configured_kernel<CS, 7, FULL, F8, HKV, Q4>(device);
        default: return configured_kernel<CS, 8, FULL, F8, HKV, Q4>(device);
    }
}

template <bool FULL, bool F8 = false, int HKV = 8, bool Q4 = false>
Kernel select(int M, int cs, int device) {
    return cs == 16 ? select<16, FULL, F8, HKV, Q4>(M, device) : select<8, FULL, F8, HKV, Q4>(M, device);
}

// attrs[0]: cluster of CS blocks; attrs[1]: programmatic dependent launch (the kernel waits with
// griddepcontrol.wait before touching anything an earlier kernel may write; a no-op without PDL)
cudaLaunchConfig_t launch_config(const Kernel& k, cudaStream_t stream, cudaLaunchAttribute* attrs, int hkv = 8) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(k.cs, hkv);
    cfg.blockDim = dim3(NT);
    cfg.dynamicSmemBytes = k.smem;
    cfg.stream = stream;
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = k.cs, attrs[0].val.clusterDim.y = 1, attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 2;
    return cfg;
}

int attn_cluster_size(int device) {
    struct Selection {
        int device = -1;
        int cs = 8;
    };
    static thread_local Selection selected;
    if (selected.device != device) {
        // Probe the variant with the largest shared-memory footprint so the choice covers all widths.
        constexpr Kernel k = kernel_of<16, MAXM, false, false>();
        cudaLaunchAttribute attrs[2];
        cudaLaunchConfig_t cfg = launch_config(k, nullptr, attrs);
        cfg.numAttrs = 1;
        int active = 0;
        if (cudaFuncSetAttribute(k.fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) != cudaSuccess ||
            cudaFuncSetAttribute(k.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(k.smem)) != cudaSuccess ||
            cudaOccupancyMaxActiveClusters(&active, k.fn, &cfg) != cudaSuccess) {
            (void)cudaGetLastError();
            active = 0;
        }
        selected = {device, active > 0 ? 16 : 8};
    }
    return selected.cs;
}

}  // namespace

std::size_t workspace_bytes_batched(int max_m, int max_batch, int hkv) {
    return batched_workspace_bytes(max_m, max_batch, hkv);
}

std::size_t workspace_bytes(int max_m, int hkv) {
    require(hkv == 8 || hkv == 4, "hkv must be 8 or 4");
    require(max_m >= 1 && max_m <= MAXM, "max_m must be in 1..8");
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return ws_bytes(max_m, attn_cluster_size(device), hkv);
}

int max_active_clusters(int M, int window) {
    require(M >= 1 && M <= MAXM, "M must be in 1..8");
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    const int cs = attn_cluster_size(device);
    const Kernel k = window == 0 ? select<true>(M, cs, device) : select<false>(M, cs, device);
    cudaLaunchAttribute attrs[2];
    cudaLaunchConfig_t cfg = launch_config(k, nullptr, attrs);
    cfg.numAttrs = 1;  // occupancy: the cluster shape only
    int n = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, k.fn, &cfg));
    return n;
}

void attn_decode(const DecodeArgs& a, cudaStream_t stream) {
    if (a.batch > 1 || a.proj16 || !a.conv_write) { attn_decode_batched(a, stream); return; }
    require(a.window == 0 || a.window == WIN, "window must be 0 (full layer) or 1024 (sliding layer)");
    const bool full = a.window == 0;
    const int M = a.M;
    require(M >= 1 && M <= MAXM, "M must be in 1..8");
    require(a.proj && a.q_scale && a.r_scale && a.rel_proj && a.k_conv && a.v_conv && a.conv_k_state &&
                a.conv_v_state && a.k_cache && a.v_cache && a.pos && a.workspace && a.out,
            "null argument");
    require(full || !a.cache_fp8, "float8_e4m3fn caches are supported on full layers (window = 0) only");
    detail::check_q4(a);
    // device positions and loop bounds are int32; 2^30 leaves headroom for the tile strides
    require(a.cache_rows >= 1 && a.cache_rows <= (int64_t(1) << 30), "caches hold 1..2^30 positions");
    if (!full) {
        require(a.cache_rows == WIN, "sliding caches must have 1024 rows");
        require(a.kpos != nullptr, "sliding layers need kpos");
    }
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    const int cs = attn_cluster_size(device);
    require(a.hkv == 8 || a.hkv == 4, "hkv must be 8 or 4");
    require(a.workspace_size >= ws_bytes(M, cs, a.hkv), "workspace too small: need workspace_bytes");

    Params p{};
    p.proj = a.proj;
    p.R = model::TpLayout(model::KV_HEADS / a.hkv).projection_stride(full);
    p.cap = a.cache_rows;
    p.log_scaling = a.log_scaling;
    p.q_scale = a.q_scale;
    p.r_scale = a.r_scale;
    p.rel_proj = a.rel_proj;
    p.k_conv = a.k_conv;
    p.v_conv = a.v_conv;
    p.conv_k = a.conv_k_state;
    p.conv_v = a.conv_v_state;
    p.k_cache = a.k_cache;
    p.v_cache = a.v_cache;
    p.k_scale = a.k_scale;
    p.v_scale = a.v_scale;
    p.kpos = full ? nullptr : reinterpret_cast<long long*>(a.kpos);
    p.pos = reinterpret_cast<const long long*>(a.pos);
    p.counter = static_cast<int*>(a.workspace);
    p.part = reinterpret_cast<float*>(static_cast<char*>(a.workspace) + WS_PART);
    p.out = a.out;
    const auto select_heads = [&]<int HKV>() {
        if (!full) return select<false, false, HKV>(M, cs, device);
        if (a.cache_q4) return select<true, false, HKV, true>(M, cs, device);
        return a.cache_fp8 ? select<true, true, HKV>(M, cs, device) : select<true, false, HKV>(M, cs, device);
    };
    const Kernel k = a.hkv == 8 ? select_heads.operator()<8>() : select_heads.operator()<4>();
    cudaLaunchAttribute attrs[2];
    const cudaLaunchConfig_t cfg = launch_config(k, stream, attrs, a.hkv);
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, k.fn, p));
}

namespace {

template <int CS, bool Full, bool Fp8, int Hkv, bool Q4 = false>
void launch_batch(BatchParams p, int batch, cudaStream_t stream, int device) {
    constexpr auto kernel = attn_decode_batch_kernel<CS, Full, Fp8, Hkv, Q4>;
    constexpr auto smem = Smem<CS, 1, Fp8, Q4>::BYTES;
    static thread_local int configured_device = -1;
    if (configured_device != device) {
        if constexpr (CS > 8)
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
        configured_device = device;
    }
    cudaLaunchAttribute attrs[2];
    auto cfg = launch_config({nullptr, CS, smem}, stream, attrs, Hkv);
    cfg.gridDim.z = batch;
    // Complete earlier writes to the slot table before reading it.
    cfg.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, p));
}

}  // namespace

void attn_decode_batch(const DecodeArgs& a, const BatchAttnSlot* slots, int batch, cudaStream_t stream) {
    const bool full = a.window == 0;
    require(batch >= 1 && batch <= MAXM && a.M == 1, "expected 1..8 independent M=1 rows");
    require(a.hkv == 8 || a.hkv == 4, "hkv must be 8 or 4");
    require(slots && a.proj && a.q_scale && a.r_scale && a.rel_proj && a.k_conv && a.v_conv &&
                a.workspace && a.out,
            "null batch argument");
    require(a.cache_rows >= 1 && a.cache_rows <= (int64_t(1) << 30), "caches hold 1..2^30 positions");
    detail::check_q4(a, true);
    require(full || (a.window == WIN && a.cache_rows == WIN && !a.cache_fp8), "invalid sliding cache");

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    const int cs = attn_cluster_size(device);
    const auto stride = ws_bytes(1, cs, a.hkv);
    require(a.workspace_size >= batch * stride, "batch workspace needs one slice per row");

    BatchParams packet{};
    packet.slots = slots;
    packet.workspace_stride = stride;
    Params& p = packet.common;
    p.proj = a.proj;
    p.R = model::TpLayout(model::KV_HEADS / a.hkv).projection_stride(full);
    p.cap = a.cache_rows;
    p.log_scaling = a.log_scaling;
    p.q_scale = a.q_scale;
    p.r_scale = a.r_scale;
    p.rel_proj = a.rel_proj;
    p.k_conv = a.k_conv;
    p.v_conv = a.v_conv;
    p.counter = static_cast<int*>(a.workspace);
    p.out = a.out;
    auto launch = [&]<int CS, int Hkv>() {
        if (!full) launch_batch<CS, false, false, Hkv>(packet, batch, stream, device);
        else if (a.cache_q4) launch_batch<CS, true, false, Hkv, true>(packet, batch, stream, device);
        else if (a.cache_fp8) launch_batch<CS, true, true, Hkv>(packet, batch, stream, device);
        else launch_batch<CS, true, false, Hkv>(packet, batch, stream, device);
    };
    if (a.hkv == 8) {
        if (cs == 16) launch.operator()<16, 8>();
        else launch.operator()<8, 8>();
    } else {
        if (cs == 16) launch.operator()<16, 4>();
        else launch.operator()<8, 4>();
    }
}

}  // namespace ingine::kernels::attn
