// Lossless storage of the 128 BF16 bit patterns of a global K or V head.
// Words and packed pairs use the cache's little-endian channel order.
#pragma once

#include <cstddef>
#include <cstdint>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define INGINE_KV12_INLINE __host__ __device__ __forceinline__
#else
#define INGINE_KV12_INLINE inline
#endif

namespace ingine::kernels::kv12 {

using std::uint8_t;
using std::uint16_t;
using std::uint32_t;

inline constexpr int dims = 128;
inline constexpr int inline_exponents = 15;

// Bytes 0..127: sign and seven fraction bits. Bytes 128..191: exponent
// nibbles, low nibble first. Bytes 192..206: escaped exponents in channel
// order. Byte 207: maximum biased exponent (at least 1), or 0 for raw.
// In raw mode the first 51 words and the 13 spill words hold all 64 BF16
// pairs. Keeping spill separate makes the common read independent of it.
struct alignas(16) Head { uint32_t words[52]; };
struct Spill { uint32_t words[13]; };
static_assert(sizeof(Head) == 208 && sizeof(Spill) == 52);

INGINE_KV12_INLINE uint8_t exponent(uint16_t bits) { return (bits >> 7) & 255; }
INGINE_KV12_INLINE uint8_t sign_fraction(uint16_t bits) { return ((bits >> 8) & 128) | (bits & 127); }
INGINE_KV12_INLINE int code(int exp, int base) {
  return exp == 0 ? 0 : base - exp < 14 ? 1 + base - exp : 15;
}
INGINE_KV12_INLINE uint16_t restore(int sm, int exp) {
  return uint16_t(((sm & 128) << 8) | (exp << 7) | (sm & 127));
}
// Device loads are coherent, including fresh rows written by another warp
// of the same CTA. The memory clobber also orders reads against PDL waits.
INGINE_KV12_INLINE uint32_t load_word(const uint32_t* p) {
#ifdef __CUDA_ARCH__
  uint32_t value;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
  return value;
#else
  return *p;
#endif
}
INGINE_KV12_INLINE int load_byte(const uint8_t* p) {
#ifdef __CUDA_ARCH__
  uint32_t value;
  asm volatile("ld.global.cg.u8 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
  return int(value);
#else
  return *p;
#endif
}
INGINE_KV12_INLINE uint32_t raw_pair(const Head& h, const Spill& s, int i) {
  return load_word(i < 51 ? &h.words[i] : &s.words[i - 51]);
}
INGINE_KV12_INLINE int count_escapes(uint32_t codes) {
  // One bit at nibble positions 0,4,... for every 0xf.
  uint32_t ones = (codes & (codes >> 1) & (codes >> 2) & (codes >> 3)) & 0x11111111u;
#ifdef __CUDA_ARCH__
  return __popc(ones);
#else
  return __builtin_popcount(ones);
#endif
}

// Independent eight-channel read; also works when callers in a warp read
// unrelated rows (ragged batches and prefill). Only escape chunks need the
// prefix count. No warp collectives on the streaming decode path.
INGINE_KV12_INLINE void unpack8(const Head& h, const Spill& s, int chunk, uint32_t (&out)[4]) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(&h);
  const int base = load_byte(bytes + 207);
  if (!base) {
    for (int j = 0; j < 4; ++j) out[j] = raw_pair(h, s, chunk * 4 + j);
    return;
  }
  const uint32_t codes = load_word(h.words + 32 + chunk);
  int rank = 0;
  if (count_escapes(codes)) {
    for (int c = 0; c < chunk; ++c) rank += count_escapes(load_word(h.words + 32 + c));
  }
  const uint32_t sm_words[2] = {load_word(h.words + 2 * chunk), load_word(h.words + 2 * chunk + 1)};
#ifdef __CUDACC__
#pragma unroll
#endif
  for (int p = 0; p < 4; ++p) {
    const uint32_t sm = sm_words[p / 2] >> (16 * (p % 2));
    uint32_t pair = 0;
#ifdef __CUDACC__
#pragma unroll
#endif
    for (int j = 0; j < 2; ++j) {
      const int c = (codes >> (8 * p + 4 * j)) & 15;
      const int exp = c == 15 ? load_byte(bytes + 192 + rank++) : c == 0 ? 0 : base + 1 - c;
      pair |= uint32_t(restore((sm >> (8 * j)) & 255, exp)) << (16 * j);
    }
    out[p] = pair;
  }
}

// Scalar producer for host caches, interoperability checks and fixtures.
inline void pack(const uint16_t* bf16, Head& h, Spill& s) {
  int base = 1;
  for (int i = 0; i < dims; ++i) {
    const int exp = exponent(bf16[i]);
    if (exp > base) base = exp;
  }
  h = {};
  auto* bytes = reinterpret_cast<uint8_t*>(&h);
  int escapes = 0;
  for (int i = 0; i < dims; ++i) {
    const int exp = exponent(bf16[i]), c = code(exp, base);
    if (c == 15) {
      if (escapes == inline_exponents) {
        for (int p = 0; p < dims / 2; ++p) {
          const auto pair = uint32_t(bf16[2 * p]) | (uint32_t(bf16[2 * p + 1]) << 16);
          (p < 51 ? h.words[p] : s.words[p - 51]) = pair;
        }
        h.words[51] = 0;
        return;
      }
      bytes[192 + escapes++] = uint8_t(exp);
    }
    bytes[i] = sign_fraction(bf16[i]);
    h.words[32 + i / 8] |= uint32_t(c) << (4 * (i % 8));
  }
  bytes[207] = uint8_t(base);
}

#ifdef __CUDACC__
// One whole warp per head. Each lane supplies four consecutive BF16 bit
// patterns in two uint32 pairs. The caller orders publication and reads.
__device__ __forceinline__ void pack4(uint2 bits, Head& h, Spill& s) {
  constexpr unsigned mask = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  uint16_t value[4] = {uint16_t(bits.x), uint16_t(bits.x >> 16), uint16_t(bits.y), uint16_t(bits.y >> 16)};
  int local_max = 1;
#pragma unroll
  for (int j = 0; j < 4; ++j) local_max = max(local_max, int(exponent(value[j])));
  const int base = __reduce_max_sync(mask, local_max);
  int n = 0;
  uint32_t sm = 0, nibbles = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int c = code(exponent(value[j]), base);
    n += c == 15;
    sm |= uint32_t(sign_fraction(value[j])) << (8 * j);
    nibbles |= uint32_t(c) << (4 * j);
  }
  const int total = __reduce_add_sync(mask, n);
  if (total > inline_exponents) {
    const int p = 2 * lane;
    (p < 51 ? h.words[p] : s.words[p - 51]) = bits.x;
    (p + 1 < 51 ? h.words[p + 1] : s.words[p + 1 - 51]) = bits.y;
    if (lane == 0) h.words[51] = 0;
    return;
  }
  h.words[lane] = sm;
  const uint32_t other = __shfl_xor_sync(mask, nibbles, 1);
  if (!(lane & 1)) h.words[32 + lane / 2] = nibbles | (other << 16);
  auto* bytes = reinterpret_cast<uint8_t*>(&h);
  if (lane == 15) bytes[207] = uint8_t(base);
  if (lane >= total && lane < inline_exponents) bytes[192 + lane] = 0;
  if (total) {
    int prefix = n;
#pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
      const int left = __shfl_up_sync(mask, prefix, delta);
      if (lane >= delta) prefix += left;
    }
    int rank = prefix - n;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      if (((nibbles >> (4 * j)) & 15) == 15) bytes[192 + rank++] = exponent(value[j]);
  }
}

__device__ __forceinline__ uint4 load8(const void* cache, const void* spill, std::size_t head, int chunk) {
  uint32_t pairs[4];
  unpack8(static_cast<const Head*>(cache)[head], static_cast<const Spill*>(spill)[head], chunk, pairs);
  return make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
}
#endif

}  // namespace ingine::kernels::kv12
#undef INGINE_KV12_INLINE
