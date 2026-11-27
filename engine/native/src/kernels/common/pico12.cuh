// PICO12 stores BF16 low bytes verbatim and encodes each high byte as a sign bit and a
// three-bit code. Optional BF16 outliers are stored in per-row CSR and decode to +0 in the dense path.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__aarch64__) && !defined(__CUDACC__)
#include <arm_neon.h>
#endif

#ifdef __CUDACC__
#include "kernels/common/l2.cuh"
#endif

namespace ingine::kernels::pico12 {

inline constexpr unsigned kTile = 32;
inline constexpr unsigned kChunk = 8;
inline constexpr unsigned kEscape = 0;
inline constexpr int kCompactCols = 1 << 16;

// The first two LUT words hold the eight upper exponent fields, in code order. In sparse
// matrices code zero is reserved for escapes.
struct Layout {
  uint32_t lut[4]{};
  int rows = 0;
  int cols = 0;
  bool escapes = false;
  size_t row_mask_offset = 0;
  size_t outlier_rows_offset = 0;
  int num_outlier_rows = 0;
};

struct Packed {
  std::unique_ptr<uint8_t[]> data;
  size_t bytes = 0;
  Layout layout;
};

struct DeviceView {
  uint32_t lut[4];
  const uint32_t* row_ptr;
  const uint32_t* entries;
  int cols = 0;
  const uint32_t* row_mask = nullptr;
  const int32_t* outlier_rows = nullptr;
  int num_outlier_rows = 0;
};

constexpr size_t padded_cols(int cols) { return (size_t(cols) + kTile - 1) / kTile * kTile; }

#ifdef __CUDACC__
__host__ __device__
#endif
constexpr size_t row_bytes(size_t padded) { return padded + padded / 2; }
constexpr size_t row_ptr_bytes(int rows) { return (size_t(rows) + 1) * sizeof(uint32_t); }
constexpr size_t row_mask_bytes(int rows) { return ((size_t(rows) + 31) / 32) * sizeof(uint32_t); }

inline DeviceView device_view(const Layout& layout, const uint8_t* device_blob) {
  DeviceView view{};
  std::memcpy(view.lut, layout.lut, sizeof(view.lut));
  view.cols = layout.cols;
  if (layout.escapes) {
    const size_t padded = padded_cols(layout.cols);
    const uint8_t* row_ptr = device_blob + size_t(layout.rows) * row_bytes(padded);
    view.row_ptr = reinterpret_cast<const uint32_t*>(row_ptr);
    view.entries = reinterpret_cast<const uint32_t*>(row_ptr + row_ptr_bytes(layout.rows));
    if (layout.num_outlier_rows > 0) {
      view.row_mask = reinterpret_cast<const uint32_t*>(device_blob + layout.row_mask_offset);
      view.outlier_rows = reinterpret_cast<const int32_t*>(device_blob + layout.outlier_rows_offset);
      view.num_outlier_rows = layout.num_outlier_rows;
    }
  }
  return view;
}

namespace detail {

constexpr uint8_t exponent_hi(uint16_t bits) { return (bits >> 8) & 0x7f; }

#ifdef __CUDACC__
__host__ __device__
#endif
constexpr uint32_t escape_markers(uint32_t codes) {
  return ~(codes | (codes >> 1) | (codes >> 2)) & 0x11111111u;
}

template <class T>
inline T read(const void* ptr) {
  T value;
  std::memcpy(&value, ptr, sizeof(T));
  return value;
}

template <class T>
inline void write(void* ptr, T value) {
  std::memcpy(ptr, &value, sizeof(T));
}

using Histogram = std::array<size_t, 128>;

struct HostCodes {
  std::array<uint8_t, 256> nibbles;
#if defined(__aarch64__) && !defined(__CUDACC__)
  uint8x16x4_t band{};
  uint8_t base = 0;
  bool use_band = false;
#endif
};

struct Partition {
  Histogram histogram{};
  int begin = 0;
  int end = 0;
  size_t entry_start = 0;
};

template <class Work>
void for_partitions(std::vector<Partition>& parts, Work work) {
  std::vector<std::jthread> threads;
  threads.reserve(parts.size() - 1);
  for (size_t i = 1; i < parts.size(); ++i) threads.emplace_back(work, std::ref(parts[i]));
  work(parts[0]);
}

template <bool Escapes>
inline uint32_t pack_chunk(const uint8_t* in, const HostCodes& codes, uint8_t* bf_low, unsigned valid) {
#if defined(__aarch64__) && !defined(__CUDACC__)
  if (valid == kChunk && codes.use_band) {
    // An exponent outside the lookup band maps to zero, the sparse escape.
    const uint16x8_t bits = vreinterpretq_u16_u8(vld1q_u8(in));
    const uint8x8_t high = vshrn_n_u16(bits, 8);
    const uint8x8_t index = vsub_u8(vand_u8(high, vdup_n_u8(0x7f)), vdup_n_u8(codes.base));
    const uint8x8_t code = vqtbl4_u8(codes.band, index);
    const uint8x8_t sign = vand_u8(vshr_n_u8(high, 4), vdup_n_u8(8));
    uint8x8_t nibbles = vorr_u8(code, sign);
    uint8x8_t low = vmovn_u16(bits);
    if constexpr (Escapes) {
      const uint8x8_t escaped = vceq_u8(code, vdup_n_u8(kEscape));
      nibbles = vbic_u8(nibbles, escaped);
      low = vbic_u8(low, escaped);
    }
    vst1_u8(bf_low, low);
    const uint8x8_t even = vuzp1_u8(nibbles, nibbles), odd = vuzp2_u8(nibbles, nibbles);
    return vget_lane_u32(vreinterpret_u32_u8(vsli_n_u8(even, odd, 4)), 0);
  }
#endif
  uint64_t low = 0;
  uint32_t nibbles = 0;
  for (unsigned i = 0; i < valid; ++i) {
    const uint16_t bits = read<uint16_t>(in + i * sizeof(uint16_t));
    const uint8_t nibble = codes.nibbles[bits >> 8];
    if (!Escapes || nibble != kEscape) low |= uint64_t(bits & 0xff) << (8 * i);
    nibbles |= uint32_t(nibble) << (4 * i);
  }
  write(bf_low, low);
  return nibbles;
}

template <bool Escapes, bool Wide = false>
void pack_rows(Partition& part, const uint8_t* source, int cols, size_t padded,
               const HostCodes& codes, uint8_t* dense, uint8_t* row_ptr, uint8_t* entries) {
  constexpr size_t entry_bytes = sizeof(uint32_t) * (Wide ? 2 : 1);
  size_t next = part.entry_start;
  for (int row = part.begin; row < part.end; ++row) {
    if constexpr (Escapes) write(row_ptr + size_t(row) * sizeof(uint32_t), uint32_t(next));
    const uint8_t* in = source + size_t(row) * cols * sizeof(uint16_t);
    uint8_t* bf_low = dense + size_t(row) * row_bytes(padded);
    uint8_t* hi4 = bf_low + padded;
    for (size_t col = 0; col < padded; col += kChunk) {
      const size_t valid_col = std::min(col, size_t(cols));
      const uint8_t* chunk = in + valid_col * sizeof(uint16_t);
      const unsigned valid = std::min(size_t(cols) - valid_col, size_t(kChunk));
      const uint32_t nibbles = pack_chunk<Escapes>(chunk, codes, bf_low + col, valid);
      write(hi4 + col / 2, nibbles);
      if constexpr (Escapes) {
        uint32_t escapes = escape_markers(nibbles);
        if (valid < kChunk) escapes &= (1u << (4 * valid)) - 1;  // Padding has code zero too.
        for (; escapes; escapes &= escapes - 1) {
          const unsigned i = std::countr_zero(escapes) / 4;
          const uint32_t bits = read<uint16_t>(chunk + i * sizeof(uint16_t));
          uint8_t* entry = entries + next++ * entry_bytes;
          if constexpr (Wide) {
            write(entry, uint32_t(col + i));
            write(entry + sizeof(uint32_t), bits);
          } else {
            write(entry, (bits << 16) | uint32_t(col + i));
          }
        }
      }
    }
  }
}

}  // namespace detail

// `bf16` points to contiguous little-endian BF16 bit patterns in row-major order. The blob can be
// uploaded with one copy. Each 32-column-padded row is [BF16 low bytes, sign and exponent codes
// (low nibble first)]. Sparse blobs append rows + 1 uint32_t CSR offsets followed by the outliers, in
// column order within each row. Each entry is a uint32_t with the BF16 bits in the high half and
// column in the low half. Matrices wider than 65536 columns use two words per entry (column, bits).
// After the CSR entries, sparse blobs append a bit mask and a sorted list of nonempty rows.
inline Packed pack_pico12_host(const void* bf16, int rows, int cols) {
  static_assert(std::endian::native == std::endian::little);
  if (!bf16 || rows <= 0 || cols <= 0) throw std::invalid_argument("PICO12: invalid BF16 matrix");
  const size_t padded = padded_cols(cols);
  if (size_t(rows) > std::numeric_limits<size_t>::max() / 2 / padded)
    throw std::length_error("PICO12: matrix is too large");
  const size_t elements = size_t(rows) * padded;
  const size_t dense_bytes = size_t(rows) * row_bytes(padded);
  const auto* source = static_cast<const uint8_t*>(bf16);

  const unsigned available = std::clamp(std::thread::hardware_concurrency(), 1u, 32u);
  const size_t workers = std::clamp(elements / (1u << 20), size_t(1), std::min<size_t>(rows, available));
  std::vector<detail::Partition> parts(workers);
  for (size_t i = 0; i < workers; ++i) {
    parts[i].begin = int(size_t(rows) * i / workers);
    parts[i].end = int(size_t(rows) * (i + 1) / workers);
  }
  detail::for_partitions(parts, [&](detail::Partition& part) {
    // Independent bins break the update dependency when consecutive weights share an exponent field.
    detail::Histogram bins[4]{};
    const size_t begin = size_t(part.begin) * cols, end = size_t(part.end) * cols;
    size_t i = begin;
    for (; end - i >= 4; i += 4)
      for (unsigned j = 0; j < 4; ++j)
        ++bins[j][detail::exponent_hi(detail::read<uint16_t>(source + (i + j) * sizeof(uint16_t)))];
    for (; i < end; ++i)
      ++bins[0][detail::exponent_hi(detail::read<uint16_t>(source + i * sizeof(uint16_t)))];
    for (const auto& bin : bins)
      for (size_t e = 0; e < bin.size(); ++e) part.histogram[e] += bin[e];
  });

  detail::Histogram counts{};
  for (const auto& part : parts)
    for (size_t e = 0; e < counts.size(); ++e) counts[e] += part.histogram[e];
  const size_t unique = std::count_if(counts.begin(), counts.end(), [](size_t n) { return n != 0; });
  Packed packed;
  packed.layout.rows = rows;
  packed.layout.cols = cols;
  packed.layout.escapes = unique > 8;
  std::array<unsigned, 128> order;
  std::iota(order.begin(), order.end(), 0u);
  // Code zero maps to the zero field unless a direct matrix needs all eight codes for
  // nonzero fields. Sparse matrices use the remaining codes for the most frequent nonzero fields.
  std::sort(order.begin() + 1, order.end(), [&](unsigned a, unsigned b) {
    return counts[a] == counts[b] ? a < b : counts[a] > counts[b];
  });
  const unsigned start = unique == 8 && counts[0] == 0 ? 1 : 0;
  std::array<uint8_t, 128> codes{};
  for (unsigned code = packed.layout.escapes ? 1 : 0; code < 8; ++code) {
    const unsigned e = order[start + code];
    codes[e] = uint8_t(code);
    packed.layout.lut[code / 4] |= uint32_t(e) << (8 * (code % 4));
  }
  detail::HostCodes host_codes;
  for (unsigned e = 0; e < codes.size(); ++e) {
    host_codes.nibbles[e] = codes[e];
    host_codes.nibbles[e + 128] = packed.layout.escapes && codes[e] == kEscape ? kEscape : codes[e] | 8;
  }
#if defined(__aarch64__) && !defined(__CUDACC__)
  unsigned low = 127, high = 0;
  for (unsigned e = 0; e < codes.size(); ++e) {
    if (counts[e] && (!packed.layout.escapes || codes[e] != kEscape)) {
      low = std::min(low, e);
      high = std::max(high, e);
    }
  }
  if (high == 0) low = 0;
  if (high - low < 64) {
    std::array<uint8_t, 64> band{};
    for (unsigned e = low; e <= high; ++e) band[e - low] = codes[e];
    for (unsigned i = 0; i < 4; ++i) host_codes.band.val[i] = vld1q_u8(band.data() + 16 * i);
    host_codes.base = uint8_t(low);
    host_codes.use_band = true;
  }
#endif

  const size_t offsets_bytes = packed.layout.escapes ? row_ptr_bytes(rows) : 0;
  const bool wide = cols > kCompactCols;
  const size_t entry_bytes = sizeof(uint32_t) * (wide ? 2 : 1);
  size_t nnz = 0;
  if (packed.layout.escapes) {
    for (auto& part : parts) {
      part.entry_start = nnz;
      for (size_t e = 0; e < codes.size(); ++e)
        if (codes[e] == kEscape) nnz += part.histogram[e];
    }
    if (nnz > std::numeric_limits<uint32_t>::max())
      throw std::length_error("PICO12: outlier count exceeds the format's index range");
  }
  const size_t remaining = std::numeric_limits<size_t>::max() - dense_bytes;
  if (offsets_bytes > remaining || nnz > (remaining - offsets_bytes) / entry_bytes)
    throw std::length_error("PICO12: packed matrix is too large");
  packed.bytes = dense_bytes + offsets_bytes + nnz * entry_bytes;
  size_t capacity = packed.bytes;
  if (nnz) {
    // Reserve the maximum row list size until the CSR offsets give the exact count.
    const size_t mask_bytes = row_mask_bytes(rows);
    const size_t max_outlier_rows = std::min(size_t(rows), nnz);
    const size_t available_bytes = std::numeric_limits<size_t>::max() - capacity;
    if (mask_bytes > available_bytes || max_outlier_rows > (available_bytes - mask_bytes) / sizeof(int32_t))
      throw std::length_error("PICO12: packed matrix is too large");
    packed.layout.row_mask_offset = packed.bytes;
    packed.layout.outlier_rows_offset = packed.bytes + mask_bytes;
    capacity = packed.layout.outlier_rows_offset + max_outlier_rows * sizeof(int32_t);
  }
  packed.data = std::make_unique_for_overwrite<uint8_t[]>(capacity);
  uint8_t* row_ptr = packed.data.get() + dense_bytes;
  uint8_t* entries = row_ptr + offsets_bytes;
  if (packed.layout.escapes) detail::write(entries - sizeof(uint32_t), uint32_t(nnz));
  detail::for_partitions(parts, [&](detail::Partition& part) {
    auto pack = [&]<bool Escapes, bool Wide = false>() {
      detail::pack_rows<Escapes, Wide>(part, source, cols, padded, host_codes, packed.data.get(), row_ptr, entries);
    };
    if (!packed.layout.escapes) pack.template operator()<false>();
    else if (wide) pack.template operator()<true, true>();
    else pack.template operator()<true>();
  });
  if (nnz) {
    uint8_t* mask = packed.data.get() + packed.layout.row_mask_offset;
    uint8_t* outlier_rows = packed.data.get() + packed.layout.outlier_rows_offset;
    std::memset(mask, 0, row_mask_bytes(rows));
    uint32_t begin = 0;
    for (int row = 0; row < rows; ++row) {
      const uint32_t end = detail::read<uint32_t>(row_ptr + (size_t(row) + 1) * sizeof(uint32_t));
      if (begin < end) {
        uint8_t* word = mask + size_t(row >> 5) * sizeof(uint32_t);
        detail::write(word, detail::read<uint32_t>(word) | (1u << (row & 31)));
        detail::write(outlier_rows + size_t(packed.layout.num_outlier_rows++) * sizeof(int32_t), int32_t(row));
      }
      begin = end;
    }
    packed.bytes = packed.layout.outlier_rows_offset + size_t(packed.layout.num_outlier_rows) * sizeof(int32_t);
  }
  return packed;
}

inline std::vector<uint16_t> unpack_pico12_host(const Packed& packed) {
  const auto& layout = packed.layout;
  const size_t padded = padded_cols(layout.cols);
  const auto view = device_view(layout, packed.data.get());
  const bool wide = layout.cols > kCompactCols;
  std::vector<uint16_t> result(size_t(layout.rows) * layout.cols);
  for (int row = 0; row < layout.rows; ++row) {
    const uint8_t* bf_low = packed.data.get() + size_t(row) * row_bytes(padded);
    const uint8_t* hi4 = bf_low + padded;
    uint16_t* dst = result.data() + size_t(row) * layout.cols;
    for (int col = 0; col < layout.cols; ++col) {
      const unsigned nibble = (hi4[col / 2] >> (4 * (col % 2))) & 15;
      const unsigned code = nibble & 7;
      const unsigned e = (layout.lut[code / 4] >> (8 * (code % 4))) & 255;
      dst[col] = uint16_t(((nibble & 8) << 12) | (e << 8) | bf_low[col]);
    }
    if (layout.escapes) {
      const uint32_t end = view.row_ptr[row + 1];
      for (uint32_t i = view.row_ptr[row]; i < end; ++i) {
        if (wide) {
          const size_t word = size_t(i) * 2;
          dst[view.entries[word]] = uint16_t(view.entries[word + 1]);
        } else {
          const uint32_t entry = view.entries[i];
          dst[entry & 0xffffu] = uint16_t(entry >> 16);
        }
      }
    }
  }
  return result;
}

#ifdef __CUDACC__

struct Chunk {
  uint2 bf_low;
  uint32_t codes;
};

struct Chunk16 {
  uint4 bf_low;
  uint2 codes;
};

template <l2::Evict E>
__device__ __forceinline__ Chunk load8(const uint8_t* row, int cols, int col, uint64_t policy) {
  Chunk chunk;
  const uint8_t* bf_low = row + col;
  const uint8_t* hi4 = row + cols + col / 2;
  if constexpr (l2::hinted<E>) {
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v2.u32 {%0,%1}, [%2], %3;"
                 : "=r"(chunk.bf_low.x), "=r"(chunk.bf_low.y) : "l"(bf_low), "l"(policy));
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.u32 %0, [%1], %2;"
                 : "=r"(chunk.codes) : "l"(hi4), "l"(policy));
  } else {
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];"
                 : "=r"(chunk.bf_low.x), "=r"(chunk.bf_low.y) : "l"(bf_low));
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(chunk.codes) : "l"(hi4));
  }
  return chunk;
}

template <l2::Evict E>
__device__ __forceinline__ Chunk16 load16(const uint8_t* row, int cols, int col, uint64_t policy) {
  const uint4 bf_low = l2::ld_stream<E>(row + col, policy);
  const uint8_t* hi4 = row + cols + col / 2;
  uint2 codes;
  if constexpr (l2::hinted<E>) {
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v2.u32 {%0,%1}, [%2], %3;"
                 : "=r"(codes.x), "=r"(codes.y) : "l"(hi4), "l"(policy));
  } else {
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];"
                 : "=r"(codes.x), "=r"(codes.y) : "l"(hi4));
  }
  return {bf_low, codes};
}

__device__ __forceinline__ uint2 decode4(const DeviceView& view, uint32_t bf_low, uint32_t hi4,
                                        uint32_t high_mask = 0xffffffffu) {
  const uint32_t exponent = __byte_perm(view.lut[0], view.lut[1], hi4 & 0x7777u);
  // Shift each nibble's sign into selector bit 2. The selected bytes are all 0x00 or all
  // 0xff, so generic PRMT's sign replication leaves the result unchanged for any bit 3.
  uint32_t sign;
  asm("prmt.b32 %0, 0, -1, %1;" : "=r"(sign) : "r"(hi4 >> 1));
  const uint32_t bf_high = (exponent | (sign & 0x80808080u)) & high_mask;
  return make_uint2(__byte_perm(bf_low, bf_high, 0x5140), __byte_perm(bf_low, bf_high, 0x7362));
}

__device__ __forceinline__ uint4 decode8(Chunk chunk, const DeviceView& view) {
  const uint2 low = decode4(view, chunk.bf_low.x, chunk.codes);
  const uint2 high = decode4(view, chunk.bf_low.y, chunk.codes >> 16);
  return make_uint4(low.x, low.y, high.x, high.y);
}

// Materializing BF16 rows needs the original bits in their original positions. Only chunks
// containing an escape look up their first outlier, then consume entries in column order.
template <bool Wide>
__device__ __forceinline__ uint4 restore8(Chunk chunk, const DeviceView& view, int row, int col) {
  const uint4 dense = decode8(chunk, view);
  const uint32_t markers = detail::escape_markers(chunk.codes);
  if (!markers) return dense;

  uint32_t first = __ldg(view.row_ptr + row), last = __ldg(view.row_ptr + size_t(row) + 1);
  while (first < last) {
    const uint32_t mid = first + (last - first) / 2;
    uint32_t entry_col;
    if constexpr (Wide) entry_col = __ldg(view.entries + size_t(mid) * 2);
    else entry_col = __ldg(view.entries + mid) & 0xffffu;
    if (entry_col < unsigned(col)) first = mid + 1;
    else last = mid;
  }
  auto next_bits = [&]() {
    if constexpr (Wide) return __ldg(view.entries + size_t(first++) * 2 + 1);
    else return __ldg(view.entries + first++) >> 16;
  };
  auto restore_pair = [&](uint32_t bits, uint32_t pair_markers) {
    if (pair_markers & 1u) bits |= next_bits();
    if (pair_markers & 16u) bits |= next_bits() << 16;
    return bits;
  };
  // Separate statements preserve column order when both pairs have outliers.
  const uint32_t a = restore_pair(dense.x, markers);
  const uint32_t b = restore_pair(dense.y, markers >> 8);
  const uint32_t c = restore_pair(dense.z, markers >> 16);
  const uint32_t d = restore_pair(dense.w, markers >> 24);
  return make_uint4(a, b, c, d);
}

// The batched head materializes individual BF16 chunks using a linear element index.
template <bool Escapes>
__device__ __forceinline__ uint4 decode8(Chunk chunk, const DeviceView& view, size_t element) {
  if constexpr (Escapes) {
    if (detail::escape_markers(chunk.codes)) {
      const int row = int(element / view.cols), col = int(element % view.cols);
      if (view.cols > kCompactCols) return restore8<true>(chunk, view, row, col);
      return restore8<false>(chunk, view, row, col);
    }
  }
  return decode8(chunk, view);
}

#endif  // __CUDACC__

}  // namespace ingine::kernels::pico12
