// Batched decoding (batch.h): request slots and one decode step over many sequences, wired as Prefiller::run with
// one row per sequence. Differences from the prefill: the projections are widened to the FP32 rows of the decode
// attention kernel, which attends over each sequence's own slot (positions, caches, conv states), the short
// convolutions continue each slot's state, and the head runs on every row.
#include "model/batch.h"

#include "core/device.h"
#include "ingine/generator.h"
#include "kernels/attn/attn.h"
#include "kernels/batch/batch.h"
#include "kernels/batch/dense.h"
#include "kernels/batch/moe.h"
#include "kernels/common/kv12.cuh"
#include "kernels/dense/dense.h"
#include "kernels/prefill/prefill.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ingine::model {
namespace pk = kernels::prefill;
namespace dense = kernels::dense;
namespace attn = kernels::attn;
namespace kb = kernels::batch;

namespace {

constexpr int H = HIDDEN;
constexpr int LD_V = BatchDecoder::kLogitsLd;  // row stride of the logits (aligned rows for cuBLASLt)

const float* f32p(const Tensor& t) { return static_cast<const float*>(t.data); }
const __nv_bfloat16* bf16p(const Tensor& t) { return static_cast<const __nv_bfloat16*>(t.data); }

Tensor zeros(std::vector<DeviceBuffer>& storage, DType dt, std::initializer_list<std::int32_t> shape) {
  std::size_t n = 1;
  for (std::int32_t d : shape) n *= std::size_t(d);
  storage.emplace_back(n * dtype_size(dt));
  CUDA_CHECK(cudaMemset(storage.back().p, 0, storage.back().bytes));
  return rm(storage.back().p, dt, shape);
}

// fast.Buffers.blocks (as prefill.cpp)
std::vector<const float*> blocks_of(int l, int p, const float* B0, const float* B1, float* const* B2, float* const* E, int coda_pass) {
  if (l < 1) return {};
  if (l < 7) return {B0};
  if (l < 13) return {B0, B1};
  if (l < 19) return {B0, B1, B2[l < LOOP_END ? p : coda_pass]};
  return {B0, B1, B2[coda_pass], E[coda_pass]};
}

}  // namespace

// ------------------------------------------------------------------------------------------------- state

std::size_t BatchState::device_bytes() const {
  std::size_t n = 0;
  for (const auto& b : storage) n += b.bytes;
  return n;
}

State BatchState::view(int s) const {
  if (s < 0 || s >= slots) throw std::out_of_range("BatchState::view: slot " + std::to_string(s));
  State v;
  v.max_ctx = ctx;
  v.loopcd = loopcd;
  v.kv_fp8 = kv_fp8;
  v.ring_rows = ring_rows;
  v.kv_q4 = kv_q4;
  v.kv_12 = kv_12;
  const std::size_t es = dtype_size(k_full.dtype);
  auto at = [](const Tensor& t, std::size_t bytes) { return static_cast<char*>(t.data) + bytes; };
  v.k_ring = rm(at(k_ring, std::size_t(s) * ring_stride() * 2), DType::BF16, {loopcd.ring_visits(), ring_rows, hkv_local, HEAD_DIM});
  v.v_ring = rm(at(v_ring, std::size_t(s) * ring_stride() * 2), DType::BF16, {loopcd.ring_visits(), ring_rows, hkv_local, HEAD_DIM});
  v.ring_pos = rm(at(ring_pos, std::size_t(s) * kpos_stride() * 8), DType::I64, {loopcd.ring_visits(), ring_rows});
  v.k_full = rm(at(k_full, std::size_t(s) * full_stride() * es), k_full.dtype, {loopcd.full_visits(), ctx, hkv_local, k_full.ne[0]});
  v.v_full = rm(at(v_full, std::size_t(s) * full_stride() * es), v_full.dtype, {loopcd.full_visits(), ctx, hkv_local, k_full.ne[0]});
  if (kv_q4 || kv_12) {
    const std::size_t bytes = std::size_t(s) * full_scale_stride() * dtype_size(k_full_scale.dtype);
    v.k_full_scale = rm(at(k_full_scale, bytes), k_full_scale.dtype, {loopcd.full_visits(), ctx, hkv_local, k_full_scale.ne[0]});
    v.v_full_scale = rm(at(v_full_scale, bytes), v_full_scale.dtype, {loopcd.full_visits(), ctx, hkv_local, v_full_scale.ne[0]});
  }
  v.conv_k = rm(at(conv_k, std::size_t(s) * conv_kv_stride() * 4), DType::FP32, {loopcd.visits(), CONV_K - 1, kv_dim_local});
  v.conv_v = rm(at(conv_v, std::size_t(s) * conv_kv_stride() * 4), DType::FP32, {loopcd.visits(), CONV_K - 1, kv_dim_local});
  v.conv_attn = rm(at(conv_attn, std::size_t(s) * conv_h_stride() * 4), DType::FP32, {loopcd.visits(), CONV_K - 1, H});
  v.conv_mlp = rm(at(conv_mlp, std::size_t(s) * conv_h_stride() * 4), DType::FP32, {loopcd.visits(), CONV_K - 1, H});
  v.length = 0;
  return v;
}

BatchState make_batch_state(int slots, int ctx, bool kv_fp8, const TpLayout& layout, bool kv_q4, bool kv_12, LoopCdConfig loopcd) {
  validate_kv_formats(kv_fp8, kv_q4, kv_12);
  if (slots < 1 || slots > 4096) throw std::invalid_argument("slots must be in 1..4096");
  if (ctx < 2 || ctx > (1 << 20)) throw std::invalid_argument("slot context must be in 2..2^20");
  validate_loopcd_tp(loopcd, layout.tp_size);
  BatchState b;
  b.loopcd = loopcd;
  b.slots = slots;
  b.ctx = ctx;
  b.kv_fp8 = kv_fp8;
  b.kv_q4 = kv_q4;
  b.kv_12 = kv_12;
  b.ring_rows = std::min(ctx, WINDOW);
  b.hkv_local = layout.hkv_local;
  b.kv_dim_local = layout.kv_dim_local;
  const DType full = (kv_q4 || kv_12) ? DType::U8 : kv_fp8 ? DType::FP8_E4M3FN : DType::BF16;
  const int full_dim = kv_12 ? int(sizeof(kernels::kv12::Head)) : kv_q4 ? HEAD_DIM / 2 : HEAD_DIM;
  b.k_ring = zeros(b.storage, DType::BF16, {slots * loopcd.ring_visits(), b.ring_rows, b.hkv_local, HEAD_DIM});
  b.v_ring = zeros(b.storage, DType::BF16, {slots * loopcd.ring_visits(), b.ring_rows, b.hkv_local, HEAD_DIM});
  b.ring_pos = zeros(b.storage, DType::I64, {slots * loopcd.ring_visits(), b.ring_rows});
  CUDA_CHECK(cudaMemset(b.ring_pos.data, 0xff, b.storage.back().bytes));  // -1
  b.k_full = zeros(b.storage, full, {slots * loopcd.full_visits(), ctx, b.hkv_local, full_dim});
  b.v_full = zeros(b.storage, full, {slots * loopcd.full_visits(), ctx, b.hkv_local, full_dim});
  if (kv_q4 || kv_12) {
    const DType side = kv_12 ? DType::U8 : DType::FP16;
    const int side_dim = kv_12 ? int(sizeof(kernels::kv12::Spill)) : 2;
    b.k_full_scale = zeros(b.storage, side, {slots * loopcd.full_visits(), ctx, b.hkv_local, side_dim});
    b.v_full_scale = zeros(b.storage, side, {slots * loopcd.full_visits(), ctx, b.hkv_local, side_dim});
  }
  b.conv_k = zeros(b.storage, DType::FP32, {slots * loopcd.visits(), CONV_K - 1, b.kv_dim_local});
  b.conv_v = zeros(b.storage, DType::FP32, {slots * loopcd.visits(), CONV_K - 1, b.kv_dim_local});
  b.conv_attn = zeros(b.storage, DType::FP32, {slots * loopcd.visits(), CONV_K - 1, H});
  b.conv_mlp = zeros(b.storage, DType::FP32, {slots * loopcd.visits(), CONV_K - 1, H});
  return b;
}

void reset_slot(BatchState& st, int s, cudaStream_t stream) {
  const State v = st.view(s);
  CUDA_CHECK(cudaMemsetAsync(v.ring_pos.data, 0xff, std::size_t(st.kpos_stride()) * 8, stream));
  CUDA_CHECK(cudaMemsetAsync(v.conv_k.data, 0, std::size_t(st.conv_kv_stride()) * 4, stream));
  CUDA_CHECK(cudaMemsetAsync(v.conv_v.data, 0, std::size_t(st.conv_kv_stride()) * 4, stream));
  CUDA_CHECK(cudaMemsetAsync(v.conv_attn.data, 0, std::size_t(st.conv_h_stride()) * 4, stream));
  CUDA_CHECK(cudaMemsetAsync(v.conv_mlp.data, 0, std::size_t(st.conv_h_stride()) * 4, stream));
}

// ------------------------------------------------------------------------------------------------ decoder

BatchDecoder::BatchDecoder(const Weights& w, const Ablation& abl, BatchState& st, int max_seqs, cudaStream_t stream,
                           int max_m, TpRuntime* tp, const Dense* unembed_override)
    : w_(w), unembed_override_(unembed_override), tp_(tp), layout_(tp ? tp->layout : TpLayout{}), abl_(abl), st_(st), max_seqs_(max_seqs), max_m_(max_m), stream_(stream),
      gemm_(std::make_unique<kb::TunedGemm>()) {
  validate_loopcd_tp(st.loopcd, layout_.tp_size);
  const bool uses_ref_coda = st.loopcd.uses_ref_coda();
  if (max_seqs < 1 || max_seqs > st.slots) throw std::invalid_argument("BatchDecoder: max_seqs must be in 1..slots");
  if (max_m < 1 || max_m > 8) throw std::invalid_argument("BatchDecoder: max_m must be in 1..8");
  const char* fp8_env = std::getenv("INGINE_BATCH_FP8_DENSE");
  const char* pico12_env = std::getenv("INGINE_BATCH_PICO12_GEMM");
  const char* gate_env = std::getenv("INGINE_BATCH_GATE_FUSE");
  // The wide FP8 kernel vector-loads adjacent weight scales. Supported TP=1 and TP=2 segment offsets are even.
  batch_fp8_dense_ = w.dense_fp8 && layout_.weight_offset(SEG_K) % 2 == 0 &&
                     (!fp8_env || !*fp8_env || std::strtol(fp8_env, nullptr, 10) != 0);
  batch_pico12_gemm_ = w.dense_pico12 && (!pico12_env || !*pico12_env || std::strtol(pico12_env, nullptr, 10) != 0);
  batch_gate_fuse_ = (w.dense_fp8 || w.dense_pico12) && (!gate_env || !*gate_env || std::strtol(gate_env, nullptr, 10) != 0);
  const std::size_t S = std::size_t(max_seqs), N = S * std::size_t(max_m);  // sequences, rows
  auto alloc = [&](std::size_t bytes) {
    storage_.emplace_back(bytes);
    return storage_.back().p;
  };
  auto f32 = [&](std::size_t n) { return static_cast<float*>(alloc(n * 4)); };
  auto bf16 = [&](std::size_t n) { return static_cast<__nv_bfloat16*>(alloc(n * 2)); };
  auto i32 = [&](std::size_t n) { return static_cast<int*>(alloc(n * 4)); };
  auto i64 = [&](std::size_t n) { return static_cast<std::int64_t*>(alloc(n * 8)); };
  tokens_ = i32(N), slots_ = i32(S), sampled_ = i32(N), rows_ = i32(S + 1);
  temps_ = f32(N);
  if (st.loopcd.auto_mode()) {
    pin_omegas_ = MappedHostBuffer(N * sizeof(float));
    omegas_dev_ = f32(N);
  }
  pos_ = i64(S), ids_ = i64(N);
  d_seed_ = static_cast<std::uint64_t*>(alloc(sizeof(std::uint64_t)));
  B0_ = f32(N * H), B1_ = f32(N * H);
  for (int p = 0; p < PASSES; ++p) B2_[p] = f32(N * H), E_[p] = f32(N * H);
  hid_ = f32(N * H), part_ = f32(N * H), y_ = f32(N * H), y2_ = f32(N * H);
  xin_ = bf16(3 * N * H), xm_ = bf16(N * H), xf_ = bf16(N * H);
  if (batch_fp8_dense_) {
    xin_fp8_q_ = static_cast<std::uint8_t*>(alloc(3 * N * H));
    xin_fp8_s_ = f32(3 * N);
    attn_fp8_q_ = static_cast<std::uint8_t*>(alloc(N * layout_.q_dim_local));
    attn_fp8_s_ = f32(N);
  }
  if (batch_gate_fuse_) {
    mixf_bf16_ = bf16(3 * N * H), msgs_bf16_ = bf16(4 * N * H), mixm_bf16_ = bf16(N * H);
  } else {
    mixf_ = f32(3 * N * H), msgs_ = f32(4 * N * H), mixm_ = f32(N * H);
  }
  proj16_ = bf16(N * layout_.projection_rows(true));
  attn_out_ = bf16(N * layout_.q_dim_local), wo_out_ = bf16(N * H);
  gin_ = bf16(3 * N * 2 * H), g16_ = bf16(3 * N * GATE_RANK);
  if (!batch_gate_fuse_) fg_ = f32(3 * N * 2 * H);
  router_logits_ = f32(N * EXPERTS), wts_ = f32(N * TOP_K);
  idx_ = i32(N * TOP_K);
  counts_ = i32(EXPERTS), tiles_n_ = i32(1), pair_token_ = i32(N * TOP_K), pair_dst_ = i32(N * TOP_K);
  tiles_ = static_cast<int4*>(alloc(std::size_t(kb::moe_tiles_capacity(int(N))) * sizeof(int4)));
  act_ = bf16(N * TOP_K * layout_.I_local), down_ = bf16(N * TOP_K * H), moe_sum_ = bf16(N * H);
  if (tp_) {
    partial_ = f32(N * H);
    down_partial_ = f32(N * TOP_K * H);
    local_cand_ = static_cast<kernels::head::Candidate*>(alloc(N * sizeof(kernels::head::Candidate)));
    gathered_cand_ = static_cast<kernels::head::Candidate*>(alloc(2 * N * sizeof(kernels::head::Candidate)));
  }
  // BF16, PICO12 and NVFP4 heads fuse sampling. FP8 rows materialize logits first.
  use_nvfp4_head_ = !unembed_override_ && w.unembed_nvfp4.codes != nullptr;
  const Dense& head_w = unembed_override_ ? *unembed_override_ : w.unembed;
  std::size_t partials_count = 0;
  if (use_nvfp4_head_) {
    partials_count = kb::head_partials_nvfp4(int(N), w.unembed_nvfp4.rows);
  } else if (head_w.fp8()) {
    logits_ = f32(N * LD_V);
  } else {
    partials_count = kb::head_partials(int(N), layout_.v_end - layout_.v_begin);
  }
  if (uses_ref_coda) {
    if (!logits_) logits_ = f32(N * LD_V);
    ref_logits_ = f32(N * LD_V);
  }
  if (partials_count) head_partials_ = static_cast<int2*>(alloc(partials_count * sizeof(int2)));
  const char* reject_env = std::getenv("INGINE_BATCH_REJECTION_SAMPLE");
  reject_enabled_ = !uses_ref_coda && max_m > 1 && partials_count > 0 &&
                    (!reject_env || !*reject_env || std::strtol(reject_env, nullptr, 10) != 0);
  if (tp_ && !tp_->agree(reject_enabled_, stream_))
    throw std::invalid_argument("BatchDecoder: TP ranks disagree on rejection sampling");
  if (reject_enabled_) {
    const std::size_t support_count = N * kb::RejectSpec::kSupport;
    pin_reject_active_ = MappedHostBuffer(N * sizeof(std::uint8_t));
    pin_reject_ids_ = MappedHostBuffer(support_count * sizeof(std::int32_t));
    pin_reject_probs_ = MappedHostBuffer(support_count * sizeof(float));
    pin_reject_picked_ = MappedHostBuffer(N * sizeof(std::int32_t));
    reject_active_ = static_cast<std::uint8_t*>(alloc(N * sizeof(std::uint8_t)));
    reject_ids_ = i32(support_count);
    reject_probs_ = f32(support_count);
    reject_picked_ = i32(N);
    reject_spec_ = {reject_active_, reject_ids_, reject_probs_, reject_picked_, f32(support_count),
                    f32(partials_count * 2)};
    if (tp_) {
      local_reject_ = static_cast<kb::RejectShardPartial*>(alloc(N * sizeof(kb::RejectShardPartial)));
      gathered_reject_ = static_cast<kb::RejectShardPartial*>(alloc(2 * N * sizeof(kb::RejectShardPartial)));
    }
  }
  if (const char* ge = std::getenv("INGINE_BATCH_GRAPH"); ge && *ge && std::atoi(ge) == 0) graph_enabled_ = false;
  // Keep the existing warmup for the path that still uses tuned dense GEMMs.
  if (!batch_fp8_dense_ && !batch_pico12_gemm_) graph_warmup_ = 2;
  if (const char* gw = std::getenv("INGINE_BATCH_GRAPH_WARMUP"); gw && *gw) {
    const int parsed = std::atoi(gw);
    if (parsed >= 1) graph_warmup_ = parsed;
  }
  if (const char* gms = std::getenv("INGINE_BATCH_GRAPH_MAX_SEQS"); gms && *gms) {
    const int parsed = std::atoi(gms);
    if (parsed >= 0) graph_max_seqs_ = parsed;
  }
  graph_max_tokens_ = graph_max_seqs_ * max_m_;
  pin_tokens_ = MappedHostBuffer(N * sizeof(std::int32_t));
  pin_slots_ = MappedHostBuffer(S * sizeof(std::int32_t));
  pin_pos_ = MappedHostBuffer(S * sizeof(std::int64_t));
  pin_rows_ = MappedHostBuffer((S + 1) * sizeof(std::int32_t));
  pin_temps_ = MappedHostBuffer(N * sizeof(float));
  pin_seed_ = MappedHostBuffer(sizeof(std::uint64_t));
  pin_sampled_ = MappedHostBuffer(N * sizeof(std::int32_t));
  // Recover BF16 weights per visit when using the unpacking/dequantization path.
  wscratch_ = ((w.dense_pico12 && !batch_pico12_gemm_) || (w.dense_fp8 && !batch_fp8_dense_))
                  ? bf16(std::size_t(layout_.projection_rows(true)) * H)
                  : nullptr;
  if (max_m > 1) {
    taps_ = bf16(N * NUM_TAPS * H);
    const int visits = st.loopcd.visits();
    hist_k_ = bf16(visits * N * layout_.kv_dim_local), hist_v_ = bf16(visits * N * layout_.kv_dim_local);
    hist_attn_ = bf16(visits * N * H), hist_mlp_ = bf16(visits * N * H);
    commit_ = i32(S);
  }
  for (auto& b : storage_) b.fill(0);
  CUDA_CHECK(cudaDeviceSynchronize());
  if (!batch_gate_fuse_) {
    // Pack forget16 | gain16 after the weights' uploads for the unfused N = 2H GEMM.
    const std::size_t gate_elems = std::size_t(H) * GATE_RANK;
    auto* fg_rows = bf16(std::size_t(LOOP_END - LOOP_START) * 4 * 2 * gate_elems);
    for (int l = LOOP_START; l < LOOP_END; ++l) {
      const Layer& layer = w.layers[l];
      __nv_bfloat16* dst = fg_rows + std::size_t(l - LOOP_START) * 4 * 2 * gate_elems;
      fg_qkv_[l] = dst, fg_mlp_[l] = dst + 3 * 2 * gate_elems;
      const std::pair<const Tensor*, const Tensor*> stacks[] = {{&layer.qkv_gate_forget16, &layer.qkv_gate_gain16},
                                                               {&layer.mlp1_gate_forget16, &layer.mlp1_gate_gain16}};
      for (int s = 0; s < 2; ++s) {
        const int G = s == 0 ? 3 : 1;
        __nv_bfloat16* d = dst + (s == 0 ? 0 : 3 * 2 * gate_elems);
        for (int half = 0; half < 2; ++half)
          CUDA_CHECK(cudaMemcpy2D(d + half * gate_elems, 2 * gate_elems * 2,
                                  half == 0 ? stacks[s].first->data : stacks[s].second->data, gate_elems * 2,
                                  gate_elems * 2, G, cudaMemcpyDeviceToDevice));
      }
    }
  }
  attn_ws_ = DeviceBuffer(attn::workspace_bytes_batched(max_m, max_seqs, layout_.hkv_local));
  attn_ws_.fill(0);  // the kernel resets its counters itself
  sampled_host_.resize(N);
  temps_host_.resize(N);
}

BatchDecoder::~BatchDecoder() = default;

std::size_t BatchDecoder::device_bytes() const {
  std::size_t n = attn_ws_.bytes + logits_buf_.bytes;
  for (const auto& b : storage_) n += b.bytes;
  return n;
}

void BatchDecoder::keep_logits(bool keep) {
  keep_logits_ = keep;
  if (keep && !logits_) {
    logits_buf_ = DeviceBuffer(std::size_t(max_seqs_) * max_m_ * LD_V * sizeof(float));
    logits_ = static_cast<float*>(logits_buf_.p);
  }
}

const float* const* BatchDecoder::block_table(const std::vector<const float*>& blocks) {
  auto it = tables_.find(blocks);
  if (it == tables_.end()) {
    DeviceBuffer b(std::max<std::size_t>(blocks.size(), 1) * sizeof(void*));
    if (!blocks.empty()) b.copy_from_host(blocks.data(), blocks.size() * sizeof(void*));
    it = tables_.emplace(blocks, std::move(b)).first;
  }
  return static_cast<const float* const*>(it->second.p);
}

kb::ConvStates BatchDecoder::conv_states() const {
  const State s0 = st_.view(0);
  return {static_cast<float*>(s0.conv_k.data),    static_cast<float*>(s0.conv_v.data),
          static_cast<float*>(s0.conv_attn.data), static_cast<float*>(s0.conv_mlp.data),
          st_.conv_kv_stride(),                   st_.conv_h_stride()};
}

void BatchDecoder::verify_rings(int n, const std::int32_t* commit) {
  const int rings = st_.loopcd.ring_visits();
  const int rows_per_ring = max_seqs_ * (max_m_ - 2);
  const std::size_t rows = std::size_t(rings) * rows_per_ring;
  const std::size_t kv = rows * layout_.kv_dim_local;
  // Mapped host memory keeps this optional rollback workspace out of the device allocation budget.
  if (!pin_ring_backup_.data()) {
    const std::size_t bytes = 2 * kv * sizeof(__nv_bfloat16) + rows * sizeof(std::int64_t);
    pin_ring_backup_ = MappedHostBuffer(bytes);
  }
  auto* backup = static_cast<__nv_bfloat16*>(pin_ring_backup_.device_ptr());
  const kb::RingRows cache{ptr<__nv_bfloat16>(st_.k_ring), ptr<__nv_bfloat16>(st_.v_ring),
                           ptr<std::int64_t>(st_.ring_pos), st_.ring_rows};
  const kb::RingRows saved{backup, backup + kv, reinterpret_cast<std::int64_t*>(backup + 2 * kv),
                           rows_per_ring};
  kb::verify_rings(cache, saved, rings, layout_.kv_dim_local, n, max_m_, slots_, pos_, rows_, commit, stream_);
}

void BatchDecoder::accept(const std::int32_t* kept) {
  if (!last_verify_) return;  // every sequence had one row: its states advanced in the step
  bool restore_rings = false;
  for (int b = 0; b < last_n_; ++b) {
    if (kept[b] < 1 || kept[b] > last_m_[std::size_t(b)])
      throw std::invalid_argument("BatchDecoder::accept: kept rows outside 1..m");
    restore_rings |= last_ring_backup_ && kept[b] + 1 < last_m_[std::size_t(b)];
  }
  CUDA_CHECK(cudaMemcpyAsync(commit_, kept, std::size_t(last_n_) * 4, cudaMemcpyHostToDevice, stream_));
  kb::conv_commit(conv_states(), last_n_, slots_, {hist_k_, hist_v_, hist_attn_, hist_mlp_},
                  rows_host_[std::size_t(last_n_)], rows_, commit_, stream_, layout_.kv_dim_local, st_.loopcd.visits());
  if (restore_rings) verify_rings(last_n_, commit_);
}

std::span<const std::int32_t> BatchDecoder::step(int n, const std::int32_t* m, const std::int32_t* tokens,
                                                 const std::int32_t* slots, const std::int64_t* pos,
                                                 const float* temperature, std::uint64_t seed,
                                                 const BatchRejectInputs* reject, const float* omegas) {
  if (n < 1 || n > max_seqs_) throw std::invalid_argument("BatchDecoder::step: n outside 1..max_seqs");
  rows_host_.resize(std::size_t(n) + 1), last_m_.assign(m, m + n);
  rows_host_[0] = 0;
  const bool may_wrap = (st_.loopcd.hidden() || st_.loopcd.auto_mode() || st_.loopcd.uses_ref_coda()) && st_.ring_rows == WINDOW;
  bool backup_rings = false;
  int M = 0;  // the most rows of a sequence
  for (int b = 0; b < n; ++b) {
    if (m[b] < 1 || m[b] > max_m_) throw std::invalid_argument("BatchDecoder::step: rows outside 1..max_m");
    if (slots[b] < 0 || slots[b] >= st_.slots) throw std::invalid_argument("BatchDecoder::step: slot out of range");
    if (pos[b] < 0 || pos[b] + m[b] > st_.ctx)
      throw std::invalid_argument("BatchDecoder::step: positions outside the slot context");
    rows_host_[std::size_t(b) + 1] = rows_host_[std::size_t(b)] + m[b];
    M = std::max(M, m[b]);
    backup_rings |= may_wrap && m[b] > 2 && pos[b] + m[b] > WINDOW;
  }
  const cudaStream_t s = stream_;
  const int T = rows_host_[std::size_t(n)];  // rows: m[b] per sequence
  const bool verify = M > 1;  // keep taps, conv inputs and the overwritten conv rows (max_m > 1)
  const bool use_reject = reject_enabled_ && reject && reject->active &&
                          std::any_of(reject->active, reject->active + T,
                                      [](std::uint8_t active) { return active != 0; });
  if (use_reject && (!reject->ids || !reject->probs || !reject->picked))
    throw std::invalid_argument("BatchDecoder::step: incomplete rejection sampling inputs");
  std::memcpy(pin_tokens_.data(), tokens, std::size_t(T) * 4);
  std::memcpy(pin_slots_.data(), slots, std::size_t(n) * 4);
  std::memcpy(pin_pos_.data(), pos, std::size_t(n) * 8);
  std::memcpy(pin_rows_.data(), rows_host_.data(), (std::size_t(n) + 1) * 4);
  auto* pt = static_cast<float*>(pin_temps_.data());
  for (int b = 0; b < n; ++b)
    std::fill(pt + rows_host_[std::size_t(b)], pt + rows_host_[std::size_t(b) + 1], temperature[b]);
  std::memcpy(pin_seed_.data(), &seed, sizeof(seed));

  CUDA_CHECK(cudaMemcpyAsync(tokens_, pin_tokens_.data(), std::size_t(T) * 4, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(slots_, pin_slots_.data(), std::size_t(n) * 4, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(pos_, pin_pos_.data(), std::size_t(n) * 8, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(rows_, pin_rows_.data(), (std::size_t(n) + 1) * 4, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(temps_, pin_temps_.data(), std::size_t(T) * 4, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(d_seed_, pin_seed_.data(), sizeof(seed), cudaMemcpyHostToDevice, s));
  if (omegas_dev_) {
    auto* dst = static_cast<float*>(pin_omegas_.data());
    for (int b = 0; b < n; ++b)
      std::fill(dst + rows_host_[b], dst + rows_host_[b + 1], omegas ? omegas[b] : 0.f);
    CUDA_CHECK(cudaMemcpyAsync(omegas_dev_, dst, std::size_t(T) * sizeof(float), cudaMemcpyHostToDevice, s));
  }
  if (use_reject) {
    const auto stage = [&](void* dst, const MappedHostBuffer& pin, const void* src, std::size_t bytes) {
      std::memcpy(pin.data(), src, bytes);
      CUDA_CHECK(cudaMemcpyAsync(dst, pin.data(), bytes, cudaMemcpyHostToDevice, s));
    };
    const std::size_t support_count = std::size_t(T) * kb::RejectSpec::kSupport;
    stage(reject_active_, pin_reject_active_, reject->active, std::size_t(T) * sizeof(std::uint8_t));
    stage(reject_ids_, pin_reject_ids_, reject->ids, support_count * sizeof(std::int32_t));
    stage(reject_probs_, pin_reject_probs_, reject->probs, support_count * sizeof(float));
    stage(reject_picked_, pin_reject_picked_, reject->picked, std::size_t(T) * sizeof(std::int32_t));
  }
  last_n_ = n, last_verify_ = verify, last_ring_backup_ = backup_rings;
  if (backup_rings) verify_rings(n, nullptr);

  const Dense& head_w_chk = unembed_override_ ? *unembed_override_ : w_.unembed;
  const bool can_graph = graph_enabled_ &&
                         (st_.loopcd.uses_ref_coda() || use_nvfp4_head_ || !head_w_chk.fp8()) &&
                         n <= graph_max_seqs_ && T <= graph_max_tokens_;
  // Collapse ragged M into attn_batch's 16-row tile boundaries (1, 5, max_m_) to avoid 7x graph key fragmentation.
  const int M_exec = !verify ? 1 : (M <= 5 ? std::min(5, max_m_) : max_m_);
  const GraphKey gkey{n, T, M_exec, keep_logits_, use_reject};
  const kb::RejectSpec* reject_spec = use_reject ? &reject_spec_ : nullptr;
  bool used_graph = false;
  const std::size_t graph_capacity = (batch_fp8_dense_ || batch_pico12_gemm_) ? 256 : 128;
  if (can_graph) {
    if (auto it = graphs_.find(gkey); it != graphs_.end()) {
      it->second.launch(s);
      used_graph = true;
    } else if (graphs_.size() < graph_capacity && graph_seen_[gkey]++ >= graph_warmup_) {
      // The first eager visit initializes block tables, kernel attributes and exact-shape GEMM plans.
      gemm_->set_capturing(true);
      DecodeGraphDefinition def;
      try {
        def.capture(s, [&] { run_body(n, T, M_exec, verify, seed, reject_spec); });
      } catch (...) {
        gemm_->set_capturing(false);
        throw;
      }
      gemm_->set_capturing(false);
      DecodeGraphExecutable exec;
      exec.instantiate(def);
      exec.upload(s);
      exec.launch(s);
      graphs_.emplace(gkey, std::move(exec));
      used_graph = true;
    }
  }
  if (!used_graph) run_body(n, T, M_exec, verify, seed, reject_spec);

  CUDA_CHECK(cudaMemcpyAsync(pin_sampled_.data(), sampled_, std::size_t(T) * 4, cudaMemcpyDeviceToHost, s));
  if (tp_) tp_->wait(s);
  else CUDA_CHECK(cudaStreamSynchronize(s));
  std::memcpy(sampled_host_.data(), pin_sampled_.data(), std::size_t(T) * 4);
  if (!used_graph) gemm_->collect();
  return {sampled_host_.data(), std::size_t(T)};
}

void BatchDecoder::run_body(int n, int T, int M, bool verify, std::uint64_t seed,
                            const kb::RejectSpec* reject) {
  const auto& loopcd = st_.loopcd;
  const bool uses_ref_coda = loopcd.uses_ref_coda();
  const Weights& w = w_;
  const cudaStream_t s = stream_;
  const std::size_t slab = std::size_t(T) * H;
  const float* attn_dirs = f32p(abl_.attn_dirs);
  const float* attn_w = f32p(abl_.attn_w);
  const float* mlp_dirs = f32p(abl_.mlp_dirs);
  const float* mlp_w = f32p(abl_.mlp_w);
  const State s0 = st_.view(0);

  auto mix = [&](const std::vector<const float*>& blocks, const float* base, const float* write, const float* abl_r,
                 const float* abl_w, float* partial_out, const float* queries, int Q, const float* norm_w,
                 __nv_bfloat16* out_bf16, float* out_f32) {
    dense::MixParams p;
    p.blocks = blocks.empty() ? nullptr : block_table(blocks);
    p.nblocks = int(blocks.size());
    p.base = base;
    p.write = write;
    p.abl_r = write ? abl_r : nullptr;
    p.abl_d = write && abl_.detect ? abl_r + std::size_t(NUM_VISITS) * H : nullptr;
    p.abl_w = write ? abl_w : nullptr;
    p.partial_out = partial_out;
    p.queries = queries;
    p.Q = Q;
    p.norm_w = norm_w;
    p.out_bf16 = out_bf16;
    p.out_f32 = out_f32;
    p.M = T;
    dense::mix(p, false, s);
  };
  // GEMM consumes BF16 rows. Recover packed weights into per-visit scratch.
  auto bf16_rows = [&](const Dense& d) -> const __nv_bfloat16* {
    if (d.is_pico12) {
      dense::unpack_pico12_rows(d.matrix(), d.rows(), wscratch_, s);
    } else if (d.fp8()) {
      dense::dequant_rows(d.w.data, f32p(d.scale), d.rows(), d.cols(), wscratch_, s);
    } else {
      return bf16p(d.w);
    }
    return wscratch_;
  };
  const std::int64_t TG = std::int64_t(T) * GATE_RANK;
  auto gate_down = [&](int G, const Tensor& down) {
    gemm_->run(gin_, 2 * H, bf16p(down), 2 * H, g16_, GATE_RANK, false, T, GATE_RANK, 2 * H, s,
               {G, std::int64_t(T) * 2 * H, std::int64_t(GATE_RANK) * 2 * H, TG});
  };
  // G gates at once: the down projections as one strided GEMM, then forget | gain (fg rows, N = 2H) as one.
  auto gates = [&](const pk::GateRows& rows, int G, const Tensor& down, const __nv_bfloat16* fg_rows,
                   const float* norm_w, __nv_bfloat16* out) {
    pk::gate_inputs(rows, G, T, gin_, s);
    gate_down(G, down);
    gemm_->run(g16_, GATE_RANK, fg_rows, GATE_RANK, fg_, 2 * H, true, T, 2 * H, GATE_RANK, s,
               {G, TG, std::int64_t(2) * H * GATE_RANK, std::int64_t(T) * 2 * H});
    kb::gate_apply(rows, G, fg_, T, norm_w, out, slab, s);
  };
  auto gates_bf16 = [&](const pk::GateRowsBf16& rows, int G, const Tensor& down, const Tensor& forget16,
                        const Tensor& gain16, const float* norm_w, __nv_bfloat16* out) {
    pk::gate_inputs_bf16(rows, G, T, gin_, s);
    gate_down(G, down);
    for (int g = 0; g < G; ++g) {
      const std::size_t weight_offset = std::size_t(g) * H * GATE_RANK;
      pk::gate_fg_apply_bf16(g16_ + g * TG, bf16p(forget16) + weight_offset, bf16p(gain16) + weight_offset,
                            rows.h[g], rows.msg[g], norm_w + std::size_t(g) * H, T, out + g * slab, s);
    }
  };

  pk::widen_ids(tokens_, T, ids_, IMAGE_PLACEHOLDER_ID, s);
  dense::embed(ids_, T, bf16p(w.embed), f32p(w.embed_norm), B0_, false, s);
  int prev_v = -1, tap = 0;

  const Dense& head_w = unembed_override_ ? *unembed_override_ : w.unembed;
  auto finish = [&](bool reference) {
    const int coda_pass = reference ? loopcd.ref_idx : PASSES - 1;
    auto* xf = reference ? xm_ : xf_;
    mix({B0_, B1_, B2_[coda_pass], E_[coda_pass]}, part_, y2_, mlp_dirs + std::size_t(prev_v) * H,
        mlp_w + 4 * std::size_t(prev_v), nullptr, f32p(w.final_query), 1, f32p(w.final_norm), xf, nullptr);
    if (!reference && max_m_ > 1)
      CUDA_CHECK(cudaMemcpy2DAsync(taps_ + std::size_t(NUM_TAPS - 1) * H, std::size_t(NUM_TAPS) * H * 2, xf,
                                  std::size_t(H) * 2, std::size_t(H) * 2, T, cudaMemcpyDeviceToDevice, s));
    if (uses_ref_coda) {
      auto* out = reference ? ref_logits_ : logits_;
      if (use_nvfp4_head_) {
        kb::head_sample_nvfp4(xf, H, w.unembed_nvfp4, T, temps_, seed, out, LD_V,
                              head_partials_, sampled_, s, nullptr, d_seed_);
      } else if (head_w.fp8()) {
        for (int r = 0; r < T; ++r)
          pk::head_logits(xf + std::size_t(r) * H, head_w.w.data, f32p(head_w.scale), UNPADDED_VOCAB,
                          out + std::size_t(r) * LD_V, s);
      } else {
        kb::head_sample(xf, H, head_w.w.data, H, T, UNPADDED_VOCAB, H, temps_, seed, out, LD_V,
                        head_partials_, sampled_, s, 0, nullptr, d_seed_, head_w.matrix().pico12);
      }
      if (use_nvfp4_head_ || !head_w.fp8()) dense::loopcd_softcap_logits(out, T, LD_V, UNPADDED_VOCAB, s);
      return;
    }
    if (tp_ && reject) {
      if (use_nvfp4_head_) {
        kb::head_sample_nvfp4(xf, H, w.unembed_nvfp4, T, temps_, seed, nullptr, 0,
                              head_partials_, nullptr, s, nullptr, d_seed_, reject, local_reject_);
      } else {
        const int n_local = layout_.v_end - layout_.v_begin;
        kb::head_sample(xf, H, head_w.w.data, H, T, n_local, H, temps_, seed, nullptr, 0,
                        head_partials_, nullptr, s, layout_.v_begin, nullptr, d_seed_, head_w.matrix().pico12,
                        reject, local_reject_);
      }
      tp_->exchange_draft_head(local_reject_, std::size_t(T) * sizeof(kb::RejectShardPartial), gathered_reject_, s);
      const int v_split = layout_.tp_rank == 0 ? layout_.v_end : layout_.v_begin;
      kb::head_reject_merge_tp2(gathered_reject_, v_split, temps_, seed, d_seed_, *reject, T, sampled_, s);
    } else if (use_nvfp4_head_) {
      if (tp_) {
        kb::head_sample_nvfp4(xf, H, w.unembed_nvfp4, T, temps_, seed, nullptr, 0,
                              head_partials_, nullptr, s, local_cand_, d_seed_);
        tp_->exchange_draft_head(local_cand_, std::size_t(T) * sizeof(kernels::head::Candidate), gathered_cand_, s);
        tp_merge_candidates(gathered_cand_, sampled_, T, s);
      } else {
        kb::head_sample_nvfp4(xf, H, w.unembed_nvfp4, T, temps_, seed, keep_logits_ ? logits_ : nullptr,
                              LD_V, head_partials_, sampled_, s, nullptr, d_seed_, reject);
      }
    } else if (head_w.fp8()) {
      for (int b = 0; b < T; ++b)
        pk::head_logits(xf + std::size_t(b) * H, head_w.w.data, f32p(head_w.scale), UNPADDED_VOCAB,
                        logits_ + std::size_t(b) * LD_V, s);
      kb::sample(logits_, LD_V, T, UNPADDED_VOCAB, false, temps_, seed, sampled_, s);
    } else if (tp_) {
      const int n_local = layout_.v_end - layout_.v_begin;
      kb::head_sample(xf, H, head_w.w.data, H, T, n_local, H, temps_, seed, nullptr, 0,
                      head_partials_, nullptr, s, layout_.v_begin, local_cand_, d_seed_, head_w.matrix().pico12);
      tp_->exchange_draft_head(local_cand_, std::size_t(T) * sizeof(kernels::head::Candidate), gathered_cand_, s);
      tp_merge_candidates(gathered_cand_, sampled_, T, s);
    } else {
      kb::head_sample(xf, H, head_w.w.data, H, T, UNPADDED_VOCAB, H, temps_, seed, keep_logits_ ? logits_ : nullptr,
                      LD_V, head_partials_, sampled_, s, 0, nullptr, d_seed_, head_w.matrix().pico12, reject);
    }
  };

  const int visits = loopcd.visits();
  for (int v = 0; v < visits; ++v) {
    if (v == NUM_VISITS) finish(false);
    const bool reference = v >= NUM_VISITS;
    const int coda_pass = reference ? loopcd.ref_idx : PASSES - 1;
    const int wv = weight_visit(v);
    const int l = VISITS[wv].layer, p = VISITS[wv].pass;
    const Layer& layer = w.layers[l];
    const bool msgs = is_loop_layer(l) && p > 0;
    const bool boundary = l % BLOCK == 0;
    const std::vector<const float*> blocks = blocks_of(l, p, B0_, B1_, B2_, E_, coda_pass);

    // ---- hidden entering this visit (the previous MLP write applied), stored where it is a block
    float* hid;
    const float* base;
    const float* write;
    if (reference && l == LOOP_END) {
      hid = E_[coda_pass], base = hid, write = nullptr;
    } else if (v == 0) {
      hid = B0_, base = B0_, write = nullptr;
    } else if (l == LOOP_START && p > 0) {
      hid = B1_, base = B1_, write = nullptr;
    } else {
      hid = l == 6 ? B1_ : l == 12 ? B2_[p] : l == LOOP_END ? E_[coda_pass] : hid_;
      base = part_, write = y2_;
    }
    const int pv = prev_v >= 0 ? prev_v : 0;
    const float* r_prev = mlp_dirs + std::size_t(pv) * H;
    const float* w_prev = mlp_w + 4 * std::size_t(pv);
    if (l == LOOP_END && (loopcd.hidden() || loopcd.auto_mode())) {
      if (abl_.active) {
        mix({}, base, write, r_prev, w_prev, hid, nullptr, 1, nullptr, nullptr, nullptr);
        base = hid, write = nullptr;
      }
      if (loopcd.auto_mode())
        dense::loopcd_hidden_rows(base, write, E_[loopcd.ref_idx], hid, B2_[PASSES - 1], B2_[loopcd.ref_idx],
                                  T, H, omegas_dev_, s);
      else
        dense::loopcd_hidden(base, write, E_[loopcd.ref_idx], hid, B2_[PASSES - 1], B2_[loopcd.ref_idx],
                             slab, loopcd.omega, s);
      base = hid, write = nullptr;
    }
    if (msgs) {
      const int i = l - LOOP_START;
      const float* src_base = i == 0 ? part_ : E_[p - 1];
      const float* src_write = i == 0 ? y2_ : nullptr;
      mix({B2_[p - 1]}, src_base, src_write, r_prev, w_prev, i == 0 ? E_[p - 1] : nullptr,
          f32p(w.message_queries) + std::size_t(i) * 4 * H, 4, nullptr, msgs_bf16_, msgs_);
      if (i == 0) write = nullptr;
    }
    // ---- attention read(s)
    const float* queries = layer.attn_query.data ? f32p(layer.attn_query) : nullptr;
    const int nq = layer.attn_query.data ? layer.attn_query.ne[1] : 1;
    float* partial_out = hid != base ? hid : nullptr;
    if (msgs) {
      mix(blocks, base, write, r_prev, w_prev, partial_out, queries, nq, nullptr, mixf_bf16_, mixf_);
      if (batch_gate_fuse_) {
        pk::GateRowsBf16 rows;
        for (int g = 0; g < 3; ++g)
          rows.h[g] = mixf_bf16_ + (layer.full ? g : 0) * slab, rows.msg[g] = msgs_bf16_ + g * slab;
        gates_bf16(rows, 3, layer.qkv_gate_down, layer.qkv_gate_forget16, layer.qkv_gate_gain16,
                   f32p(layer.attn_norm3), xin_);
      } else {
        pk::GateRows rows;
        for (int g = 0; g < 3; ++g) rows.h[g] = mixf_ + (layer.full ? g : 0) * slab, rows.msg[g] = msgs_ + g * slab;
        gates(rows, 3, layer.qkv_gate_down, fg_qkv_[l], f32p(layer.attn_norm3), xin_);
      }
    } else {
      mix(blocks, base, write, r_prev, w_prev, partial_out, l > 0 ? queries : nullptr, l > 0 ? nq : 1,
          f32p(layer.attn_norm_mix), xin_, nullptr);
    }
    if (tap < 3 && v == TAP_VISITS[tap]) {
      if (max_m_ > 1) pk::store_tap(hid, T, taps_ + std::size_t(tap) * H, std::int64_t(NUM_TAPS) * H, s);
      ++tap;
    }

    // ---- projections (fast.proj_layout): one GEMM for a single input, else q | gate | rel | sink from input 0,
    // then k, v (inputs 1, 2) and k_dyn, v_dyn if present. The decode attention kernel reads BF16 projection rows.
    const int R = layout_.projection_rows(layer.full);
    const bool single = !layer.full && !msgs;
    const int k_off = layout_.weight_offset(SEG_K);
    const int v_off = layout_.weight_offset(SEG_V);
    if (batch_fp8_dense_ && layer.wproj.fp8()) {
      const auto* wp = static_cast<const std::uint8_t*>(layer.wproj.w.data);
      const float* wp_scale = f32p(layer.wproj.scale);
      pk::quantize_fp8_rows(xin_, H, single ? T : 3 * T, H, xin_fp8_q_, xin_fp8_s_, s);
      auto project = [&](int input, int offset, int rows) {
        pk::fp8_dense_gemm(xin_fp8_q_ + input * slab, xin_fp8_s_ + std::size_t(input) * T, H,
                           wp + std::size_t(offset) * H, wp_scale + offset, H, proj16_ + offset, R, false,
                           T, rows, H, s);
      };
      if (single) {
        project(0, 0, R);
      } else {
        project(0, 0, k_off);
        for (int seg = SEG_K; seg < num_segs(layer.full); ++seg)
          project(SEG_INPUT[seg], layout_.weight_offset(seg), layout_.segment_rows(seg));
      }
    } else if (batch_pico12_gemm_ && layer.wproj.is_pico12) {
      const auto& layout = *layer.wproj.matrix().pico12;
      auto project = [&](int input, int offset, int rows) {
        kb::pico12_gemm(xin_ + input * slab, H, layer.wproj.w.data, layout, offset, proj16_ + offset, R, false,
                        T, rows, H, s);
      };
      if (single) {
        project(0, 0, R);
      } else {
        project(0, 0, k_off);
        for (int seg = SEG_K; seg < num_segs(layer.full); ++seg)
          project(SEG_INPUT[seg], layout_.weight_offset(seg), layout_.segment_rows(seg));
      }
    } else {
      const __nv_bfloat16* wp = bf16_rows(layer.wproj);
      if (single) {
        gemm_->run(xin_, H, wp, H, proj16_, R, false, T, R, H, s);
      } else {
        static_assert(SEG_INPUT[SEG_Q] == 0 && SEG_INPUT[SEG_SINK] == 0 && SEG_INPUT[SEG_K] == 1 &&
                      SEG_INPUT[SEG_V] == 2 && SEG_INPUT[SEG_K_DYN] == 1 && SEG_INPUT[SEG_V_DYN] == 2);
        static_assert(SEG_ROWS[SEG_K] == SEG_ROWS[SEG_V] && SEG_ROWS[SEG_K_DYN] == SEG_ROWS[SEG_V_DYN]);
        gemm_->run(xin_, H, wp, H, proj16_, R, false, T, k_off, H, s);
        for (const int seg : {SEG_K, SEG_K_DYN}) {
          const int seg_off = layout_.weight_offset(seg);
          if (seg_off >= R) break;
          const int rows = layout_.segment_rows(seg);
          gemm_->run(xin_ + slab, H, wp + std::size_t(seg_off) * H, H, proj16_ + seg_off, R, false, T,
                     rows, H, s, {2, std::int64_t(slab), std::int64_t(rows) * H, rows});
        }
      }
    }
    if (verify)  // the k/v conv inputs (the states are written after acceptance)
      kb::keep_kv(proj16_, R, T, k_off, v_off, hist_k_ + std::size_t(v) * T * layout_.kv_dim_local,
                  hist_v_ + std::size_t(v) * T * layout_.kv_dim_local, s, layout_.kv_dim_local);

    // ---- attention: every sequence over its own slot
    attn::DecodeArgs a;
    a.proj16 = proj16_;
    a.conv_write = !verify;
    a.M = M;
    a.hkv = layout_.hkv_local;
    a.q_scale = f32p(layer.q_scale);
    a.r_scale = f32p(layer.r_scale);
    a.rel_proj = f32p(layer.rel_proj);
    a.conv_k_state = s0.conv_k_at(v);
    a.conv_v_state = s0.conv_v_at(v);
    a.pos = pos_;
    a.workspace = attn_ws_.p;
    a.workspace_size = attn_ws_.bytes;
    a.out = attn_out_;
    a.batch = n;
    a.rows = rows_;
    a.slots = slots_;
    a.conv_stride = st_.conv_kv_stride();
    if (layer.full) {
      const int fi = full_cache_index(v);
      a.k_conv = f32p(layer.k_static);
      a.v_conv = f32p(layer.v_static);
      a.k_cache = s0.k_full_at(fi).data;
      a.v_cache = s0.v_full_at(fi).data;
      a.cache_fp8 = st_.kv_fp8;
      a.cache_q4 = st_.kv_q4;
      a.cache_kv12 = st_.kv_12;
      if (a.cache_q4 || a.cache_kv12) {
        a.k_scale = s0.k_full_scale_at(fi).data;
        a.v_scale = s0.v_full_scale_at(fi).data;
        a.scale_stride = st_.full_scale_stride();
      }
      a.cache_rows = st_.ctx;
      a.kpos = nullptr;
      a.window = 0;
      a.log_scaling = true;
      a.cache_stride = st_.full_stride();
    } else {
      const int ri = ring_cache_index(v);
      a.k_conv = f32p(layer.k_conv);
      a.v_conv = f32p(layer.v_conv);
      a.k_cache = s0.k_ring_at(ri).data;
      a.v_cache = s0.v_ring_at(ri).data;
      a.cache_fp8 = false;
      a.cache_rows = st_.ring_rows;
      a.kpos = s0.ring_pos_at(ri);
      a.window = WINDOW;
      a.log_scaling = false;
      a.cache_stride = st_.ring_stride();
      a.kpos_stride = st_.kpos_stride();
    }
    attn::attn_decode(a, s);

    // ---- output projection + post-attention conv (each slot's state); a verify keeps the conv inputs
    __nv_bfloat16* const wo_dst = verify ? hist_attn_ + std::size_t(v) * slab : wo_out_;
    const bool wo_fp32 = tp_ && !tp_->bf16_wire(2 * v);
    void* wo_prod = wo_fp32 ? static_cast<void*>(partial_) : static_cast<void*>(wo_dst);
    const int q = layout_.q_dim_local;
    if (batch_fp8_dense_ && layer.wo.fp8()) {
      pk::quantize_fp8_rows(attn_out_, q, T, q, attn_fp8_q_, attn_fp8_s_, s);
      pk::fp8_dense_gemm(attn_fp8_q_, attn_fp8_s_, q, layer.wo.w.data, f32p(layer.wo.scale), q, wo_prod, H, wo_fp32,
                         T, H, q, s);
    } else if (batch_pico12_gemm_ && layer.wo.is_pico12) {
      kb::pico12_gemm(attn_out_, q, layer.wo.w.data, *layer.wo.matrix().pico12, 0, wo_prod, H, wo_fp32, T, H, q, s);
    } else {
      gemm_->run(attn_out_, q, bf16_rows(layer.wo), q, wo_prod, H, wo_fp32, T, H, q, s);
    }
    if (tp_) {
      tp_->reduce(partial_, wo_dst, std::size_t(T) * H, TpSchedule::Prefill, 2 * v, s);
      if (wo_fp32) tp_narrow_hidden(partial_, wo_dst, std::size_t(T) * H, s);
    }
    kb::conv(wo_dst, H, n, rows_, H, f32p(layer.attn_conv), s0.conv_attn_at(v), st_.conv_h_stride(), slots_, !verify,
             y_, s);

    // ---- MLP read: this visit's attention write, ablated, on top of hid (or a fresh block)
    std::vector<const float*> mlp_blocks = blocks;
    if (boundary) mlp_blocks.push_back(hid);
    const float* mbase = boundary ? nullptr : hid;
    if (msgs) {
      mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(wv) * H, attn_w + 4 * std::size_t(wv), part_,
          f32p(layer.mlp_query), 1, nullptr, mixm_bf16_, mixm_);
      if (batch_gate_fuse_) {
        pk::GateRowsBf16 rows;
        rows.h[0] = mixm_bf16_, rows.msg[0] = msgs_bf16_ + 3 * slab;
        gates_bf16(rows, 1, layer.mlp1_gate_down, layer.mlp1_gate_forget16, layer.mlp1_gate_gain16,
                   f32p(layer.mlp_norm1), xm_);
      } else {
        pk::GateRows rows;
        rows.h[0] = mixm_, rows.msg[0] = msgs_ + 3 * slab;
        gates(rows, 1, layer.mlp1_gate_down, fg_mlp_[l], f32p(layer.mlp_norm1), xm_);
      }
    } else {
      mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(wv) * H, attn_w + 4 * std::size_t(wv), part_,
          f32p(layer.mlp_query), 1, f32p(layer.mlp_norm1), xm_, nullptr);
    }

    // ---- routed experts (model._moe_grouped) + post-MLP conv (each slot's state)
    gemm_->run(xm_, H, bf16p(layer.router_bf16), H, router_logits_, EXPERTS, true, T, EXPERTS, H, s);
    const pk::MoeWork work{counts_, tiles_n_, tiles_, pair_token_, pair_dst_};
    kb::moe_select(router_logits_, f32p(layer.router_bias), T, idx_, wts_, work, s);
    const Experts& ex = layer.experts;
    const pk::MoeExperts me{ex.fmt == ExpertFormat::NVFP4 ? 0 : 1, ex.w13.data, ex.w13_scale.data,
                            f32p(ex.w13_factor), ex.w2.data, ex.w2_scale.data, f32p(ex.w2_factor), layout_.I_local};
    kb::moe_gate_up(me, xm_, T, work, act_, s);
    __nv_bfloat16* const moe_dst = verify ? hist_mlp_ + std::size_t(v) * slab : moe_sum_;
    if (tp_) {
      kb::moe_down_partial(me, act_, T, work, down_partial_, s);
      const bool moe_fp32 = !tp_->bf16_wire(2 * v + 1);
      pk::moe_combine_partial(down_partial_, wts_, T, moe_fp32 ? partial_ : nullptr, moe_fp32 ? nullptr : moe_dst, s);
      tp_->reduce(partial_, moe_dst, std::size_t(T) * H, TpSchedule::Prefill, 2 * v + 1, s);
      if (moe_fp32) tp_narrow_hidden(partial_, moe_dst, std::size_t(T) * H, s);
    } else {
      kb::moe_down(me, act_, T, work, down_, s);
      pk::moe_combine(down_, wts_, T, moe_dst, s);
    }
    kb::conv(moe_dst, H, n, rows_, H, f32p(layer.mlp_conv), s0.conv_mlp_at(v), st_.conv_h_stride(), slots_, !verify,
             y2_,
             s);
    prev_v = wv;
  }

  if (uses_ref_coda) {
    finish(true);
    dense::loopcd_combine_logits(logits_, ref_logits_, T, LD_V, UNPADDED_VOCAB, loopcd.mode, loopcd.omega, s);
    kb::sample(logits_, LD_V, T, UNPADDED_VOCAB, false, temps_, seed, sampled_, s, d_seed_);
  } else {
    finish(false);
  }
}

}  // namespace ingine::model
