#include "model/dflash2.h"
#include "model/tp_runtime.h"

#include "core/device.h"
#include "core/host_memory.h"
#include "core/safetensors.h"
#include "kernels/dense/dense.h"
#include "kernels/head/head.h"
#include "kernels/prefill/prefill.h"

#include <algorithm>

#include <charconv>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <stdexcept>
#include <string_view>

#ifdef INGINE_HAS_TP_COMM
#include <inkling_comm/comm_backend.h>
#endif

namespace ingine::model {
namespace {

int env_int(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (!value) return fallback;
  const std::string_view input(value);
  int result;
  const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), result);
  if (error != std::errc{} || end != input.data() + input.size())
    throw std::invalid_argument(std::string("invalid integer in ") + name);
  return result;
}

bool is_gb10() {
  int device, major, minor;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  return major == 12 && minor == 1;
}

#ifdef INGINE_HAS_TP_COMM
namespace comm = inkling::comm;
namespace head = kernels::head;

constexpr unsigned kReductions = 2 * NUM_VISITS;
constexpr unsigned kHeadBefore = 1 + 2 * kReductions;
constexpr unsigned kHeadAfter = kHeadBefore + 1;
constexpr unsigned kStepEnd = kHeadAfter + 1;
constexpr unsigned kStageCount = static_cast<unsigned>(VerifyStage::Count);
constexpr unsigned kStageSlots = 2 * NUM_VISITS;
constexpr unsigned stage_event(VerifyStage stage, int occurrence, bool end) {
  return kStepEnd + 1 + (static_cast<unsigned>(stage) * kStageSlots + occurrence) * 2 + end;
}
constexpr unsigned kEventCount = kStepEnd + 1 + kStageCount * kStageSlots * 2;
constexpr std::size_t kVocabularyShard = UNPADDED_VOCAB / 2;

int head_width(const RuntimeOptions& options) {
  return std::max(options.max_batch, options.spec ? 8 : 1);
}
static_assert(sizeof(head::Candidate) == 8);

std::uint64_t checkpoint_fingerprint(const RuntimeOptions& options) {
  // Checkpoint identity guard for rendezvous, independent of node-local paths.
  // Use the safetensors headers and a prefix/suffix of each payload. Ablations are small.
  std::uint64_t hash = 14695981039346656037ull;
  auto mix = [&](std::span<const std::byte> data) {
    for (std::byte x : data) hash = (hash ^ std::to_integer<unsigned>(x)) * 1099511628211ull;
  };
  auto read = [&](const SafetensorsFile& file, std::uint64_t offset, std::size_t size) {
    std::vector<std::byte> bytes(size);
    file.read_range(offset, size, bytes.data());
    mix(bytes);
  };
  const SafetensorsModel model(options.model_dir);
  for (const auto& file : model.files()) {
    read(*file, 0, file->data_offset());
    const auto sample = std::min<std::uint64_t>(file->file_size() - file->data_offset(), 4096);
    read(*file, file->data_offset(), sample);
    read(*file, file->file_size() - sample, sample);
  }
  if (!options.ablation.empty()) {
    const SafetensorsFile file(options.ablation);
    read(file, 0, file.file_size());
  }
  return hash;
}

comm::CommPlan make_plan(const RuntimeOptions& options) {
  const bool bf16 = options.tp.wire == TpWire::Bf16;
  const auto bytes = std::size_t(options.prefill_chunk) * HIDDEN * (bf16 ? 2 : 4);
  // Reserve twelve bits for the chunk length and five for mode, precision and head flags.
  const std::uint64_t schedule = 0x4d34000000000000ull |
      (std::uint64_t(options.max_ctx) << 17) | (std::uint64_t(options.prefill_chunk) << 5) |
      (options.spec << 4) | (options.head_nvfp4 << 3) | (options.dense_fp8 << 2) | (options.kv_fp8 << 1) | options.tp.full_logits;
  const std::uint64_t wire_hash = bf16 ? 0x200b : options.tp.wire == TpWire::Bf16Moe ? 0x200c : 0x200f;
  const std::uint64_t layout_hash = wire_hash | (std::uint64_t(options.max_batch - 1) << 16);
  const auto gather_bytes = kVocabularyShard * sizeof(float) * options.max_batch;
  return {std::max(bytes, gather_bytes), layout_hash, checkpoint_fingerprint(options),
          schedule | (std::uint64_t(options.draft_type != "dspark" &&
                    options.head_nvfp4 && !dflash2_full_head()) << 49) | (std::uint64_t(options.tp.serve) << 48) | (std::uint64_t(options.spec_widths) << 46)};
}

comm::CommOptions transport_options(const RuntimeOptions& options) {
  auto net = comm::comm_options_from_env();
  net.world_size = options.tp.size;
  net.rank = options.tp.rank;
  CUDA_CHECK(cudaGetDevice(&net.cuda_device));
  return net;
}
#endif

}  // namespace

TpOptions tp_options(std::optional<int> size, std::optional<int> rank, std::string wire) {
  TpOptions options;
  options.size = size ? *size : env_int("INKLING_TP_SIZE", 1);
  options.rank = rank ? *rank : env_int("INKLING_TP_RANK", 0);
  if (wire.empty()) {
    const char* env = std::getenv("INKLING_TP_WIRE");
    wire = env ? env : "fp32";
  }
  if (wire == "fp32") options.wire = TpWire::Fp32;
  else if (wire == "bf16") options.wire = TpWire::Bf16;
  else if (wire == "bf16-moe") options.wire = TpWire::Bf16Moe;
  else throw std::invalid_argument("TP wire must be bf16, fp32 or bf16-moe");
  if ((options.size != 1 && options.size != 2) || options.rank < 0 || options.rank >= options.size)
    throw std::invalid_argument("TP requires size 1 or 2 and rank in [0, size)");
  return options;
}

struct TpRuntime::Impl {
#ifdef INGINE_HAS_TP_COMM
  Impl(const RuntimeOptions& options)
      : backend(comm::make_comm_backend(make_plan(options), transport_options(options))),
        logits(!options.head_nvfp4 || options.tp.full_logits || options.tp.serve ? kVocabularyShard * head_width(options) * sizeof(float) : 0),
        local(head_width(options) * sizeof(head::Candidate)),
        candidates(2 * local.bytes),
        index(head_width(options) * sizeof(int32_t)),
        head_scratch(options.head_nvfp4 && !options.tp.full_logits
                         ? head::nvfp4_candidate_workspace(head_width(options), int(kVocabularyShard)) : 0) {
    CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    if (options.tp.profile) {
      events.resize(kEventCount);
      for (auto& e : events) CUDA_CHECK(cudaEventCreate(&e));
    }
  }

  ~Impl() {
    cudaEventDestroy(done);
    for (auto e : events)
      if (e) cudaEventDestroy(e);
  }

  std::unique_ptr<comm::CommBackend> backend;
  DeviceBuffer logits, local, candidates, index, head_scratch;
  PinnedHostBuffer request_hashes{2 * sizeof(std::uint64_t)};
  cudaEvent_t done = nullptr;
  std::vector<cudaEvent_t> events;
#endif
};

TpRuntime::TpRuntime(const RuntimeOptions& options)
    : layout(options.tp.size, options.tp.rank), wire(options.tp.wire),
      full_logits(options.tp.full_logits), profile(options.tp.profile),
      gb10_spec(options.spec && is_gb10()), serving(options.tp.serve) {
#ifdef INGINE_HAS_TP_COMM
  impl_ = std::make_unique<Impl>(options);
#else
  throw std::runtime_error("TP=2 requires a build linked to inkling_comm::inkling_comm");
#endif
}

TpRuntime::~TpRuntime() = default;

void TpRuntime::reduce(float* fp32, __nv_bfloat16* bf16, std::size_t count, TpSchedule schedule,
                       unsigned op, cudaStream_t stream) {
#ifdef INGINE_HAS_TP_COMM
  const bool timed = profile && schedule == TpSchedule::Decode;
  if (timed) mark(1 + 2 * op, stream);
  const comm::CollectiveSite site{static_cast<unsigned>(schedule), op};
  if (bf16_wire(op)) impl_->backend->allreduce_bf16({bf16, count}, {bf16, count}, site, stream);
  else impl_->backend->allreduce_f32({fp32, count}, {fp32, count}, site, stream);
  if (timed) mark(2 + 2 * op, stream);
#endif
}

void TpRuntime::head(const Weights& w, const __nv_bfloat16* x, float* logits, int32_t* token, int m,
                     TpSchedule schedule, cudaStream_t stream) {
#ifdef INGINE_HAS_TP_COMM
  using Candidate = kernels::head::Candidate;
  const bool timed = profile && schedule == TpSchedule::Decode;
  auto* local = static_cast<Candidate*>(impl_->local.p);
  auto* shard = static_cast<float*>(impl_->logits.p);
  if (timed) mark_stage(VerifyStage::Head, 0, false, stream);
  if (w.unembed_nvfp4.codes) {
    head::head_nvfp4(x, m, w.unembed_nvfp4, full_logits ? shard : nullptr,
                     full_logits ? nullptr : local, static_cast<Candidate*>(impl_->head_scratch.p),
                     false, stream);
  } else {
    if (schedule == TpSchedule::Decode) {
      const kernels::dense::Matrix matrix{w.unembed.w.data, ptr<float>(w.unembed.scale),
                                          w.unembed.rows(), w.unembed.cols()};
      kernels::dense::head(x, m, matrix, shard, false, stream);
    } else {
      kernels::prefill::head_logits(x, w.unembed.w.data, ptr<float>(w.unembed.scale),
                                    kVocabularyShard, shard, stream);
    }
    if (!full_logits) {
      auto* index = static_cast<int32_t*>(impl_->index.p);
      head::argmax(shard, m, kVocabularyShard, kVocabularyShard, index, stream);
      tp_store_candidate(shard, index, layout.v_begin, m, kVocabularyShard, local, stream);
    }
  }
  if (timed) {
    mark_stage(VerifyStage::Head, 0, true, stream);
    mark(kHeadBefore, stream);
  }
  const comm::CollectiveSite site{static_cast<unsigned>(schedule), kReductions};
  if (full_logits && serving) {
    // The local shard is already on the sampling GPU. Populate its half of each row
    // with a pitched device copy and gather only the remote shard into the other half.
    constexpr std::size_t shard_bytes = kVocabularyShard * sizeof(float);
    constexpr std::size_t row_bytes = UNPADDED_VOCAB * sizeof(float);
    const bool root = layout.tp_rank == 0;
    if (root)
      CUDA_CHECK(cudaMemcpy2DAsync(logits, row_bytes, shard, shard_bytes, shard_bytes,
                                   m, cudaMemcpyDeviceToDevice, stream));
    impl_->backend->gather_rows_to_root({root ? nullptr : shard, root ? 0 : m * kVocabularyShard},
        {root ? reinterpret_cast<std::byte*>(logits + kVocabularyShard) : nullptr,
         root ? m * row_bytes - shard_bytes : 0},
        {static_cast<std::size_t>(m), {0, kVocabularyShard}, UNPADDED_VOCAB}, 0, site, stream);
  } else if (full_logits) {
    impl_->backend->gather_rows_to_root({shard, m * kVocabularyShard},
        {reinterpret_cast<std::byte*>(logits), logits ? m * UNPADDED_VOCAB * sizeof(float) : 0},
        {static_cast<std::size_t>(m), {kVocabularyShard, kVocabularyShard}, 0}, 0, site, stream);
  } else {
    impl_->backend->allgather_small({static_cast<const std::byte*>(impl_->local.p), m * sizeof(Candidate)},
        {static_cast<std::byte*>(impl_->candidates.p), 2 * m * sizeof(Candidate)}, site, stream);
  }
  if (timed) mark(kHeadAfter, stream);
  if (!full_logits) tp_merge_candidates(static_cast<const Candidate*>(impl_->candidates.p), token, m, stream);
  if (timed) mark(kStepEnd, stream);
#endif
}

void TpRuntime::allreduce_bf16(__nv_bfloat16* bf16, std::size_t count, TpSchedule schedule,
                               unsigned op, cudaStream_t stream) {
#ifdef INGINE_HAS_TP_COMM
  const comm::CollectiveSite site{static_cast<unsigned>(schedule), op};
  impl_->backend->allreduce_bf16({bf16, count}, {bf16, count}, site, stream);
#else
  (void)bf16; (void)count; (void)schedule; (void)op; (void)stream;
#endif
}

void TpRuntime::broadcast_proposal(void* device, std::size_t bytes, cudaStream_t stream) {
#ifdef INGINE_HAS_TP_COMM
  impl_->backend->broadcast_tokens({static_cast<std::byte*>(device), bytes}, 0,
      {static_cast<unsigned>(TpSchedule::Proposal), 0}, stream);
#endif
}

void TpRuntime::exchange_draft_head(void* device, std::size_t bytes, void* gathered, cudaStream_t stream) {
#ifdef INGINE_HAS_TP_COMM
  const comm::CollectiveSite site{static_cast<unsigned>(TpSchedule::DraftHead), gathered ? 1u : 0u};
  auto* data = static_cast<std::byte*>(device);
  if (gathered)
    impl_->backend->allgather_small({data, bytes}, {static_cast<std::byte*>(gathered), 2 * bytes}, site, stream);
  else
    impl_->backend->broadcast_tokens({data, bytes}, 0, site, stream);
#endif
}

bool TpRuntime::agree(std::uint64_t fingerprint, cudaStream_t stream) {
#ifdef INGINE_HAS_TP_COMM
  auto* hashes = static_cast<std::uint64_t*>(impl_->request_hashes.data());
  hashes[0] = fingerprint;
  CUDA_CHECK(cudaMemcpyAsync(impl_->local.p, hashes, sizeof(fingerprint), cudaMemcpyHostToDevice, stream));
  impl_->backend->allgather_small({static_cast<const std::byte*>(impl_->local.p), sizeof(fingerprint)},
      {static_cast<std::byte*>(impl_->candidates.p), 2 * sizeof(fingerprint)},
      {static_cast<unsigned>(TpSchedule::Request), 0}, stream);
  CUDA_CHECK(cudaMemcpyAsync(hashes, impl_->candidates.p, 2 * sizeof(fingerprint), cudaMemcpyDeviceToHost, stream));
  wait(stream);
  return hashes[0] == hashes[1];
#else
  return true;
#endif
}

void TpRuntime::verify_request(std::uint64_t fingerprint, cudaStream_t stream) {
  if (!agree(fingerprint, stream)) throw std::invalid_argument("TP ranks submitted different request schedules");
}

void TpRuntime::wait(cudaStream_t stream) {
#ifdef INGINE_HAS_TP_COMM
  CUDA_CHECK(cudaEventRecord(impl_->done, stream));
  CUDA_CHECK(cudaEventSynchronize(impl_->done));
  check();
#endif
}

void TpRuntime::check() const {
#ifdef INGINE_HAS_TP_COMM
  const auto state = impl_->backend->status();
  if (!state.ok)
    throw std::runtime_error("TP collective failed at epoch " + std::to_string(state.epoch) +
                             ", schedule " + std::to_string(state.site.schedule_id) +
                             ", operation " + std::to_string(state.site.op_index) + ": " + state.message);
#endif
}

void TpRuntime::mark(unsigned boundary, cudaStream_t stream) {
#ifdef INGINE_HAS_TP_COMM
  // External records survive stream capture as timed event nodes and are read after replay completion.
  cudaStreamCaptureStatus capture;
  CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
  const unsigned flags = capture == cudaStreamCaptureStatusActive ? cudaEventRecordExternal : cudaEventRecordDefault;
  CUDA_CHECK(cudaEventRecordWithFlags(impl_->events[boundary], stream, flags));
#endif
}

void TpRuntime::mark_stage(VerifyStage stage, int occurrence, bool end, cudaStream_t stream) {
#ifdef INGINE_HAS_TP_COMM
  mark(stage_event(stage, occurrence, end), stream);
#endif
}

void TpRuntime::print_verify_profile(int width) const {
#ifdef INGINE_HAS_TP_COMM
  constexpr const char* names[] = {
      "mix", "message_gates", "wproj", "attn", "wo", "router", "moe_up", "moe_down_partial", "conv_epilogue", "head"};
  auto elapsed = [&](unsigned begin, unsigned end) {
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, impl_->events[begin], impl_->events[end]));
    return ms;
  };
  double attn_reduce = 0, moe_reduce = 0;
  for (unsigned visit = 0; visit < kReductions; ++visit)
    (visit % 2 ? moe_reduce : attn_reduce) += elapsed(1 + 2 * visit, 2 + 2 * visit);
  std::printf("verify_profile M=%d graph=%.6f wo_allreduce=%.6f moe_allreduce=%.6f head_exchange=%.6f",
              width, elapsed(0, kStepEnd), attn_reduce, moe_reduce, elapsed(kHeadBefore, kHeadAfter));
  for (unsigned stage = 0; stage < kStageCount; ++stage) {
    auto kind = static_cast<VerifyStage>(stage);
    double total = 0;
    for (unsigned visit = 0; visit < NUM_VISITS; ++visit) {
      if (kind == VerifyStage::Head && visit != 0) continue;
      if (kind == VerifyStage::MessageGates &&
          (!is_loop_layer(VISITS[visit].layer) || VISITS[visit].pass == 0)) continue;
      const bool twice = kind == VerifyStage::Conv || kind == VerifyStage::Mix || kind == VerifyStage::MessageGates;
      const int occurrence = twice ? 2 * visit : visit;
      const auto ms = [&](int i) { return elapsed(stage_event(kind, i, false), stage_event(kind, i, true)); };
      total += ms(occurrence);
      if (twice) total += ms(occurrence + 1);
    }
    std::printf(" %s=%.6f", names[stage], total);
  }
  std::printf("\n");
#endif
}

DecodeTiming TpRuntime::timing() const {
#ifdef INGINE_HAS_TP_COMM
  if (profile) {
    auto elapsed = [&](unsigned begin, unsigned end) {
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, impl_->events[begin], impl_->events[end]));
      return ms;
    };
    DecodeTiming result{elapsed(0, kStepEnd), 0, elapsed(kHeadBefore, kHeadAfter), 1};
    for (unsigned i = 0; i < kReductions; ++i) result.reduce_ms += elapsed(1 + 2 * i, 2 + 2 * i);
    return result;
  }
#endif
  return {};
}
}  // namespace ingine::model
