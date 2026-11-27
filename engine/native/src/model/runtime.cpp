// Model runtime: weights + ablation + state + decoder, decode graphs per call width (engine.Engine._capture),
// state snapshots and speculative rollback (engine.Engine._snapshot/_restore/_rollback), pipelined width-1
// steps and the switch graphs of device-selected verify widths.
#include "model/runtime.h"

#include "core/device.h"
#include "kernels/dense/dense.h"

#include <cstdlib>
#include <cstring>
#include <string_view>
#include <algorithm>
#include <array>
#include <stdexcept>

namespace ingine::model {
namespace {

constexpr int kMaxWidth = 8;

bool enable_cert_head(const RuntimeOptions& options) {
  if (options.tp.size != 1 || options.head_nvfp4 || options.dense_fp8 || loopcd_config(options).uses_ref_coda()) return false;
  if (const char* value = std::getenv("INGINE_CERT_HEAD")) {
    if (std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument("INGINE_CERT_HEAD must be 0 or 1");
  }
  return options.cert_head || options.dense_pico12;
}

TpLayout runtime_layout(const RuntimeOptions& options) {
  validate_loopcd_tp(loopcd_config(options), options.tp.size);
  validate_dense_formats(options.dense_fp8, options.dense_pico12);
  validate_kv_formats(options.kv_fp8, options.kv_q4, options.kv_12);
  if (options.tp.size != 1 && options.tp.size != 2)
    throw std::invalid_argument("runtime supports TP sizes 1 and 2");
  if (options.tp.size == 2 && options.vision)
    throw std::invalid_argument("TP=2 supports text decoding");
  if (options.tp.size == 2 && options.spec && options.tp.full_logits)
    throw std::invalid_argument("TP=2 speculation requires the distributed greedy head");
  if (options.spec_verify_width) {
    const auto widths = verify_width_grid(options.spec_widths);
    if ((options.tp.size != 2 && options.draft_type != "dflash2") || !options.spec || options.spec_widths == SpecWidths::Coarse ||
        std::find(widths.begin(), widths.end(), options.spec_verify_width) == widths.end())
      throw std::invalid_argument("fixed verify width must belong to the selected speculation width grid");
  }
  if (options.tp.size == 2 && (options.prefill_chunk < 1 || options.prefill_chunk > 8192))
    throw std::invalid_argument("TP=2 prefill chunk must be in 1..8192");
  return TpLayout(options.tp.size, options.tp.rank);
}

DeviceBuffer device_copy(const Tensor& t, cudaStream_t stream) {
  DeviceBuffer b(t.bytes());
  CUDA_CHECK(cudaMemcpyAsync(b.p, t.data, t.bytes(), cudaMemcpyDeviceToDevice, stream));
  return b;
}

// Carves consecutive 256-byte aligned pieces out of one allocation.
struct Carve {
  char* p;
  template <class T>
  T* take(std::size_t count) {
    T* r = reinterpret_cast<T*>(p);
    p += (count * sizeof(T) + 255) / 256 * 256;
    return r;
  }
};
constexpr std::size_t carved(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

std::size_t undo_bytes(int visits, int rings) {
  return carved(std::size_t(visits) * CONV_ROW_FLOATS * 4) +
         2 * carved(std::size_t(rings) * KV_DIM * 2) + carved(std::size_t(rings) * 8);
}
std::size_t backup_bytes(int visits, int rings) {
  return 2 * carved(std::size_t(visits) * 3 * KV_DIM * 4) +
         2 * carved(std::size_t(visits) * 3 * HIDDEN * 4) +
         2 * carved(std::size_t(rings) * kMaxWidth * KV_DIM * 2) +
         carved(std::size_t(rings) * kMaxWidth * 8);
}

}  // namespace

GraphExec::~GraphExec() {
  if (exec_) cudaGraphExecDestroy(exec_);
}

GraphExec& GraphExec::operator=(GraphExec&& o) noexcept {
  if (this != &o) {
    if (exec_) cudaGraphExecDestroy(exec_);
    exec_ = o.exec_;
    o.exec_ = nullptr;
  }
  return *this;
}

void GraphExec::launch(cudaStream_t stream) const {
  if (!exec_) throw std::logic_error("GraphExec::launch: empty graph");
  CUDA_CHECK(cudaGraphLaunch(exec_, stream));
}

Runtime::Runtime(const RuntimeOptions& options) : options_(options) {
  if (options.max_ctx < 16) throw std::invalid_argument("max_ctx must be at least 16");
  const auto layout = runtime_layout(options);
  const bool cert_head = enable_cert_head(options);
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  weights_ = load_weights(options.model_dir, options.dense_fp8, stream_, options.head_nvfp4, layout,
                          options.dense_pico12, cert_head);
  ablation_ = load_ablation(options.ablation);
  init();
}

Runtime::Runtime(const RuntimeOptions& options, Weights weights, Ablation ablation)
    : options_(options), weights_(std::move(weights)), ablation_(std::move(ablation)) {
  if (options.tp.size != 1 || options.tp.rank != 0)
    throw std::invalid_argument("Runtime over caller-supplied weights requires TP=1");
  runtime_layout(options);
  if (options.max_ctx < 16) throw std::invalid_argument("max_ctx must be at least 16");
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  init();
}

void Runtime::init() {
  const RuntimeOptions& options = options_;
  const int batch_cap = options.batch_spec > 0 ? 128 : kMaxWidth;
  if (options.max_batch < 1 || options.max_batch > batch_cap)
    throw std::invalid_argument(options.batch_spec > 0 ? "max_batch must be in 1..128" : "max_batch must be in 1..8");
  if (options.batch_spec == 0 && options.max_batch > 1 && options.tp.size > 1 && (!options.tp.serve || options.tp.profile))
    throw std::invalid_argument("TP batch graphs require unprofiled serving mode");
  const auto layout = runtime_layout(options);
  state_ = make_state(options.max_ctx, options.kv_fp8, layout, options.kv_q4, options.kv_12, loopcd_config(options));
  const int visits = state_.loopcd.visits();
  const int rings = state_.loopcd.ring_visits();
  const bool ref_coda = state_.loopcd.uses_ref_coda();
  if (layout.tp_size > 1) tp_ = std::make_unique<TpRuntime>(options);
  if (state_.loopcd.auto_mode()) {
    d_loopcd_omega_ = DeviceBuffer(sizeof(float));
    CUDA_CHECK(cudaMemsetAsync(d_loopcd_omega_.p, 0, sizeof(float), stream_));
  }
  decoder_ = std::make_unique<Decoder>(weights_, ablation_, state_, true, nullptr, tp_.get(),
                                       static_cast<const float*>(d_loopcd_omega_.p));
  pos_ = DeviceBuffer(sizeof(std::int64_t));
  pos_.fill(0);
  token_ = DeviceBuffer(kMaxWidth * sizeof(std::int64_t));
  token_.fill(0);
  CUDA_CHECK(cudaStreamCreateWithFlags(&capture_stream_, cudaStreamNonBlocking));
  if (!tp_) {
    for (cudaEvent_t& e : chain_done_) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    chain_host_ = MappedHostBuffer(2 * sizeof(std::int32_t));
    for (int i = 0; i < 2; ++i) {
      undo_storage_[i] = DeviceBuffer(undo_bytes(visits, rings));
      Carve c{static_cast<char*>(undo_storage_[i].p)};
      undo_[i].conv = c.take<float>(std::size_t(visits) * CONV_ROW_FLOATS);
      undo_[i].k = c.take<__nv_bfloat16>(std::size_t(rings) * KV_DIM);
      undo_[i].v = c.take<__nv_bfloat16>(std::size_t(rings) * KV_DIM);
      undo_[i].pos = c.take<std::int64_t>(rings);
    }
  }

  buffers_.emplace(1, make_buffers(1, layout, options.tp.full_logits, options.tp.serve, has_cert_head(), ref_coda));
  history_[1] = false;
  if (options.spec) {
    for (int w : verify_width_grid(options.spec_widths)) {
      if (w > 1) {
        buffers_.emplace(w, make_buffers(w, layout, options.tp.full_logits, options.tp.serve, has_cert_head(), ref_coda));
        history_[w] = true;
      }
      verify_widths_.push_back(w);
    }
    backup_storage_ = DeviceBuffer(backup_bytes(visits, rings));
    Carve c{static_cast<char*>(backup_storage_.p)};
    const int cols[] = {layout.kv_dim_local, layout.kv_dim_local, HIDDEN, HIDDEN};
    for (int i = 0; i < 4; ++i) backup_.conv[i] = c.take<float>(std::size_t(visits) * 3 * cols[i]);
    backup_.k = c.take<__nv_bfloat16>(std::size_t(rings) * kMaxWidth * layout.kv_dim_local);
    backup_.v = c.take<__nv_bfloat16>(std::size_t(rings) * kMaxWidth * layout.kv_dim_local);
    backup_.pos = c.take<std::int64_t>(std::size_t(rings) * kMaxWidth);
  }
  // every width reads its tokens from the shared buffer (set_inputs, chain_advance and the verify selector
  // write one place)
  for (auto& [m, b] : buffers_) b.token = rm(token_.p, DType::I64, {m});
  for (const auto& [m, history] : history_) capture(m, history, graphs_);
  if (has_cert_head())
    for (const auto& [m, history] : history_) capture(m, history, certified_graphs_, true);
  if (tp_ && options.tp.serve) {
    tp_->full_logits = true;
    for (const auto& [m, history] : history_) capture(m, history, sampled_graphs_);
    tp_->full_logits = false;
  }
  if (options.max_batch > 1 && options.batch_spec == 0) init_batch(layout);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

State& Runtime::state_slot(int slot) {
  return slot == 0 ? state_ : extra_slots_.at(slot - 1);
}

void Runtime::set_batch_inputs(std::span<const BatchInput> rows) {
  auto& table = *static_cast<BatchSlotTable*>(host_batch_table_.data());
  const int visits = state_.loopcd.visits();
  auto* ref = reinterpret_cast<BatchRefSlotTable*>(&table + 1);
  auto* omegas = reinterpret_cast<float*>(&table + 1);  // Auto and the reference coda are mutually exclusive.
  unsigned used = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto [slot, token, omega] = rows[i];
    if (slot < 0 || slot >= options_.max_batch || (used & (1u << slot)))
      throw std::invalid_argument("batch slots must be distinct and in range");
    used |= 1u << slot;
    auto& state = state_slot(slot);
    if (state.length < 0 || state.length >= state.max_ctx)
      throw std::invalid_argument("batch slot position outside context");
    if (token < 0 || token >= UNPADDED_VOCAB)
      throw std::invalid_argument("batch token outside vocabulary");
    table.tokens[i] = token;
    if (state_.loopcd.auto_mode()) omegas[i] = omega;
    for (int v = 0; v < visits; ++v) {
      const bool reference = v >= NUM_VISITS;
      const int r = v - NUM_VISITS;
      auto& entry = reference ? ref->attn[r][i] : table.attn[v][i];
      entry.position = state.length;
      entry.conv_k_state = state.conv_k_at(v);
      entry.conv_v_state = state.conv_v_at(v);
      if (const int full = full_cache_index(v); full >= 0) {
        entry.k_cache = state.k_full_at(full).data;
        entry.v_cache = state.v_full_at(full).data;
        if (state.kv_q4 || state.kv_12) {
          entry.k_scale = state.k_full_scale_at(full).data;
          entry.v_scale = state.v_full_scale_at(full).data;
        }
        entry.kpos = nullptr;
      } else {
        const int ring = ring_cache_index(v);
        entry.k_cache = state.k_ring_at(ring).data;
        entry.v_cache = state.v_ring_at(ring).data;
        entry.kpos = state.ring_pos_at(ring);
      }
      (reference ? ref->attn_conv[r][i] : table.attn_conv[v][i]) = state.conv_attn_at(v);
      (reference ? ref->mlp_conv[r][i] : table.mlp_conv[v][i]) = state.conv_mlp_at(v);
    }
  }
  CUDA_CHECK(cudaMemcpyAsync(batch_table_.p, &table, batch_table_.bytes, cudaMemcpyHostToDevice, stream_));
}

void Runtime::init_batch(const TpLayout& layout) {
  decoder_->allocate_batch_workspace(options_.max_batch);
  for (int i = 1; i < options_.max_batch; ++i)
    extra_slots_.push_back(make_state(options_.max_ctx, options_.kv_fp8, layout, options_.kv_q4, options_.kv_12, state_.loopcd));
  const bool ref_coda = state_.loopcd.uses_ref_coda();
  const std::size_t bytes = sizeof(BatchSlotTable) + (ref_coda ? sizeof(BatchRefSlotTable) : 0) +
      (state_.loopcd.auto_mode() ? options_.max_batch * sizeof(float) : 0);
  batch_table_ = DeviceBuffer(bytes);
  host_batch_table_ = MappedHostBuffer(bytes);
  std::memset(host_batch_table_.data(), 0, bytes);
  auto* table = static_cast<BatchSlotTable*>(batch_table_.p);
  std::array<BatchInput, kMaxWidth> inputs{};
  for (int i = 0; i < options_.max_batch; ++i) inputs[i] = {i, 0};
  set_batch_inputs(std::span(inputs).first(options_.max_batch));
  for (int width = 1; width <= options_.max_batch; ++width) {
    auto& entry = batch_widths_[width];
    entry.buffers = make_buffers(width, layout, true, bool(tp_), has_cert_head(), ref_coda);
    entry.buffers.token = rm(table->tokens, DType::I64, {width});
    auto capture = [&](bool full_logits) {
      if (tp_) tp_->full_logits = full_logits;
      decoder_->decode_batch(entry.buffers, table, stream_, !full_logits && has_cert_head());
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      if (tp_) tp_->check();
      cudaGraph_t graph = nullptr;
      CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
      try {
        decoder_->decode_batch(entry.buffers, table, stream_, !full_logits && has_cert_head());
      } catch (...) {
        cudaStreamEndCapture(stream_, &graph);
        if (graph) cudaGraphDestroy(graph);
        throw;
      }
      CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
      cudaGraphExec_t exec;
      CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
      CUDA_CHECK(cudaGraphDestroy(graph));
      return GraphExec(exec);
    };
    entry.greedy = capture(false);
    if (tp_ || has_cert_head()) entry.sampled = capture(true);
  }
  if (tp_) tp_->full_logits = false;
  for (int i = 0; i < options_.max_batch; ++i) reset_state(state_slot(i), stream_);
}

Buffers& Runtime::run_batch(std::span<const BatchInput> rows, bool full_logits, bool greedy_only) {
  auto& entry = batch_widths_.at(int(rows.size()));
  set_batch_inputs(rows);
  if (tp_) tp_->full_logits = full_logits;
  const bool use_full_graph = tp_ ? full_logits : has_cert_head() && (full_logits || !greedy_only);
  (use_full_graph ? entry.sampled : entry.greedy).launch(stream_);
  if (tp_) tp_->wait(stream_);
  else CUDA_CHECK(cudaStreamSynchronize(stream_));
  for (const auto& row : rows) ++state_slot(row.slot).length;
  return entry.buffers;
}

Runtime::~Runtime() {
  if (stream_) cudaStreamSynchronize(stream_);
  for (auto& [m, exec] : graphs_) cudaGraphExecDestroy(exec);
  graphs_.clear();
  for (auto& [m, exec] : sampled_graphs_) cudaGraphExecDestroy(exec);
  batch_widths_.clear();
  decoder_.reset();
  for (cudaEvent_t e : chain_done_)
    if (e) cudaEventDestroy(e);
  if (capture_stream_) cudaStreamDestroy(capture_stream_);
  if (stream_) cudaStreamDestroy(stream_);
}

bool Runtime::has_width(int m) const { return buffers_.count(m) != 0; }

std::span<const int> Runtime::verify_widths() const { return verify_widths_; }

Buffers& Runtime::buffers(int m) {
  auto it = buffers_.find(m);
  if (it == buffers_.end()) throw std::invalid_argument("no decode buffers for width " + std::to_string(m));
  return it->second;
}

StatePtrs Runtime::state_ptrs() const {
  StatePtrs p;
  p.kv_dim = state_.conv_k.ne[0];
  p.ref_coda = state_.loopcd.uses_ref_coda();
  p.conv[0] = ptr<float>(state_.conv_k);
  p.conv[1] = ptr<float>(state_.conv_v);
  p.conv[2] = ptr<float>(state_.conv_attn);
  p.conv[3] = ptr<float>(state_.conv_mlp);
  p.k_ring = ptr<__nv_bfloat16>(state_.k_ring);
  p.v_ring = ptr<__nv_bfloat16>(state_.v_ring);
  p.ring_pos = ptr<std::int64_t>(state_.ring_pos);
  return p;
}

void Runtime::capture(int m, bool history, std::map<int, cudaGraphExec_t>& dest, bool greedy_only) {
  // warm up so the pointer tables and kernel attributes are set outside capture; the scratch writes to the
  // conv states and ring positions are undone afterwards (the cache rows it wrote are never valid)
  const Snapshot saved = snapshot();
  Buffers& b = buffers(m);
  for (int i = 0; i < 2; ++i) decoder_->decode(b, pos(), history, stream_, TpReduction::Reconstruct, greedy_only);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  if (tp_) tp_->check();
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
  try {
    decoder_->decode(b, pos(), history, stream_, TpReduction::Reconstruct, greedy_only);
  } catch (...) {
    cudaStreamEndCapture(stream_, &graph);
    if (graph) cudaGraphDestroy(graph);
    throw;
  }
  CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
  cudaGraphExec_t exec = nullptr;
  const cudaError_t err = cudaGraphInstantiate(&exec, graph, 0);
  cudaGraphDestroy(graph);
  CUDA_CHECK(err);
  CUDA_CHECK(cudaGraphUpload(exec, stream_));
  dest[m] = exec;
  restore(saved);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Runtime::set_loopcd_omega(float omega) {
  if (!d_loopcd_omega_.p || omega == loopcd_omega_) return;
  kernels::dense::set_loopcd_omega(static_cast<float*>(d_loopcd_omega_.p), omega, stream_);
  loopcd_omega_ = omega;
}

void Runtime::set_inputs(std::span<const std::int32_t> tokens, int position) {
  const int m = int(tokens.size());
  buffers(m);  // the width must exist
  set_step_inputs(tokens.data(), m, position, this->tokens(), pos(), stream_);
}

const float* Runtime::run_step(int m, bool greedy_only) {
  const bool certified = greedy_only && has_cert_head();
  const auto& graphs = certified ? certified_graphs_ :
      tp_ && tp_->full_logits && options_.tp.serve ? sampled_graphs_ : graphs_;
  auto it = graphs.find(m);
  if (it == graphs.end()) throw std::invalid_argument("no decode graph for width " + std::to_string(m));
  CUDA_CHECK(cudaGraphLaunch(it->second, stream_));
  return certified ? nullptr : static_cast<const float*>(buffers(m).logits.data);
}

const float* Runtime::run_step_eager(int m) {
  Buffers& b = buffers(m);
  decoder_->decode(b, pos(), history_.at(m), stream_);
  return static_cast<const float*>(b.logits.data);
}

const float* Runtime::decode_logits(std::int32_t token, int position) {
  const std::int32_t t[1] = {token};
  set_inputs(t, position);
  return run_step(1);
}

void Runtime::chain_advance(const std::int32_t* sampled, int position) {
  const int k = position & 1;
  model::chain_advance(sampled, tokens(), pos(), position, static_cast<std::int32_t*>(chain_host_.device_ptr()) + k,
                       state_ptrs(), undo_[k], stream_);
  CUDA_CHECK(cudaEventRecord(chain_done_[k], stream_));
}

std::int32_t Runtime::chain_token(int position) {
  const int k = position & 1;
  CUDA_CHECK(cudaEventSynchronize(chain_done_[k]));
  return static_cast<const volatile std::int32_t*>(chain_host_.data())[k];
}

void Runtime::chain_undo(int position) { model::chain_undo(state_ptrs(), undo_[position & 1], position, stream_); }

Runtime::Snapshot Runtime::snapshot() {
  Snapshot s;
  for (const Tensor* t : {&state_.ring_pos, &state_.conv_k, &state_.conv_v, &state_.conv_attn, &state_.conv_mlp})
    s.tensors.push_back(device_copy(*t, stream_));
  return s;
}

void Runtime::restore(const Snapshot& snap) {
  const Tensor* ts[5] = {&state_.ring_pos, &state_.conv_k, &state_.conv_v, &state_.conv_attn, &state_.conv_mlp};
  for (int i = 0; i < 5; ++i)
    CUDA_CHECK(cudaMemcpyAsync(ts[i]->data, snap.tensors[i].p, ts[i]->bytes(), cudaMemcpyDeviceToDevice, stream_));
}

namespace {
// the state tensors a prompt checkpoint holds, in checkpoint order
std::array<const Tensor*, 7> checkpoint_tensors(const State& st) {
  return {&st.k_ring, &st.v_ring, &st.ring_pos, &st.conv_k, &st.conv_v, &st.conv_attn, &st.conv_mlp};
}
}  // namespace

std::size_t Runtime::checkpoint_bytes() const {
  std::size_t n = 0;
  for (const Tensor* t : checkpoint_tensors(state_)) n += carved(t->bytes());
  return n;
}

void Runtime::save_checkpoint(void* dst) {
  char* p = static_cast<char*>(dst);
  for (const Tensor* t : checkpoint_tensors(state_)) {
    CUDA_CHECK(cudaMemcpyAsync(p, t->data, t->bytes(), cudaMemcpyDefault, stream_));
    p += carved(t->bytes());
  }
}

void Runtime::restore_checkpoint(const void* src) {
  const char* p = static_cast<const char*>(src);
  for (const Tensor* t : checkpoint_tensors(state_)) {
    CUDA_CHECK(cudaMemcpyAsync(t->data, p, t->bytes(), cudaMemcpyDefault, stream_));
    p += carved(t->bytes());
  }
}

std::size_t Runtime::full_rows_bytes(int n) const {
  if (n < 0 || n > options_.max_ctx)
    throw std::invalid_argument("full_rows_bytes: n = " + std::to_string(n) + " outside [0, max_ctx = " +
                                std::to_string(options_.max_ctx) + "]");
  const std::size_t row = state_.k_full.nb[2] + (state_.k_full_scale.data ? state_.k_full_scale.nb[2] : 0);
  return 2 * std::size_t(state_.loopcd.full_visits()) * row * std::size_t(n);
}

// One 2D copy per cache, with one row of n positions per full-attention visit, packed in
// the parked copy (K visits, V visits, then K and V side data for Q4 or KV12).
void Runtime::save_full_rows(void* dst, int n) {
  if (!full_rows_bytes(n)) return;
  char* p = static_cast<char*>(dst);
  for (const Tensor* t : {&state_.k_full, &state_.v_full, &state_.k_full_scale, &state_.v_full_scale}) {
    if (!t->data) continue;
    const std::size_t width = std::size_t(n) * t->nb[2];
    CUDA_CHECK(cudaMemcpy2DAsync(p, width, t->data, t->nb[3], width, state_.loopcd.full_visits(), cudaMemcpyDefault, stream_));
    p += width * state_.loopcd.full_visits();
  }
}

void Runtime::restore_full_rows(const void* src, int n) {
  if (!full_rows_bytes(n)) return;
  const char* p = static_cast<const char*>(src);
  for (const Tensor* t : {&state_.k_full, &state_.v_full, &state_.k_full_scale, &state_.v_full_scale}) {
    if (!t->data) continue;
    const std::size_t width = std::size_t(n) * t->nb[2];
    CUDA_CHECK(cudaMemcpy2DAsync(t->data, t->nb[3], p, width, width, state_.loopcd.full_visits(), cudaMemcpyDefault, stream_));
    p += width * state_.loopcd.full_visits();
  }
}

void Runtime::save_verify(int P, int m) {
  if (verify_widths_.empty()) throw std::logic_error("save_verify needs the speculative widths (options.spec)");
  if (m > 1) verify_save(state_ptrs(), backup_, nullptr, P, m, stream_);
}

void Runtime::rollback(int P, int commit, int m) {
  if (commit == m) return;
  if (commit < 1 || commit > m) throw std::invalid_argument("rollback: commit out of range");
  const Buffers& b = buffers(m);
  const float* hist[4] = {ptr<float>(b.hist_k), ptr<float>(b.hist_v), ptr<float>(b.hist_attn), ptr<float>(b.hist_mlp)};
  verify_rollback(state_ptrs(), backup_, hist, nullptr, P, nullptr, commit, m, stream_);
}

void Runtime::record_step(int m, cudaStream_t stream, bool greedy_only) {
  if (!graphs_.count(m)) throw std::invalid_argument("record_step: no decode graph for width " + std::to_string(m));
  decoder_->decode(buffers(m), pos(), history_.at(m), stream, TpReduction::Reconstruct, greedy_only);
}

void Runtime::record_save_verify(int m, cudaStream_t stream) {
  if (m == 1) return;
  if (verify_widths_.empty()) throw std::logic_error("record_save_verify needs the speculative widths (options.spec)");
  verify_save(state_ptrs(), backup_, pos(), 0, m, stream);
}

void Runtime::record_rollback(int m, const std::int32_t* accepted, cudaStream_t stream) {
  if (m == 1) return;
  if (!history_.count(m) || !history_.at(m)) throw std::invalid_argument("record_rollback: width without history");
  const Buffers& b = buffers(m);
  const float* hist[4] = {ptr<float>(b.hist_k), ptr<float>(b.hist_v), ptr<float>(b.hist_attn), ptr<float>(b.hist_mlp)};
  verify_rollback(state_ptrs(), backup_, hist, pos(), 0, accepted, 0, m, stream);
}

GraphExec Runtime::capture_switch(const std::function<void(cudaStream_t, cudaGraphConditionalHandle)>& head,
                                  const std::function<void(cudaStream_t, int)>& branch) {
  if (verify_widths_.empty()) throw std::logic_error("capture_switch needs the speculative widths (options.spec)");
  const cudaStream_t s = capture_stream_;
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaGraphCreate(&graph, 0));
  struct Owned {
    cudaGraph_t g;
    ~Owned() { cudaGraphDestroy(g); }
  } owned{graph};
  // run `body` captured into `g` (an exception ends the capture first)
  auto captured = [&](cudaGraph_t g, const std::function<void()>& body) {
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(s, g, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
    try {
      body();
    } catch (...) {
      cudaGraph_t out = nullptr;
      cudaStreamEndCapture(s, &out);
      throw;
    }
    cudaGraph_t out = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(s, &out));
  };

  cudaGraphConditionalHandle handle;
  CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0, 0));
  cudaGraph_t* bodies = nullptr;
  captured(graph, [&] {
    head(s, handle);
    cudaStreamCaptureStatus status;
    cudaGraph_t g = nullptr;
    const cudaGraphNode_t* deps = nullptr;
    const cudaGraphEdgeData* edges = nullptr;
    std::size_t n = 0;
    CUDA_CHECK(cudaStreamGetCaptureInfo(s, &status, nullptr, &g, &deps, &edges, &n));
    cudaGraphNodeParams p{};
    p.type = cudaGraphNodeTypeConditional;
    p.conditional.handle = handle;
    p.conditional.type = cudaGraphCondTypeSwitch;
    p.conditional.size = unsigned(verify_widths_.size());
    cudaGraphNode_t node = nullptr;
    CUDA_CHECK(cudaGraphAddNode(&node, g, deps, edges, n, &p));
    bodies = p.conditional.phGraph_out;
    CUDA_CHECK(cudaStreamUpdateCaptureDependencies(s, &node, nullptr, 1, cudaStreamSetCaptureDependencies));
  });
  for (std::size_t i = 0; i < verify_widths_.size(); ++i)
    captured(bodies[i], [&] { branch(s, verify_widths_[i]); });

  cudaGraphExec_t exec = nullptr;
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  GraphExec result(exec);
  CUDA_CHECK(cudaGraphUpload(exec, stream_));
  return result;
}

}  // namespace ingine::model
