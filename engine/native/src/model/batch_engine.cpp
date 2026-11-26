#include "model/batch_engine.h"

#include "core/device.h"
#include "kernels/batch/batch.h"
#include "model/batch_dflash2.h"
#include "model/batch_drafter.h"
#include "model/dflash2.h"
#include "model/prefill.h"
#include "model/runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ingine::model {
namespace {

constexpr int G = BatchDrafter::GAMMA;
constexpr int TAP_ROW = NUM_TAPS * HIDDEN;  // bf16 per row of taps

// the Gumbel noise of sampling call `draw`
std::uint64_t seed_of(std::uint64_t draw) { return draw * 0x9E3779B97F4A7C15ull + 0x2545F4914F6CDD1Dull; }

std::string resolve_batch_draft_type(const BatchEngineOptions& o) {
  std::string dt = o.batch_draft_type;
  if (dt.empty()) {
    if (const char* env = std::getenv("INGINE_BATCH_DRAFT_TYPE"); env && *env) dt = env;
    else if (!o.dflash2_dir.empty()) dt = "dflash2";
    else dt = "dspark";
  }
  if (dt != "dspark" && dt != "dflash2" && dt != "hybrid")
    throw std::invalid_argument("BatchEngine: batch_draft_type must be dspark, dflash2, or hybrid");
  return dt;
}

std::filesystem::path resolve_dflash2_dir(const BatchEngineOptions& o) {
  if (!o.dflash2_dir.empty()) return o.dflash2_dir;
  if (const char* env = std::getenv("INKLING_DFLASH2_DIR"); env && *env) return env;
  const auto sibling = o.model_dir.parent_path() / "Inkling-2-Mini-DFlash2";
  if (std::filesystem::exists(sibling / "model.safetensors")) return sibling;
  throw std::invalid_argument("BatchEngine: DFlash2 requires --dflash2-dir or INKLING_DFLASH2_DIR");
}

}  // namespace

BatchEngine::Stream::Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
BatchEngine::Stream::~Stream() {
  if (s) cudaStreamDestroy(s);
}

static std::unique_ptr<TpRuntime> make_batch_tp(const BatchEngineOptions& o) {
  if (o.tp.size <= 1) return nullptr;
  RuntimeOptions ro;
  ro.model_dir = o.model_dir;
  ro.max_ctx = o.ctx;
  ro.dense_fp8 = false;
  ro.kv_fp8 = o.kv_fp8;
  ro.head_nvfp4 = o.head_nvfp4;
  ro.spec = o.spec > 0;
  ro.tp = o.tp;
  ro.tp.full_logits = false;
  ro.tp.serve = false;
  return std::make_unique<TpRuntime>(ro);
}

BatchEngine::BatchEngine(const BatchEngineOptions& o)
    : ablated_(!o.ablation.empty()),
      prefill_chunk_(o.prefill_chunk),
      spec_(o.spec),
      spec_threshold_(o.spec_threshold),
      spec_min_(std::max(0, std::min(o.spec_min, o.spec))),
      tp_(make_batch_tp(o)),
      w_(load_weights(o.model_dir, /*dense_fp8=*/false, stream_.s, o.head_nvfp4,
                      tp_ ? tp_->layout : TpLayout{})),
      abl_(load_ablation(o.ablation)),
      st_(make_batch_state(o.slots, o.ctx, o.kv_fp8, tp_ ? tp_->layout : TpLayout{})),
      pf_(std::make_unique<Prefiller>(w_, abl_, o.prefill_chunk, stream_.s, tp_.get())),
      dec_(std::make_unique<BatchDecoder>(w_, abl_, st_, o.slots, stream_.s, 8, tp_.get())),
      chunk_(std::size_t(std::max(o.prefill_chunk, 1)) * 4),
      logits_(std::size_t(UNPADDED_VOCAB) * 4),
      first_(4) {
  if (spec_ < 0 || spec_ > G) throw std::invalid_argument("BatchEngine: spec must be in 0..7");
  if (!(spec_threshold_ >= 0.f && spec_threshold_ < 1.f))
    throw std::invalid_argument("BatchEngine: spec_threshold must be in [0, 1)");
  if (spec_ > 0) {
    batch_draft_type_ = resolve_batch_draft_type(o);
    const bool hybrid = batch_draft_type_ == "hybrid";
    if (hybrid) dflash2_max_ctx_ = batch_dflash2_max_ctx(std::max(1, o.dflash2_max_ctx), tp_.get(), stream_.s);
    const bool use_dspark = batch_draft_type_ == "dspark" || hybrid;
    const bool use_dflash2 = batch_draft_type_ == "dflash2" || (hybrid && dflash2_max_ctx_ > 0);
    if (use_dspark) {
      const bool shard_batch_drafter = [] {
        const char* e = std::getenv("INGINE_SHARD_BATCH_DRAFTER");
        return !e || !*e || std::atoi(e) != 0;
      }();
      if (!tp_ || shard_batch_drafter || tp_->layout.tp_rank == 0) {
        std::vector<DeviceBuffer> tmp_head;
        Dense full_head;
        if (tp_ || !w_.unembed.w.data || w_.unembed.fp8())
          full_head = load_head(o.model_dir, false, false, stream_.s, tmp_head).dense;
        drafter_ = std::make_unique<BatchDrafter>(o.model_dir, w_, o.slots, o.ctx, o.slots, stream_.s,
                                                  BatchDrafter::Precision::MXFP8, full_head.w.data ? &full_head : nullptr,
                                                  shard_batch_drafter ? tp_.get() : nullptr);
      }
    }
    if (use_dflash2) {
      if (tp_ && o.head_nvfp4 && !dflash2_full_head()) {
        dflash2_head_ = std::make_unique<BatchDFlash2Head>(*tp_, w_.unembed_nvfp4, o.slots, stream_.s);
      }
      if (!tp_ || tp_->layout.tp_rank == 0) {
        const auto dir = resolve_dflash2_dir(o);
        const kernels::head::Nvfp4Matrix* head_ovr = nullptr;
        kernels::head::Nvfp4Matrix loaded_head{};
        if (!dflash2_head_ && (tp_ || !w_.unembed_nvfp4.codes)) {
          loaded_head = load_head(o.model_dir, false, true, stream_.s, dflash2_head_storage_).nvfp4;
          head_ovr = &loaded_head;
        }
        dflash2_ = std::make_unique<BatchDFlash2>(dir, w_, o.slots, o.slots, stream_.s,
                                                  BatchDFlash2::default_precision(), BatchDFlash2::default_kv_fp8(),
                                                  head_ovr, dflash2_head_.get());
      }
    }
    taps_chunk_ = DeviceBuffer(std::size_t(prefill_chunk_) * TAP_ROW * 2);
    gathered_ = DeviceBuffer(std::size_t(o.slots) * 8 * TAP_ROW * 2);
    rows_ = DeviceBuffer(std::size_t(o.slots) * 8 * 4);
    if (tp_) draft_bcast_ = DeviceBuffer(std::size_t(o.slots) * G * 8);
    drafts_.assign(std::size_t(o.slots) * G, 0);
    conf_.assign(std::size_t(o.slots) * G, 0.f);
    if (!o.spec_trace.empty() && !(trace_ = std::fopen(o.spec_trace.c_str(), "w")))
      throw std::runtime_error("BatchEngine: cannot write " + o.spec_trace.string());
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_.s));
}

BatchEngine::~BatchEngine() {
  cudaStreamSynchronize(stream_.s);
  if (trace_) std::fclose(trace_);
}

std::string BatchEngine::model_id() const { return ablated_ ? "inkling-2-mini-abliterated" : "inkling-2-mini"; }

std::size_t BatchEngine::device_bytes() const {
  std::size_t extra = 0;
  for (const auto& b : dflash2_head_storage_) extra += b.bytes;
  return st_.device_bytes() + dec_->device_bytes() + pf_->device_bytes() + chunk_.bytes + logits_.bytes + first_.bytes +
         (drafter_ ? drafter_->device_bytes() : 0) + (dflash2_ ? dflash2_->device_bytes() : 0) +
         (dflash2_head_ ? dflash2_head_->device_bytes() : 0) + extra + taps_chunk_.bytes + gathered_.bytes + rows_.bytes;
}

std::vector<std::int32_t> BatchEngine::admit(std::span<const Admission> requests) {
  std::vector<std::int32_t> first(requests.size());
  std::vector<std::size_t> together;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const Admission& a = requests[i];
    if (a.slot < 0 || a.slot >= st_.slots) throw std::invalid_argument("BatchEngine::admit: slot out of range");
    const int P = int(a.prompt.size());
    if (P < 1 || P + lookahead() + 1 > st_.ctx)
      throw std::invalid_argument("BatchEngine::admit: a prompt of " + std::to_string(P) +
                                  " tokens does not fit a slot of " + std::to_string(st_.ctx) + " positions");
    reset_slot(st_, a.slot, stream_.s);
    if (drafter_) drafter_->reset_slot(a.slot);
    if (dflash2_) dflash2_->reset_slot(a.slot);
    if (P <= kStepPrefill && P + 8 <= st_.ctx) together.push_back(i);
    else prefill_one(a, first[i]);
  }
  if (!together.empty()) prefill_together(requests, together, first);
  CUDA_CHECK(cudaStreamSynchronize(stream_.s));
  if (spec_ > 0 && !requests.empty()) {  // drafts after the first tokens
    const int n = int(requests.size());
    slot_.resize(std::size_t(n)), anchor_.resize(std::size_t(n)), next_pos_.resize(std::size_t(n));
    for (int i = 0; i < n; ++i)
      slot_[i] = requests[i].slot, anchor_[i] = first[i], next_pos_[i] = std::int64_t(requests[i].prompt.size());
    draft(n);
  }
  return first;
}

void BatchEngine::prefill_one(const Admission& a, std::int32_t& first) {
  const cudaStream_t s = stream_.s;
  const int P = int(a.prompt.size());
  State v = st_.view(a.slot);
  const bool sampled = a.temperature > 0.f;
  for (int p = 0; p < P; p += prefill_chunk_) {
    const int T = std::min(prefill_chunk_, P - p);
    const bool final_chunk = (p + T == P);
    CUDA_CHECK(cudaMemcpyAsync(chunk_.p, a.prompt.data() + p, std::size_t(T) * 4, cudaMemcpyHostToDevice, s));
    Tensor taps = rm(taps_chunk_.p, DType::BF16, {T, NUM_TAPS, HIDDEN});
    if (tp_) tp_->full_logits = sampled;
    float* log_dst = (!tp_ || (sampled && tp_->layout.tp_rank == 0)) ? static_cast<float*>(logits_.p) : nullptr;
    std::int32_t* grd_dst = (tp_ && !sampled) ? static_cast<std::int32_t*>(first_.p) : nullptr;
    const bool need_taps = drafter_ || dflash2_;
    pf_->run(v, static_cast<const std::int32_t*>(chunk_.p), T, p, log_dst,
             need_taps ? &taps : nullptr, {}, grd_dst, final_chunk);
    if (need_taps) {
      const std::int32_t slot = a.slot, count = T;
      const std::int64_t start = p;
      const auto* tp_ptr = static_cast<const __nv_bfloat16*>(taps_chunk_.p);
      if (drafter_) drafter_->commit(1, &slot, &start, &count, tp_ptr);
      if (dflash2_) dflash2_->commit(1, &slot, &start, &count, tp_ptr);
    }
  }
  const std::uint64_t seed = seed_of(++draws_);
  if (!tp_) {
    kernels::batch::sample(static_cast<const float*>(logits_.p), UNPADDED_VOCAB, 1, UNPADDED_VOCAB, false, a.temperature,
                           seed, static_cast<std::int32_t*>(first_.p), s);
  } else if (sampled) {
    if (tp_->layout.tp_rank == 0)
      kernels::batch::sample(static_cast<const float*>(logits_.p), UNPADDED_VOCAB, 1, UNPADDED_VOCAB, false, a.temperature,
                             seed, static_cast<std::int32_t*>(first_.p), s);
    tp_->broadcast_proposal(first_.p, 4, s);
  }
  CUDA_CHECK(cudaMemcpyAsync(&first, first_.p, 4, cudaMemcpyDeviceToHost, s));
  if (tp_) tp_->wait(s);
}

void BatchEngine::prefill_together(std::span<const Admission> requests, std::span<const std::size_t> which,
                                   std::vector<std::int32_t>& first) {
  // per step, every prompt with tokens left feeds its next min(8, left) of them
  std::vector<int> done(which.size(), 0);
  std::vector<std::size_t> member;  // the step's sequence i is which[member[i]]
  for (;;) {
    tok_.clear(), slot_.clear(), pos_.clear(), temp_.clear(), m_.clear(), member.clear();
    for (std::size_t j = 0; j < which.size(); ++j) {
      const Admission& a = requests[which[j]];
      const int left = int(a.prompt.size()) - done[j];
      if (left <= 0) continue;
      const int k = std::min(left, 8);
      tok_.insert(tok_.end(), a.prompt.begin() + done[j], a.prompt.begin() + done[j] + k);
      slot_.push_back(a.slot), pos_.push_back(done[j]), temp_.push_back(a.temperature), m_.push_back(k);
      member.push_back(j);
    }
    const int n = int(slot_.size());
    if (n == 0) break;
    const auto got = dec_->step(n, m_.data(), tok_.data(), slot_.data(), pos_.data(), temp_.data(), seed_of(++draws_));
    offsets(n);
    for (int i = 0; i < n; ++i) {
      const std::size_t j = member[std::size_t(i)];
      done[j] += m_[std::size_t(i)];
      if (done[j] == int(requests[which[j]].prompt.size()))
        first[which[j]] = got[std::size_t(off_[std::size_t(i) + 1]) - 1];
    }
    keep_ = m_;
    dec_->accept(keep_.data());  // every prompt row is kept
    if (drafter_ || dflash2_) commit_kept(n);
  }
}

void BatchEngine::offsets(int n) {
  off_.resize(std::size_t(n) + 1);
  off_[0] = 0;
  for (int i = 0; i < n; ++i) off_[std::size_t(i) + 1] = off_[std::size_t(i)] + m_[std::size_t(i)];
}

void BatchEngine::commit_kept(int n) {
  if (!drafter_ && !dflash2_) return;
  rows_host_.clear();
  for (int i = 0; i < n; ++i)
    for (int r = 0; r < keep_[std::size_t(i)]; ++r) rows_host_.push_back(off_[std::size_t(i)] + r);
  CUDA_CHECK(cudaMemcpyAsync(rows_.p, rows_host_.data(), rows_host_.size() * 4, cudaMemcpyHostToDevice, stream_.s));
  kernels::batch::gather_rows(dec_->taps(), TAP_ROW, static_cast<const std::int32_t*>(rows_.p), int(rows_host_.size()),
                              static_cast<__nv_bfloat16*>(gathered_.p), stream_.s);
  const auto* gptr = static_cast<const __nv_bfloat16*>(gathered_.p);
  if (drafter_) drafter_->commit(n, slot_.data(), pos_.data(), keep_.data(), gptr);
  if (dflash2_) dflash2_->commit(n, slot_.data(), pos_.data(), keep_.data(), gptr);
}

void BatchEngine::draft(int n) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::size_t cnt = std::size_t(n) * G;
  draft_host_.resize(cnt), conf_host_.resize(cnt);

  // Partition indices into DFlash2 vs DSpark subsets (in dflash2 mode all -> df, in dspark mode all -> ds).
  std::vector<int> df_idx, ds_idx;
  if (batch_draft_type_ == "dflash2") {
    df_idx.resize(std::size_t(n));
    for (int i = 0; i < n; ++i) df_idx[std::size_t(i)] = i;
  } else if (batch_draft_type_ == "hybrid") {
    for (int i = 0; i < n; ++i) {
      if (dflash2_max_ctx_ > 0 && next_pos_[std::size_t(i)] <= dflash2_max_ctx_) df_idx.push_back(i);
      else ds_idx.push_back(i);
    }
  } else {
    ds_idx.resize(std::size_t(n));
    for (int i = 0; i < n; ++i) ds_idx[std::size_t(i)] = i;
  }

  auto gather_sub = [&](const std::vector<int>& idx, std::vector<std::int32_t>& sl, std::vector<std::int32_t>& an,
                        std::vector<std::int64_t>& np) {
    sl.resize(idx.size()), an.resize(idx.size()), np.resize(idx.size());
    for (std::size_t k = 0; k < idx.size(); ++k) {
      const int i = idx[k];
      sl[k] = slot_[std::size_t(i)];
      an[k] = anchor_[std::size_t(i)];
      np[k] = next_pos_[std::size_t(i)];
    }
  };

  auto scatter_sync = [&](const std::vector<int>& idx, const std::int32_t* dev_tok, const float* dev_conf) {
    const std::size_t sub_cnt = idx.size() * G;
    std::vector<std::int32_t> dt(sub_cnt);
    std::vector<float> dc(sub_cnt);
    if (tp_) {
      auto* base = static_cast<std::byte*>(draft_bcast_.p);
      if (tp_->layout.tp_rank == 0 && dev_tok) {
        CUDA_CHECK(cudaMemcpyAsync(base, dev_tok, sub_cnt * 4, cudaMemcpyDeviceToDevice, stream_.s));
        CUDA_CHECK(cudaMemcpyAsync(base + sub_cnt * 4, dev_conf, sub_cnt * 4, cudaMemcpyDeviceToDevice, stream_.s));
      }
      tp_->broadcast_proposal(draft_bcast_.p, sub_cnt * 8, stream_.s);
      CUDA_CHECK(cudaMemcpyAsync(dt.data(), base, sub_cnt * 4, cudaMemcpyDeviceToHost, stream_.s));
      CUDA_CHECK(cudaMemcpyAsync(dc.data(), base + sub_cnt * 4, sub_cnt * 4, cudaMemcpyDeviceToHost, stream_.s));
      tp_->wait(stream_.s);
    } else {
      CUDA_CHECK(cudaMemcpyAsync(dt.data(), dev_tok, sub_cnt * 4, cudaMemcpyDeviceToHost, stream_.s));
      CUDA_CHECK(cudaMemcpyAsync(dc.data(), dev_conf, sub_cnt * 4, cudaMemcpyDeviceToHost, stream_.s));
      CUDA_CHECK(cudaStreamSynchronize(stream_.s));
    }
    for (std::size_t k = 0; k < idx.size(); ++k) {
      const int sl = slot_[std::size_t(idx[k])];
      std::copy_n(dt.begin() + std::ptrdiff_t(k) * G, G, drafts_.begin() + std::ptrdiff_t(sl) * G);
      std::copy_n(dc.begin() + std::ptrdiff_t(k) * G, G, conf_.begin() + std::ptrdiff_t(sl) * G);
    }
  };

  if (!df_idx.empty()) {
    const int ndf = int(df_idx.size());
    if (dflash2_) {
      std::vector<std::int32_t> sl, an;
      std::vector<std::int64_t> np;
      gather_sub(df_idx, sl, an, np);
      const BatchDFlash2::Drafts d = dflash2_->propose(ndf, sl.data(), an.data(), np.data());
      scatter_sync(df_idx, d.tokens, d.conf);
    } else if (tp_) {
      if (dflash2_head_) dflash2_head_->run(ndf, nullptr);
      scatter_sync(df_idx, nullptr, nullptr);
    }
  }

  if (!ds_idx.empty()) {
    const int nds = int(ds_idx.size());
    if (drafter_) {
      std::vector<std::int32_t> sl, an;
      std::vector<std::int64_t> np;
      gather_sub(ds_idx, sl, an, np);
      const BatchDrafter::Drafts d =
          drafter_->draft(nds, sl.data(), an.data(), np.data(), 0.f, seed_of(++draws_), spec_);
      scatter_sync(ds_idx, d.tokens, d.conf);
    } else if (tp_) {
      ++draws_;
      scatter_sync(ds_idx, nullptr, nullptr);
    }
  }
  draft_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ++draft_calls_;
}

void BatchEngine::step(std::span<Seq> seqs, std::vector<std::vector<std::int32_t>>& emitted) {
  const int n = int(seqs.size());
  emitted.resize(std::size_t(n));
  if (n == 0) return;
  // drafts verified per sequence: spec, or with spec_threshold the leading drafts whose estimated acceptance (the
  // product of their confidences) exceeds the threshold scaled by the running share of the slots (a row costs the
  // same, a step's tokens fall with fewer sequences), at most spec
  m_.resize(std::size_t(n));
  const float theta = std::max(0.05f, spec_threshold_ * float(n) / float(st_.slots));
  for (int i = 0; i < n; ++i) {
    int k = spec_;
    if (spec_threshold_ > 0.f) {
      const float* c = conf_.data() + std::size_t(seqs[i].slot) * G;
      float p = 1.f;
      k = 0;
      while (k < spec_ && ((p *= c[k]) > theta || k < spec_min_)) ++k;
    }
    m_[std::size_t(i)] = k + 1;
  }
  tok_.clear(), slot_.resize(std::size_t(n)), pos_.resize(std::size_t(n)), temp_.resize(std::size_t(n));
  for (int i = 0; i < n; ++i) {
    const Seq& q = seqs[i];
    tok_.push_back(q.pending);
    for (int k = 1; k < m_[std::size_t(i)]; ++k) tok_.push_back(drafts_[std::size_t(q.slot) * G + std::size_t(k - 1)]);
    slot_[i] = q.slot, pos_[i] = q.pos, temp_[i] = q.temperature;
  }
  const auto tv0 = std::chrono::steady_clock::now();
  const auto got =
      dec_->step(n, m_.data(), tok_.data(), slot_.data(), pos_.data(), temp_.data(), seed_of(++draws_));
  verify_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - tv0).count();
  offsets(n);
  if (spec_ == 0) {
    for (int i = 0; i < n; ++i) {
      emitted[i].assign(1, got[std::size_t(i)]);
      seqs[i].pending = got[std::size_t(i)];
      seqs[i].pos += 1;
    }
    return;
  }
  keep_.resize(std::size_t(n)), anchor_.resize(std::size_t(n)), next_pos_.resize(std::size_t(n));
  for (int i = 0; i < n; ++i) {
    const int K = m_[std::size_t(i)] - 1;                            // this sequence's drafts
    const std::int32_t* row = tok_.data() + off_[std::size_t(i)];    // pending, drafts
    const std::int32_t* out = got.data() + off_[std::size_t(i)];     // the target's token after each row
    int a = 0;
    while (a < K && row[1 + a] == out[a]) ++a;
    emitted[i].assign(row + 1, row + 1 + a);
    emitted[i].push_back(out[a]);
    keep_[i] = a + 1;
    spec_accepted_ += std::uint64_t(a);
    spec_rows_ += std::uint64_t(K + 1);
    if (trace_) {
      const float* c = conf_.data() + std::size_t(slot_[i]) * G;
      std::fprintf(trace_, "%d %d %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", slot_[i], K, a, c[0], c[1], c[2], c[3], c[4],
                   c[5], c[6]);
    }
    seqs[i].pending = anchor_[i] = out[a];
    seqs[i].pos = next_pos_[i] = pos_[i] + a + 1;
  }
  spec_steps_ += std::uint64_t(n);
  dec_->accept(keep_.data());
  commit_kept(n);
  draft(n);
}

}  // namespace ingine::model
