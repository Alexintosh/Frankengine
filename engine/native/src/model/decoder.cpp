// fast.FastDecoder.decode with fast.KernelOps: the fixed kernel sequence of one decode step for M
// consecutive tokens (48 visits: attention prologue mix (+ message gates), proj, attn, proj_conv, MLP
// prologue mix (+ gate), router, moe_up, moe_down; then the final read, norm and head).
#include "model/runtime.h"

#include "core/device.h"
#include "kernels/attn/attn.h"
#include "kernels/dense/dense.h"
#include "kernels/moe/moe.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ingine::model {
namespace dense = kernels::dense;
namespace moe = kernels::moe;
namespace attn = kernels::attn;

namespace {

bool fuse_mlp_prologue_enabled() {
  if (const char* flag = std::getenv("INGINE_DECODER_FUSE_MLP_PROLOGUE")) {
    if (std::strcmp(flag, "0") == 0) return false;
    if (std::strcmp(flag, "1") == 0) return true;
  }
  if (const char* rma = std::getenv("INGINE_ROUTER_MMA"); rma && rma[0] == '0') return false;
  return dense::mlp_prologue_default_enabled();
}

const float* f32p(const Tensor& t) { return static_cast<const float*>(t.data); }
float* f32m(const Tensor& t) { return static_cast<float*>(t.data); }
__nv_bfloat16* bf16m(const Tensor& t) { return static_cast<__nv_bfloat16*>(t.data); }

// fast.proj_layout(full, single_input) as an int32 [S, 3] table (row_begin, row_end, input)
std::vector<int> proj_layout(bool full, bool single_input, const TpLayout& tp) {
  std::vector<int> t;
  for (int s = 0; s < num_segs(full); ++s) {
    t.push_back(tp.weight_offset(s));
    t.push_back(tp.weight_offset(s + 1));
    t.push_back(single_input ? 0 : SEG_INPUT[s]);
  }
  return t;
}

DeviceBuffer upload_ints(const std::vector<int>& v) {
  DeviceBuffer b(v.size() * sizeof(int));
  b.copy_from_host(v.data(), v.size() * sizeof(int));
  return b;
}

}  // namespace

Decoder::Decoder(const Weights& w, const Ablation& abl, State& st, bool pdl, TpSimulation* simulation, TpRuntime* tp,
                 const float* loopcd_omega)
    : w_(w), abl_(abl), st_(st),
      pdl_(pdl && !simulation && (!tp || (tp->gb10_spec && !tp->profile))),
      simulation_(simulation), tp_(tp), loopcd_omega_(loopcd_omega),
      down_workspace_(tp && tp->gb10_spec ? sizeof(float) * std::size_t(8) * TOP_K * HIDDEN : 0),
      cert_head_ws_(!simulation && !tp && !st.loopcd.uses_ref_coda() && w.unembed_certified.bounds
          ? kernels::head::cert_head_workspace(w.unembed.rows()) : 0),
      attn_ws_(simulation ? 0 : attn::workspace_bytes(8, tp ? tp->layout.hkv_local : KV_HEADS)) {
  validate_loopcd_tp(st.loopcd, tp ? tp->layout.tp_size : simulation ? 2 : 1);
  if (st.loopcd.auto_mode() && !loopcd_omega_)
    throw std::invalid_argument("Decoder requires a device weight for LoopCD Auto");
  if (simulation_) return;
  attn_ws_.fill(0);  // zeroed once; the kernel resets its counter itself
  const TpLayout layout = tp_ ? tp_->layout : TpLayout{};
  segments_[0] = upload_ints(proj_layout(false, true, layout));
  segments_[1] = upload_ints(proj_layout(false, false, layout));
  full_segments_ = upload_ints(proj_layout(true, false, layout));
}

Decoder::~Decoder() = default;

const std::int64_t* Decoder::block_table(const std::vector<const float*>& blocks) {
  auto it = tables_.find(blocks);
  if (it == tables_.end()) {
    // built on first use, i.e. in the eager warm-up before capture (KernelOps._block_ptrs)
    DeviceBuffer b(std::max<std::size_t>(blocks.size(), 1) * sizeof(void*));
    if (!blocks.empty()) b.copy_from_host(blocks.data(), blocks.size() * sizeof(void*));
    it = tables_.emplace(blocks, std::move(b)).first;
  }
  return static_cast<const std::int64_t*>(it->second.p);
}

void Decoder::decode(Buffers& b, const std::int64_t* pos, bool history, cudaStream_t stream,
                     TpReduction reduction, bool greedy_only) {
  decode_impl(b, pos, history, stream, reduction, nullptr, greedy_only);
}

void Decoder::allocate_batch_workspace(int max_batch) {
  if (simulation_ || max_batch < 2 || max_batch > 8)
    throw std::invalid_argument("invalid batch workspace configuration");
  const int hkv = tp_ ? tp_->layout.hkv_local : KV_HEADS;
  batch_attn_ws_ = DeviceBuffer(max_batch * attn::workspace_bytes(1, hkv));
  batch_attn_ws_.fill(0);
}

void Decoder::decode_batch(Buffers& b, const BatchSlotTable* slots, cudaStream_t stream, bool greedy_only) {
  if (simulation_ || !slots || b.m < 1 || b.m > 8 || !batch_attn_ws_.p)
    throw std::invalid_argument("invalid batch decode configuration");
  decode_impl(b, nullptr, false, stream, TpReduction::Reconstruct, slots, greedy_only);
}

void Decoder::decode_impl(Buffers& b, const std::int64_t* pos, bool history, cudaStream_t stream,
                          TpReduction reduction, const BatchSlotTable* slots, bool greedy_only, int loop_passes) {
  const bool profile = tp_ && tp_->profile;
  auto stage = [&](VerifyStage kind, int occurrence, bool end) {
    if (profile) tp_->mark_stage(kind, occurrence, end, stream);
  };
  if (profile) tp_->mark(0, stream);
  const TpLayout layout = tp_ ? tp_->layout : TpLayout{};
  const int m = b.m;
  const int H = HIDDEN;
  const bool pdl = pdl_;
  const bool fuse_mlp = !simulation_ && !profile && m >= 2 && m <= 8 && fuse_mlp_prologue_enabled();
  const Weights& w = w_;
  const Ablation& abl = abl_;
  State& st = st_;
  const auto& loopcd = st.loopcd;
  const bool uses_ref_coda = loopcd.uses_ref_coda();
  if (uses_ref_coda && !b.ref_logits.data)
    throw std::invalid_argument("Decoder requires reference coda buffers for this state");
  const auto* ref_slots = slots && uses_ref_coda ? reinterpret_cast<const BatchRefSlotTable*>(slots + 1) : nullptr;
  const std::size_t slab = std::size_t(m) * H;  // one [m, H] slab of a [G, m, H] buffer

  auto reduce_conv = [&](float* fp32, __nv_bfloat16* bf16, int visit, dense::ConvKind kind,
                         const Tensor& weights, float* state, float* out, float* hist) {
    const unsigned op = 2 * visit + (kind == dense::ConvKind::Mlp);
    if (tp_) tp_->reduce(fp32, bf16, slab, TpSchedule::Decode, op, stream);
    stage(VerifyStage::Conv, op, false);
    if (slots) {
      const bool ref = visit >= NUM_VISITS;
      const int r = visit - NUM_VISITS;
      auto states = kind == dense::ConvKind::Mlp
          ? (ref ? ref_slots->mlp_conv[r] : slots->mlp_conv[visit])
          : (ref ? ref_slots->attn_conv[r] : slots->attn_conv[visit]);
      dense::conv_residual_epilogue_batch(fp32, bf16, m, kind, f32p(weights), states, out, pdl, stream);
    } else {
      dense::conv_residual_epilogue(fp32, bf16, m, kind, f32p(weights), state, out, hist, pdl, stream);
    }
    stage(VerifyStage::Conv, op, true);
  };

  auto mix_params = [&](const std::vector<const float*>& blocks, const float* base, const float* write,
                        const float* abl_r, const float* abl_w, float* partial_out, const Tensor* queries,
                        const float* norm_w, __nv_bfloat16* out_bf16, float* out_f32) {
    dense::MixParams p;
    p.blocks = blocks.empty() ? nullptr : reinterpret_cast<const float* const*>(block_table(blocks));
    p.nblocks = int(blocks.size());
    p.base = base;
    p.write = write;
    // KernelOps.mix: the ablation applies to `write` only
    p.abl_r = write ? abl_r : nullptr;
    p.abl_d = write && abl.detect ? abl_r + std::size_t(NUM_VISITS) * H : nullptr;  // the detector of the same visit
    p.abl_w = write ? abl_w : nullptr;
    p.partial_out = partial_out;
    p.queries = queries ? f32p(*queries) : nullptr;
    p.Q = queries ? queries->ne[1] : 1;
    p.norm_w = norm_w;
    p.out_bf16 = out_bf16;
    p.out_f32 = out_f32;
    p.M = m;
    return p;
  };

  dense::embed(static_cast<const std::int64_t*>(b.token.data), m, static_cast<const __nv_bfloat16*>(w.embed.data),
               f32p(w.embed_norm), f32m(b.B0), pdl, stream);
  int prev_v = -1;
  int tap = 0;
  const float* attn_dirs = f32p(abl.attn_dirs);
  const float* attn_w = f32p(abl.attn_w);
  const float* mlp_dirs = f32p(abl.mlp_dirs);
  const float* mlp_w = f32p(abl.mlp_w);
  __nv_bfloat16* taps = bf16m(b.taps);

  const int main_pass = loop_passes - 1;
  auto finish = [&](bool reference) {
    const int coda_pass = reference ? loopcd.ref_idx : main_pass;
    const Tensor final_norm1 = rm(w.final_norm.data, DType::FP32, {1, H});
    auto* xf = bf16m(reference ? b.xm : b.xf);
    dense::MixParams fp = mix_params(b.blocks(LAYERS, 0, coda_pass), f32p(b.part), f32p(b.y2),
                                    mlp_dirs + std::size_t(prev_v) * H, mlp_w + 4 * std::size_t(prev_v),
                                    nullptr, &w.final_query, f32p(final_norm1), xf, nullptr);
    if (!reference) {
      fp.out_copy = taps + std::size_t(NUM_TAPS - 1) * H;
      fp.out_copy_stride = std::int64_t(NUM_TAPS) * H;
    }
    dense::mix(fp, pdl, stream);
    auto* logits = f32m(reference ? b.ref_logits : b.logits);
    if (simulation_)
      simulation_->head(b, stream);
    else if (tp_)
      tp_->head(w, xf, logits, ptr<int32_t>(b.greedy), m, TpSchedule::Decode, stream);
    else if (!uses_ref_coda && greedy_only && w.unembed_certified.bounds)
      kernels::head::cert_head(xf, m, w.unembed.matrix(), w.unembed_certified, logits,
                               cert_head_ws_.p, ptr<int32_t>(b.greedy), stream);
    else if (w.unembed_nvfp4.codes)
      kernels::head::head_nvfp4(xf, m, w.unembed_nvfp4, logits, nullptr, nullptr, pdl, stream);
    else
      dense::head(xf, m, w.unembed.matrix(), logits, pdl, stream);
  };

  const int coda_before = LOOP_START + loop_passes * (LOOP_END - LOOP_START);
  const int visits = loopcd.visits();
  for (int v = 0; v < visits; ++v) {
    if (loop_passes < PASSES && v == coda_before) v = CODA_FIRST;  // shortened forward used by the independent test oracle
    if (v == NUM_VISITS) finish(false);
    const bool reference = v >= NUM_VISITS;
    const int coda_pass = reference ? loopcd.ref_idx : main_pass;
    const int wv = weight_visit(v);
    const int l = VISITS[wv].layer, p = VISITS[wv].pass;
    const Layer& layer = w.layers[l];
    const bool loop = is_loop_layer(l);
    const bool msgs = loop && p > 0;
    const bool boundary = l % BLOCK == 0;
    const std::vector<const float*> blocks = b.blocks(l, p, coda_pass);

    // ---- hidden entering this visit (the previous MLP write applied), stored where it is a block
    float* hid;
    const float* base;
    const float* write;
    if (reference && l == LOOP_END) {
      // E[ref] already includes the last loop MLP write, completed at the next pass's layer 6.
      hid = f32m(b.E[coda_pass]), base = hid, write = nullptr;
    } else if (v == 0) {
      hid = f32m(b.B0), base = f32p(b.B0), write = nullptr;
    } else if (l == LOOP_START && p > 0) {
      hid = f32m(b.B1), base = f32p(b.B1), write = nullptr;
    } else {
      hid = l == 6 ? f32m(b.B1) : l == 12 ? f32m(b.B2[p]) : l == LOOP_END ? f32m(b.E[coda_pass]) : f32m(b.hid);
      base = f32p(b.part), write = f32p(b.y2);
    }
    const int pv = prev_v >= 0 ? prev_v : 0;
    const float* r_prev = mlp_dirs + std::size_t(pv) * H;
    const float* w_prev = mlp_w + 4 * std::size_t(pv);
    if (l == LOOP_END && (loopcd.hidden() || loopcd.auto_mode())) {
      if (abl.active) {
        dense::mix(mix_params({}, base, write, r_prev, w_prev, hid, nullptr, nullptr, nullptr, nullptr), pdl, stream);
        base = hid, write = nullptr;
      }
      const float* ref_stream = f32p(b.E[loopcd.ref_idx]);
      float* depth = f32m(b.B2[main_pass]);
      const float* ref_depth = f32p(b.B2[loopcd.ref_idx]);
      if (!loopcd.auto_mode())
        dense::loopcd_hidden(base, write, ref_stream, hid, depth, ref_depth, slab, loopcd.omega, stream);
      else if (slots)
        dense::loopcd_hidden_rows(base, write, ref_stream, hid, depth, ref_depth, m, H,
                                  reinterpret_cast<const float*>(slots + 1), stream);
      else
        dense::loopcd_hidden_device(base, write, ref_stream, hid, depth, ref_depth, slab, loopcd_omega_, stream);
      base = hid, write = nullptr;
    }
    dense::MixParams msg_mix;  // message visits: launched together with the attention read (independent of it)
    if (msgs) {
      const int i = l - LOOP_START;
      const float* src_base = i == 0 ? f32p(b.part) : f32p(b.E[p - 1]);
      const float* src_write = i == 0 ? f32p(b.y2) : nullptr;
      // messages from pass p-1: its layer-12 block and its final hidden (E[p-1], completed at i == 0)
      const Tensor mq = rm(f32m(w.message_queries) + std::size_t(i) * 4 * H, DType::FP32, {4, H});
      msg_mix = mix_params({f32p(b.B2[p - 1])}, src_base, src_write, r_prev, w_prev,
                           i == 0 ? f32m(b.E[p - 1]) : nullptr, &mq, nullptr, nullptr, f32m(b.msgs));
      if (i == 0) write = nullptr;  // pass p-1's last write went into E[p-1]; layer 6 restarts from B1
    }
    // ---- attention read(s); at a tap visit the same launch writes b.taps[:, tap] = bf16(hid)
    const Tensor* queries = layer.attn_query.data ? &layer.attn_query : nullptr;
    float* partial_out = hid != base ? hid : nullptr;
    dense::MixParams ap =
        msgs ? mix_params(blocks, base, write, r_prev, w_prev, partial_out, queries, nullptr, nullptr, f32m(b.mixf))
             : mix_params(blocks, base, write, r_prev, w_prev, partial_out, l > 0 ? queries : nullptr,
                          f32p(layer.attn_norm_mix), bf16m(b.xin), nullptr);
    if (tap < 3 && wv == TAP_VISITS[tap]) {
      ap.new_bf16 = taps + std::size_t(tap) * H;
      ap.new_bf16_stride = std::int64_t(NUM_TAPS) * H;
      ++tap;
    }
    stage(VerifyStage::Mix, 2 * v, false);
    if (msgs) dense::mix_pair(ap, msg_mix, pdl, stream);
    else dense::mix(ap, pdl, stream);
    stage(VerifyStage::Mix, 2 * v, true);
    if (msgs) {  // sliding layers mix one query: all three gates read b.mixf[0]
      stage(VerifyStage::MessageGates, 2 * v, false);
      dense::message_gates(f32p(b.mixf), !layer.full, f32p(b.msgs), 3, m,
                           static_cast<const __nv_bfloat16*>(layer.qkv_gate_down.data), layer.qkv_gate_forget16.data,
                           layer.qkv_gate_gain16.data, false, f32p(layer.attn_norm3), bf16m(b.xin), nullptr, pdl,
                           stream);
      stage(VerifyStage::MessageGates, 2 * v, true);
    }
    // ---- attention
    const bool single = !layer.full && !msgs;
    const Tensor& proj_out = layer.full ? b.proj_full : b.proj_sliding;
    const int R = layout.projection_stride(layer.full);
    if (simulation_) {
      simulation_->attention(layer, b, v, single, history, pos, reduction, stream);
    } else {
      const DeviceBuffer& seg = layer.full ? full_segments_ : segments_[single ? 0 : 1];
      stage(VerifyStage::Wproj, v, false);
      dense::proj(bf16m(b.xin), m, layer.wproj.matrix(), static_cast<const int*>(seg.p), num_segs(layer.full),
                  f32m(proj_out), pdl, stream);
      stage(VerifyStage::Wproj, v, true);
      // b.hist_k/hist_v[v] (K and V segments of proj_out) are copied by the MLP-side mix below
      attn::DecodeArgs a;
      a.proj = f32p(proj_out);
      a.M = slots ? 1 : m;
      a.hkv = layout.hkv_local;
      a.q_scale = f32p(layer.q_scale);
      a.r_scale = f32p(layer.r_scale);
      a.rel_proj = f32p(layer.rel_proj);
      a.conv_k_state = st.conv_k_at(v);
      a.conv_v_state = st.conv_v_at(v);
      a.pos = pos;
      auto& workspace = slots ? batch_attn_ws_ : attn_ws_;
      a.workspace = workspace.p;
      a.workspace_size = workspace.bytes;
      a.out = bf16m(b.attn_out);
      if (layer.full) {
        const int fi = full_cache_index(v);
        a.k_conv = f32p(layer.k_static);
        a.v_conv = f32p(layer.v_static);
        a.k_cache = st.k_full_at(fi).data;
        a.v_cache = st.v_full_at(fi).data;
        a.cache_fp8 = st.kv_fp8;
        a.cache_q4 = st.kv_q4;
        a.cache_kv12 = st.kv_12;
        if (a.cache_q4 || a.cache_kv12) {
          a.k_scale = st.k_full_scale_at(fi).data;
          a.v_scale = st.v_full_scale_at(fi).data;
        }
        a.cache_rows = st.max_ctx;
        a.kpos = nullptr;
        a.window = 0;
        a.log_scaling = true;
      } else {
        const int ri = ring_cache_index(v);
        a.k_conv = f32p(layer.k_conv);
        a.v_conv = f32p(layer.v_conv);
        a.k_cache = st.k_ring_at(ri).data;
        a.v_cache = st.v_ring_at(ri).data;
        a.cache_fp8 = false;
        a.cache_rows = WINDOW;
        a.kpos = st.ring_pos_at(ri);
        a.window = WINDOW;
        a.log_scaling = false;
      }
      stage(VerifyStage::Attention, v, false);
      if (slots)
        attn::attn_decode_batch(a, reference ? ref_slots->attn[v - NUM_VISITS] : slots->attn[v], m, stream);
      else attn::attn_decode(a, stream);
      stage(VerifyStage::Attention, v, true);
      float* hist = history ? f32m(b.hist_attn) + std::size_t(v) * slab : nullptr;
      if (tp_ || slots) {
        const bool bf16 = tp_ && tp_->bf16_wire(2 * v);
        float* raw = bf16 ? nullptr : f32m(b.y);
        auto* raw16 = bf16 ? bf16m(b.xf) : nullptr;
        stage(VerifyStage::Wo, v, false);
        dense::proj_raw(bf16m(b.attn_out), m, layer.wo.matrix(), raw, raw16, pdl, stream);
        stage(VerifyStage::Wo, v, true);
        reduce_conv(raw, raw16, v, dense::ConvKind::Attention, layer.attn_conv, st.conv_attn_at(v), f32m(b.y), hist);
      } else {
        dense::proj_conv(bf16m(b.attn_out), m, layer.wo.matrix(), f32p(layer.attn_conv), st.conv_attn_at(v), f32m(b.y),
                         hist, pdl, stream);
      }
    }
    // ---- MLP read: this visit's attention write, ablated, on top of hid (or a fresh block)
    std::vector<const float*> mlp_blocks = blocks;
    if (boundary) mlp_blocks.push_back(hid);
    const float* mbase = boundary ? nullptr : hid;
    const bool fuse_visit = fuse_mlp && !msgs && layer.mlp_query.ne[1] == 1 && mlp_blocks.size() <= 5;
    dense::MixParams mp =
        msgs ? mix_params(mlp_blocks, mbase, f32p(b.y), attn_dirs + std::size_t(wv) * H, attn_w + 4 * std::size_t(wv), f32m(b.part),
                          &layer.mlp_query, nullptr, nullptr, f32m(b.mixm))
             : mix_params(mlp_blocks, mbase, f32p(b.y), attn_dirs + std::size_t(wv) * H, attn_w + 4 * std::size_t(wv), f32m(b.part),
                          &layer.mlp_query, f32p(layer.mlp_norm1), bf16m(b.xm), nullptr);
    if (history && !simulation_) {  // b.hist_k/hist_v[v] = proj_out[:, K/V segment]
      int c = 0;
      for (auto [dst, s] : {std::pair{&b.hist_k, SEG_K}, std::pair{&b.hist_v, SEG_V}})
        mp.copies[c++] = dense::MixRowCopy{f32p(proj_out) + layout.activation_offset(s), R,
                                           f32m(*dst) + std::size_t(v) * m * layout.kv_dim_local,
                                           layout.kv_dim_local, layout.kv_dim_local};
    }
    if (fuse_visit) {
      dense::mix_router(mp, static_cast<const __nv_bfloat16*>(layer.router_bf16.data), f32p(layer.router_bias),
                        static_cast<int*>(b.idx.data), f32m(b.wts), pdl, stream);
    } else {
      stage(VerifyStage::Mix, 2 * v + 1, false);
      dense::mix(mp, pdl, stream);
      stage(VerifyStage::Mix, 2 * v + 1, true);
      if (msgs) {
        stage(VerifyStage::MessageGates, 2 * v + 1, false);
        dense::message_gates(f32p(b.mixm), false, f32p(b.msgs) + 3 * slab, 1, m,
                             static_cast<const __nv_bfloat16*>(layer.mlp1_gate_down.data), layer.mlp1_gate_forget16.data,
                             layer.mlp1_gate_gain16.data, false, f32p(layer.mlp_norm1), bf16m(b.xm), nullptr, pdl,
                             stream);
        stage(VerifyStage::MessageGates, 2 * v + 1, true);
      }
      stage(VerifyStage::Router, v, false);
      dense::router(bf16m(b.xm), m, layer.router_bf16.data, false, f32p(layer.router_bias),
                    static_cast<int*>(b.idx.data), f32m(b.wts), pdl, stream);
      stage(VerifyStage::Router, v, true);
    }
    if (simulation_) {
      simulation_->mlp(layer, b, v, history, reduction, stream);
    } else {
      const Experts& ex = layer.experts;
      const moe::Format fmt = ex.fmt == ExpertFormat::NVFP4 ? moe::Format::NVFP4 : moe::Format::MXFP8;
      stage(VerifyStage::MoeUp, v, false);
      moe::moe_up(bf16m(b.xm), m, static_cast<const int*>(b.idx.data), ex.w13.data, ex.w13_scale.data,
                  f32p(ex.w13_factor), fmt, bf16m(b.act), pdl, stream, layout.I_local);
      stage(VerifyStage::MoeUp, v, true);
      float* hist = history ? f32m(b.hist_mlp) + std::size_t(v) * slab : nullptr;
      if (tp_ || slots) {
        const bool bf16 = tp_ && tp_->bf16_wire(2 * v + 1);
        float* raw = bf16 ? nullptr : f32m(b.y2);
        auto* raw16 = bf16 ? bf16m(b.xf) : nullptr;
        stage(VerifyStage::MoeDown, v, false);
        moe::moe_down_partial(bf16m(b.act), m, ptr<int>(b.idx), f32p(b.wts), ex.w2.data, ex.w2_scale.data,
                              f32p(ex.w2_factor), fmt, layout.I_local, raw16, raw, nullptr, pdl, stream,
                              static_cast<float*>(down_workspace_.p));
        stage(VerifyStage::MoeDown, v, true);
        reduce_conv(raw, raw16, v, dense::ConvKind::Mlp, layer.mlp_conv, st.conv_mlp_at(v), f32m(b.y2), hist);
      } else {
        // With PDL, moe_down prefetches weights before waiting for moe_up.
        moe::moe_down(bf16m(b.act), m, static_cast<const int*>(b.idx.data), f32p(b.wts), ex.w2.data, ex.w2_scale.data,
                      f32p(ex.w2_factor), fmt, f32p(layer.mlp_conv), st.conv_mlp_at(v), f32m(b.y2), hist, pdl, stream);
      }
    }
    prev_v = wv;
  }
  if (uses_ref_coda) {
    finish(true);
    dense::loopcd_combine_logits(f32m(b.logits), f32p(b.ref_logits), m, UNPADDED_VOCAB,
                                UNPADDED_VOCAB, loopcd.mode, loopcd.omega, stream);
  } else {
    finish(false);
  }
}

}  // namespace ingine::model
