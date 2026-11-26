#include "model/spec.h"

#include "core/device.h"
#include "kernels/draft/draft.h"
#include "model/dflash2.h"
#include "model/model_kernels.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ingine::model {

namespace kd = kernels::draft;
namespace kh = kernels::head;

namespace {

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static_assert(sizeof(VerifyRecord::drafts) / sizeof(std::int32_t) == kd::GAMMA);

Widths verify_widths(const Runtime& rt) {
  const std::span<const int> w = rt.verify_widths();
  if (w.empty() || w.size() > kd::BLOCK) throw std::invalid_argument("SpecLoop: the runtime has no verify widths (spec off)");
  Widths r;
  r.n = int(w.size());
  std::copy(w.begin(), w.end(), r.w);
  if (r.w[r.n - 1] < kd::BLOCK) throw std::invalid_argument("SpecLoop: the widest verify call must hold a block");
  return r;
}

}  // namespace

SpecLoop::SpecLoop(Runtime& rt, Drafter* drafter, DFlash2* dflash2, DFlash2Head* dflash2_head)
    : rt_(rt), drafter_(drafter), dflash2_(dflash2), dflash2_head_(dflash2_head),
      host_verify_(rt.tp() || dflash2 || rt.options().draft_type == "hybrid"),
      width_(sizeof(std::int32_t)), record_(sizeof(VerifyRecord)), forced_window_(kd::BLOCK * sizeof(std::int32_t)) {
  const bool sharded_dflash2 = dflash2 && dflash2->is_sharded();
  if (drafter && rt.options().tp.rank != 0)
    throw std::invalid_argument("SpecLoop: expected the DSpark drafter on rank 0");
  if (rt.options().tp.rank == 0 && !drafter && !dflash2)
    throw std::invalid_argument("SpecLoop: expected a drafter on rank 0");
  if (rt.options().tp.rank != 0 && dflash2 && !sharded_dflash2)
    throw std::invalid_argument("SpecLoop: unsharded DFlash2 is only valid on rank 0");
  if (drafter && drafter->stream() != rt.stream())
    throw std::invalid_argument("SpecLoop: drafter and runtime streams differ");
  const Widths widths = verify_widths(rt);
  if (drafter) {
    // The draft outputs live at fixed addresses. One warm draft names them.
    draft_ = drafter->draft(0, 0, 0.f);
    drafter->reset();
  }
  width_.fill(0);
  if (host_verify_) proposal_ = DeviceBuffer(sizeof(VerifyProposal));
  if (rt_.tp() || dflash2_)
    host_accept_ = DeviceBuffer(sizeof(kd::AcceptResult));
  if (dflash2_ && (!rt_.tp() || rt_.options().tp.rank == 0))
    target_tokens_ = DeviceBuffer(kd::BLOCK * sizeof(std::int32_t));
  if (!host_verify_) {
    const float threshold = rt.options().spec_threshold;
    verify_ = rt.capture_switch(
        [&](cudaStream_t s, cudaGraphConditionalHandle h) {
          verify_select(draft_.conf, draft_.tokens, threshold, widths, rt_.tokens(),
                        static_cast<std::int32_t*>(width_.p), h, s);
        },
        [&](cudaStream_t s, int m) {
          rt_.record_save_verify(m, s);
          rt_.record_step(m, s);
        });
  }
  // where a round's outcome lives on the device: after the acceptance the drafter commits the accepted
  // rows and drafts for the next anchor from these (one graph, no host round trip in between)
  if (drafter) {
    kd::RoundInputs round;
    round.nwidths = widths.n;
    for (int i = 0; i < widths.n; ++i) {
      round.widths[i] = widths.w[i];
      round.taps[i] = ptr<const __nv_bfloat16>(rt.buffers(widths.w[i]).taps);
    }
    auto* result = rt_.tp() ? static_cast<kd::AcceptResult*>(host_accept_.p) : drafter->accept_scratch().result;
    round.m = host_verify_ ? &static_cast<VerifyProposal*>(proposal_.p)->width
                      : static_cast<const std::int32_t*>(width_.p);
    round.accepted = &result->accepted;
    round.next_greedy = &result->next;
    round.next_sampled = &static_cast<VerifyRecord*>(record_.device_ptr())->next;  // sample_from_probs
    drafter->bind_round(round);
  }
  CUDA_CHECK(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
  CUDA_CHECK(cudaStreamSynchronize(rt.stream()));
}

SpecLoop::~SpecLoop() {
  if (done_) cudaEventDestroy(done_);
}

void SpecLoop::prepare_accept(const Sampling& sampling) {
  const bool forced = !sampling.forced.empty();
  const bool greedy = sampling.temperature <= 0.f || forced;
  const AcceptKey key = greedy ? AcceptKey{true, forced, 0.f, 1.f, 0}
                               : AcceptKey{false, false, sampling.temperature, sampling.top_p, sampling.top_k};
  if (accept_ && key == accept_key_) return;
  const Widths widths = verify_widths(rt_);
  const int V = drafter_->vocab();
  const Drafter::AcceptScratch& scratch = drafter_->accept_scratch();
  const kh::SamplingParams params{sampling.temperature, sampling.top_k, sampling.top_p};
  const auto* width = static_cast<const std::int32_t*>(width_.p);
  auto* record = static_cast<VerifyRecord*>(record_.device_ptr());
  // forced replay: the drafts are compared with the forced tokens (the argmax still runs, for the timing)
  const auto* target = forced ? static_cast<const std::int32_t*>(forced_window_.device_ptr()) : scratch.target;
  accept_ = GraphExec();
  accept_ = rt_.capture_switch(
      [&](cudaStream_t s, cudaGraphConditionalHandle h) { width_select(width, widths, h, s); },
      [&](cudaStream_t s, int m) {
        // engine._accept over the m verify rows (anchor + m - 1 drafts)
        const float* logits = static_cast<const float*>(rt_.buffers(m).logits.data);
        const int n = m - 1;
        if (greedy) {
          kh::argmax(logits, m, V, V, scratch.target, s);
          kd::greedy_accept(target, draft_.tokens, n, scratch.result, s);
        } else {
          kh::sampling_probs(logits, m, V, V, params, scratch.probs, s);
          kd::sampled_accept(scratch.probs, draft_.logits, V, draft_.tokens, n, sampling.temperature, draft_.qstats,
                             scratch.uniforms, scratch.ws, scratch.resid, scratch.result, s);
        }
        rt_.record_rollback(m, &scratch.result->accepted, s);
        verify_publish(width, &scratch.result->accepted, greedy ? &scratch.result->next : nullptr, rt_.tokens(), record,
                       s);
      });
  accept_key_ = key;
}

void SpecLoop::generate(std::int32_t token, int max_new_tokens, const Sampling& sampling, kh::SamplerRng& rng,
                        std::span<const std::int32_t> stop_ids,
                        const std::function<bool(std::span<const std::int32_t>)>& on_tokens, GenerateStats& stats,
                        std::vector<std::int32_t>& tokens_in_state) {
  // Hybrid can switch between greedy and arbitrary sampling settings without
  // recapturing acceptance. The target uses its startup per-width graphs.
  if (host_verify_) {
    generate_host_verified(token, max_new_tokens, sampling, rng, stop_ids, on_tokens, stats, tokens_in_state);
    return;
  }
  const cudaStream_t stream = rt_.stream();
  int P = int(tokens_in_state.size());  // position of the anchor (sampled, not yet processed)
  if (drafter_->length() != P) throw std::logic_error("SpecLoop: drafter context does not match the state");

  const bool forced = !sampling.forced.empty();
  const bool greedy = sampling.temperature <= 0.f || forced;  // forced: acceptance by equality
  const int V = drafter_->vocab();
  const Drafter::AcceptScratch& scratch = drafter_->accept_scratch();
  auto* record_dev = static_cast<VerifyRecord*>(record_.device_ptr());
  const auto* record = static_cast<const VerifyRecord*>(record_.data());
  prepare_accept(sampling);
  if (sampling.temperature > 0.f) drafter_->set_seed(splitmix64(rng.seed ^ splitmix64(rng.counter++)));

  int emitted = 0, rounds = 0, accepted = 0;
  std::vector<std::int32_t> out{token};
  // the last verify call (none before the first round): its position, width and accepted drafts
  int round_P = -1, round_m = 0, round_acc = 0;
  // forced replay: per round, events around the verify + acceptance and around the next draft
  struct RoundEvents {
    cudaEvent_t e[3];
  };
  std::vector<RoundEvents> events;
  auto record_event = [&](int i) {
    CUDA_CHECK(cudaEventCreate(&events.back().e[i]));
    CUDA_CHECK(cudaEventRecord(events.back().e[i], stream));
  };
  // the first draft; later ones are enqueued by the previous round (they leave the target state alone, so
  // the last one is wasted when the generation ends) and run while the host emits
  drafter_->draft(token, P, sampling.temperature);
  for (;;) {
    const std::int32_t anchor = out.back();

    // emit this round's tokens up to a stop token or the budget
    std::size_t k = 0;
    bool done = false;
    while (k < out.size()) {
      const std::int32_t tok = out[k++];
      ++emitted;
      if (std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end() || emitted == max_new_tokens) {
        done = true;
        break;
      }
    }
    stats.generated_tokens = emitted;
    const bool go_on = on_tokens(std::span<const std::int32_t>(out.data(), k));
    drafter_->note_tokens(std::span<const std::int32_t>(out.data(), k));  // for the drafts after the next one
    if (done || !go_on) {
      // The last emitted token stays unprocessed, as in plain decoding. If the generation ended inside the
      // round's accepted drafts (out[k - 1] with k - 1 < accepted), the state, the drafter and
      // tokens_in_state hold drafts past it: keep the verify call's first k rows (anchor + k - 1 drafts).
      const int keep = int(k) - 1;
      if (round_P >= 0 && keep < round_acc) {
        rt_.rollback(round_P, keep + 1, round_m);
        drafter_->commit(ptr<const __nv_bfloat16>(rt_.buffers(round_m).taps), keep + 1, round_P);
        tokens_in_state.resize(tokens_in_state.size() - std::size_t(round_acc - keep));
      }
      break;
    }

    // verify the confident drafts in the width the device picks, accept, roll back the rejected rows; then
    // commit the accepted rows and draft for the next anchor, all from device values
    const std::int32_t inputs[1] = {anchor};
    rt_.set_inputs(inputs, P);
    if (forced) {  // the forced tokens after the anchor (forced[emitted - 1]); the accept graph reads them
      auto* w = static_cast<std::int32_t*>(forced_window_.data());
      for (int i = 0; i < kd::BLOCK; ++i)
        w[i] = std::size_t(emitted + i) < sampling.forced.size() ? sampling.forced[emitted + i] : 0;
      events.emplace_back();
      record_event(0);
    }
    if (!greedy) kh::uniform(kd::GAMMA, rng, scratch.uniforms, stream);
    verify_.launch(stream);
    accept_.launch(stream);
    if (!greedy) kh::sample_from_probs(scratch.resid, 1, V, V, rng, &record_dev->next, stream);
    CUDA_CHECK(cudaEventRecord(done_, stream));
    if (forced) record_event(1);
    drafter_->advance(P);
    if (forced) record_event(2);
    CUDA_CHECK(cudaEventSynchronize(done_));  // the round's only host synchronisation
    VerifyRecord r;
    std::memcpy(&r, record, sizeof r);
    const int acc = r.accepted;
    const int commit = acc + 1;
    round_P = P;
    round_m = r.m;
    round_acc = acc;
    if (forced) {  // the run all 7 drafts would have reached
      int all = 0;
      while (all < kd::GAMMA && std::size_t(emitted + all) < sampling.forced.size() &&
             r.drafts[all] == sampling.forced[emitted + all])
        ++all;
      stats.spec_trace.push_back({r.m, acc, all, 0.f, 0.f});
    }

    drafter_->advanced(commit);
    tokens_in_state.push_back(anchor);
    tokens_in_state.insert(tokens_in_state.end(), r.drafts, r.drafts + acc);
    P += commit;
    ++rounds;
    accepted += acc;
    out.assign(r.drafts, r.drafts + acc);
    out.push_back(r.next);
  }
  if (forced) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < events.size(); ++i) {
      if (i < stats.spec_trace.size()) {
        CUDA_CHECK(cudaEventElapsedTime(&stats.spec_trace[i].verify_ms, events[i].e[0], events[i].e[1]));
        CUDA_CHECK(cudaEventElapsedTime(&stats.spec_trace[i].draft_ms, events[i].e[1], events[i].e[2]));
      }
      for (cudaEvent_t e : events[i].e) cudaEventDestroy(e);
    }
  }
  stats.spec_rounds = rounds;
  stats.spec_accepted = accepted;
}

void SpecLoop::generate_host_verified(std::int32_t token, int max_new_tokens, const Sampling& sampling, kh::SamplerRng& rng,
                           std::span<const std::int32_t> stop_ids,
                           const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                           GenerateStats& stats, std::vector<std::int32_t>& tokens_in_state) {
  auto* tp = rt_.tp();
  const auto stream = rt_.stream();
  const Widths widths = verify_widths(rt_);
  const int fixed_width = rt_.options().spec_verify_width;
  const bool forced = !sampling.forced.empty();
  if (fixed_width && std::find(widths.w, widths.w + widths.n, fixed_width) == widths.w + widths.n)
    throw std::invalid_argument("Fixed verify width is outside the configured grid");
  if (fixed_width && !forced && rt_.options().draft_type != "dflash2") throw std::invalid_argument("fixed verify widths require forced replay");
  const bool trace = (forced && drafter_) || dflash2_ || (drafter_ && rt_.options().spec_trace);
  const bool sampled = sampling.temperature > 0.f && !forced;
  if (sampled && drafter_) drafter_->set_seed(splitmix64(rng.seed ^ splitmix64(rng.counter++)));
  int position = int(tokens_in_state.size());
  if (drafter_ && drafter_->length() != position)
    throw std::logic_error("SpecLoop: drafter context does not match the state");

  const bool profile_decision = rt_.options().tp.serve && tp && tp->profile;
  // Reuse events for forced replay and optional serving decision profiling.
  struct Events {
    cudaEvent_t e[6]{};
    ~Events() {
      for (auto event : e)
        if (event) cudaEventDestroy(event);
    }
  } events;
  if (trace)
    for (int i = 0; i < 4; ++i) CUDA_CHECK(cudaEventCreate(&events.e[i]));
  if (profile_decision)
    for (int i = 4; i < 6; ++i) CUDA_CHECK(cudaEventCreate(&events.e[i]));
  auto event = [&](int i) {
    if (events.e[i]) CUDA_CHECK(cudaEventRecord(events.e[i], stream));
  };

  auto* proposal = static_cast<VerifyProposal*>(proposal_.p);
  auto* result = !tp && drafter_ ? drafter_->accept_scratch().result
                                 : static_cast<kd::AcceptResult*>(host_accept_.p);
  auto* record_dev = static_cast<VerifyRecord*>(record_.device_ptr());
  const auto* record_host = static_cast<const VerifyRecord*>(record_.data());
  auto exchange = [&](bool stop) {
    if (drafter_)
      verify_proposal(draft_.conf, draft_.tokens, rt_.options().spec_threshold, widths, stop ? -1 : fixed_width,
                      proposal, stream);
    if (dflash2_ && (!tp || tp->layout.tp_rank == 0)) {
      const bool hybrid = rt_.options().draft_type == "hybrid";
      const bool adaptive = hybrid || rt_.options().dflash2_adaptive;
      const float threshold = hybrid ? rt_.options().hybrid_dflash2_threshold : rt_.options().spec_threshold;
      verify_proposal(dflash2_->confidence(), draft_.tokens, threshold, widths,
                      stop ? -1 : (fixed_width ? fixed_width : adaptive ? 0 : 8), proposal, stream);
    }
    if (tp) tp->broadcast_proposal(proposal, sizeof(VerifyProposal), stream);
    CUDA_CHECK(cudaMemcpyAsync(proposal_host_.data(), proposal, sizeof(VerifyProposal), cudaMemcpyDeviceToHost,
                               stream));
    if (tp) tp->wait(stream);
    else CUDA_CHECK(cudaStreamSynchronize(stream));
    return *static_cast<const VerifyProposal*>(proposal_host_.data());
  };

  // Both ranks process the same stop IDs and budget. Rank 0 owns callback cancellation and sends a
  // width-zero proposal when the worker would otherwise enter another round.
  std::vector<std::int32_t> out{token};
  int last_width = 0, last_accepted = 0;
  if (drafter_) {
    event(0);
    drafter_->draft(token, position, sampled ? sampling.temperature : 0.f);
    event(1);
  }
  for (;;) {
    int count = 0;
    bool done = false;
    for (const int32_t id : out) {
      ++count;
      ++stats.generated_tokens;
      if (stats.generated_tokens == max_new_tokens ||
          std::find(stop_ids.begin(), stop_ids.end(), id) != stop_ids.end()) {
        done = true;
        break;
      }
    }
    const auto emitted = std::span(out).first(count);
    const bool keep_running = on_tokens(emitted);
    const bool stop = done || (!tp && !keep_running);
    if (stop) {
      const int keep = count - 1;  // The last emitted token remains the unprocessed anchor.
      if (keep < last_accepted) {
        const int start = position - last_accepted - 1;
        rt_.rollback(start, keep + 1, last_width);
        if (drafter_)
          drafter_->commit(ptr<const __nv_bfloat16>(rt_.buffers(last_width).taps), keep + 1, start);
        tokens_in_state.resize(tokens_in_state.size() - (last_accepted - keep));
      }
    }
    if (drafter_) drafter_->note_tokens(emitted);
    if (stop) break;

    const int offset = stats.generated_tokens;
    const int32_t anchor = out.back();
    // Both ranks finish the draft collectives before rank 0 can publish cancellation.
    if (dflash2_ && (keep_running || dflash2_head_ || dflash2_->is_sharded())) {
      event(0);
      if (last_width) {
        const int n = last_accepted + 1;
        dflash2_->commit(ptr<const __nv_bfloat16>(rt_.buffers(last_width).taps), n, position - n, true);
      }
      draft_.tokens = dflash2_->propose(anchor);
      event(1);
    } else if (dflash2_head_) {
      dflash2_head_->run();
    }
    const VerifyProposal p = exchange(!keep_running);
    if (p.width == 0) break;
    if (std::find(widths.w, widths.w + widths.n, p.width) == widths.w + widths.n)
      throw std::runtime_error("SpecLoop: received an unsupported verify width");
    const int m = p.width;
    int32_t inputs[kd::BLOCK] = {anchor};
    std::copy_n(p.drafts, kd::GAMMA, inputs + 1);
    rt_.set_inputs(inputs, position);
    const int32_t* target = tp ? ptr<const int32_t>(rt_.buffers(m).greedy) : nullptr;
    if (forced) {
      auto* w = static_cast<int32_t*>(forced_window_.data());
      for (int i = 0; i < m; ++i)
        w[i] = std::size_t(offset + i) < sampling.forced.size() ? sampling.forced[offset + i] : 0;
      target = static_cast<const int32_t*>(forced_window_.device_ptr());
    }
    event(2);
    rt_.save_verify(position, m);
    const float* logits = rt_.run_step(m);
    if (!tp && !forced && !sampled) {
      auto* ids = drafter_ ? drafter_->accept_scratch().target : static_cast<int32_t*>(target_tokens_.p);
      kh::argmax(logits, m, UNPADDED_VOCAB, UNPADDED_VOCAB, ids, stream);
      target = ids;
    }
    const int eligible = forced ? std::min(m - 1, int(sampling.forced.size()) - offset) : m - 1;
    event(4);
    if (sampled) {
      if (drafter_) {
        const auto& scratch = drafter_->accept_scratch();
        const int V = drafter_->vocab();
        const kh::SamplingParams params{sampling.temperature, sampling.top_k, sampling.top_p};
        if (tp && eligible == 0) {
          CUDA_CHECK(cudaMemsetAsync(&result->accepted, 0, sizeof(result->accepted), stream));
          kh::sample(logits, 1, V, V, params, rng, &result->next, stream);
        } else {
          kh::uniform(kd::GAMMA, rng, scratch.uniforms, stream);
          kh::sampling_probs(logits, m, V, V, params, scratch.probs, stream);
          kd::sampled_accept(scratch.probs, draft_.logits, V, draft_.tokens, eligible, sampling.temperature,
                             draft_.qstats, scratch.uniforms, scratch.ws, scratch.resid, result, stream);
          kh::sample_from_probs(scratch.resid, 1, V, V, rng, &result->next, stream);
        }
      } else if (dflash2_ && (!tp || tp->layout.tp_rank == 0)) {
        // Exact lossless target sampling (temperature / top_p / top_k) verified against DFlash2 drafts.
        const kh::SamplingParams params{sampling.temperature, sampling.top_k, sampling.top_p};
        auto* ids = static_cast<std::int32_t*>(target_tokens_.p);
        kh::sample(logits, m, UNPADDED_VOCAB, UNPADDED_VOCAB, params, rng, ids, stream);
        kd::greedy_accept(ids, proposal->drafts, eligible, result, stream);
      }
      if (tp) tp->broadcast_proposal(result, sizeof(*result), stream);
    } else {
      kd::greedy_accept(target, proposal->drafts, eligible, result, stream);
    }
    event(5);
    rt_.record_rollback(m, &result->accepted, stream);
    verify_publish(&proposal->width, &result->accepted, &result->next, rt_.tokens(), record_dev, stream);
    event(3);
    CUDA_CHECK(cudaEventRecord(done_, stream));
    // The draft timing events are reused only after the current timings have been read.
    if (drafter_ && !trace && (!tp || !tp->profile)) drafter_->advance(position);
    CUDA_CHECK(cudaEventSynchronize(done_));
    if (tp) tp->check();

    if (tp && tp->profile) {
      tp->print_verify_profile(m);
      if (profile_decision && drafter_) {
        float elapsed = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, events.e[4], events.e[5]));
        std::printf("verify_decision M=%d sampled=%d ms=%.6f\n", m, sampled, elapsed);
      }
      if (drafter_) {
        drafter_->print_draft_profile(m);
        if (!trace) drafter_->advance(position);
      }
    }
    const VerifyRecord r = *record_host;
    const int commit = r.accepted + 1;
    if (trace) {
      int all = 0;
      while (all < kd::GAMMA && std::size_t(offset + all) < sampling.forced.size() &&
             r.drafts[all] == sampling.forced[offset + all]) ++all;
      SpecRoundRecord row{m, r.accepted, all, 0.f, 0.f};
      CUDA_CHECK(cudaEventElapsedTime(&row.verify_ms, events.e[2], events.e[3]));
      CUDA_CHECK(cudaEventElapsedTime(&row.draft_ms, events.e[0], events.e[1]));
      CUDA_CHECK(cudaEventElapsedTime(&row.control_ms, events.e[1], events.e[2]));
      stats.spec_trace.push_back(row);
      if (drafter_) {
        event(0);
        drafter_->advance(position);
        event(1);
      }
    }
    if (drafter_) drafter_->advanced(commit);
    tokens_in_state.push_back(anchor);
    tokens_in_state.insert(tokens_in_state.end(), r.drafts, r.drafts + r.accepted);
    position += commit;
    last_width = m;
    last_accepted = r.accepted;
    ++stats.spec_rounds;
    stats.spec_accepted += r.accepted;
    out.assign(r.drafts, r.drafts + r.accepted);
    out.push_back(r.next);
  }
  if (drafter_) {
    CUDA_CHECK(cudaEventRecord(done_, stream));
    CUDA_CHECK(cudaEventSynchronize(done_));
  }
}

}  // namespace ingine::model
