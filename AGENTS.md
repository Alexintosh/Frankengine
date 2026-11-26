# Frankengine Engineering & Promotion Protocol

This repository serves `thinkingmachines/Inkling-2-Mini` (`30B-A3.6B` looped MoE) in production on Blackwell GPUs (`TP=1` and Dual-Node `TP=2`).
Every optimization or serving change MUST follow this strict quality-first protocol before promotion to `main` or live production.

## Non-Negotiable Rules
1. **Strict-Lossless Default:** Production defaults MUST remain strict-lossless (`TP_WIRE=fp32`, official `NVFP4/MXFP8` weights, `KV_FP8=1`). Any lossy optimization (e.g. `--kv-q4`) MUST be strictly opt-in behind an explicit CLI flag and gated by KLD validation.
2. **Never Trust Micro-Benchmarks Alone:** Isolated synthetic benchmarks have repeatedly missed real coding-agent regressions (e.g. `c=2` batch trapping, Solo telemetry blind spots, chunked-prefill starvation, long-context acceptance drift). Every change MUST be qualified with **authentic multi-turn `omp` coding-agent sessions** in `tmux`.
3. **Separate Branches & Clean A/B:** Implement each improvement on its own feature branch (`feat/<name>`). Compare against a saved baseline JSON using `scripts/real_agent_eval.py --baseline <baseline.json>` before merging or promoting.
4. **Safe Process Lifecycle:** Never use `pkill -f`. Always stop (`./stop.sh`) *before* rebuilding the serving binary in-place, then run `./prepare.sh && ./start.sh`.
5. **Privacy Hygiene:** Never commit private hostnames, LAN/Tailscale IPs, private domain names, or local usernames to this public repository.

## Mandatory Pre-Promotion Checklist
Run all 4 gates in order and record exact numbers:

1. **C++ Unit Tests:**
   ```bash
   ctest --test-dir engine/native/build-gb10-tp -R "serve_scheduler|serve_server" --output-on-failure
   ```
2. **Numerical Parity (`ingine-validate`, 330 stages):**
   ```bash
   ingine-validate --model-dir <MODEL_DIR> --golden engine/native/tests/golden/model \
     --dense-fp8 --head-nvfp4 --draft-nvfp4 --spec-widths fine2
   ```
   - Gate: `PASS` (`top1 >= 0.950`, `KL mean <= 0.010`, `KL max <= 0.250`).
3. **Launcher / Lifecycle Tests (on TP=2 host):**
   ```bash
   PYTHONPATH=scripts python3 -m unittest scripts/test_lifecycle.py -v
   ```
4. **End-to-End Authentic Coding-Agent Regression Suite (Suites 1–5 + Probe C):**
   ```bash
   python3 scripts/real_agent_eval.py \
     --base-url http://127.0.0.1:8834/v1 \
     --server-log <PATH_TO_RANK0_LOG> \
     --omp-model <PROVIDER>/inkling-2-mini \
     --baseline <BASELINE_RESULTS_JSON>
   ```
   - **Hard Quality & Stability Gates (must ALL pass):**
     - Zero special-token leaks (`<|...|>`).
     - `100%` functional pass on real `omp` multi-turn tasks (`unittest`, multi-file refactor, `>BATCH_CTX` huge prompt, `c=4` parallel `omp` fan-out).
     - `interrupted=false` on all `c=1` and `c=2` turns (no accidental `BatchDecoder` drop).
     - Solo `/v1/telemetry` counters active (`total_decode_tokens > 0`, `total_prefill_tokens > 0`).
     - Prefill pacing under concurrent `2×15K` cold burst keeps streaming decode `>= 15 tok/s` (no single-digit starvation).
     - `100%` exact Needle-in-a-Haystack passkey retrieval (`16K` and `65K`) and clean 4-way mid-stream cancel recovery (`active=0, decoding=0`).
