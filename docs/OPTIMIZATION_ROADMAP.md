# Inkling-2-Mini Engine — Living Optimization Roadmap & Experiment Ledger

> **Purpose:** Single source of truth for all architectural R&D, lossless/near-lossless engine optimization, speculative-decoding research, configuration hill-climbing, and datacenter scaling on the custom `Inkling-2-Mini` inference engine (`~/code/inkling-2-mini-engine`). Designed for multi-session, multi-agent continuity.
>
> **Last updated:** 2026-11-24 (v2 — added Byte-Level Kernel Roofline Audit & Architecture-First Phased Plan)
> **Primary cluster:** `2× NVIDIA DGX Spark (GB10, sm_121a)` over dual-rail ConnectX-7 RoCEv2 (`rank0` rank 0, `rank1` rank 1)
> **Model:** `thinkingmachines/Inkling-2-Mini` (looped MoE Transformer: `H=2304`, `I=1152`, `V=200,058`, 24Q/8KV GQA 3:1, 48 layer visits = `0..5`, `6..17 × 3`, `18..23`, 128 experts top-8, NVFP4 layers `0..19`, MXFP8 layers `20..23`, 4-tap causal conv1d, DSpark `mtp.safetensors` drafter `Γ=7 / BLOCK=8`)

---

## 0. Multi-Agent Protocol & Execution Philosophy

### Strategic Ordering ("Architecture First, Lossless First, Config Tuning After")
Do **not** jump into fine-grained config sweeps until the major architectural ceilings have been tested in order:
1. **Phase A — Strict-Lossless Architectural Levers (`KL ≈ 0` / bit-exact):** Squeeze communication overlap, zero-proxy RDMA, lossless GPU weight decompression, exact distributed sampling, async drafting, and lossless-target `Γ=15` drafter distillation.
2. **Phase B — Near-Lossless Precision & Auxiliary Layout (`KL ≤ 0.005`, `33/33` golden chat):** Calibrated FP8/NVFP4 on remaining BF16/FP8 tensors (`message_gates`, `Wproj`, MoE `20..23`) and long-context attention/prefill optimizations.
3. **Phase C — Config Hill-Climbing & Production Qualification:** Noise-controlled interleaved A/B sweeps (`ctl → cand → ctl`) and full-context needle-in-a-haystack stability gates using the **`inference-optimisation`** skill (`scripts/bench.py`, `scripts/queue.sh`, `scripts/mem.sh`).
4. **Phase D — Controlled Lossy / Aggressive Compression (Deferred):** Only explore dynamic `top-k` expert pruning or sub-4-bit MoE (`EXL3`/`QTIP`/Trellis) *after* Phases A–C approach the lossless physical ceiling.
5. **Phase E — Datacenter Scaling:** `NvlsBackend` + `sm_100a` `tcgen05` (`1×/8× B200` → `64× NVL72` `20k+ tok/s`).

### Mandatory Rules for Every Agent / Session
1. **Read before running:** Check [Section 2 (Roofline Audit)](#2-byte-level-stack--kernel-roofline-audit), [Section 4 (Phased Roadmap)](#4-phased-optimization-roadmap), and [Section 5 (Dead Ends / Do-Not-Repeat)](#5-dead-ends--negative-results-do-not-repeat) before starting any experiment. Never re-run a documented `NO-GO` without a concrete hypothesis that changes its root cause.
2. **Use `inference-optimisation` skill discipline:** Compute theoretical ceiling first; change one variable per experiment; run interleaved controls (`ctl → cand → ctl`) with medians; verify outputs (never speed alone); run long-context needle retrieval before crowning a new best.
3. **Respect concurrency & hardware locks:**
   - Acquire `flock /tmp/frankengine_gpu.lock` before any GPU job on `rank0` (verify no live server owns the GPU first).
   - Never use `pkill -f` over SSH; clean up exact PIDs on both `rank0` and `rank1` and confirm zero orphaned GPU processes.
   - Do **not** touch cluster ops/lifecycle scripts (`start.sh`, `stop.sh`, `status.sh`, `prepare.sh`, `scripts/lifecycle.py`) unless explicitly assigned.
4. **Preserve reference trees:** Keep `ingine/` (`gb10-baseline`) and `ingine-phase0/` (`phase0-telemetry`) untouched. Production TP=2 code lives in `ingine-tp/` (`dual-spark-tp2`) and `tp_comm/`. Isolated R&D prototypes live in `spec_rd/` (or dedicated feature branches).
5. **Qualification gates before upstreaming to `ingine-tp/`:**
   - Official logit gate: `mean KL ≤ 0.01` (prefer `≤ 0.005`), `max KL ≤ 0.25`, `top-1 ≥ 0.95` (current M5 FP32 reference: `65/66` top-1, mean KL `0.001927`).
   - Official greedy chat generation: `33/33` tokens exact match.
   - Real hardware wall-clock win across target workloads (`2K`, `32K`, `128K`, `Chat`, `Code`).
6. **Update this document at the end of every session:** Update the track status badge, append a row (`EXP-XXX`) to [Section 6](#6-chronological-experiment--session-log), and record any rejected idea in [Section 5](#5-dead-ends--negative-results-do-not-repeat).

---

## 1. Current Verified State (Baseline vs. Production)

### Hardware Roofline (`2× DGX Spark`)
- **Measured single-GB10 streaming bandwidth (`512 MiB` locked probe):** `231.38 GB/s` (48 SMs, 24 MiB L2, 18 MiB max persisting L2, 100 KiB shared mem/SM).
- **Aggregate `2× GB10` bandwidth:** `462.76 GB/s` across dual-rail ConnectX-7 RoCEv2 (`rocep1s0f1` + `roceP2p1s0f1`, MTU 4096, GID 3, zero-copy `cudaHostAllocMapped` RNIC DMA).
- **Key silicon asymmetry:** GB10 pairs **low LPDDR5X memory bandwidth (`231 GB/s`)** with **massive `sm_121a` SM/Tensor compute surplus**. During single-stream decode (`M=1..8`), GPU compute is practically free — any lossless technique trading spare compute for fewer DRAM bytes or overlapped communication is high-leverage.
- **Achieved TP=2 parallel efficiency (speculative decode):** **88–91%** (`1.77×–1.82×` over `1× GB10` stock speculative).

### Qualified CLI Throughput (`ingine-bench`, Original Weights)

| Configuration | Built-in Chat | Built-in Code | 2K (3 items) | 32K (3 items) | 128K (2 items) | Validation |
|---|---:|---:|---:|---:|---:|---|
| `1× GB10` stock plain (`FP8` dense/KV) | 55.21 | 54.85 | 55.23 | 49.02 | 36.39 | Reference (`158691b`) |
| `1× GB10` stock speculative (`0.80`, coarse) | 72.43 | 71.89 | 91.37 | 85.62 | 60.76 | Reference (`158691b`) |
| `2× GB10` TP=2 plain (`FP32` wire, NVFP4 head) | 82.84 | 82.85 | — | — | 58.65 | `33/33` chat, `65/66` top-1 |
| `2× GB10` M4 DSpark best published | 116.11¹ | 120.47 | 150.23 | 140.70 | 98.31² | `33/33` chat (`FP32`), KL `0.00193` |
| **`2× GB10` M5 DSpark qualified peak** | **118.46** | **126.86** | **166.02** | **151.47** | **101.42** | **`33/33` chat, `65/66` top-1** |
| **M5 speedup vs. `1×` stock plain** | **2.15×** | **2.31×** | **3.01×** | **3.09×** | **2.79×** | — |
| **M5 speedup vs. `1×` stock spec** | **1.64×** | **1.76×** | **1.82×** | **1.77×** | **1.67×** | — |

¹ M4 best chat used full BF16 (`5/33` golden); best FP32 M4 chat was `113.53 tok/s`. ² Matched 2-item 128K rerun.

### Live OpenAI HTTP SSE Serving (`ingine-serve`, M6, ~1.8K Coding Prompt, FP32 Wire)

| Mode & Request State | Reused Tokens | TTFT | Decode tok/s | Notes |
|---|---:|---:|---:|---|
| DSpark greedy, cold | 0 | 944 ms | **113.44** | Free-running 300-tok coding turn |
| DSpark greedy, follow-up / parked | 2,088–2,103 | **9–74 ms** | **152.68–153.16** | Prefix-cache reuse active |
| DSpark sampled (`T=0.7, p=0.9, k=40`), cold | 0 | 944 ms | **103.16** | `-9.1%` vs. greedy (exact gather) |
| DSpark sampled, follow-up / parked | 1,791–2,105 | **11–251 ms** | **138.44–139.09** | Exact same-seed repeatability |
| Plain greedy / sampled, cold | 0 | 938–945 ms | **81.46 / 79.72** | Non-speculative reference |

### Repository & Branch Map (`rank0:~/Frankengine/`)

| Path | Branch / Commit | Role |
|---|---|---|
| `.` (workspace root) | `main` | `tp_comm/`, `scripts/`, `docs/`, `results/`, `spec_rd/`, `bench/`, cluster ops |
| `ingine-tp/` | `dual-spark-tp2` @ `8306c64` | **Production TP=2 engine** (M2–M6: NVFP4 head/drafter, M5 Split W2 + PDL + `fine2`, M6 `ingine-serve` + exact sampling) |
| `ingine-phase0/` | `phase0-telemetry` @ `bfda58b` | Frozen Phase 0 routing, loop-reuse, tree-16, and NVFP4 parity telemetry |
| `ingine/` | `gb10-baseline` @ `158691b` | Frozen minimal `sm_121a` / `cluster=8` patch on upstream `556acc7` |
| `spec_rd/ingine-wide/` | Isolated prototype | Track 3B TP=1 wide-block (`B=8/12/16`) & suffix-copy prototype (not upstreamed) |

---

## 2. Byte-Level Stack & Kernel Roofline Audit

Before pursuing micro-optimizations, we audited exact DRAM bytes read per rank across all 48 visits (`44` NVFP4 visits + `4` MXFP8 visits; Phase 0 `E_uniq(2)=13.0`, `E_uniq(4)=20.4`, `E_uniq(8)=30.6`) against the measured **`231.38 GB/s`** GB10 ceiling:

| Stage (48 visits, per rank) | Bytes (`M=2` / `M=4` / `M=8`) | Pure DRAM Floor (`ms`) | Measured (`ms`) | Achieved BW (`M=8`) | % of Silicon Ceiling |
|---|---:|---:|---:|---:|---:|
| **MoE `W13` (`moe_up`)** | 996 / 1,564 / **2,345 MB** | 4.31 / 6.76 / **10.14** | 4.66 / 6.98 / **10.57** | **221.9 GB/s** (`224.1` @ `M=4`) | **95.9% – 96.9%** |
| **NVFP4 Verifier Head** | **129.6 MB** (all widths) | **0.56** | 0.60 / 0.62 / **0.62** | **210.1 GB/s** (`215.7` @ `M=2`) | **90.8% – 93.2%** |
| **`Wproj` (FP8 QKV+gate)** | **477.3 MB** (all widths) | **2.06** | 2.53 / 2.48 / **2.52** | **189.6 GB/s** | **82.0% – 83.1%** |
| **MoE `W2` (M5 Split W2)** | 498 / 782 / **1,173 MB** | 2.15 / 3.38 / **5.07** | 3.14¹ / 4.63¹ / **6.24** | **188.0 GB/s** | **81.3%** |
| **`Wo` (FP8 out proj)** | **169.9 MB** (all widths) | **0.73** | 1.05 / 1.03 / **1.08** | **157.3 GB/s** | **68.0% – 71.1%** |
| **Sum of 5 Weight GEMVs** | **2.27 / 3.12 / 4.29 GB** | **9.82 / 13.49 / 18.56** | **11.97 / 15.74 / 21.02** | — | **~88% aggregate** |
| **All Other Verify Stages** | — | — | **4.97 / 5.97 / 9.66 ms above floor** | — | RoCE, gates, attn, mix, router, conv |

¹ `M=2`/`M=4` `W2` timings are pre-Split-W2 (M4 event profile); `M=8` `6.24 ms` is post-M5 Split W2.

### Architectural Audit Verdicts
1. **Software stack (`ingine-tp` C++/CUDA + `tp_comm`):** **OPTIMAL — DO NOT REPLACE.** The native kernels already use `CL=8` thread-block clusters, DSMEM (`st.async` + `mbarrier`), PDL (`griddepcontrol.wait`), and warp-level NVFP4/MXFP8 dequantization, hitting **96–97%** of physical DRAM bandwidth on `MoE W13` and **91–93%** on the NVFP4 head. Moving to vLLM/SGLang/TRT-LLM/Triton would regress speed by 1.5×–3× on this looped-MoE + 4-tap-conv + RoCE-UMA workload.
2. **`TP=2` vs `EP=2` on `2× Spark`:** **`TP=2` WINS.** Pure `EP=2` (64 experts/rank) reads the same average weight bytes and still needs 1 AllReduce per layer (inputs are already replicated after `Wo`), while suffering a `+12–15%` binomial load-imbalance straggler penalty (`max(E_0, E_1) ≈ 17.5` vs `15.3`). `EP` only pays off at `≥8–16` GPUs (`NVL72` `TP=8 × EP=8`).
3. **Where real headroom actually lives:**
   - **Non-GEMV overhead (`9.66 ms` @ `M=8`, `34%` of verify + `1.57 ms` serial draft):**
     - `96 RoCE AllReduces`: **`2.18–3.64 ms`** (`12–15%`) — 100% serialized behind GPU compute and bouncing through a CPU proxy (`CPU 14`).
     - `Message Gates` (`36` loop visits): **`1.03–1.58 ms`** — reads **`170 MB` of replicated `BF16` weights** (`qkv_gate_*` + `mlp1_gate_*`) on *both* ranks!
     - `Attention` (2K): **`0.98–1.26 ms`** — scales up heavily at `32K–128K`.
     - `Hidden mixes`: `0.79 ms` · `Conv1d epilogue`: `0.70 ms` · `Router` (`28.3 MB` replicated BF16): `0.64 ms`.
     - `DSpark NVFP4 Draft`: **`1.55–1.57 ms`** serial on Rank 0 while Rank 1 sits 100% idle.
     - `M6 Sampled Head Exchange`: **`+0.63 ms` (`M=2`) / `+2.26 ms` (`M=8`)** due to single-CTA `gather_rows_to_root`.
   - **Uncompressed / 8–16 bit weights (`~800+ MB` still in BF16/FP8/MXFP8):** `Wproj` (`477 MB` FP8), `Wo` (`170 MB` FP8), `message_gates` (`170 MB` BF16), `router` (`28 MB` BF16), `MoE layers 20..23` (`MXFP8`, `~15%` of MoE bytes).
   - **Speculative horizon (`α`):** Stock `mtp.safetensors` caps at `Γ=7`, yielding `α ≈ 1.0–1.5` (Chat), `2.6–2.7` (Coding), `3.9–4.1` (Refactor).

---

## 3. Hardware Scaling Projection (Calibrated Model)

| Topology | Interconnect / Memory BW | Stock DSpark (`α ≈ 2.6–4.0`, `Γ=7`) | Distilled `Γ=15` (`α ≥ 9–11`) | Primary Bottleneck |
|---|---|---|---|---|
| `1× GB10` (measured) | `231 GB/s` UMA | `72–91 tok/s` (refactor `128`) | ~`110–145 tok/s` | Weight & MoE bandwidth |
| `2× GB10` TP=2 (measured) | `463 GB/s` + 2× CX-7 RoCE | **`118–166 tok/s`** | ~`200–250 tok/s` | MoE `W13/W2` BW + `2.2–3.6 ms` RoCE |
| `1× B200` (projected) | `8 TB/s` HBM3e | ~`2,400 tok/s` | ~`3,800–4,800 tok/s` | HBM3e weight BW |
| `8× B200` TP=8 (projected) | `64 TB/s` + NVLink 5 SHARP | ~`6,800 avg / 10,000 peak` | ~`11,000–15,000 tok/s` | 96× NVLS latency + kernel launch |
| `16× B200` (projected) | 2 nodes over InfiniBand | ~`3,600 tok/s` (**slower than 8×!**) | — | IB latency on 96 AllReduces |
| `64× GB200/300 NVL72` (`TP=8×EP=8`) | Full NVLink fabric | ~`9,400–10,000 tok/s` | **`20,000+ tok/s` (target)** | Needs `α ≥ 9` @ `0.50 ms` or `α ≥ 11` @ `0.60 ms` |

---

## 4. Phased Optimization Roadmap

Status legend: `[TODO]` · `[IN PROGRESS]` · `[DONE]` · `[ON HOLD]` · `[DEFERRED]` · `[NO-GO]`

---

### Phase A — Strict-Lossless Architectural Levers (Execute First!)
*Zero degradation to target model output distribution (`KL ≈ 0` / bit-exact or exact seeded sampling).*

#### `ARCH-01` · Split-H Compute + RoCE AllReduce Overlap `[DONE → NO-GO]` — *Wave 1*
- **Branch & Commit:** `ingine-tp` branch `arch/01-split-h-overlap` @ `fcbaca4` (preserved, NOT upstreamed to `dual-spark-tp2`).
- **Implementation:** Opt-in `INKLING_TP_SPLIT_H=1` splitting `MoE W2` + conv1d into two 1,152-channel halves (`72` 16-ch tiles each) with CUDA-graph event fork/join overlapping Slice 0 RoCE AllReduce on `comm_stream` with Slice 1 `W2` compute on `compute_stream`.
- **Parity:** **Strict 100% PASS** — `33/33` golden chat, `65/66` top-1, identical `KL 0.00192697476039`, bit-exact W2 FP32 + conv output/state/history across 21 layer cases (`M=2/4/8`), exact token-array match on all 2K/32K continuations.
- **Measured Interleaved Results (`ctl1 → cand → ctl2` vs matched M5 `8306c64`):**
  - Fixed-width verify (`M=2 / 4 / 8`): `14.47/19.38/26.74` → **`15.00/19.98/27.26`** → `14.42/19.17/26.77 ms` (**`+1.9% to +4.2%` slower**).
  - End-to-end (`2K / 32K / Chat / Code`): `171.7/156.5/120.6/128.8` → **`167.1/152.8/117.3/125.7`** → `170.7/155.8/121.1/129.1 tok/s` (**`-1.9% to -3.1%` slower**).
- **Root Cause (Nsight Systems 2025.3.2, 912 traced visits @ `M=8`):** Slice 1 `W2` (`72 × 64` CTAs on 48 GB10 SMs) saturated resident SM slots, delaying Slice 0's RoCE CTA launch by **50.74 µs** median after Slice 1 W2 started → only **4.93 µs** median overlap per visit (**`0.232 ms`** summed across all 48 visits!), completely outweighed by `+48` doubled RoCE collective startups/handshakes and extra launches (`+0.50 ms` net penalty).

#### `ARCH-02` · Zero-CPU-Proxy / GPU-Initiated RDMA (`IBGDA` / `mlx5dv`) `[DONE → NO-GO]` — *Wave 1*
- **Branch & Commit:** root branch `arch/02-zero-proxy-rdma` @ `5843626` (preserved, inline kept disabled).
- **Hardware / Driver Feasibility on GB10 (`CX-7 fw 28.45.4028`, `CUDA 13.0.88`, `rdma-core 50.0`):**
  - Both nodes report `GPUDirect RDMA = 0`, `DMA-BUF = 0`, `nvidia-peermem` absent, no DOCA/GPUNetIO/NVSHMEM/GDRCopy.
  - DevX BF/NC UAR allocation + `cudaHostRegisterIoMemory` + `cudaHostGetDevicePointer` returned `cudaSuccess` on both nodes, and CPU-origin manual RC RDMA WRITE doorbells passed (`CQE opcode=0 payload=0123456789abcdef`).
  - **Hard Operational Failure:** The first GPU-origin `st.mmio.relaxed.sys.global.b64` + `__threadfence_system()` doorbell store **hung/rebooted `rank0`** (no CUDA error or CQE returned). **Verdict: `NO-GO`** for GPU-initiated RDMA on GB10 UMA.
- **CPU Proxy Latency Anatomy (Eager Stage Trace):**
  - `CPU seen → first post`: **`0.080–0.208 µs`** · `ibv_post_send`: **`0.160–0.224 µs/rail`** · `CQ → CPU ready`: **`0.032–0.048 µs`**.
  - **Key Insight:** Total CPU proxy software overhead is **`<0.6 µs` per collective** — M1's pinned-host + CPU proxy is already near the physical wire/coherence floor on GB10 UMA!
- **`IBV_SEND_INLINE` (192 B budget) A/B:** Passed all 60 rank/width checks + `0` mismatches across `301,989,888` words/rank stress test, but produced **no repeatable latency gain** (`M=1..16` BF16 `1.029–2.928 ms`, FP32 `1.194–4.533 ms` vs matched controls) → **`NO-GO`**, keep inline disabled.


#### `ARCH-03` · Lossless GPU Weight Compression (DFloat11 / Entropy Bit-Packing) `[STEP 1 DONE → PARTIAL GO / LOW PRIORITY]`
- **Branch & Commit:** `arch/03-entropy-audit` @ `729197a` (`scripts/arch03_entropy_audit.py`).
- **Measured Step 1 Audit Results (Real `Inkling-2-Mini` Checkpoint, CPU-only, 1,024-symbol tiles incl. 8-substream metadata):**

| Tensor Class (Rank-0 stream) | Raw Bytes | Shannon Bits/Sym | Net Compression @ 1024 | Saved Bytes @ `M=8` | Max Theoretical `ms` Saved (`M=2` / `M=8`) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| **NVFP4 MoE `W13` / `W2` codes** | 3.40 / 1.70 GB | **3.968 / 3.969** (of 4) | **0.00%** (fallback raw) | 0 B | `0.000 / 0.000 ms` | **`NO-GO`** |
| **NVFP4 MoE `W13` + `W2` `E4M3` scales** | 425 + 212 MB | **4.47 / 4.35** (of 8) | **-44.28% / -44.94%** | **148.31 MB** (`-4.89%` NVFP4) | `0.273 / 0.641 ms` | **`PARTIAL GO`** (tight `0.14 ms` ALU budget for `≥0.5 ms` net) |
| **FP8 `Wproj` / `Wo`** | 238 / 85 MB | 6.57 / 6.57 | **-15.33% / -15.34%** | 73.1 + 26.2 MB | `0.429 / 0.429 ms` combined | **`PARTIAL GO` / `NO-GO`** |
| **MXFP8 MoE (`layers 20..23`) codes + `E8M0` scales** | 2.04 GB + 63.7 MB | 6.53 / **0.70–0.73** | **-15.7%** / **-82.8%** | 83.2 MB | `0.161 / 0.360 ms` combined | **`PARTIAL GO`** |
| **BF16 `message_gates` (24 repeat visits)** | 56.6 MB (`113.2 MB`/rd) | 11.07 (exp 3.11) | **-29.17%** | 33.0 MB | `0.143 / 0.143 ms` | **`NO-GO`** *(Bonus finding: `0.49 ms` DRAM floor vs `1.58 ms` measured → **latency/occupancy bound**, not DRAM bound!)* |
| **BF16 `router` / NVFP4 `head` / Draft** | 14.2 / 129.6 / 254 MB | — | -32.5% / -5.97% / -11–16% | — | `0.040 + 0.033 + 0.128 ms` | **`NO-GO`** |
| **All measured streams summed (ideal)** | — | — | — | — | `1.207 ms` (`M=2`) / **`1.774 ms` (`M=8`)** | Upper bound before ALU decode cost |

- **Step 2 Decision:** Deprioritized behind `ARCH-06a` (DFlash 2), `ARCH-01`, `ARCH-02`, and `ARCH-04`. Only prototype fused **NVFP4 `E4M3` scale-only** Huffman decoding if higher-ROI tracks plateau, because `0.641 ms` @ `M=8` leaves only `0.141 ms` of exposed decode overhead to net `≥0.5 ms`.

#### `ARCH-04` · Fast Exact Distributed Sampled Serving (`OPT-01`) `[PHASE A DONE → GO CANDIDATE]` — *Wave 2*
- **Branches & Commits:** `arch/04-fast-sampled-gather-comm` @ `e084e56` (`tp_comm` 32-CTA 128-bit vectorized `gather_rows_to_root`), `arch/04-fast-sampled-gather` @ `fb4f0fd` (engine report).
- **Measured Phase A Results (Bit-Exact FP32, `2× GB10`, interleaved `ctl1 → cand → ctl2`):**
  - **Isolated `gather-bench` (rank-0 CUDA event median, 64 replays):**
    - `M=1`: `165.5 / 166.0 µs` → **`100.0 µs`** (`1.66×`)
    - `M=2`: `418.3 / 319.9 µs` → **`117.2 µs`** (`2.7–3.6×`)
    - `M=4`: `630.4 / 630.2 µs` → **`153.6 µs`** (`4.1×`)
    - `M=8`: `1,247.8 / 1,249.4 µs` → **`226.7 µs`** (**`5.5×`**)
  - **Live M6 `ingine-serve` (`--tp-profile`, 10 cold+warm coding requests across `T=0/0.7/1.0`, `top_k=0/40`, seeds `42/43`):**
    - `M=2` head exchange: `0.472–0.477 ms` → **`0.151 ms`** (`-0.324 ms`, **`3.1×`**)
    - `M=4` head exchange: `0.905–0.911 ms` → **`0.190 ms`** (`-0.718 ms`, **`4.8×`**)
    - `M=8` head exchange: `1.791–1.799 ms` → **`0.286 ms`** (`-1.509 ms`, **`6.3×`**, beats `<0.45 ms` gate!)
    - **All 8/8 sampled requests (cold & warm, `k=40` AND `k=0`) beat BOTH controls by `+2.0% to +2.8%`** (e.g., `T=0.7 k=40` warm `106.7 → 109.6`, `T=0.7 k=0` warm `119.4 → 122.1`, `T=1.0 k=40` warm `108.0 → 111.1 tok/s` with profiling enabled).
- **Parity:** Bit-exact FP32 across widths `1, 2, 4, 8` (direct & cache-bypass visibility), 2-rank smoke tests + 4 CTests (`test_tp2_kernels`, `spec_accept`, `serve_scheduler`, `serve_server`) PASS, **10/10 byte-identical HTTP responses**. *(Note: run 33/33 golden chat + greedy interleave gate before merging `e084e56` into main `tp_comm/`.)*
- **Phase B (`top_k` exchange) Verdict:** **Deferred for `TP=2` GB10** (Phase A already shrunk full-vocab exchange to `0.15–0.29 ms`, and M6 keeps all cutoff ties requiring overflow fallback; keep Phase B for `TP=8` NVL72 where full-vocab AllGather is `5.60 MB/rank` vs `17.9 KB`!).

#### `ARCH-05` · Asynchronous / Speculative-Pipelined Drafting `[TODO — Wave 3]`
- **Target:** `ingine-tp/native/src/model/spec.cpp`, `drafter.cpp`, `decoder.cpp`.
- **Hypothesis:** Rank 0's `1.55–1.57 ms` draft runs strictly serially after verify finishes (while Rank 1 sits 100% idle). Optimistic lookahead drafting on a secondary CUDA stream (or sharding `ARCH-06a` DFlash 2 across both Sparks!) removes the serial idle bubble.
- **Expected gain:** `-0.5 to -1.3 ms/round` (`+4 to +10 tok/s`, lossless).

#### `ARCH-06a` · Port Pretrained `incoai/Inkling-2-Mini-DFlash2` Drafter into `ingine-tp` `[DONE → GO CANDIDATE (Short/Med Context + Hybrid Routing)]`
- **Branch & Worktree:** `arch/06a-dflash2-drafter` @ `e98e06f` (`ingine-tp`, opt-in `--draft-type dflash2` + `INKLING_DFLASH2_SHARDED_HEAD=1`).
- **Source:** [`incoai/Inkling-2-Mini-DFlash2`](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) (Apache-2.0 open weights, `0.5B` params, `block_size=8`, rev `4f9088ea`).
- **Part 4 Final Architecture (`e98e06f`):**
  - **FP8 Decode Commit (`8×` faster):** Packed `fc` + 5-layer K/V to `E4M3` using vectorized 8-row FP8 MMA kernels → cut decode commit from **`1.85–2.42 ms` → `0.294 ms` (`ctx=64`) / `0.298 ms` (`ctx=2K`)**, beating the `<0.35 ms` gate!
  - **Live TP=2 Sharded NVFP4 Target Head:** Reuses each rank's existing verifier NVFP4 head (`32 KB` BF16 hidden bcast + `896 B/rank` local top-16 exchange) → **`0.77–0.85 ms`** live (vs `1.43 ms` full rank-0) and **eliminated `259.3 MB`** of duplicate packed weights from rank 0!
  - **Nsight 2K `conv+norm` Audit:** Confirmed Part 3's 2K `2.1–2.5 ms` jump was a CUDA-event profiling artifact; Nsight measured **`0.286 ms` (`ctx=64`) / `0.297 ms` (`ctx=2K`)** (`conv` `1.26 µs`).
  - **Cumulative Confidence (`∏ p_i`) `fine2` Policy:** Cut unused verify rows at 2K from **`2.68 → 1.06`**, lifting 2K DFlash 2 from `143.8 → 161.1 tok/s` (`+12.0%`!).
  - **Total Draft Time:** **`3.18–3.26 ms`** (short) / **`3.77–3.83 ms`** (2K–32K) — down from `8.4 ms` (Part 2) and `4.83 ms` (Part 3). Remaining breakdown: `0.29 ms` commit + `1.70 ms` 5L FP8 proj + `0.20–0.77 ms` 5L attn + `0.30 ms` conv+norm + `0.78 ms` TP=2 head + `0.035 ms` selector.
- **Verified Part 4 Interleaved Results (`2× GB10`, `ctl1 → cand → ctl2` vs M5 `8306c64`):**
  - **Built-in Chat (`FP8 Graph + TP=2 Head`):** **`119.36` → `157.58 tok/s` (`+32.0%`, `0.35`, `α+1=4.390`)**, follow-up `0.50` hit **`159.98 tok/s` (`+34.0%` NEW ALL-TIME RECORD!)**
  - **Built-in Code (`FP8 Graph + TP=2 Head`):** **`124.21` → `132.61 tok/s`** (`0.55`, `draft 3.18 ms`), follow-up `0.45` hit **`136.18 tok/s` (`+9.6%`)** *(Part 3 BF16-commit `0.55` reached `137.9 tok/s`)*
  - **2K Coding (`product-0.35` cumulative):** `α+1` **`3.539 → 4.151` (`+17.3%`)**, decode **`161.10 tok/s`** vs M5 **`165.72 tok/s`** (within `2.8%`; on `1× Spark` where verify is `~2×` longer, `+0.61` accepted tokens easily wins!)
  - **32K Coding (`0.55`):** `α+1` `3.482 → 3.236`, decode **`127.42 tok/s`** vs M5 **`153.54 tok/s`** — **PROVED BUG-FREE**: Round-20 oracle dump (`pos=32,825`, after 57 decode commits & ring evictions) matched **`7/7` exact PyTorch proposals** in BOTH BF16 & FP8 and matched all **2,048 live rows** in all 5 K/V rings (`BF16 K RMSE 0.00148–0.00153`, `V RMSE 0.00176–0.00204`). The 32K gap is the pretrained checkpoint's fixed `2,048`-token sliding window vs M5 DSpark's global target-hidden-state drafter (`1.55 ms`).
- **Parity:** **`33/33` golden chat** (all modes incl. `fine2 0.35/0.55`), **`65/66` top-1 & identical `KL 0.00192697476039`**, **`7/7` real-tap PyTorch parity** at `ctx=30`, `ctx=3,000`, AND `ctx=32,825` (round 20), and bit-identical token-13 target logit hashes on both ranks.
- **Production Strategy (`OPT-03`):** Upstream opt-in `--draft-type dflash2` + TP=2 sharded head + cumulative `∏ p_i` policy; auto-route short/medium contexts (`ctx ≤ 2K–4K`, Chat/Code) to DFlash 2 (`+10–34%`) and long contexts (`ctx > 2K–4K`) to M5 DSpark (`154–166 tok/s`, `101 tok/s` @ 128K) until `ARCH-06b` trains a wider-window/longer-horizon drafter. Also port cumulative `∏ p_i` to M5 DSpark in Phase C!

#### `ARCH-06b` · Distilled `Γ=15` (`BLOCK=16`) DFlash2-Style Drafter Pilot `[TODO — Strategic for 64× NVL72 20k tok/s]`
- **Target:** Extending `DFlash 2`'s 2-tap dynamic conv + path selector architecture from `BLOCK=8` to `BLOCK=16` (`Γ=15`) via on-policy teacher-tap distillation (`1M → 4M → 16M` positions).
- **Why still needed after `ARCH-06a`:** `DFlash 2` (`BLOCK=8`) caps at `α ≤ 7` (measured `4.8–5.0`), whereas `64× NVL72` at `0.50–0.60 ms/round` needs **`α ≥ 9–11`** to hit **`20,000+ tok/s`**. `DFlash 2`'s architecture gives us a massively stronger starting recipe than vanilla DSpark.

---

### Phase B — Near-Lossless Precision & Auxiliary Layout (`KL ≤ 0.005`, `33/33` Golden)
*Test via fast offline KL/parity sweeps first before writing kernels.*

#### `ARCH-07` · Calibrated Quantization / Sharding of Remaining BF16 & FP8 Tensors `[DONE → OPT-IN ONLY (Combined Stack NO-GO for Prod Rollout)]`
- **Branch & Commit:** `arch/07-near-lossless-quant` @ `cd08fdc` (reports: `arch07_near_lossless_audit.md`, `arch07_near_lossless_step2.md`).
- **Step 1 Real-Engine Audit (`c8858c8`):** Proved sensitivity hierarchy (`Wo` `KL 0.0072–0.0084` NO-GO; `router` FP8 `91.1%` exact top-8 DEFER; `Wproj SmoothQuant α=0.7` `KL 0.00480` thin margin; all-in stack `KL 0.00630` `63/66` NO-GO). Promoted clean `MoE 20..22 W13` + `MoE 20..23 W2` + `message_gates` (`FP8`/`H16 NVFP4`) to Step 2.
- **Step 2 Real Packed Kernels & Interleaved `2× GB10` Benchmarks (`cd08fdc`):**
  - **Quality Gates:** `MoE W13 20..22 + W2` missed top-1 (`64/66`, `NO-GO`). Three standalone levers + one combined stack passed:
    - `MoE W13 20..22` (existing NVFP4 kernel): `KL 0.00246`, `65/66`, `33/33` → cut `W13` by `-0.13/-0.21/-0.37/-0.52 ms` (`M=1/2/4/8`), total verify `-0.51/-0.47 ms` at `M=4/8` (`-1.5% to -2.2%`), neutral/better cold prefill; minor extra protocol markers on unexecuted tool-call fixture.
    - `Packed FP8 row message_gates`: `KL 0.00284` (exact simulator match), `65/66`, `33/33` → faster gate stage at ALL widths (`0.82→0.69`, `0.90→0.75`, `1.00→0.91`, `1.28→1.26 ms`).
    - `Packed H16 NVFP4 message_gates`: **`KL 0.00245`, `66/66` top-1**, `33/33` → fastest at `M=1/2` (`0.82→0.60`, `0.89→0.70 ms`), **regressed at `M=8` (`1.29→1.49 ms`)** due to H16 butterfly work.
    - `Combined W13 20..22 + H16 gates`: **`KL 0.00268`, `66/66`**, `33/33` → beat total verify at all widths (`-0.08/-0.47/-0.60/-0.32 ms`), E2E `+2%` (2K `168.4`), `+11%` (32K `158.9`), `+4%` (128K `104.1`), **but regressed Code `-12%` (`143→126 tok/s`, `α+1` `2.89→2.37`) and Chat `-4%`**.
  - **Rollout Decision:** **Keep unified `TP=2` production 100% strict-lossless (`KL 0.001927`)** — `ARCH-07` Step 2's real `M=4/8` verify gain (`~0.3–0.6 ms` = `~1.5–2%`) does not justify trajectory drift / `-12%` Code acceptance swings in default production. Preserve `arch/07-near-lossless-quant` @ `cd08fdc` as an opt-in experimental branch (re-evaluate for `TP=1` Frankengine where byte savings double!).

#### `ARCH-08` · Non-Saturated Kernels (`ARCH-08a`) & Long-Context / Prefill (`ARCH-08b`) `[ARCH-08a DONE → NO-GO; ARCH-08b DONE → GO CANDIDATE]`
- **`ARCH-08a` (`arch/08a-aux-kernels` @ `1cb483d`) — `NO-GO` (all 6 lossless micro-variants rejected after 108 dual-Spark runs):**
  - Cleanly established actual M5 `8306c64` bandwidths: **`Wproj` is already `208–211 GB/s` (`90–91%` BW!)**, **`Wo` is `179–184 GB/s` (`78–80%` BW)**, **`message_gates` is `0.88–1.27 ms` (`89–129 GB/s`)**, **`router` is `0.35–0.46 ms`**, **`conv1d` `0.47–0.51 ms`**.
  - Tested and rejected (each ablated `ctl1 → cand → ctl2`):
    1. `message_gates` 2-token staging (regressed `M=4/8` by `+0.05–0.07 ms`: still 1 CTA/SM, added sync barrier) & warp RMS broadcast (`-6..15 µs` gate stage, regressed `M=4` total verify, `≤0.2%` noise E2E; MLP gate uses only one 8-CTA cluster on 48 SMs and cannot be sharded without adding a collective).
    2. Basalt-inspired 2-tile/CTA (`32` rows/CTA) on `Wo` & `Wproj` at `M=2, 3, 4`: M5 FP8 MMA *already* computes 16 rows/CTA, so pairing halved CTA parallelism (`Wproj` `+0.097 ms` @ `M=3`; `Wo` neutral/slower).
    3. Decode `router` direct BF16 global loads (`M=4` `-0.02 ms` failed to reproduce on repeat) & Basalt small-tile `16×8` MMA short-prefill router (`78.3 ms` fell between `77.5/78.4 ms` controls; fresh-process `394.5` vs `381.0/384.5 ms`).
    4. `conv1d` 64-thread 36-CTA epilogue (neutral/slower).
  - **Key Takeaway:** M5's lossless compute kernels are **already microarchitecturally saturated** on GB10; further speed on `message_gates`/`Wproj`/`MoE 20..23` requires **reducing DRAM bytes** (`ARCH-07` Phase B quantization), NOT lossless micro-tiling!
- **`ARCH-08b` (`arch/08b-longctx-prefill` @ `bd126f9`) — `GO CANDIDATE` (`+45.5% to +51.4%` bit-exact cold prefill speedup!):**
  - **Root-Cause Prefill Audit (Nsight Systems on `2× Spark`, 128K cold prefill = `77.6 s`):** **All compute combined** (`MoE W13` 6.31s + `W2` 5.06s + `Wproj` 2.50s + `Wo` 0.75s + `Attn` 10.28s + `message_gates` 5.57s + `conv/router/mix` 4.68s + `combine` 2.19s) = `37.34 s`, while **96×/chunk RoCE `AllReduce` ALONE took `39.554 s` (`51.0%` of total prefill!)** because `tp_comm` used a single-CTA pack/reduce kernel on `18.9 MB` (`2,048 × 2,304 × 4B`) chunks (`2.93 GB/s` effective!).
  - **Fix (`Parallel FP32 RoCE`, 48-CTA pack/reduce for `≥1 MiB` payloads, identical decode path):**
    - **2K prefill:** `1.171 s` (`1,750 tok/s`) → **`0.785 s` (`2,610 tok/s`, `+48.9%`!)**
    - **8K prefill:** `4.617 s` (`1,774 tok/s`) → **`3.113 s` (`2,631 tok/s`, `+48.1%`!)**
    - **32K prefill:** `17.795 s` (`1,841 tok/s`) → **`11.714 s` (`2,797 tok/s`, `+51.4%`!)**
    - **128K prefill:** `77.070 s` (`1,701 tok/s`) → **`52.978 s` (`2,474 tok/s`, `+45.5%` — `-24.1 s` saved!)**
    - **Strict Parity:** **`33/33` golden chat**, **`65/66` top-1**, identical **`KL 0.00192697476039`**, **`768/768` exact IDs at 2K, `768/768` at 32K, `512/512` at 128K**, and zero decode regression (within `0.8%` across 2K/32K/128K/Chat/Code).
  - **Rejected Prefill Variants:** Prefill-only `BF16 MoE` wire diverged on 2K & 32K continuations and dropped decode speed (`-3.6%` to `-6.7%`) → **REJECTED**; chunk sizes `C=128/256/512/3072` all trailed `C=2048` → **REJECTED**.
  - **32K & 128K Decode Attention Audit:** Full FP8 decode attention **ALREADY sustains `195.5–233.5 GB/s` (`84.5%–100.9%` of GB10's `231.38 GB/s` physical DRAM ceiling!)** across `M=1..8` (`32K`: `1.20–1.37 ms` vs `1.16 ms` floor; `128K`: `4.60–4.83 ms` vs `4.64 ms` floor) → already saturated, kept intact! *(Note: bulk RoCE still averages `~7.4 GB/s` vs `50 GB/s` dual-rail line rate during prefill — future `tp_comm` pipelined multi-chunk RDMA could shave another `~10–12 s` off 128K TTFT!)*

---

### Phase C — Unified Strict-Lossless TP=2 Integration, Config Hill-Climbing & Qualification (`inference-optimisation` Skill)
*All strict-lossless Phase A + Phase B winners (`ARCH-04` + `ARCH-08b` + `ARCH-06a`) unified, SASS-verified byte-identical on the hot decode path, and qualified across CLI, live HTTP SSE, and 65K/122K needle retrieval.*

#### `OPT-03` · Unified Strict-Lossless TP=2 Stack, Context-Aware Speculation & Full-Context Qualification `[DONE → PROMOTED TO PRODUCTION (256K + Hybrid DFlash2)]`
- **Production Branches & Commits:**
  - Root `main` @ `9b8e039` (`tp_comm/` promoted from `release/tp2-unified-comm` `f89e1a00`: `ARCH-04` 32-CTA gather in dedicated `publish_gather` + `ARCH-08b` 48-CTA bulk FP32 AllReduce `≥1 MiB`; **stock decode `collective<bypass>` SASS byte-identical to M5 `459366d` at 40 regs / 0 spills!**)
  - `ingine-tp` `dual-spark-tp2` @ `165520b` (fast-forwarded from `release/tp2-unified`: `ARCH-06a` FP8 Graph DFlash 2 with eager startup capture + live TP=2 sharded NVFP4 head `-259 MB` + context-aware `--draft-type hybrid` routing `ctx ≤ 1,536` greedy → DFlash 2 `fine2 0.50`, `>1,536` or sampled → DSpark; **all 28 target GEMV entry points SASS byte-identical to M5 `8306c64`!**)
  - Production `start.sh` (`scripts/config.sh` + `scripts/lifecycle.py`): defaults upgraded to **`MAX_CTX=262144` (256K)**, **`PARK_MB=2048`**, **`DRAFT_TYPE=hybrid`**, `DFLASH2_ADAPTIVE=1`, `DFLASH2_THRESHOLD=0.50`, `DFLASH2_MAX_CTX=1536`.
- **Verified Interleaved CLI Results (`M5 ctl1 → Unified Hybrid → M5 ctl2`):**
  - **Built-in Chat:** `120.37` → **`158.56 tok/s` (`+31.7%`!)** (`α+1`: `2.257 → 3.821`)
  - **Built-in Code:** `124.87` → **`135.08 tok/s` (`+8.2%`!)** (`α+1`: `2.345 → 3.000`)
  - **2K:** Prefill `1.167 s` (`1,755`) → **`0.784 s` (`2,613 tok/s`, `+48.9%`!)** · Decode **`165.29 tok/s`** vs `165.32–165.43` (`768/768` exact)
  - **32K:** Prefill `17.799 s` (`1,841`) → **`11.697 s` (`2,802 tok/s`, `+52.2%`!)** · Decode **`153.11 tok/s`** vs `153.34–153.89` (`768/768` exact)
  - **128K:** Prefill `77.362 s` (`1,694`) → **`53.035 s` (`2,471 tok/s`, `+45.9%`, `-24.3 s`!)** · Decode **`102.30 tok/s`** vs `102.52–102.92` (`512/512` exact)
- **Verified Interleaved Live `ingine-serve` OpenAI HTTP/SSE (`M6 ctl1 → Unified Hybrid → M6 ctl2`):**
  - **160-tok Chat (DFlash 2):** `117.14` → **`142.83 tok/s` (`+21.9%`!)** *(benign `M`-width tie-break `"stores"` vs `"maps"` after 323 identical bytes; unified DSpark matches M6 byte-for-byte)*
  - **1.8K Coding Greedy (cold/warm):** TTFT **`0.939 → 0.617 s` (`-34.3%`)** · Decode **`122.41 / 123.14 tok/s`** (inside/above M6 `122.04–122.90` & `120.80–122.83`), `1,784` cached tokens preserved across DFlash2 route switch, byte-identical!
  - **1.8K Coding Sampled `k=40` (cold/warm):** TTFT **`0.938 → 0.605 s` (`-35.5%`)** · Decode **`116.21 / 117.87 tok/s`** (`+2.5% / +3.4%`), byte-identical!
  - **1.8K Coding Sampled `k=0` (cold/warm):** TTFT **`0.937 → 0.616 s` (`-34.3%`)** · Decode **`107.25 / 106.79 tok/s`** (`+3.2% / +2.3%`), byte-identical!
  - *Known nuance on tiny 12-tok trivial probe (`"2+2"`):* both drafters accept 6 tokens across 6 rounds (`α+1=2.0`), so DFlash 2 adds `~9 ms` (`70.64` vs `87.6 tok/s`, `~156 vs ~126 ms` total wall time).
- **Full-Context Stability & Correctness Gates (100% PASS):**
  - `52,815,312`-byte FP32 target logit dump **SHA-256 byte-identical** to M5 (`65/66` top-1, `KL 0.00192697476039`), **`33/33` golden chat** in `dspark`, `dflash2`, and `hybrid`, 4/4 CTests, 64/64 `tp_comm` mixed collectives + bit-exact gather `M=1/2/4/8`.
  - **Live 65,495-tok Needle (40%):** exact `CORMORANT-78416-QUARTZ`, TTFT **`23.71 s`**, decode **`181.63 tok/s`**, peak UMA `13,966 / 12,401 MiB`.
  - **Live 121,559-tok Needle (84%):** exact `MANGROVE-29573-COBALT`, TTFT **`47.38 s`**, decode **`129.21 tok/s`**, peak UMA `14,588 / 12,401 MiB`.

---

### Phase D — Controlled Lossy / Aggressive Compression `[DEFERRED — Only After Lossless Ceiling]`

#### `LOSSY-01` · Dynamic `top-k` Expert Pruning (`top-8 → dynamic top-6`) `[DEFERRED]`
- Drop low-weight tail experts on loop passes `1/2` to cut `E_uniq(8)` by 20–25% (`-3 to -4 ms/round`) with zero weight reformatting. Strict offline KL + coding/reasoning eval required first.

#### `LOSSY-02` · Sub-4-bit MoE Quantization (`EXL3` / `QTIP` / Trellis 2.5–3 bit on Loop Experts `6..17`) `[DEFERRED]`
- Exploit GB10's idle SM compute to cut `W13 + W2` (`3.52 GB` @ `M=8`) from `4.5 bpw` to `2.5–3.0 bpw` (`-5 to -7 ms/round`).

#### `OPT-08` · Context-Gated Verbatim Suffix Lookup `[ON HOLD — Gated After Track 3B NO-GO]`
- Only revisit for `≥32K` verbatim copy spans (`match_len ≥ 16`), never overriding high-confidence drafter proposals (`p > 0.85`), with exact `M=1` greedy parity.

---

### Phase E — Datacenter Hardware Porting (`1× B200` → `8× B200` → `64× NVL72`)

#### `DC-01` · `NvlsBackend` (NVLink SHARP) + `sm_100a` `tcgen05` Port `[TODO — Awaits B200 Access]`
- Swap `RoceUmaBackend` for `NvlsBackend` (`multimem.ld_reduce`/`st`) behind `CommBackend`; enable `TP=8` + `sm_100a` `tcgen05` + `ARCH-06a` DFlash 2. Target: `~3.2k tok/s` (`1× B200`), `~9k–13k tok/s` (`8× B200`).

#### `DC-02` · `64× NVL72` Hybrid `TP=8 × EP=8` + `ARCH-06b` Distilled `Γ=15` (`20k+ tok/s`) `[TODO — Awaits NVL72 + ARCH-06b]`
- Math gates: `@ 0.40 ms/round` → `α ≥ 7` (`q ≈ 0.897`); `@ 0.50 ms` → `α ≥ 9` (`q ≈ 0.933`); `@ 0.60 ms` → `α ≥ 11` (`q ≈ 0.960`).

---

## 5. External SOTA Benchmark & Ecosystem Analysis

### A. `slatesys/lunomi-inkling` (Basalt — #1 on LocalMaxxing, `1× RTX 5090`)
- **Measured #1 Verified Results (400 W host-capped RTX 5090, 262K ctx, greedy, Basalt `b8702c7` + custom router/MMVQ):**
  - `code-v1`: **`526.2 tok/s`** peak (6-session median **`492.3 tok/s`**), TTFT `63.5 ms`, short-prompt prefill `3,824 tok/s`, long prefill `~15,900 tok/s`.
  - `reasoning-v1`: **`481.0 tok/s`** (median **`483.8 tok/s`**), TTFT `65.7 ms`.
  - Weights: native NVFP4/MXFP8 experts + `Q8_0` attention projections & head (`KL 0.00297`, `97.83%` top-1; GSM8K `96.51%` vs vLLM `96.66%`; HumanEval `95.12%` vs `96.34%`).
- **Apples-to-Apples Bandwidth-Normalized Comparison vs Our `ingine-tp` (`2× GB10`):**
  - `1× RTX 5090` (`1,792 GB/s` GDDR7): `492–526 tok/s` = **`0.275–0.294 tok/s per GB/s`**.
  - Our `2× GB10` M5 (`462.8 GB/s` aggregate LPDDR5X **+ RoCEv2 network hops**!): `126.9` (built-in code) to `166.0 tok/s` (2K coding) = **`0.274–0.359 tok/s per GB/s`**.
  - **Takeaway:** Per GB/s of memory bandwidth, our Dual-Spark `ingine-tp` engine is **already matching or outperforming** the #1 RTX 5090 leaderboard engine — thanks to our **NVFP4 sharded head** (2× smaller & better KL `0.00193` than Basalt's `Q8_0` `0.00297`), `fine2` adaptive verification, Split W2, and PDL! Meanwhile, we adopt Basalt's two lossless micro-wins (small-tile short-prompt router + 2-row/CTA `M=3` dense linear) into `ARCH-08`.

### B. `incoai/Inkling-2-Mini-DFlash2` (Open Apache-2.0 Block-Diffusion Drafter)
- **Measured Acceptance Length & SGLang H200 `c=1` Throughput (`block_size=8`, `T=1.0`, identical target):**

| Benchmark | Autoregressive (H200) | Stock DSpark (`α+1` / tok/s) | **DFlash 2 (`α+1` / tok/s)** | **DFlash 2 Gain vs DSpark** |
|---|---:|---:|---:|---:|
| GSM8K | 180.1 | `3.52` / 375.0 | **`4.96` / 482.1** | **`+40.9%` `α+1` / `+28.6%` tok/s** |
| MATH-500 | 190.5 | `3.47` / 375.7 | **`4.97` / 525.8** | **`+43.2%` `α+1` / `+40.0%` tok/s** |
| HumanEval | 190.2 | `3.66` / 374.7 | **`4.80` / 475.9** | **`+31.1%` `α+1` / `+27.0%` tok/s** |
| MBPP | 181.7 | `3.49` / 393.8 | **`4.88` / 503.4** | **`+39.8%` `α+1` / `+27.8%` tok/s** |
| MT-Bench (Chat) | 190.7 | `2.95` / 297.6 | **`3.90` / 387.3** | **`+32.2%` `α+1` / `+30.1%` tok/s** |

- **Takeaway:** Directly answers why stock DSpark felt capped (`α ≈ 2.6–3.5`) — **2-tap dynamic convs + top-16 candidate path selector** unlock **+31% to +43%** more accepted tokens per verify at the *exact same* `block_size=8`, reusing our existing `[5, 23, 41, 48]` taps and target `lm_head`. Promoted to **`ARCH-06a`** as our #1 highest-ROI lossless lever!

### C. `tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark` (`1× DGX Spark`: vLLM & TensorFold + DFlash 2)
- **Measured `1× GB10` Single-Stream (`c=1`, `T=0`, 8 prompt types) Results:**
  - `1× Spark` vLLM + DSpark: **`61.4 tok/s` mean** (`36.6–37.3` prose/narrative, `71.5` coding, `55.9` math, `75.8` json).
  - `1× Spark` **vLLM + DFlash 2**: **`88.2 tok/s` mean** (`+44%` vs DSpark; `71.8–73.0` prose/narrative, `94.6` coding `[AL 5.50]`, `82.2` math `[AL 4.83]`). At `c=6`: `226.5 tok/s` aggregate (`38–47 tok/s` per stream).
  - `1× Spark` TensorFold + DSpark (Mia's AI Lab CUDA recipe): **`68.2 tok/s` mean** (`52.7–57.7` prose, `68.8` coding, `67.5` math); prefill `3,127–3,354 tok/s` (`7.7K–30.7K`) via Blackwell FP4/MXFP8 tensor-core prefill.
  - `1× Spark` **TensorFold + DFlash 2** (patch `0018`: FP8 drafter layers + 4-bit draft head + acceptance-guided verification): **`105.5 tok/s` mean (`+55%` vs TF DSpark!)** — **`90.8–91.9`** prose/narrative (`+57%–74%`!), **`99.4`** coding (`+44%`), **`109.1`** math (`+62%`), **`114.0`** json, **`152.2`** counting.
- **Head-to-Head vs Our `2× DGX Spark` `ingine-tp`:**
  - **Already today (BEFORE DFlash 2!):** Our `2× Spark` M5 with stock DSpark (`118–121` Chat, `127–129` Code, `166–172` 2K coding) **already beats `1× Spark` TensorFold + DFlash 2 (`91` prose / `99.4` coding) by `+29% to +67%`**!
  - **With `ARCH-06a` (`2× Spark` `ingine-tp` + NVFP4 DFlash 2 + `fine2`):** Applying the real GB10 DFlash 2 multiplier (`+44–74%`, with our NVFP4 `227 MB` drafter even lighter than TensorFold's FP8 `384 MB` + our `0.60 ms` TP=2 NVFP4 head!), our Dual-Spark engine projects to **`~165–195 tok/s` Chat/Prose** and **`~185–235 tok/s` Coding** (`~1.8×–2.3×` faster than the fastest `1× Spark` DFlash 2 stack!).
  - **Key engineering lessons confirmed/adopted:** (1) Validates our exact 3-pillar design (quantized DFlash2 backbone + 4-bit draft head + acceptance-guided adaptive verify `fine2`); (2) Confirms Inkling's `embed_norm(embed_tokens(ids))` + `after_correction` softcap + `200,058` valid-vocab clamp + tiny `~42 MiB/stream` 5-layer sliding KV ring; (3) Highlights TensorFold's `~3.2–3.4K tok/s` Blackwell tensor-core prefill as a benchmark target for our `ARCH-08b` prefill optimization.

---

## 6. Cross-Architecture Learnings & Transferability Matrix

> **Living knowledge base:** Separates what only applies to `2× DGX Spark (GB10)` from architectural principles that transfer to `B200`/`NVL72`/`H100`/`MI300X`/`Apple Silicon` and other LLMs/MoEs. Updated after every experiment.

| Finding / Technique | `2× GB10` (UMA + CX-7 RoCE) | `B200` / `H100` / `NVL72` | `MI300X` / `Apple Silicon` | General LLM / MoE vs. `Inkling-2-Mini`-Only |
|---|---|---|---|---|
| **1. `NVFP4` 4-bit codes are near-maximal entropy (`3.97/4.00` bits); only block scales (`E4M3` `-45%`, `E8M0` `-83%`) & BF16 (`-30%`) compress losslessly** (`ARCH-03`) | `NVFP4` codes: `NO-GO`. Scale-only saves `0.64 ms` @ `M=8` (tight ALU budget). | Even tighter decode budget at 3.35–8 TB/s HBM; DFloat11 shines on **BF16/FP16** models, NOT already-FP4 weights. | Same symbol statistics for identical bit encodings; Apple Silicon UMA benefits most on BF16/FP8 models. | **General across all FP4/FP8/BF16 models.** Always audit entropy before building decompression kernels. |
| **2. Packed `NVFP4` LM head (`200k` vocab) beats `FP8`/`Q8_0` head with negligible KL (`0.0003–0.0019`)** (`M2/M3`) | `1.68×` faster (`0.60 ms`/rank, `93%` DRAM BW); beats Basalt's `Q8_0` head on both size and KL. | Transfers directly (`sm_100a` Blackwell FP4); big win on any large-vocab (`≥128k`) LLM. | Transfers directly to any 4-bit capable GPU/Metal SoC. | **General for all large-vocabulary LLMs.** |
| **3. Tree speculation fails on fine-grained MoEs (`E_uniq` explosion)** (`Phase 0`, `DEAD-01`) | `M=1→8→16` explodes `E_uniq` from `8→30.6→45.3`, killing BW-bound decode. | Same on `1× B200` / `H100` BW-bound decode; only relaxes on high-EP `NVL72` (`TP=8×EP=8`) where per-GPU expert count shrinks `8×`. | Same on Apple Silicon UMA and MI300X (`batch=1`). | **General for all fine-grained MoEs (`≥64` experts).** Use narrow adaptive chains (`fine2` `{2..8}`), not wide trees. |
| **4. 2-tap dynamic conv + top-16 candidate path selector (`DFlash 2`) beats simple Markov block drafters by `+31–43%` `α+1`** (`ARCH-06a`) | Massive lossless win (`~166 → 200–220 tok/s` projected) once draft MLP is NVFP4-quantized. | Transfers **100%** (already proven `+28–40%` on H200 SGLang; essential building block for NVL72 `Γ=15`). | Transfers **100%** to any unified-memory / desktop GPU serving engine. | **General block-diffusion speculative decoding architecture** (reuses target `lm_head` + 4 feature taps). |
| **5. `Wo` precision before causal conv1d must stay `FP32` across TP reductions** (`M2/M3/M5`, `DEAD-02`) | Full BF16 wire breaks greedy (`5/33`); hybrid `bf16-moe` (`Wo` FP32) preserves `33/33`. | Same numerical sensitivity on any TP interconnect (`NVLink`/`IB`/`RoCE`) when a recurrent/conv state consumes the reduced hidden vector. | Same on multi-GPU MI300X. | **Applies to all hybrid conv/SSM/GDN/looped-state models** under TP (Mamba, DeltaNet, Inkling). |
| **6. `TP=2` beats pure `EP=2` on 2 GPUs; `EP` only wins at `≥8–16` GPUs** (`Audit`, `DEAD-10`) | `TP=2` achieves `88–91%` efficiency; `EP=2` loses `12–15%` to binomial expert straggler imbalance. | Same math on 2×/4× H100/B200; switch to hybrid `TP×EP` only at large scale (`NVL72`). | Same on any 2-GPU topology. | **General MoE parallelism law** for single-stream decode (`batch=1`). |
| **7. Latency-bound small auxiliary kernels (`message_gates` `1.58 ms` vs `0.49 ms` floor, `router` `0.64 ms`) matter disproportionately at small `M`** (`ARCH-03` + `lunomi-inkling`) | On GB10 (48 SMs), `message_gates` runs at only `~72 GB/s`; small-tile router + 2-row/CTA `M=3` linears cut short-window latency. | Even more critical on B200/NVL72 where GEMVs shrink `35×` and kernel launch/cluster barriers dominate! | High impact on Apple Silicon / RTX 5090 (proven `-3 ms` TTFT in Basalt). | **General rule:** once main GEMVs hit `>85%` DRAM BW, auxiliary/small-`M` kernels dominate next gains. |
| **8. `16× B200` over IB is slower than `8× B200` NVLink for 96-AllReduce models** (`Roofline`, `DEAD-09`) | Explains why `2× Spark` RoCE (`2.2–3.6 ms` = 12–15%) is already near-optimal after `ARCH-01`/`ARCH-02`. | Never cross an IB/RoCE boundary for high-frequency TP AllReduces when an NVLink domain is available. | Same for PCIe-only multi-GPU without P2P/NVLink. | **General rule for deep / looped-depth models** with high AllReduce counts per token. |
| **9. Full-grid compute kernels starve concurrent low-CTA comm streams on small-SM GPUs; splitting small collectives doubles fixed RTT/handshake overhead** (`ARCH-01`, `DEAD-12`) | On GB10 (48 SMs), `72×64`-CTA Slice 1 `W2` delayed Slice 0's RoCE CTA by `50.7 µs` → only `4.93 µs` median overlap (`0.232 ms` total) vs `+0.50 ms` doubled collective cost (`-2%` to `-3%`). | On B200/NVL72, NVSwitch `NVLS` (`multimem`) offloads reductions in-fabric without fighting SMs, and large SM counts (`148+` SMs) allow green-context SM partitioning if overlap is needed. | Same SM-saturation physics on RTX 5090 / Apple Silicon when compute fills all SMs/cores. | **General CUDA-graph concurrency rule:** graph event forks express eligibility, NOT scheduling priority — always verify real overlap in Nsight Systems. |
| **10. On UMA + CX-7 (`GB10`), pinned-host + lightweight CPU proxy adds `<0.6 µs` SW overhead and beats GPU-MMIO; discrete GPUs use IBGDA instead** (`ARCH-02`, `DEAD-13`) | `ibv_post_send` is `0.16–0.22 µs/rail` (`<0.6 µs` total SW cost); GPU MMIO doorbell `st.mmio` hard-hangs/reboots `rank0` (no GPUDirect RDMA on GB10 UMA). `IBV_SEND_INLINE` gives 0 gain. | Discrete H100/B200 + CX-7 support GPUDirect RDMA / IBGDA (NVSHMEM GDAKI) where GPU-issued WQEs bypass PCIe host round-trips, especially at high-rank EP. NVL72 uses NVLS inside the NVLink domain. | Apple Silicon has no RDMA NIC; discrete AMD MI300X + CX-7 follows discrete PCIe/GDA rules. | **Architecture-specific transport law:** UMA SoCs favor zero-copy pinned-host + fast CPU proxy; discrete PCIe/HBM GPUs favor GPUDirect/IBGDA or NVLink SHARP. |
| **11. Multi-CTA 128-bit logit gather crushes `TP=2` sampled overhead (`6.3×` faster, `1.79 → 0.29 ms`); `TP=8` needs distributed `top_k` exchange** (`ARCH-04`) | On `2× GB10`, 32-CTA vectorized `gather_rows_to_root` cuts full-vocab `M=8` (`3.2 MB`) exchange to `0.286 ms` (`+2–3%` live sampled tok/s, bit-exact for all `top_k` incl. `k=0`). | At `TP=8` (`V=200k`), full-vocab FP32 AllGather grows to `5.60 MB/rank` @ `M=8` vs `17.9 KB/rank` for `top_k=40` (`312×` smaller — need overflow fallback for cutoff ties). | Same scaling on multi-GPU MI300X (`V ≥ 128k`). | **General large-vocab TP sampling law:** vectorize full-vocab gather at small TP; switch to exact distributed `top_k` (with tie-overflow fallback) at `TP ≥ 4–8`. |
| **12. Kernel micro-tricks do NOT transfer blindly across kernel families (`Q8_0` scalar MMVQ vs FP8 MMA 16-row CTAs); once kernels hit 80–97% BW, only cutting DRAM bytes (`ARCH-07`) or boosting `α` (`ARCH-06a`) moves the needle** (`ARCH-08a`, `DEAD-14`) | M5 `Wproj` (`90–91%` BW) and `Wo` (`78–80%` BW) already compute 16 rows/CTA; pairing tiles or tweaking single-cluster `message_gates`/`router`/`conv1d` gave 0 repeatable E2E gain across 108 runs. | On `8 TB/s` B200, `113 MB` `message_gates` has a `14 µs` transfer floor → fixed single-cluster/barrier launch overhead dominates even more; NVLink allows repartitioning consumers. | Basalt's 2-row/CTA & small-tile router won on scalar/Q8 MMVQ on RTX 5090 / PRO 6000 (`1,792 GB/s`), not on FP8 MMA kernels. | **General roofline discipline:** always measure actual per-kernel BW and compare tile structure before porting another engine's micro-tricks. |
| **13. Block-diffusion drafters (`DFlash 2`) need FP8 (not naive RTN NVFP4) to preserve `α`, cumulative `∏ p_i` gating to avoid wasting verify rows, and context-aware routing when their sliding window (`2K`) is shorter than target context (`32K–128K`)** (`ARCH-06a`) | FP8 + TP=2 sharded NVFP4 head (`3.18 ms`, `-259 MB`) lifted Chat **`+34%` (`160 tok/s`)** & Code **`+10%` (`136 tok/s`)**; `∏ p_i` cut 2K unused rows `2.68→1.06` (`+12%` → `161.1 tok/s`); 32K proved bug-free (`7/7` @ round 20) but 2K window trails global M5 DSpark (`153.5 tok/s`). On slower `1× Spark` verify, `+0.61` `α` wins even at 2K! | On B200/NVL72, `∏ p_i` adaptive verify + FP8 drafter + sharded target head transfer **100%**; for long-context production (`32K–128K`), route by context length or distill a wider-window drafter (`ARCH-06b`). | Transfers **100%** to `1× Spark` ("Frankengine" TP=1), RTX 5090, MI300X, and Apple Silicon. | **General speculative decoding law:** drafter win depends on the ratio `Δα / (1 + ΔT_draft/T_verify)` AND context-window match; always use cumulative `∏ p_i` over per-position thresholding! |
| **14. Single-CTA decode-tuned collectives catastrophically bottleneck multi-MB prefill AllReduces (`51%` of 128K TTFT!); switching `≥1 MiB` payloads to 48-CTA pack/reduce gave `+45–51%` bit-exact prefill speed** (`ARCH-08b`) | 1-CTA staging ran at `2.93 GB/s` on `18.9 MB` FP32 prefill chunks (`39.6 / 77.6 s`); 48-CTA parallel FP32 cut 128K TTFT `77.1→53.0 s` (`+45.5%`), 32K `17.8→11.7 s` (`+51.4%`), bit-exact `768/768` & `512/512` IDs! Meanwhile `ingine-tp` 32K–128K FP8 decode attention is ALREADY `94–100.9%` of DRAM BW (`218–233.5 GB/s`). | Same dual-path rule on B200/NVL72: low-latency 1-CTA / LL kernels for tiny decode messages (`<64 KB`), full-grid / multi-CTA or NVLS bulk kernels for multi-MB prefill collectives. | On `1× Spark` ("Frankengine" TP=1), eliminating RoCE AllReduces completely gives `~2.2–3.0K tok/s` native prefill for free! | **General TP collective law:** NEVER reuse a 1-CTA decode-latency collective kernel for multi-MB chunked prefill! Always split small-message vs bulk-payload paths (`≥1 MiB`). |
| **15. Near-lossless quantization sensitivity follows a strict hierarchy: `Wo` & `router` are fragile; `message_gates` loves `H16 NVFP4` (`66/66` top-1!) or per-row FP8; `MoE 20..23` tolerates per-matrix NVFP4 (`W13` `20..22` / `W2` `20..23`); `Wproj` needs `SmoothQuant α=0.7` and cannot blindly stack with everything else** (`ARCH-07` Step 1) | `MoE 20..22 W13` (`KL 0.00246`) & `MoE 20..23 W2` (`KL 0.00244`) already run through existing NVFP4 kernels (`65/66`, `33/33`); `H16 NVFP4` `message_gates` beat M5 (`66/66`, `KL 0.00287`, `-81 MB`); `Wproj SmoothQuant α=0.7` passed alone (`KL 0.00480`, `-209 MB`) but stacking all pushed `KL` to `0.00630` (`63/66`). | Transfers **100%** to `sm_100a` B200/NVL72 (identical weight sensitivity & KL profile across hardware!). | Transfers **100%** to `1× Spark` ("Frankengine" TP=1) where byte savings are **2× larger** per GPU! | **General quantization law:** always audit per-tensor KL & router-flip rate (`91.1%` on FP8 router!) before writing kernels, and verify *stacking* KL separately from single-tensor KL. |

---

## 7. Dead Ends & Negative Results (DO NOT REPEAT)

| ID | Rejected Idea | Tested In | Measured Failure / Root Cause | Report / Source |
|---|---|---|---|---|
| `DEAD-01` | **Tree-16 Speculative Verification** | Phase 0 | MoE unique-expert explosion (`E_uniq` `+44–48%`, `30.6 → 45.3` experts/layer) for only `+4.8–11%` accepted tokens → net slowdown. | `results/phase0_summary.md` |
| `DEAD-02` | **Full BF16 TP=2 Wire (`Wo` in BF16)** | M2, M3, M5 | BF16 rounding before 4-tap causal conv1d drifts conv history (`12,933/241,920 >1 ULP`) → breaks greedy chat (`5/33`). Keep `Wo` FP32. | `results/m2_tp2_kernels_and_head_nvfp4.md`, `results/m3_tp2_plain_e2e.md` |
| `DEAD-03` | **RoCE Receive-Finish + Conv1d Fusion** | M5 | 64/64 bit-exact, but slower at every FP32 width (`+0.22/+0.52/+0.47 ms` at `M=2/4/8`): single-CTA finish under-utilizes SMs. | `results/m5_dual_spark_peak.md` |
| `DEAD-04` | **GB10 L2 Persisting Window for Loop Layers `6..17`** | M5 | 1 loop pass touches `~215 MB` vs `18 MiB` persisting L2 cap → severe thrashing (`20.8 → 38.7 ms` @ `M=4`, `29.0 → 53.8 ms` @ `M=8`). | `results/m5_dual_spark_peak.md` |
| `DEAD-05` | **Packed 2/3/4 Experts in Shared Memory (`W2`)** | M5 | `72–88` regs/thread still blocked a 3rd resident CTA/SM; slower than 5-warp Split W2 (`20.72–21.15` vs `20.52 ms` @ `M=4`). | `results/m5_dual_spark_peak.md` |
| `DEAD-06` | **`W13` Shared/Interleaved, `Wo`/`Wproj` Split-K, CPU 19 Proxy Affinity** | M5 | All fixed-width A/B variants slower than baseline. Moreover, `W13` is already **95.9–96.9%** of physical DRAM BW! | `results/m5_dual_spark_peak.md`, Sec. 2 |
| `DEAD-07` | **Pretrained `mtp.safetensors` Widened to `Γ=11/15` (`B=12/16`) Without Retraining** | Track 3B | Full-block attention diluted `k=1..7` (`-0.097` to `-0.274` tok), live adaptive throughput dropped `-4.4%` to `-11.2%`. Must distill (`ARCH-06b`). | `results/spec_rd_high_alpha_gpu.md` |
| `DEAD-08` | **Zero-Training C++ Suffix-Copy Hybrid (`B1` ext / `B2` first)** | Track 3B | `B2` lost `-15.8%` at 2K and `-15.5% to -26.1%` on refactor prompts (overriding strong stock `B=8` `α=3.89–4.08`), failed `M=1` parity. | `results/spec_rd_high_alpha_gpu.md` |
| `DEAD-09` | **Naive `16× B200` TP over InfiniBand** | Roofline | 96 AllReduces/verify over IB makes `16× B200` (`~3.6k tok/s`) slower than single-node `8× B200 NVLink` (`~6.8k–10k tok/s`). | `docs/charts/scaling_to_20k_analysis.html` |
| `DEAD-10` | **Replacing `ingine-tp` with vLLM/SGLang/TRT-LLM or Pure `EP=2` on `2× Spark`** | Audit | Native kernels hit 88–97% DRAM BW with zero-copy RoCE UMA graphs; `EP=2` on 2 GPUs adds `+12–15%` straggler imbalance. | Sec. 2 |
| `DEAD-11` | **Lossless Entropy Compression of `NVFP4` 4-Bit Weight Codes (`W13`/`W2`/`Head`/`Draft`)** | `ARCH-03` Step 1 | Measured `3.968–3.973` Shannon bits per 4-bit symbol (`99.2%` uniform) → `0.00%` compression (falls back to raw). Only `E4M3`/`E8M0` scales & BF16/FP8 compress. | `arch/03-entropy-audit` (`729197a`) |
| `DEAD-12` | **50/50 Split-H `MoE W2` + RoCE AllReduce Overlap on GB10** | `ARCH-01` | Strict parity passed (`33/33`, bit-exact), but `-1.9% to -3.1%` slower everywhere (`+0.50 ms` @ `M=8`): Slice 1 `W2` fills all 48 SMs, delaying Slice 0 RoCE CTA by `50.7 µs` (`4.93 µs` median overlap = `0.23 ms` total vs `+48` doubled collective startups). | `arch/01-split-h-overlap` (`fcbaca4`) |
| `DEAD-13` | **GPU-Initiated CX-7 MMIO Doorbell & `IBV_SEND_INLINE` on GB10 UMA** | `ARCH-02` | GPU `st.mmio.relaxed.sys.global.b64` to mapped CX-7 UAR doorbell **hard-hung/rebooted `rank0`** (no GPUDirect RDMA on GB10). CPU proxy SW overhead is already `<0.6 µs` (`0.16–0.22 µs/rail` post). `IBV_SEND_INLINE` (192 B) gave 0 repeatable gain. | `arch/02-zero-proxy-rdma` (`5843626`) |
| `DEAD-14` | **Lossless Micro-Tiling on M5 `message_gates`, `router`, `Wo`/`Wproj` (2-tile/CTA), `conv1d` (64-thr)** | `ARCH-08a` | All 6 variants rejected across 108 clean dual-Spark runs (`ctl1 → cand → ctl2`). M5 `Wproj` (`90–91%` BW) & `Wo` (`78–80%` BW) already use 16-row FP8 MMA CTAs; MLP `message_gates` is bounded by 1-cluster (8 CTAs/48 SMs) DSM topology; short-prefill MMA router tied controls (`78.3` vs `77.5/78.4 ms`). | `arch/08a-aux-kernels` (`1cb483d`) |

---

## 8. Chronological Experiment & Session Log

> **Append-only ledger.** Add a new row at the bottom after every benchmark, optimization, or R&D session.

| ID | Date | Track | Hardware / Branch & SHA | Summary of Change / Experiment | Key Measured Result vs. Baseline | Verdict | Report |
|---|---|---|---|---|---|---|---|
| `EXP-001` | 2026-11-24 | Baseline | `1× GB10` · `ingine` (`158691b`) | Minimal `sm_121a` + `cluster=8` port of upstream `556acc7` | Plain: `55.2 / 49.0 / 36.4 tok/s` (2K/32K/128K). Spec 0.8: `91.4 / 85.6 / 60.8` (Chat `72.4`, Code `71.9`) | `DONE` (Baseline) | `results/gb10_baseline.md` |
| `EXP-002` | 2026-11-24 | Phase 0 | `1× GB10` · `ingine-phase0` (`bfda58b`) | MoE `E_uniq(M)`, loop reuse, Tree-16, NVFP4 head/dense parity telemetry | `E_uniq`: `8.0→30.6→62.3` (`M=1→8→32`). Approved `head-NVFP4` (`KL 0.00054`); killed Tree-16; held naive `dense-NVFP4` (`KL 0.0118`) | `DONE` | `results/phase0_summary.md` |
| `EXP-003` | 2026-11-24 | M1 | `2× GB10` · `tp_comm/` (`459366d`) | Dual-rail ConnectX-7 RoCEv2 zero-copy UMA CUDA-graph `CommBackend` | 70.1B BF16 elements verified (`0` mismatches). 96 BF16 SUMs: `1.041 ms` (`M=1`) to `1.798 ms` (`M=8`) | `DONE` | `results/m1_roce_uma_graph.md` |
| `EXP-004` | 2026-11-24 | M2 | `1× GB10` · `ingine-tp` (`dual-spark-tp2`) | Sharded TP=2 kernels (`4KV/12Q`, `I=576`, `100k` vocab) + packed `NVFP4` head | Bit-exact reconstruction vs TP=1; `NVFP4` head `1.68×` faster (`0.63 ms/rank`); isolated `Wo` BF16 drift | `DONE` | `results/m2_tp2_kernels_and_head_nvfp4.md` |
| `EXP-005` | 2026-11-24 | M3 | `2× GB10` · `ingine-tp` (`dual-spark-tp2`) | Live 2-node TP=2 plain inference + official golden validation | Chat/Code `81.8 / 83.6 tok/s`, 128K `58.65 tok/s` (`1,670` prefill); `33/33` chat, `65/66` top-1, `KL 0.00193` (FP32 wire) | `DONE` | `results/m3_tp2_plain_e2e.md` |
| `EXP-006` | 2026-11-24 | M4 | `2× GB10` · `ingine-tp` (`dual-spark-tp2`) | Live 2-node TP=2 + DSpark (`coarse`/`fine` 1..8) + `NVFP4` drafter | `150.23` (2K) / `140.70` (32K) / `98.31` (128K) / `116.11` (Chat) / `120.47` (Code) tok/s; 0 `cudaStreamSynchronize` in hot path | `DONE` | `results/m4_tp2_dspark_e2e.md` |
| `EXP-007` | 2026-11-24 | M5 | `2× GB10` · `ingine-tp` (`8306c64`) | Stage profile + `fine2` (`{2..8}`) + local PDL + 5-warp Split W2 + optional `bf16-moe` wire (and 12 A/B rejections) | **Peak: `166.02` (2K) / `151.47` (32K) / `101.42` (128K) / `118.46` (Chat) / `126.86` (Code) tok/s** (`1.82×` 1× spec, `3.09×` 1× plain); `33/33` chat, `65/66` top-1 | `DONE` (Prod) | `results/m5_dual_spark_peak.md` |
| `EXP-008` | 2026-11-24 | M6 | `2× GB10` · `ingine-tp` (`8306c64`) | Live `ingine-serve` OpenAI HTTP/SSE, exact TP=2 seeded sampling, cancellation, queue, 1 GiB/rank prefix parking | Warm greedy `153.2 tok/s` (`9–74 ms` TTFT); warm sampled `139.1 tok/s` (`-9.1%`, `gather_rows_to_root` `+0.63–2.26 ms`); M5 CLI within `0.5%` | `DONE` (Prod) | `results/m6_dual_spark_serve.md` |
| `EXP-009` | 2026-11-24 | Track 3A/3B | `1× GB10` · `spec_rd/ingine-wide` | Offline replay + live GPU test of widened pretrained `B=12/16` (`Γ=11/15`) and C++ Suffix-Copy `B1`/`B2` | `M=12/16` verify `1.32×/1.40×` vs `M=8`; pretrained `B=12/16` `-4.4%` to `-11.2%`; `B1/B2` `-15.5%` to `-26.1%` on refactor; derived NVL72 `α ≥ 9–11` distillation gates | `NO-GO` (zero-train) / `GO` (`ARCH-06b`) | `results/spec_rd_high_alpha.md`, `results/spec_rd_high_alpha_gpu.md` |
| `EXP-010` | 2026-11-24 | Audit | `2× GB10` · `ingine-tp` (`8306c64`) | Byte-level DRAM roofline audit across all 5 weight GEMVs + auxiliary BF16/FP8 tensors | `MoE W13` **95.9–96.9%** BW (`224 GB/s`), `Head` **91–93%**, `Wproj` **82–83%**, `W2` **81%**, `Wo` **68–71%**; verified custom C++/CUDA stack & TP=2 optimal | `DONE` | Sec. 2 |
| `EXP-011` | 2026-11-24 | `ARCH-03` Step 1 | `rank0` CPU · `arch/03-entropy-audit` (`729197a`) | Full-checkpoint lossless Shannon & tiled-Huffman entropy audit (256/512/1024 tiles) | `NVFP4` codes `3.968/4` bits (`0%` `NO-GO`); `NVFP4` `E4M3` scales `-44.3/-44.9%` (`148.3 MB` = `0.641 ms` @ `M=8`, tight `PARTIAL GO`); `Wproj+Wo` `-15.3%` (`0.43 ms`); `message_gates` only `113.2 MB` (`0.49 ms` floor vs `1.58 ms` meas → latency-bound!) | `PARTIAL GO` (low ROI) | Sec. 4 (`ARCH-03`), `arch03_entropy_audit.md` |
| `EXP-012` | 2026-11-24 | External SOTA | Web / HF / GitHub | Analyzed `incoai/Inkling-2-Mini-DFlash2` and `slatesys/lunomi-inkling` (#1 LocalMaxxing RTX 5090) | `DFlash 2` delivers **`+31–43%`** higher acceptance (`4.80–4.97` vs `3.47–3.66` DSpark) reusing exact `[5,23,41,48]` taps + target `lm_head` → promoted to `ARCH-06a`! `lunomi-inkling` (`492–526 tok/s` on `1,792 GB/s` = `0.275–0.294 tok/s per GB/s`) confirms our `2× GB10` (`0.274–0.359 tok/s per GB/s`) already matches/beats #1 BW efficiency; extracted 2 lossless micro-tricks into `ARCH-08`. | `DONE` → `ARCH-06a` GO | Sec. 5 |
| `EXP-013` | 2026-11-24 | `ARCH-01` | `2× GB10` · `arch/01-split-h-overlap` (`fcbaca4`) | 50/50 Split-H `MoE W2` + CUDA-graph `comm_stream` RoCE AllReduce overlap (`INKLING_TP_SPLIT_H=1`) + Nsight Systems 912-visit trace | Strict parity PASS (`33/33`, `65/66`, `KL 0.001927`, bit-exact W2/conv). Slower on all workloads: `M=2/4/8` verify `+1.9%..+4.2%`, E2E 2K `167.1` vs `171.7/170.7`, Chat `117.3` vs `120.6/121.1`, Code `125.7` vs `128.8/129.1 tok/s`. Nsight proved Slice 1 W2 starved Slice 0 RoCE CTA (`4.93 µs` median overlap = `0.23 ms` total vs `+48` doubled collectives). | `NO-GO` | Sec. 4 (`ARCH-01`), `arch01_split_h_overlap.md` |
| `EXP-014` | 2026-11-24 | `ARCH-02` | `2× GB10` · `arch/02-zero-proxy-rdma` (`5843626`) | CX-7 DevX UAR GPU-MMIO doorbell probe + CPU proxy stage latency trace + `IBV_SEND_INLINE` (192 B) interleaved A/B | CUDA accepted `cudaHostRegisterIoMemory` & CPU manual doorbell passed, but first GPU `st.mmio` doorbell **hung/rebooted `rank0`**. Eager trace proved CPU proxy SW cost is only **`<0.6 µs`** (`ibv_post_send` `0.16–0.22 µs/rail`). `IBV_SEND_INLINE` passed `302M`-word stress test (`0` mismatches) with **0 repeatable gain**. | `NO-GO` | Sec. 4 (`ARCH-02`), `arch02_zero_proxy_rdma.md` |
| `EXP-015` | 2026-11-24 | `ARCH-04` Phase A | `2× GB10` · `arch/04-fast-sampled-gather-comm` (`e084e56`) | 32-CTA 128-bit vectorized `gather_rows_to_root` staging/unpack in `tp_comm` (bit-exact FP32, zero engine logic changes) | **Live M6 `M=2/4/8` head exchange:** `0.475→0.151 ms` (`3.1×`), `0.908→0.190 ms` (`4.8×`), **`1.795→0.286 ms` (`6.3×`)**. All **8/8 sampled HTTP requests** (`k=40` & `k=0`, `T=0.7/1.0`, cold & warm) beat both controls **`+2.0% to +2.8%`** with **10/10 byte-identical responses** & bit-exact FP32 replay. | **`GO` Candidate** | Sec. 4 (`ARCH-04`), `arch04_fast_sampled_gather.md` |
| `EXP-016` | 2026-11-24 | `ARCH-06a` Part 2 | `2× GB10` · `arch/06a-dflash2-drafter` (`e686bfc`) | First native BF16 & NVFP4 DFlash 2 drafter E2E on `2× Spark` (`33/33` golden chat, `7/7` real-tap PyTorch token parity!) | **Built-in Chat BF16 ALREADY beat M5:** `α+1` **`2.26 → 4.47` (`+98%`!)**, **`120.8 → 134.4 tok/s`** even with slow `6.22 ms` draft! Isolated 3 bugs for Part 3: (1) sliding-attn adds `+4.60 ms` at 2K/32K (`9 GB/s` vs `0.18 ms` floor); (2) `32K` ring/RoPE wraparound collapsed `α+1` (`4.32→3.37`); (3) naive NVFP4 on all 35 draft mats hurt Chat `α+1` (`4.47→3.49`) → switching to FP8/hybrid + fast split-KV attn + CUDA graph in Part 3. | `IN PROGRESS` (Part 3) | Sec. 4 (`ARCH-06a`), `arch06a_dflash2_drafter.md` |
| `EXP-017` | 2026-11-24 | `ARCH-06a` Part 3 | `2× GB10` · `arch/06a-dflash2-drafter` (`942d6e1`) | Split-KV attention (`2.44→0.70 ms`, GQA TC `0.31 ms`), FP8/hybrid precision, CUDA graph draft, isolated TP=2 sharded head (`0.754 vs 1.434 ms`, 112/112 exact), token-13 & `ctx=3000` parity | **NEW ALL-TIME RECORDS ON CHAT & CODE:** Built-in Chat **`119.9 → 151.4 tok/s` (`+26.3%`!, `α+1` `2.26→4.39`)**, Built-in Code **`125.1 → 137.9 tok/s` (`+10.2%`!)**; `33/33` golden chat, `65/66` top-1, `7/7` real-tap parity at `ctx=30 & 3000`, token-13 same-schedule bit-identical. Stage profile exposed **Commit (`1.85–2.42 ms` naive BF16!) + unwired Rank-0 head (`1.30 vs 0.75 ms`)** as the exact remaining `~2.5 ms` holding back 2K (`144.6` vs `166.3`) → fixing in Part 4! | `IN PROGRESS` (Part 4) | Sec. 4 (`ARCH-06a`), `arch06a_dflash2_drafter.md` |
| `EXP-018` | 2026-11-24 | `ARCH-08a` | `2× GB10` · `arch/08a-aux-kernels` (`1cb483d`) | Ablated 6 lossless auxiliary/dense micro-kernels (`message_gates` 2-tok staging & warp RMS bcast, `Wo`/`Wproj` 2-tile/CTA, `router` direct loads & Basalt `16×8` MMA short-prefill, `conv1d` 64-thr) across 108 dual-Spark runs | **0/6 qualified (`NO-GO`):** both finalists passed strict parity (`33/33`, `65/66`, `KL 0.001927`, `768/768` 2K & 32K tokens) but gave `≤0.2%` E2E noise / `M=4` regressions; short-prefill router tied controls (`78.3` vs `77.5/78.4 ms`). Confirmed M5 `Wproj` is already **`90–91%` BW (`211 GB/s`)** & `Wo` **`78–80%` BW (`184 GB/s`)** → lossless kernel ceiling reached! | `NO-GO` | Sec. 4 (`ARCH-08`), `arch08a_aux_kernels.md` |
| `EXP-019` | 2026-11-24 | `ARCH-06a` Part 4 | `2× GB10` · `arch/06a-dflash2-drafter` (`e98e06f`) | FP8 8-row MMA decode commit (`fc` + K/V), live opt-in TP=2 sharded NVFP4 head (`-259 MB`), cumulative `∏ p_i` `fine2` policy, Nsight 2K `conv+norm` audit, 32K round-20 oracle check | **FP8 commit `2.42→0.294 ms` (`8×`)**; **live TP=2 head `0.77–0.85 ms`**; best draft **`3.18 ms`**. **Built-in Chat `157.6` (`0.35`) / `160.0 tok/s` (`0.50`, `+34%` NEW RECORD!)**, **Code `136.2 tok/s` (`+9.6%`)**, **2K `product-0.35` `161.1 tok/s`** (`α+1=4.15`, within `2.8%` of M5 `165.7`, cut unused rows `2.68→1.06`!). **32K round-20 (`pos=32,825`) matched `7/7` PyTorch proposals & all 5 K/V rings (`RMSE 0.0015`)** → bug-free (gap is 2K window vs M5 global state). `33/33` golden, `65/66` top-1, `KL 0.001927`. | **`GO` Candidate** (Short/Med + Hybrid) | Sec. 4 (`ARCH-06a`), `arch06a_dflash2_drafter.md` |
| `EXP-020` | 2026-11-24 | `ARCH-08b` | `2× GB10` · `arch/08b-longctx-prefill` (`bd126f9`) | Nsight 2K–128K prefill & 32K/128K decode attention roofline audit + 48-CTA Parallel FP32 bulk RoCE AllReduce (`≥1 MiB`) + chunk-size & prefill-only BF16 MoE ablations | **Audit proved `39.55/77.60 s` (`51%`) of 128K prefill was 1-CTA RoCE staging (`2.93 GB/s`), while 32K–128K FP8 decode attention is ALREADY `195.5–233.5 GB/s` (`84.5–100.9%` DRAM BW!)** **48-CTA Parallel FP32 cut 128K TTFT `77.07→52.98 s` (`+45.5%`), 32K `17.80→11.71 s` (`+51.4%`), 8K `4.62→3.11 s` (`+48.1%`), 2K `1.17→0.785 s` (`+48.9%`)** with bit-exact parity (`33/33`, `65/66`, `KL 0.001927`, `768/768` 2K & 32K, `512/512` 128K, zero decode regression). Rejected prefill-only BF16 MoE (continuation drift) & `C≠2048`. | **`GO` Candidate** (`tp_comm` bulk FP32) | Sec. 4 (`ARCH-08`), `arch08b_longctx_prefill.md` |
| `EXP-021` | 2026-11-24 | `ARCH-07` Step 1 | `2× GB10` · `arch/07-near-lossless-quant` (`c8858c8`) | Real-engine near-lossless sensitivity & calibration audit across `Wproj`, `Wo`, `message_gates`, `MoE 20..23` (real NVFP4 kernels!), and `router` (`mean KL ≤ 0.005`, `max ≤ 0.10`, `top-1 ≥ 65/66`, `33/33`) | **Found 3 clean individual winners:** `MoE 20..22 W13` (`KL 0.00246`, `65/66`, `33/33`) & `MoE 20..23 W2` (`KL 0.00244`, `65/66`, `33/33`, ZERO new kernels!); `message_gates H16 NVFP4` (**`66/66` top-1**, `KL 0.00287`, `-81 MB`) / `per-row FP8` (`KL 0.00284`, `65/66`); `Wproj SmoothQuant α=0.7` (`KL 0.00480`, `65/66`, `-209 MB`). Rejected `Wo` (`KL 0.0072–0.0084`), deferred `router` (`91.1%` exact top-8). All-in naive stack failed (`KL 0.00630`, `63/66`) → Step 2 testing clean `MoE + gates` stack! | **`PARTIAL GO`** → Step 2 | Sec. 4 (`ARCH-07`), `arch07_near_lossless_audit.md` |
| `EXP-022` | 2026-11-25 | `ARCH-07` Step 2 | `2× GB10` · `arch/07-near-lossless-quant` (`cd08fdc`) | Real packed CUDA kernels (`FP8 row` & `H16 NVFP4` `message_gates`) + real NVFP4 `MoE W13 20..22` dispatch, fixed-width verify + full 5-workload interleaved E2E | `W13 20..22` cut verify `-0.51/-0.47 ms` @ `M=4/8` (`KL 0.00246`, `65/66`, `33/33`); `FP8 gates` beat BF16 at all widths; `H16 gates` hit **`66/66`** & `0.60 ms` @ `M=1` but regressed `M=8` (`1.49 vs 1.29 ms`). Combined `W13+H16` (`66/66`, `KL 0.00268`) gained `+2%/+11%/+4%` at 2K/32K/128K but regressed Code `-12%` (`α+1` `2.89→2.37`) & Chat `-4%` → **kept opt-in only; unified TP=2 stays 100% strict-lossless (`KL 0.001927`)!** | **OPT-IN ONLY** (Combined `NO-GO` for prod) | Sec. 4 (`ARCH-07`), `arch07_near_lossless_step2.md` |
| `EXP-023` | 2026-11-25 | Phase C (`OPT-03`) | `2× GB10` · `release/tp2-unified-comm` (`f89e1a00`) + `release/tp2-unified` (`165520ba`) | Unified all 3 strict-lossless winners (`ARCH-04` + `ARCH-08b` + `ARCH-06a` eager-captured FP8 DFlash2 + TP=2 sharded head + `--draft-type hybrid` `ctx≤1536`), restored byte-identical M5 SASS on decode RoCE (`40 regs`) & all 28 target GEMVs, full CLI + live SSE + 65K/122K needle qualification | **CLI:** Chat **`158.6 tok/s` (`+31.7%`)**, Code **`135.1` (`+8.2%`)**, Prefill 2K/32K/128K **`+48.9%/+52.2%/+45.9%`** (`128K TTFT 77.4→53.0 s`), Decode 2K/32K/128K **`165.3/153.1/102.3`** (inside control spread, `768/768` & `512/512` exact). **Live SSE:** 160-tok Chat **`142.8 tok/s` (`+21.9%`)**, 1.8K coding TTFT **`-35%`**, greedy **`122.4/123.1`** (zero regression), sampled **`+2.3%..+3.4%`** (byte-identical). **Needles:** 65.5K (`181.6 tok/s`) & 121.6K (`129.2 tok/s`) exact match! `52.8 MB` FP32 target logits SHA-256 identical, `33/33` golden in all 3 modes. | **`DONE` (Promoted to Prod)** | Sec. 4 (`OPT-03`), `phase_c_tp2_unified_qualification.md` |
| `EXP-024` | 2026-11-25 | Track 1 (`TP=1` Frankengine) | `1× GB10` · `feat/tp1-frankengine` (`a62aa5a`) | Enabled native `TP=1` eager-captured FP8 Graph `DFlash 2` + `hybrid` (`0.50`, `ctx≤1536`) with zero-copy NVFP4 LM head reuse (`-259.3 MB`), full CLI (`Chat/Code/2K/32K/128K`), live `256K` SSE, and `65.5K/121.6K` needles on single DGX Spark | **Live Short Code:** **`155.35 tok/s`** (`+25.5%` vs TP=1 DSpark `123.8`, **`+33.3%` over TensorFold `116.5 tok/s`!**), TTFT `0.397→0.207 s`. **CLI:** Chat **`86.84`** (`+9.7%`), Code **`95.51`** (`+13.4%`), 2K **`104.8`** (prefill **`2,889 tok/s`**, `0.709 s`), 32K **`94.7`** (`2,770 tok/s`), 128K **`64.8`** (`2,329 tok/s`). **Live 256K Needles:** `65.5K` (`25.1 s`) & `121.6K` (`51.5 s`) exact passkey, peak NVML **`26,733 MiB`** (`26.1 GiB`). `4/4` CTests, `33/33` golden in all 3 modes, `TP=2` SASS byte-identical! | **`DONE` (Unified TP=1 + TP=2)** | `track1_tp1_frankengine.md` |
| `EXP-025` | 2026-11-25 | Track 2b (`Suffix-Copy Spec`) | `1× & 2× GB10` · `feat/track2b-suffix-spec` (`0b4b108`) | Zero-VRAM incremental CPU suffix-copy matcher (`2.4–5.5 µs`/step at 256K, `--suffix-spec`, `min-match=4`) skipping neural draft on hits with 1-round exact tap commit; interleaved standard + coding-edit/refactor/tool-JSON suites | **Failed suite-wide throughput & exact-continuation gates:** blind `M=8` verify on 4-token suffix matches hurt MoE verify efficiency (`TP=2` 2K `162.7→158.4`, 32K `153.2→143.7`, refactor `178.3→167.3`; `TP=1` 2K `104.4→100.7`, 32K `94.9→89.1`). Won only on ultra-repetitive JSON tool calls (`75–80%` hit rate: `TP=2` DFlash2 `226.7→242.9`, hybrid `224.2→233.6`; `TP=1` DFlash2 `150.2→155.7`). Preserved on experimental branch only; **NOT promoted**. | **`NO-GO` (Experimental Branch Only)** | `track2b_suffix_spec.md` |
| `EXP-026` | 2026-11-25 | Track 2c (`Prefill Roofline`) | `1× & 2× GB10` · `feat/track2c-prefill-engine` (`1a7ea47`) | Full Nsight Systems millisecond breakdown of `2K/32K` cold prefill on `TP=1` & `TP=2` + isolated 4-slice bulk FP32 RoCE probe + 2-stage `wgemm.cu` probe | **Key Roofline Discovery (`32K`):** Grouped MoE GEMMs (`wgemm.cu`, software-decoded BF16 MMA) are the #1 compute stage (**`4.825 s` / 40.2%** on `TP=1`, **`2.884 s` / 25.1%** on `TP=2`), directly explaining TensorFold's `TP=1` prefill lead (`3,395` vs `2,770 tok/s`). `TP=2` bulk FP32 RoCE (`3.083 s`, **`15.03–15.23 GB/s`** isolated) dropped `-13%` (`13.04 GB/s`) with 4 independent communicators → `NO-GO`. 2-stage `wgemm` lost `7/8` brackets → `NO-GO`. Future prefill lever = native Blackwell `mxf4nvf4` Tensor Core MMA in `wgemm.cu`. Decode SASS byte-identical (`40 regs / 0 spills`). | **`AUDIT DONE` (Probes `NO-GO`)** | `track2c_prefill_accel.md` |

### Template for Next Entry
```markdown
| `EXP-0XX` | YYYY-MM-DD | `ARCH-XX` / `OPT-XX` | `<HW>` · `<branch>` (`<sha>`) | <What changed / tested> | <Exact tok/s, ms, KL, parity vs. M5/M6 baseline> | `DONE` / `NO-GO` / `PARTIAL` | Sec. / Report |
```

---

## 9. Reference Index (Docs, Reports, External Links & Visuals)

- **Master Document (`docs/`):** `docs/OPTIMIZATION_ROADMAP.md` — *Single consolidated living roadmap, results ledger, SOTA benchmarks, and cross-architecture transferability matrix.*
- **Architecture & Historical Plans (`docs/`):** `docs/tp2_architecture.md` · `docs/phase0_telemetry_plan.md` · `docs/m4_max_throughput_plan.md` · `docs/hardware_scaling_and_megakernel_audit.md` · `docs/RTX5090_AGENT_RUNBOOK.md`
- **External SOTA References:**
  - `incoai/Inkling-2-Mini-DFlash2`: https://huggingface.co/incoai/Inkling-2-Mini-DFlash2 (Apache-2.0 0.5B block-diffusion drafter)
  - `slatesys/lunomi-inkling` (Basalt): https://github.com/slatesys/lunomi-inkling (#1 LocalMaxxing RTX 5090 Inkling-2-Mini engine)
  - `tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark`: https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark (`1× DGX Spark` vLLM + TensorFold + DFlash 2 benchmarks)
  - `sybil-solutions/inference-optimisation-skill`: https://github.com/sybil-solutions/inference-optimisation-skill (installed locally & on `rank0`)
- **Detailed Per-Milestone Reports (`results/`):** `gb10_baseline.md` · `phase0_summary.md` · `m1_roce_uma_graph.md` · `m2_tp2_kernels_and_head_nvfp4.md` · `m3_tp2_plain_e2e.md` · `m4_tp2_dspark_e2e.md` · `m5_dual_spark_peak.md` · `m6_dual_spark_serve.md` · `spec_rd_high_alpha.md` · `spec_rd_high_alpha_gpu.md` · `arch01_split_h_overlap.md` · `arch02_zero_proxy_rdma.md` · `arch03_entropy_audit.md` · `arch04_fast_sampled_gather.md` · `arch06a_dflash2_drafter.md` · `arch07_near_lossless_audit.md` · `arch07_near_lossless_step2.md` · `arch08a_aux_kernels.md` · `arch08b_longctx_prefill.md` · `phase_c_tp2_unified_qualification.md` · `track1_tp1_frankengine.md` · `track2b_suffix_spec.md` · `track2c_prefill_accel.md`
- **Interactive Charts (`docs/charts/`):** `inkling_roofline_review.html` · `gb10-context-scaling.html` · `scaling_to_20k_analysis.html` · `dual_spark_m5_results.html` · `track3_high_alpha_findings.html`
