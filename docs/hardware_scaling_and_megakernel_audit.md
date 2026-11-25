# Inkling-2-Mini Hardware Scaling & Blackwell Megakernel Engineering Audit

**Date:** 2026-11-25  
**Model:** `Inkling-2-Mini` (`30B` total / `3.6B` active MoE, `128` experts top-8, `24` layers / `48` loop visits, `8` full-attention visits, `200,058` unpadded vocab, `1M` native context)  
**Empirical Anchor:** Custom C++/CUDA `TP=2` engine (`ingine-tp` @ `165520b` + `tp_comm` @ `9b8e039`) measured on `2× NVIDIA DGX Spark (GB10, sm_121a, 231.38 GB/s measured DRAM BW/GPU)` at `256K` live serving context.

---

## 1. Executive Summary

Now that our `2× DGX Spark` (`GB10`) engine saturates **80–100% of physical DRAM bandwidth** across every major kernel (`NVFP4 MoE W13` ~96%, `FP8` full attention `32K–128K` 94–101%, `FP8 Wproj` 90–91%, `NVFP4 LM Head` ~90%) and serves `256K` context live at **`153–243 tok/s` decode** and **`2.8–3.2K tok/s` cold prefill**, we can project larger Blackwell and next-generation hardware using **exact byte counts, microsecond latency budgets, and real measured speculative acceptance (`α + 1`)** rather than naive bandwidth ratios.

### Key Bottom-Line Conclusions
1. **Bandwidth does NOT scale single-stream (`c1`) decode linearly past `1× B200`.** Moving from `231 GB/s` (`GB10`) to `8,000 GB/s` (`B200`) exposes two fixed latency walls that were invisible on Spark: the **48-visit CUDA kernel-launch/dependency floor (`~0.35–0.50 ms`)** and **`96×` AllReduce collective latency** (`~0.20–0.25 ms` on 8-GPU NVLink 5 SHARP; `~0.6–0.8 ms` cross-node InfiniBand).
2. **`1× B200` (and `8× B200` run as `TP=1`/`TP=2` replicas for concurrency) is the highest-ROI sweet spot:**
   - **`1× B200`** (`192 GB HBM3e`, zero AllReduce) delivers **`~1,800–2,100 tok/s`** sustained coding (`2K–32K`), **`~2,500–2,900 tok/s`** short-code peak, and **`~45K–75K tok/s`** cold prefill (`128K` in `~2.2 s`) with zero megakernel rewrite.
3. **Honest Megakernel + `GB200 NVL72` Reality Check:**
   - **Realistic production target** with a persistent Megakernel + 1-shot parallel `B=16` DFlash-style drafter: **`6,500–9,500 tok/s`** average agentic coding (`6,000–8,000 tok/s` on a single `8× B200` node), with **`11,000–15,000 tok/s`** peaks on repetitive boilerplate/refactoring.
   - **`20,000 tok/s` single-stream is a laboratory ceiling, NOT a realistic operating expectation** — it simultaneously demands a `≤0.55 ms` round floor across 48 sequential visits and `α + 1 ≥ 11–12` (which requires `≥93%` consecutive exact-match probability per token).
4. **Megakernel ROI depends sharply on topology:**
   - **`1× B200`:** Low ROI (`+20–25%` gain for high engineering cost).
   - **`8× B200` (`TP=8` NVLink 5):** **High ROI (`+40–50%` gain)** — fusing visits with inline `multimem.ld_reduce` cuts round time from `~1.05 ms` to `~0.70–0.80 ms`.
   - **`64× GB200 NVL72` for 1 user:** **Poor economic ROI** — cross-tray 64-GPU NVLink synchronization (`3–5 µs × 48`) caps single-stream speedup vs `8× B200` to `~15–25%` at `8×` the hardware cost. On `NVL72`, split into **9–18 `TP=4`/`TP=8` islands** for massive concurrent agent swarms (`3,000–6,000 tok/s` per stream).

---

## 2. Empirical Byte & Roofline Baseline (`Inkling-2-Mini`)

All projections below are anchored to the exact `TP=1` total DRAM traffic per verify step across all 48 visits (derived directly from our verified `TP=2` byte audit on `2× GB10`):

| Component (48 visits, `TP=1` total) | Precision | `M=1` | `M=2` | `M=4` | `M=8` | `M=16` (est.) |
|---|---|---:|---:|---:|---:|---:|
| **MoE `W13` (`moe_up`)** | NVFP4 (L0–19) / MXFP8 (L20–23) | ~1.22 GB | 1.99 GB | 3.13 GB | 4.69 GB | ~8.1 GB |
| **MoE `W2` (`moe_down`)** | NVFP4 (L0–19) / MXFP8 (L20–23) | ~0.61 GB | 1.00 GB | 1.56 GB | 2.35 GB | ~4.1 GB |
| **`Wproj` (QKV + gate)** | FP8 | 955 MB | 955 MB | 955 MB | 955 MB | 955 MB |
| **`Wo` (out proj)** | FP8 | 340 MB | 340 MB | 340 MB | 340 MB | 340 MB |
| **LM Head** | NVFP4 | 259 MB | 259 MB | 259 MB | 259 MB | 259 MB |
| **`message_gates` (24 repeat visits)** | BF16 (replicated) | 113 MB | 113 MB | 113 MB | 113 MB | 113 MB |
| **`router`** | BF16 (replicated) | 28 MB | 28 MB | 28 MB | 28 MB | 28 MB |
| **Total Target Weight Traffic (short ctx)** | — | **~3.52 GB** | **~4.69 GB** | **~6.39 GB** | **~8.74 GB** | **~13.9 GB** |

- **Full-Attention KV Cache Traffic (`FP8 E4M3`, 8 full visits, 8 KV heads, `head_dim=128`, K+V):**
  - `16 KB / token` (`TP=1`) → **`+537 MB` @ 32K**, **`+2.15 GB` @ 128K**, **`+4.29 GB` @ 256K**.
- **Drafter Traffic:**
  - `DSpark` (`NVFP4`, autoregressive): `~320 MB` (`1.56 ms` measured on `GB10`).
  - `DFlash 2` (`FP8` 5-layer parallel block-diffusion + sharded `NVFP4` head): `~550 MB` (`3.18 ms` measured on `2× GB10`).
- **Active Prefill Compute:** `~7.2 GFLOP/token` (dense + active MoE) plus quadratic attention term on the 8 full-attention visits.

---

## 3. Calibrated Hardware Scaling Projections (Single-Stream `c1`)

Assumptions: sustained small-batch HBM3e/GDDR7 efficiency of **75–82%** on large-SM GPUs (`~6.0–6.5 TB/s` sustained on `8 TB/s` B200, `~1.43 TB/s` on `1.79 TB/s` RTX 5090/PRO 6000); fixed CUDA Graph launch/dependency floor of `~0.35–0.50 ms` without megakernel fusion.

| Hardware Topology | Memory BW / Interconnect | `c1` Coding Avg (`2K–32K`, current engine `α+1 ≈ 3.5`) | `c1` Short-Code Peak (`DFlash2`, `α+1 ≈ 5.5`) | `c1` Next-Gen (`B=16` DFlash + Tree, `α+1 ≈ 9–10`) | Cold Prefill (`tok/s` & `128K` TTFT) | Dominant Bottleneck |
|---|---|---:|---:|---:|---:|---|
| **`2× DGX Spark` (measured)** | `463 GB/s` agg + 2× CX-7 RoCE | **`153–165 tok/s`** | **`200–243 tok/s`** | `~260–320` | **`2.5–3.2K`** (`53 s`) | DRAM BW + `2.4 ms` RoCE |
| **`1× RTX 5090` / `PRO 6000`** | `1,792 GB/s` GDDR7 (no TP) | **`~580–680 tok/s`** | **`~850–1,050 tok/s`** | `~1,100–1,400` | **`~15–25K`** (`~6–8 s`) | GDDR7 BW (`5090` 32 GB tight; `PRO 6000` 96 GB ideal for 256K+) |
| **`1× NVIDIA B200`** *(ROI sweet spot)* | `8,000 GB/s` HBM3e (**0 AllReduce**) | **`~1,800–2,100 tok/s`** | **`~2,500–2,900 tok/s`** | `~3,600–4,500` | **`~45–75K`** (`~2.0–2.8 s`) | HBM3e (`~1.05 ms`) + launch floor (`~0.45 ms`) + draft (`~0.30 ms`) |
| **`8× B200` (`TP=8`, CUDA Graph)** | `64 TB/s` + NVLink 5 SHARP | **`~3,200–4,200 tok/s`** | **`~4,800–6,000 tok/s`** | `~6,500–8,500` | **`~220–350K`** (`~0.4–0.6 s`) | Launch floor (`0.40 ms`) + `96× NVLS` (`0.22 ms`) |
| **`8× B200` (`TP=8` + Megakernel)** | `64 TB/s` + inline `multimem` | **`~4,500–6,000 tok/s`** | **`~6,500–8,500 tok/s`** | **`~10,000–13,000`** *(refactor)* | **`~220–350K`** (`~0.4–0.6 s`) | `96× NVLS` + grid barriers (`~0.70–0.80 ms/round`) |
| **`16× B200` (2 nodes, InfiniBand)** | Cross-node NDR/XDR IB | **`~2,500–3,500 tok/s`** | **`~4,000–5,000 tok/s`** | — | `~400K+` | **Slower than `8× B200`** (`96×` IB latency `~0.6–0.8 ms`) |
| **`GB200/GB300 NVL72`** (`TP=8×EP=8` + Megakernel) | 72-GPU NVLink 5 fabric | **`~5,500–7,500 tok/s`** | **`~8,000–10,000 tok/s`** | **`~11,000–15,000`** *(refactor; 20K lab ceiling)* | **`~1M+`** (`<0.2 s`) | 48 sequential visits × cross-tray NVLink latency |
| **`1× Vera Rubin` (`R200`, HBM4)** | `~13–20 TB/s` HBM4 (no TP) | **`~3,000–4,200 tok/s`** | **`~4,500–6,000 tok/s`** | `~7,000–10,000` | **`~100–180K`** (`~0.8–1.2 s`) | Single-GPU launch floor (`~0.35 ms`) |
| **SRAM ASICs** *(Cerebras / Groq / MatX)* | `80 TB/s – PB/s` on-chip SRAM | **`~3,500–6,000 tok/s`** | **`~8,000–15,000+ tok/s`** | — | **`~500K+`** | Multi-session `256K–1M` KV SRAM capacity |

### Long-Context Resilience (`128K` Decode Penalty)
- On **`2× GB10`**, reading `+2.15 GB` of `FP8` KV cache at `128K` adds `+4.6 ms/round` (`165 → 102 tok/s`, `-38%`).
- On **`1× B200`** (`6.2 TB/s` sustained), the exact same `2.15 GB` adds only **`+0.35 ms/round`** (`~1,900 → ~1,500 tok/s`, `-20%`).
- On **`8× B200 TP=8`**, sharded KV reads add only **`+0.05 ms/round`** — decode speed remains virtually flat from `2K` through `256K`.

---

## 4. Microsecond-by-Microsecond Audit: Megakernel on `GB200 NVL72`

To hit **`20,000 tok/s`** at `α + 1 = 10`, total round time must be **`≤ 500 µs`**. Subtracting even an aggressive `120 µs` for parallel drafting leaves **`380 µs / 48 visits = 7.9 µs per layer visit`**.

### Realistic Per-Visit Breakdown (`M=16`, `64× GB200 NVL72`, `TP=8 × EP=8`, Persistent Megakernel)

| Sequential Stage Inside 1 Visit (×48) | Realistic Time (`µs`) | Physical Cause |
|---|---:|---|
| **Attention sub-layer** (`gates` + `norm` + `Wproj` + `conv1d` + `attn` + `Wo`) | `1.5 – 2.2 µs` | Weight HBM read is `~0.5 µs`; remainder is register/SRAM pipeline & attention reduction |
| **`AllReduce #1` (`TP=8` NVLink 5 `multimem`)** | `1.8 – 2.5 µs` | `74–147 KB` payload; real hardware latency for 8-GPU NVLS multicast + barrier |
| **MoE sub-layer** (`router` + `W13` + `W2`, `EP=8` straggler) | `2.0 – 3.0 µs` | At `M=16`, `~55/128` unique experts activate; binomial straggler across 8 EP groups executes `~10–11` experts (`~4.2 MB` NVFP4 / `8.4 MB` MXFP8 + MMA/scatter) |
| **`AllReduce #2` (64-GPU cross-tray NVL72)** | `3.0 – 5.0 µs` | Rack-scale 64-GPU NVLink Switch multicast/reduction across trays is materially slower than single-tray 8-GPU NVLS |
| **Megakernel grid-wide SM barriers (2–3×/visit)** | `0.8 – 1.5 µs` | Global SM synchronization between dependent sub-stages |
| **Total per layer visit** | **`9.5 – 14.0 µs`** | **Cannot physically reach `7.9 µs` in real silicon** |
| **× 48 visits + sharded LM head (`~30 µs`)** | **`485 – 700 µs`** | **Production engineering reality: `~600–700 µs` verify** |

---

## 5. The Speculative Acceptance Wall (`α + 1` Reality Check)

Even with a world-class `~600 µs` megakernel verifier, reaching `14K–20K tok/s` hits two hard speculative-decoding realities:

1. **Autoregressive drafters (`DSpark`) cannot scale to `Γ=15` at sub-millisecond verify times:**
   - 15 sequential draft steps × `25–40 µs/step` = `375–600 µs`, consuming the entire round budget.
   - **Mandatory requirement:** A **1-shot parallel block-diffusion drafter (`DFlash 2`-style, `B=16`)** whose 5 layers execute in one forward pass (`~115–150 µs` on `TP=8` B200 including sharded NVFP4 head). This part *is* physically achievable.
2. **Geometric decay caps `α + 1` on open-ended tasks:**
   $$\mathbb{E}[\alpha + 1] = \frac{1 - p^{\Gamma + 1}}{1 - p}$$
   - Empirically, `DFlash 2` (`Γ=7`) achieves `α + 1 = 3.0–4.1` on general coding/chat, corresponding to per-token exact match **`p ≈ 0.75–0.80`**.
   - At `p = 0.78`, doubling draft length from `Γ=7` to `Γ=15` only raises `α + 1` from **`4.0 → 4.5`** — **not** `10`.
   - Achieving **`α + 1 = 10`** at `Γ=15` requires **`p ≥ 0.925` across 15 consecutive tokens**. That occurs on **heavy refactoring, boilerplate, structured JSON/tool edits, or context-copy (suffix/tree-16)** — not general reasoning or chat.

### Honest Workload-Stratified Expectation (`0.65 – 0.85 ms/round`)

| Workload Class | Realistic `α + 1` (`B=16` DFlash + Tree/Suffix) | `8× B200` (`TP=8` + Megakernel) | `64× GB200 NVL72` (`TP=8×EP=8` + Megakernel) | Assessment |
|---|---:|---:|---:|---|
| **Open Chat / Reasoning** | `4.0 – 5.0` | **`~5,000 – 6,500 tok/s`** | **`~5,500 – 7,500 tok/s`** | Solid, repeatable |
| **Real Agentic Coding** | `5.0 – 6.5` | **`~6,000 – 8,000 tok/s`** | **`~6,500 – 9,500 tok/s`** | **Realistic flagship target** |
| **Refactor / Boilerplate / Copy** | `8.5 – 11.0` | **`~10,000 – 13,000 tok/s`** | **`~11,000 – 15,000 tok/s`** | Achievable peak on predictable code |
| **Synthetic / Lab Record (`20K`)** | `≥ 12` @ `0.55 ms` | Out of reach | **`~18,000 – 20,000 tok/s`** | Theoretical ceiling only; do not commit as SLA |

---

## 6. Engineering & Economic Verdict

1. **Portability of `ingine-tp` (`~90%` direct reuse across Blackwell):**
   - `GB10 (sm_121a)`, `RTX 5090 / PRO 6000 (sm_120)`, and `B200 / GB200 (sm_100a)` share the same Blackwell NVFP4/FP8 Tensor Core MMA, Cluster DSMEM, and PDL execution model.
   - Moving to `B200` / `8× B200` requires only **three targeted changes** before any megakernel work:
     1. Recompile for `sm_100a` and retune CTA wave/grid sizing from `48 SMs` (`GB10`) to `148–160 SMs` (`B200`), especially `message_gates`, `router`, and small-M MoE/dense GEMVs.
     2. Add an **NVLink 5 / NVLS (`multimem`) / NVSHMEM** backend under our existing decoupled `tp_comm` interface (replacing `roce_uma`).
     3. Enable continuous batching (`cN`) to monetize the massive unused HBM bandwidth and `192 GB/GPU` capacity on `B200`.
2. **When to build a Persistent Megakernel (and when NOT to):**
   - **Do NOT build for `1× B200` (or `RTX PRO 6000` / `DGX Spark`):** CUDA Graph + PDL already gets within `~20–25%` of the single-GPU hardware floor with clean, maintainable, SASS-verified code.
   - **DO build for `8× B200` (`TP=8` NVLink 5) if single-stream ultra-low-latency (`5K–10K tok/s`) is the product goal:** Here launch overhead (`~0.40 ms`) dominates DRAM (`~0.15 ms`), so fusing each visit with inline `multimem.ld_reduce` delivers a real **`+40–50%` speedup**.
   - **Do NOT dedicate `64× GB200 NVL72` to a single stream:** Cross-tray 64-GPU synchronization erodes single-stream scaling; partition `NVL72` into **9–18 `TP=4`/`TP=8` serving islands** for `>200,000+` aggregate tok/s across concurrent coding agents at `3,000–6,000 tok/s` each.
3. **Never span `TP` across InfiniBand for a 30B-A3.6B model:** Keep `TP` strictly inside one NVLink domain (`≤8` GPUs on HGX B200) and scale out horizontally via replicas or disaggregated prefill/decode.
