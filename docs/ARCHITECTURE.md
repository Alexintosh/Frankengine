# Frankengine — Architecture & Kernels

Frankengine is a C++/CUDA inference engine built for **`Inkling-2-Mini` (`30B-A3.6B` Looped MoE)** on NVIDIA Blackwell GPUs (`sm_120a` RTX 5090 / RTX PRO 6000, `sm_121a` DGX Spark GB10, `sm_100a` B200 / GB200).

## Core Technical Highlights

### 1. Native Blackwell Kernels & Prefill v3
- **Block-Scaled MoE:** Native `mxf4nvf4` and `mxf8f6f4` Tensor Core kernels for both prefill and batched decode (`BatchDecoder::expert_kernel` hits `99.3%` of GB10's `231.38 GB/s` streaming DRAM roofline).
- **Custom FP8 Dense GEMM (`fp8_gemm.cu`):**
  - `128×128` prefill MMA (`mma.sync.m16n8k32`, `149.4 TFLOP/s` on GB10) with fused `OUTER_VEC_32F` epilogue.
  - High-occupancy small-grid decode tiles (`64×64`, `32×32`, `16×32`, `16×16`, `16×8`, zero spills) that lift narrow `BatchDecoder` projections (`K`/`V`/`K_dyn`/`V_dyn`/`Wo`) by **`2.1–2.7×`** on 48-SM GPUs while leaving the `128×128` prefill machine code byte-identical.
- **FP8 Global Prefill Attention (`BR64_BK64`):** `ldmatrix.b8.trans` FP8 attention reaching **`3,930 tok/s`** cold 128K prefill on `1× DGX Spark` and **`18,743 tok/s`** on `1× RTX PRO 6000`.
- **Fused BF16 Gates & Relative Logits:** Fused `gate_fg_apply_bf16` (both FP8 and PICO12 paths), BF16 MMA `rel_logits`, and zero-copy `NVFP4` LM head (`head_sample_nvfp4` / `head_topk16_nvfp4`).

### 2. Bit-Exact 100% Lossless Mode (`PICO12` + `CERT_HEAD`)
- **`PICO12` (`--dense-pico12`):** Bit-exact 12-bit BF16-native dense weight packing (`75.1%` of BF16 bytes, `0` bits reconstruction error via per-row CSR outliers and 7-op `decode4`), paired with a fused tiled BF16 MMA GEMM (`kb::pico12_gemm`, zero spills, **`1.75–1.84×`** faster than unpack+cuBLASLt on `wproj`, `-38.1 MiB` scratch VRAM).
- **`CERT_HEAD` (`--cert-head`):** Mathematically certified Cauchy-Schwarz greedy LM head using an NVFP4 proxy (`~259 MB`) plus outward-rounded row error bounds to discard provably suboptimal rows and verify all remaining candidates with exact PICO12/BF16 arithmetic — guaranteeing **100% bit-exact greedy (`T=0`) parity**.

### 3. `BatchDFlash2`, AngelSpec `D-cut` & Single-Pass Exact Rejection Sampling
- **`BatchDFlash2` (`GAMMA=7`):** 5-layer batched block-diffusion drafter (`MXFP8` weights + `FP8` KV rings = `1.545 GB` for 32 slots, WMMA GQA, auto-selected when `--dflash2-dir` and `--batch-spec >= 5`), with `32×32K` continuous batching and automatic `256K` solo fallback.
- **Zero-Logit-Buffer Single-Pass Exact Leviathan Rejection Sampling (`INGINE_BATCH_REJECTION_SAMPLE=1`):**
  - `BatchDFlash2` samples from the temperature-scaled top-16 support and returns device-side `(support_ids, support_probs, support_picked)` (`+29 KiB` VRAM).
  - The fused `BF16`, `PICO12`, and `NVFP4` batch heads accumulate exact target support logits and outside-support online softmax/Gumbel-max statistics in a **single pass** with **zero vocab-sized logits buffer** and **zero register spills**.
  - Preserves the **100% exact target distribution** (verified over 8,192 CUDA graph replays and `3×295` `inkqc`) while raising `avg_accept_len` from **`2.86 → 3.74`** at `T=0.8`.
- **AngelSpec-Inspired `D-cut` Verification (`INGINE_BATCH_DCUT=1`):** Global marginal-gain prefix allocation across all active batch lanes (`theta=0.15` auto-tuned for GB10 MoE) with 16-row CUDA graph bucketing (`INGINE_BATCH_DCUT_BUCKET=16`) and collapsed `M_exec` keys.

### 4. Batched Short-Prompt Prefill & Zero-VRAM Persistent Prefix Cache
- **GPU `clone_slot` Forking (`INGINE_MIN_FORK_PREFIX=32`):** Automatically forks shared prefixes (`>=32` net tokens) across concurrent arrivals in a single pass.
- **Batched Short-Prompt Prefill (`INGINE_BATCH_SHORT_PREFILL=1`):** Prefills short prompt suffixes (`<=256` tokens) across all ready slots simultaneously via `BatchDecoder::step` followed by a single batched `batch_draft_slots` call, cutting `c=24` burst TTFT by **45–60%**.
- **Persistent Prefix Anchors:** LRU anchor pool detecting `<|end_message|>` / `<|im_end|>` boundaries with zero extra VRAM allocation (up to **`19.9×`** TTFT reduction on repeated 6K system prompts).

### 5. Dual-Node `TP=2` RoCEv2 & Live Observability
- **Dual-Rail 200 GbE RoCEv2 UMA Backend (`tp_comm/`):** 48-CTA FP32/BF16 AllReduce, CUDA-Graph-safe `K=4` pipelined bulk RDMA (`>=8 MiB`), `bf16-moe` wire (`Wo` stays FP32 for conv1d precision), `6.3×` 32-CTA sampled gather, and 50/50 Megatron-sharded solo `DFlash 2` (`-22.9%` draft latency). See [`docs/tp2_architecture.md`](tp2_architecture.md).
- **Vision ON & Live Telemetry:** Built-in ViT multimodal support enabled by default (`TP=1`), lock-free `GET /v1/telemetry` (JSON) & `GET /metrics` (Prometheus), plus bundled [`.omp/extensions/frankengine-parallel.ts`](../.omp/extensions/frankengine-parallel.ts) for single-chat multi-lane fan-out.

## Repository Layout

```text
Frankengine/
├── engine/         # C++/CUDA Blackwell engine (TP=1 & TP=2, FP8/PICO12/NVFP4, DFlash2, BatchDFlash2, Prefix Anchors)
├── tp_comm/        # Custom dual-rail RoCEv2 UMA RDMA backend
├── scripts/        # TP=1 & TP=2 launchers, colab-g4 RTX PRO 6000 kit, bench_matrix.py, llmstat
├── .omp/           # Single-chat parallel fan-out extension (frankengine-parallel.ts)
├── docs/           # Benchmarks, architecture, RTX 5090/Blackwell runbook, optimization roadmap
├── reports/        # Full interleaved qualification & ablation reports
├── LICENSE         # Apache-2.0
└── NOTICE          # Upstream & third-party copyright and license attributions
```
