# Frankengine

**High-throughput, strict-lossless C++/CUDA inference engine for `Inkling-2-Mini` (`30B-A3.6B` Looped MoE) on NVIDIA Blackwell GPUs (`GB10` DGX Spark `sm_121a`, `RTX 5090`/`PRO 6000` `sm_120`, and `B200`/`GB200` `sm_100a`).**

Built on top of upstream [`ingine`](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine), Frankengine adds a custom **dual-rail 200 GbE RoCE UMA RDMA collective backend (`tp_comm/`)**, **Tensor Parallel (`TP=2`)** execution, **true GPU step-level continuous batching (`--max-batch 1..8`)**, **sharded `NVFP4` LM head**, **eager-captured `FP8` Graph `DFlash 2` block-diffusion drafting**, **context-aware hybrid speculation (`DFlash 2` + global `DSpark`)**, and **256K live OpenAI-compatible serving**.

<p align="center">
  <img src="docs/assets/inkling2_dual_dgx_spark_infographic.png" alt="Inkling-2-Mini on 2× NVIDIA DGX Spark Infographic" width="680"/>
</p>

---

## Verified Production Benchmarks (`2× NVIDIA DGX Spark`, GB10 `sm_121a`)

Each DGX Spark (`GB10`) has **128 GB LPDDR5X unified memory** with **`231.38 GB/s` measured streaming DRAM bandwidth** (vs. `1,792 GB/s` RTX 5090 / `8,000 GB/s` NVIDIA B200), linked via dual-rail 200 GbE RoCE.

### Live OpenAI HTTPS Serving (`256K` Context, `DRAFT_TYPE=hybrid`)
| Workload | Prefill / TTFT | Decode Throughput | Notes |
|---|---:|---:|---|
| **Short Code (greedy, `DFlash 2`)** | `88 ms` (`36` tok) | **`243.1 tok/s`** | `248.7 tok/s` warm (`2 ms` TTFT, `400` tok) |
| **32K Context Sampled Code (`temp=0.7`)** | **`9 ms`** (prefix cache) | **`199.9 tok/s`** | Live OpenAI API, thinking off |
| **Short Code Sampled (`temp=0.7`, thinking on)** | `141 ms` | **`161.8 tok/s`** | Exact TP=2 sampled head (`6.3×` 32-CTA gather) |
| **Cold Prefill (`4K / 8K / 16K`)** | `1.39 / 2.76 / 5.43 s` | — | **`3,200 / 3,097 / 3,081 tok/s`** |
| **Real Multi-Turn Coding Agent (`24K–27K` ctx)** | `86–208 ms` | **`125–220 tok/s`** | Prefix-cache `extend` hit every turn |
| **Needle-in-a-Haystack (`65.5K` / `121.6K` live)** | `23.7 / 47.4 s` | **`181.6 / 129.2 tok/s`** | **100% exact passkey retrieval** (`27.2 GiB` / `256 GiB` used) |

### Interleaved CLI Qualification (`M5 Baseline → Unified TP=2`)
| Workload | M5 Baseline | Unified TP=2 (`hybrid`) | Delta | Parity |
|---|---:|---:|---:|---|
| **Built-in Chat** (6-run median) | `120.4 tok/s` | **`158.6 tok/s`** (`α+1=3.82`) | **`+31.7%`** | `33/33` golden in all modes |
| **Built-in Code** (6-run median) | `124.9 tok/s` | **`135.1 tok/s`** (`α+1=3.00`) | **`+8.2%`** | `65/66` top-1, `KL 0.001927` |
| **2K** (Cold Prefill / Decode) | `1,755` / `165.4 tok/s` | **`2,613` / `165.3 tok/s`** | **`+48.9%` prefill** | `768/768` exact IDs |
| **32K** (Cold Prefill / Decode) | `1,841` / `153.6 tok/s` | **`2,802` / `153.1 tok/s`** | **`+52.2%` prefill** | `768/768` exact IDs |
| **128K** (Cold Prefill / Decode) | `1,694` (`77.4s`) / `102.7` | **`2,471` (`53.0s`) / `102.3`** | **`+45.9%` (`-24.3s`)** | `512/512` exact IDs |

- **Strict-lossless verified:** `52,815,312`-byte FP32 target logit dump is **SHA-256 byte-identical** to baseline; all 28 target GEMV kernels and decode RoCE collectives match stock SASS byte-for-byte.
- **Physical DRAM saturation on GB10:** `NVFP4 MoE W13` ~96%, `FP8 Full Attn (32K–128K)` 94–101%, `FP8 Wproj` 90–91%, `NVFP4 LM Head` ~90%.

### Single-GPU `TP=1` Benchmarks (`1× NVIDIA DGX Spark`, `256K` Context, Zero-Copy NVFP4 Head)
| Workload (`1× GB10`) | `TP=1` DSpark Control | **`TP=1` Frankengine (`hybrid`)** | vs. Control / SOTA |
|---|---:|---:|---|
| **Live Short Code (greedy, `DFlash 2`)** | `123.8 tok/s` (`397 ms` TTFT) | **`155.35 tok/s`** (`207 ms` TTFT) | **`+25.5%`** vs DSpark · **`+33.3%`** over TensorFold (`116.5 tok/s`) |
| **CLI Built-in Chat / Code** (6-run median) | `79.1 / 84.2 tok/s` | **`86.8 / 95.5 tok/s`** | **`+9.7% / +13.4%`** |
| **CLI `2K / 32K / 128K` Decode** | `105.3 / 94.6 / 64.4 tok/s` | **`104.8 / 94.7 / 64.8 tok/s`** | Exact parity (`768/768`, `512/512`) |
| **CLI `2K / 32K / 128K` Cold Prefill** | `2,904 / 2,747 / 2,318 tok/s` | **`2,889 / 2,770 / 2,329 tok/s`** | Zero RoCE overhead |
| **Live `65.5K / 121.6K` Needles @ `256K`** | — | **100% Exact Passkey** (`25.1 / 51.5 s` TTFT) | **`26.1 GiB`** (`26,733 MiB`) peak NVML (`0 MB` extra head) |

### True GPU Continuous Batching (`--max-batch 8`, `256K × 8`, `c=1..8` Live OpenAI HTTP/SSE)
Step-level continuous batching with chunked prefill (`2048`-token chunks interleaved with active decode steps), per-slot prefix reuse, automatic solo-slot `DFlash 2`/`DSpark` interruption & resume, and zero `c=1` SASS/throughput regression (`TP=2` Short Code `250.73 tok/s`; `TP=1` `154.99 tok/s`).

| Concurrency | `TP=2` Short Greedy (`agg / stream`) | `TP=2` Short Sampled 256-tok (`agg / stream`) | `TP=1` Short Greedy (`agg / stream`) | `TP=1` Short Sampled 256-tok (`agg / stream`) | HTTP 200 |
|---:|---:|---:|---:|---:|---:|
| **`c = 1`** | `135.1 / 141.0 tok/s` | `173.2 / 239.4 tok/s` | `76.4 / 82.1 tok/s` | `156.4 / 182.5 tok/s` | `100%` |
| **`c = 2`** | `126.4 / 65.2 tok/s` | `122.3 / 62.8 tok/s` | `79.4 / 40.8 tok/s` | `81.2 / 41.7 tok/s` | `100%` |
| **`c = 4`** | `171.6 / 44.6 tok/s` | `182.6 / 47.4 tok/s` | `105.8 / 27.4 tok/s` | `118.0 / 30.6 tok/s` | `100%` |
| **`c = 6`** | `205.3 / 35.8 tok/s` | `231.1 / 40.4 tok/s` | `125.0 / 21.8 tok/s` | `153.6 / 26.9 tok/s` | `100%` |
| **`c = 8`** | **`226.5 / 29.9 tok/s`** | **`266.3 / 34.9 tok/s`** | **`138.3 / 18.2 tok/s`** | **`181.8 / 24.2 tok/s`** | **`100%` (`40/40` queued)** |

- **Peak NVML memory at `262K × 8` slots:** `29.21 GiB` (rank 0) / `27.01 GiB` (rank 1) on `TP=2` (`~92+ GiB` free per Spark); `53.74 GiB` on `TP=1`.
- **4K context sampled (`c=1/4/8`):** `TP=2` `77.1 → 102.1 → 103.7 tok/s` agg (`29.0/stream` at `c=8`); `TP=1` `52.9 → 70.3 → 73.7 tok/s` agg.

---

## Repository Layout

```text
Frankengine/
├── engine/         # C++/CUDA Blackwell engine (TP=2 + TP=1, NVFP4/FP8/MXFP8, DSpark + DFlash 2)
├── tp_comm/        # Custom dual-rail RoCE UMA RDMA backend (48-CTA bulk FP32 AllReduce + 32-CTA 6.3x gather)
├── scripts/        # Generic TP=1 & TP=2 launchers (config.sh, local.sh.example, serve_tp1.sh, serve_tp2.sh, run_tp2.sh, llmstat)
├── docs/           # Engineering roadmap, TP=2 architecture, B200/GB200 Megakernel scaling audit, infographic
├── reports/        # Full interleaved qualification & ablation reports (TP=2 + TP=1)
├── LICENSE         # Apache-2.0
└── NOTICE          # Full upstream and third-party copyright & license attributions
```

---

## Quickstart (`DGX Spark` `sm_121a` · `TP=1` or `TP=2`)

### 1. Build Engine & `tp_comm`
```bash
cmake -S engine/native -B engine/native/build-gb10-tp \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=121a \
  -DINKLING_COMM_CUDA_ARCHITECTURES=121a \
  -DINKLING_COMM_SOURCE_DIR="$(pwd)/tp_comm" \
  -DINGINE_BUILD_TESTS=ON

cmake --build engine/native/build-gb10-tp -j 12 \
  --target ingine-serve ingine-bench ingine-validate
```

### 2. Configure Your Node Paths (Gitignored Local Override)
```bash
cp scripts/local.sh.example scripts/local.sh
# Edit scripts/local.sh with your model paths (and worker SSH alias / RoCE IP for TP=2)
```

### 3. Serve Live (`256K` Context + Hybrid `DFlash 2` / `DSpark`)
```bash
# Single-GPU (TP=1):
./scripts/serve_tp1.sh

# Dual-Node Tensor Parallel (TP=2):
./scripts/serve_tp2.sh
```
Measure live TTFT, prefill, and decode throughput from any `curl` command using the bundled [`scripts/llmstat`](scripts/llmstat):
```bash
curl -sSN http://127.0.0.1:8834/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": "Write a concise Python LRU cache class using OrderedDict with get and put methods."}],
    "temperature": 0,
    "max_tokens": 160,
    "enable_thinking": false,
    "stream": true,
    "stream_options": {"include_usage": true}
  }' | ./scripts/llmstat
```

---

## Key Documentation
- [**`docs/RTX5090_AGENT_RUNBOOK.md`**](docs/RTX5090_AGENT_RUNBOOK.md) — **Autonomous agent runbook** for RTX 5090 / RTX PRO 6000 / single-GPU Blackwell (`sm_120a` build, pinned HF downloads, 256K launch, and turnkey `scripts/bench_matrix.py` suite).
- [**`docs/OPTIMIZATION_ROADMAP.md`**](docs/OPTIMIZATION_ROADMAP.md) — Living experiment ledger (`EXP-001`–`EXP-027`), byte-level roofline audits, dead-end registry, and optimization roadmap.
- [**`docs/hardware_scaling_and_megakernel_audit.md`**](docs/hardware_scaling_and_megakernel_audit.md) — Calibrated hardware scaling projections (`RTX 5090/PRO 6000` → `1×/8× B200` → `GB200 NVL72` → `Vera Rubin`) and microsecond-by-microsecond Megakernel engineering verdict.
- [**`docs/tp2_architecture.md`**](docs/tp2_architecture.md) — Tensor-parallel sharding and RoCE collective design.
- [**`reports/phase_c_tp2_unified_qualification.md`**](reports/phase_c_tp2_unified_qualification.md) — Final Dual-Spark `TP=2` CLI, live SSE, SASS parity, and 65K/122K needle qualification report.
- [**`reports/track1_tp1_frankengine.md`**](reports/track1_tp1_frankengine.md) — Single-Spark `TP=1` Frankengine qualification report (`155.35 tok/s` live short code, `256K` live needles).
- [**`reports/track2a_continuous_batching.md`**](reports/track2a_continuous_batching.md) — True GPU continuous batching (`--max-batch 1..8`) qualification report (`266.3 tok/s` `TP=2` & `181.8 tok/s` `TP=1` at `c=8`, zero `c=1` SASS regression).
- [**`reports/track2b_suffix_spec.md`**](reports/track2b_suffix_spec.md) & [**`reports/track2c_prefill_accel.md`**](reports/track2c_prefill_accel.md) — Suffix-copy speculation evaluation & Nsight 32K prefill roofline audit.

---

## Credits & Code Lineage

Frankengine stands on the shoulders of outstanding open-source work. All original licenses and copyright notices are preserved in [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE):

### Core Engine Lineage (Apache License 2.0)
1. **[`Neroued/ninfer`](https://github.com/Neroued/ninfer) (NInfer)** — Foundational C++/CUDA inference engine design.
2. **[`satellitedown/cinference`](https://github.com/satellitedown/cinference)** (commit `383e5db`) — C++ engine evolution derived from NInfer.
3. **[`satellitedown/fngine`](https://github.com/satellitedown/fngine)** — CUDA-graph runtime, weight-loading, OpenAI HTTP server, and Jinja/tokenizer pipeline.
4. **[`satellitedown/inkling-2-mini-abliterated-ingine`](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine) (`ingine`, base commit `556acc7`)** — Single-GPU `sm_120a` Blackwell C++/CUDA engine for `Inkling-2-Mini` (NVFP4/MXFP8 MoE kernels, FP8 dense projections, prefill, DSpark speculative loop, and OpenAI server) from which Frankengine's `engine/` directory is forked and extended for `sm_121a`, `TP=2`, sharded NVFP4 head/drafter, and hybrid `DFlash 2`.

### Models & Speculative Drafters
- **[`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** (Thinking Machines, Apache-2.0) — Base `30B-A3.6B` looped MoE model and `DSpark` (`mtp.safetensors`) drafter.
- **[`incoai/Inkling-2-Mini-DFlash2`](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2)** (Inco AI, Apache-2.0) — 0.5B 5-layer parallel block-diffusion drafter weights and architecture implemented natively in Frankengine's FP8 CUDA-graph + TP=2 sharded NVFP4 head path.

### Benchmarks, Architecture Cross-Checks & Methodology
- **[`slatesys/lunomi-inkling`](https://github.com/slatesys/lunomi-inkling) (Basalt)** — Reference RTX 5090 Inkling-2-Mini engine comparisons and optimization insights.
- **[`tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark`](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark)** — Single-Spark vLLM + TensorFold + DFlash 2 reference benchmarks.
- **[`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill)** — Disciplined roofline-first, noise-controlled interleaved A/B hill-climbing and full-context qualification methodology.

### Vendored Third-Party Dependencies (`engine/native/third_party/`)
All vendored libraries retain their original licenses and copyright notices (see [`NOTICE`](NOTICE)): `cpp-httplib` (MIT), `libwebp` (BSD-3-Clause), `llama-jinja` (MIT), `nlohmann/json` (MIT), `spdlog`/`fmt` (MIT), `stb_image` (MIT/PD), and `utf8proc` (MIT).

*Unofficial community project; not affiliated with or endorsed by Thinking Machines, Inco AI, or NVIDIA.*
