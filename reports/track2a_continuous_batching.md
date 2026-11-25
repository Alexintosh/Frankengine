# Track 2a: True GPU Continuous Batching (`--max-batch 1..8`)

## Architecture & Scheduler Design

The server enables continuous batching with `--max-batch 2..8`. The default capacity is 1 and uses the original single-stream scheduler. At capacity 8 and queue size 32, the scheduler accepts 40 simultaneous jobs and sizes the HTTP thread pool to 48 threads.

Each active request owns its sampler and current anchor token. Free slots retain their committed token histories; admission prefers the longest reusable prefix and breaks ties by slot index. Prefill work is bounded to 2048 tokens per chunk, interleaved with active decode steps when other streams are ready. A `TP=2` batch selects the greedy LM head when every active request is greedy; a mixed or sampled batch gathers logits to rank 0, which samples each row independently.

A request starting alone in slot 0 runs the existing speculative generator (`DFlash 2` / `DSpark` hybrid) and checkpoint store. The scheduler checks for newly arriving requests between speculative emissions or prefill chunks; when interrupted, the generator hands off its committed history, unprocessed anchor token, and RNG counter to the batched decode loop until completion. Once active load drains, the next solo slot-0 request automatically re-enables speculative decoding. `TpControl` sends explicit lockstep frames for solo start, chunked prefill, batched decode, and slot release. Vision requests on `TP=1` use slot 0 exclusively.

### GPU Infrastructure (`Part 1`, commit `a3c2e9b`)

- Dedicated batched decode attention (`attn_decode_batch`) launches one cluster per KV head per batch row. Each row indexes its own device table entry for sequence position, KV cache pointers, and conv state pointers, with a private split-attention workspace.
- `conv_residual_epilogue_batch` executes independent rows in one launch and updates each slot's three lag rows independently.
- `Runtime` allocates additional slot states only when `RuntimeOptions::max_batch > 1` (slot 0 is the existing state). Before replaying the width-`B` CUDA graph (`B=1..8`), the runtime packs active tokens and per-visit slot pointers/positions into one device table copy.
- `Decoder` reuses all weight-reading layers. `TP=2` sizes head scratch and collective payloads for `max_batch` and captures both greedy and full-logits head graph variants. `TP=1` returns logits for all rows.
- `/health` reports live `active` and `max_batch` fields alongside `waiting`, `running`, and `max_queue`.

## Qualification Setup

Built with CUDA 13.0.88, `sm_121a`, CTests enabled, and the dual-rail RoCE UMA `tp_comm` backend on `2× NVIDIA DGX Spark (GB10)` (`rank0` + `rank1`). Candidate `ingine-serve` binaries passed SHA-256 verification on both ranks.

Live qualification runs used FP8 dense & KV, NVFP4 LM head, FP32 TP wire, `--no-vision`, `262144`-token context & model limit, `2048 MiB` parking, `--max-batch 8`, and `--max-queue 32`, with hybrid `DFlash 2` + `DSpark` speculation enabled for concurrency scaling and Short Code non-regression.

## Tests & SASS Non-Regression

All six targeted CTests passed, including 40-job queue capacity, capacity rejection, chunk interleaving, width-8 execution, mid-stream HTTP disconnects, and graceful drain:

| CTest | Result |
| --- | --- |
| `test_tp2_kernels` | Pass |
| `kernels_batch` | Pass |
| `model_runtime_batch` | Pass |
| `serve_scheduler` | Pass |
| `serve_server` | Pass |
| `spec_accept` | Pass |

Encoded SASS instructions match `a62aa5a` byte-for-byte across all 12 width-one attention variants, all 246 original dense functions, 24 head functions, 13 MoE functions, and both decode RoCE collective variants (`40` registers, `0` stack, `0` local bytes).

The BF16-KV Golden Chat fixture matched **33/33** greedy tokens on both `TP=2` and `TP=1` with hybrid speculation and an 8-slot runtime.

## Live Cross-Slot Isolation, Parity Diagnosis & Lifecycle

- **Deterministic high-margin isolation (`41..8,893` tokens):** 8 distinct word-repetition prompts (lengths 41, 41, 41, 2250, 2263, 41, 8880, 8893) submitted simultaneously matched standalone greedy non-speculative outputs **8/8** on `TP=2` and **8/8** on `TP=1` (`100%` HTTP 200).
- **Open-ended coding prompts (`52..8,827` tokens) & root-cause verification:** Simultaneous open-ended coding prompts matched 3/8 (`TP=2`) and 4/8 (`TP=1`) against solo `M=1` greedy outputs, diverging coherently at near-tied tokens (e.g., `"I can help..."` vs `"I can implement..."` at token 2 on a 52-token prompt). To distinguish cross-slot contamination from benign `M=1` vs `M>=2` floating-point reduction differences (`gemv_kernel` scalar dot-product at `M=1` vs tensor-core MMA at `M>=2`, and `embed_kernel` block-width reduction `512/256/128/64`), we ran an 8-way **homogeneous** batch (`8×` identical copy of Prompt 2) against the **heterogeneous** 8-way mixed batch (`52..8,827` tokens):
  - All `8/8` slots in the homogeneous batch produced the **identical** `"I can implement a red-black tree in Python, but I can't..."` prefix (`UNIQUE COUNT: 1`).
  - Slot 1 in the heterogeneous mixed batch matched the homogeneous `B=4`/`B=8` output **100% character-for-character across all 48 generated tokens** (`mixed8[1] == homo4[0] == homo8[0]`).
  - **Conclusion:** Zero cross-slot state leakage or corruption; the difference vs solo `M=1` is strictly benign `M=1` GEMV vs `M>=2` tensor-core MMA floating-point tie-breaking.
- **Lifecycle (`TP=2`):** An initial solo `DFlash 2` stream emitted a speculative round before three concurrent requests arrived; mixed greedy + sampled batch decoding ran cleanly, two clients disconnected mid-stream (after 8 and 9 tokens), the remaining two streams completed 256 tokens, active/waiting drained to `0/0`, and the subsequent solo request resumed `DFlash 2` (80 speculative rounds) with zero TP hangs.

## Live Concurrency Scaling (`c = 1..8`)

Aggregate throughput uses elapsed wall time from the first simultaneous client start to the last completion. Per-stream mean uses each client's decode window from first text delta to completion. TTFT measures client time to first text delta. Cap = `256` tokens, thinking disabled, `temp=0.7` with `seed=27`.

### Short Greedy (52-token Coding Prompts)

| Clients | `TP=2` agg / stream tok/s | TTFT p50 / p95 s | HTTP 200 | `TP=1` agg / stream tok/s | TTFT p50 / p95 s | HTTP 200 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 135.11 / 141.03 | 0.087 / 0.087 | 1/1 | 76.40 / 82.08 | 0.244 / 0.244 | 1/1 |
| 2 | 126.43 / 65.16 | 0.130 / 0.172 | 2/2 | 79.38 / 40.82 | 0.194 / 0.258 | 2/2 |
| 4 | 171.62 / 44.57 | 0.219 / 0.369 | 4/4 | 105.79 / 27.42 | 0.339 / 0.572 | 4/4 |
| 6 | 205.30 / 35.84 | 0.317 / 0.576 | 6/6 | 125.02 / 21.76 | 0.482 / 0.888 | 6/6 |
| 8 | **226.52 / 29.89** | 0.439 / 0.815 | 8/8 | **138.33 / 18.17** | 0.650 / 1.232 | 8/8 |

### Short Sampled (`temp=0.7`, 52-token Coding Prompts)

| Clients | `TP=2` agg / stream tok/s | TTFT p50 / p95 s | HTTP 200 | `TP=1` agg / stream tok/s | TTFT p50 / p95 s | HTTP 200 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 107.60 / 111.53 | 0.093 / 0.093 | 1/1 | 74.01 / 76.70 | 0.134 / 0.134 | 1/1 |
| 2 | 124.08 / 63.93 | 0.131 / 0.174 | 2/2 | 79.11 / 40.76 | 0.207 / 0.271 | 2/2 |
| 4 | 159.60 / 46.21 | 0.226 / 0.373 | 4/4 | 104.37 / 27.04 | 0.340 / 0.570 | 4/4 |
| 6 | 191.25 / 36.59 | 0.343 / 0.597 | 6/6 | 124.00 / 21.57 | 0.486 / 0.890 | 6/6 |
| 8 | **217.77 / 30.58** | 0.424 / 0.792 | 8/8 | **132.58 / 18.65** | 0.657 / 1.226 | 8/8 |

### Short Sampled (`temp=0.7`, Full 256-Token Completion Target, 21/21 Full Completions)

| Clients | `TP=2` agg / stream tok/s | TTFT p50 / p95 s | HTTP 200 | `TP=1` agg / stream tok/s | TTFT p50 / p95 s | HTTP 200 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 173.19 / 239.40 | 0.413 / 0.413 | 1/1 | 156.40 / 182.45 | 0.239 / 0.239 | 1/1 |
| 2 | 122.31 / 62.76 | 0.117 / 0.155 | 2/2 | 81.23 / 41.72 | 0.181 / 0.240 | 2/2 |
| 4 | 182.62 / 47.41 | 0.201 / 0.339 | 4/4 | 118.03 / 30.63 | 0.315 / 0.524 | 4/4 |
| 6 | 231.08 / 40.44 | 0.297 / 0.532 | 6/6 | 153.63 / 26.94 | 0.465 / 0.833 | 6/6 |
| 8 | **266.32 / 34.92** | 0.320 / 0.672 | 8/8 | **181.83 / 24.17** | 0.611 / 1.139 | 8/8 |

### 4K Context Sampled (`temp=0.7`, 4,327-token Coding Prompts)

| Clients | `TP=2` agg / stream tok/s | TTFT p50 / p95 s | HTTP 200 | `TP=1` agg / stream tok/s | TTFT p50 / p95 s | HTTP 200 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 77.08 / 135.17 | 1.435 / 1.435 | 1/1 | 52.94 / 80.00 | 1.648 / 1.648 | 1/1 |
| 4 | 102.11 / 39.22 | 4.096 / 4.338 | 4/4 | 70.31 / 24.11 | 4.696 / 5.030 | 4/4 |
| 8 | **103.71 / 28.96** | 10.852 / 11.451 | 8/8 | **73.70 / 16.73** | 12.392 / 13.242 | 8/8 |

### Queue Capacity & Memory Footprint (`262K × 8` Slots)

- **Queue burst:** `40/40` simultaneous requests returned `HTTP 200` on both `TP=2` and `TP=1` (`8` active + `32` queued, zero `503`s).
- **Peak NVML memory (`TP=2`):** **`29,910 MiB` (`29.21 GiB`)** on `rank0`, **`27,655 MiB` (`27.01 GiB`)** on `rank1`.
- **Peak NVML memory (`TP=1`):** **`55,034 MiB` (`53.74 GiB`)** on `rank0`.

## Single-Stream (`c = 1`) Short Code Non-Regression

160-token greedy OrderedDict LRU prompt (1 warmup + 3 timed runs, hybrid `DFlash 2` + packed `DSpark` `0.65`), comparing `--max-batch 8` against `--max-batch 1` on the same binary:

| TP | Capacity 8 median decode tok/s | Capacity 1 median decode tok/s | Delta |
| ---: | ---: | ---: | ---: |
| 2 | **250.73** | 247.31 | `+1.39%` |
| 1 | **154.99** | 155.39 | `-0.26%` |
