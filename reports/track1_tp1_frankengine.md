# Track 1 — Frankengine TP=1 on DGX Spark

Base `165520babf74295c6494883004eec77cde13408f`. Built on rank0 with CUDA 13.0.88, `sm_121a` and `INGINE_BUILD_TESTS=ON`. The build used the production `tp_comm` source at `92437d9e82211e28b32099a171222d99e496d5d4`.

TP=1 DFlash 2 uses the existing host verify loop with local target argmax and no collectives. The runtime-owned 200,058-row NVFP4 head is bound directly. Shared codes and scales account for 259.3 MB of avoided duplication. Hybrid uses the FP8 graph drafter for short greedy prompts and DSpark for sampled or longer prompts. When hybrid selects DSpark it uses the same per-width target graphs and launches acceptance with the request parameters. The standalone DSpark control uses its existing conditional verify and accept graphs. The full-width TP=1 W2 kernel keeps its per-expert BF16 rounding. The M5 split kernel still applies to the half-width TP=2 path.

## Correctness

- CTest passed `test_tp2_kernels`, `spec_accept`, `serve_scheduler` and `serve_server` after configuring the test checkpoint directory.
- DSpark, DFlash 2 and hybrid each matched 33/33 golden chat IDs.
- Logits matched the pinned stock TP=1 results exactly. Both were 65/66 top-1 with mean KL 0.00390126280368 and max KL 0.155381744257. Both exceeded the requested mean KL of approximately 0.00193.
- TP=2 hybrid smoke matched 33/33 golden chat IDs. The build’s `sm_121a` instruction words for MoE, GEMV and DFlash 2 matched the pinned production build.
- The FP8 DFlash 2 probe repeated the same seven proposal IDs after reset at lengths 1, 32 and 64 with fixed taps and anchor. It recorded no capture after warmup.

## CLI benchmarks

Built-in prompts used greedy decoding with the CLI chat template, a 256-token budget and six cold runs per prompt. The table reports median prefill and decode and the last run’s tokens per verify. GPU GB is the CLI CUDA free-memory delta across model load and startup capture.

| Policy | Workload | TTFT (s) | Prefill tok/s | Decode tok/s | α+1 | GPU GB |
|---|---|---:|---:|---:|---:|---:|
| DSpark control 1 | chat | 0.117 | 257.2 | 79.18 | 2.13 | 21.55 |
| DSpark control 1 | code | 0.124 | 322.0 | 84.51 | 2.43 | 21.55 |
| Hybrid .35 | chat | 0.115 | 261.8 | 78.34 | 3.40 | 23.10 |
| Hybrid .35 | code | 0.125 | 321.0 | 90.82 | 3.82 | 23.10 |
| Hybrid .40 | chat | 0.115 | 261.3 | 79.73 | 3.32 | 23.04 |
| Hybrid .40 | code | 0.124 | 321.9 | 92.96 | 3.76 | 23.04 |
| Hybrid .45 | chat | 0.114 | 262.3 | 81.68 | 3.32 | 23.15 |
| Hybrid .45 | code | 0.124 | 322.0 | 97.79 | 3.76 | 23.15 |
| Hybrid .50 sweep | chat | 0.114 | 263.2 | 86.82 | 3.28 | 23.04 |
| Hybrid .50 sweep | code | 0.124 | 322.3 | 95.64 | 3.41 | 23.04 |
| Hybrid .50 final | chat | 0.122 | 246.1 | 86.84 | 3.28 | 23.48 |
| Hybrid .50 final | code | 0.124 | 323.2 | 95.51 | 3.41 | 23.48 |
| Standalone NVFP4 DFlash 2 fixed M=8 | chat | 0.115 | 261.4 | 74.01 | 3.40 | 21.17 |
| Standalone NVFP4 DFlash 2 fixed M=8 | code | 0.125 | 319.4 | 79.67 | 3.51 | 21.17 |
| Standalone FP8 DFlash 2 fixed M=8 | chat | 0.117 | 255.7 | 74.86 | 3.40 | 21.48 |
| Standalone FP8 DFlash 2 fixed M=8 | code | 0.124 | 322.1 | 80.61 | 3.63 | 21.48 |
| DSpark control 2 | chat | 0.114 | 262.3 | 79.09 | 2.21 | 21.99 |
| DSpark control 2 | code | 0.123 | 324.2 | 83.99 | 2.43 | 21.99 |

The default remains 0.50 with the 1,536-token route limit. Among the four adaptive settings, 0.50 had the highest mean throughput across Chat and Code. Both built-in prompts exceeded both DSpark controls at 0.50. At 2K, FP8 adaptive DFlash 2 delivered 103.50 tok/s against the two DSpark controls at 105.05 and 105.52 tok/s, so extending the cutoff through 2K has no measured upside. At 32K, adaptive DFlash 2 was 79.31 tok/s versus 95.17 and 94.12 tok/s for the controls.

Context runs are cold, greedy free continuations using the corpus prompt IDs and budget. Each context ran control 1, the initial hybrid candidate and control 2 in that order. The table uses the final hybrid rerun after the per-width verification change. Prefill columns are item medians. Decode and α+1 are pooled across items, using emitted IDs after the first token for decode throughput.

| Context | Policy | Items | TTFT (s) | Prefill tok/s | Decode tok/s | α+1 | GPU GB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2,048 | ctl1 | 3 | 0.705 | 2904.1 | 105.05 | 3.47 | 21.22 |
| 2,048 | hybrid | 3 | 0.709 | 2889.0 | 104.78 | 3.47 | 22.45 |
| 2,048 | ctl2 | 3 | 0.705 | 2903.7 | 105.52 | 3.47 | 21.72 |
| 2,048 | dflash2 | 3 | 0.734 | 2788.8 | 89.65 | 4.24 | 20.69 |
| 2,048 | dflash2-adaptive | 3 | 0.713 | 2874.0 | 103.50 | 4.09 | 21.23 |
| 32,768 | ctl1 | 3 | 11.785 | 2780.4 | 95.17 | 3.28 | 22.07 |
| 32,768 | hybrid | 3 | 11.830 | 2770.0 | 94.74 | 3.28 | 23.28 |
| 32,768 | ctl2 | 3 | 12.079 | 2712.8 | 94.12 | 3.28 | 22.12 |
| 32,768 | dflash2 | 3 | 11.883 | 2757.6 | 69.36 | 3.47 | 20.77 |
| 32,768 | dflash2-adaptive | 3 | 11.838 | 2768.1 | 79.31 | 3.23 | 21.41 |
| 131,072 | ctl1 | 2 | 56.626 | 2314.7 | 64.75 | 2.69 | 23.84 |
| 131,072 | hybrid | 2 | 56.287 | 2328.6 | 64.75 | 2.69 | 24.93 |
| 131,072 | ctl2 | 2 | 56.468 | 2321.2 | 63.96 | 2.64 | 23.81 |

- 2,048 tokens, DSpark-routed hybrid vs control 1 matched 768/768 continuation IDs.
- 32,768 tokens, DSpark-routed hybrid vs control 1 matched 768/768 continuation IDs.
- 131,072 tokens, DSpark-routed hybrid vs control 1 matched 512/512 continuation IDs.
- At 131,072 tokens, the two stock DSpark controls matched 369/512 continuation IDs against each other.

## Verify and draft roofline

Measured CUDA event times on the 2K free continuation. The reference bandwidth is 231.38 GB/s. Sixteen 1 GiB device-to-device copies sustained 216.60 GB/s counting reads and writes. Verify times include the target and device acceptance. Draft times include the DFlash 2 context commit and proposal.

| Verify width | Verify ms | Draft ms | Round share in draft | Max tok/s at measured costs |
|---:|---:|---:|---:|---:|
| 1 | 16.732 | 4.264 | 20.3% | 47.6 |
| 2 | 22.644 | 4.335 | 16.1% | 74.1 |
| 4 | 30.757 | 4.277 | 12.2% | 114.2 |
| 8 | 42.835 | 4.300 | 9.1% | 169.7 |

The last column assumes full acceptance at each width and excludes control overhead. The total is a ceiling for the measured verify/draft pair at that context. Fine2 supports widths 2–8, so width 1 was measured with the fine grid.

The Nsight Systems short hybrid trace included eager and capture warmup. Per kernel medians were approximately 100 µs for width-8 FP8 Wproj, 40 µs for width-8 FP8 Wo, 224 µs and 118 µs for NVFP4 W13 and W2, and 1.19–1.22 ms for full NVFP4 head projections. The head’s packed codes and scales alone imply 1.20 ms at measured copy bandwidth. NVFP4 prefill expert GEMM medians were 1.03 ms for W13 and 0.50 ms for W2.

A full-width split-W2 trial kept per-expert BF16 rounding and passed 33/33 golden chat. Six-run Chat/Code medians were split-ctl1 87.05/95.77 tok/s, split-w2 87.44/96.24 tok/s, split-ctl2 86.75/95.54 tok/s. The gain was within run variation, so the fused TP=1 kernel with PDL stays in the branch.

## Live OpenAI HTTP/SSE

One request at a time at `--max-ctx 262144 --max-model-len 262144 --park-mb 2048`, with the FP8 KV cache. Client wall-clock TTFT ends at the first text delta. Decode counts completion tokens after the first token.

| Workload | DSpark TTFT (s) | DSpark tok/s | Hybrid TTFT (s) | Hybrid tok/s | Hybrid cached tokens |
|---|---:|---:|---:|---:|---:|
| short-code-160tok | 0.397 | 123.80 | 0.207 | 155.35 | 0 |
| short-chat-160tok | 0.121 | 74.31 | 0.131 | 69.86 | 0 |
| 0/0 cold | 0.641 | 77.36 | 0.637 | 77.33 | 0 |
| 0/0 warm | 0.003 | 77.28 | 0.003 | 77.30 | 1784 |
| short route switch | — | — | 0.120 | 46.37 | 0 |
| greedy restored after DFlash2 | — | — | 0.005 | 77.56 | 1784 |
| 0.7/40 cold | 0.649 | 77.30 | 0.641 | 77.43 | 0 |
| 0.7/40 warm | 0.002 | 77.55 | 0.003 | 77.30 | 1786 |
| 0.7/0 cold | 0.641 | 75.82 | 0.639 | 75.89 | 0 |
| 0.7/0 warm | 0.003 | 75.99 | 0.002 | 75.94 | 1786 |

Short Code reached 155.35 tok/s, 33.3% above the supplied TensorFold reference of 116.5 tok/s and 63.9% of the supplied TP=2 reference of 243 tok/s. The TensorFold number is the supplied reference.

| Needle prompt tokens | TTFT (s) | Exact response | Peak NVML MiB |
|---:|---:|---|---:|
| 65,495 | 25.125 | `CORMORANT-78416-QUARTZ` | 26733 |
| 121,559 | 51.512 | `MANGROVE-29573-COBALT` | 26733 |

Peak sampled per-process NVML residency at 256K configured capacity was 26,733 MiB (28.03 GB).
Both needles returned the exact passkey. The greedy route switch restored the 1,784-token parked prefix and reproduced the warm response.
Both sampled cold/warm pairs matched the fixed-seed DSpark control text.
The 1.8K cold and warm greedy responses matched the DSpark control. The earlier conditional-verify candidate diverged on the cold response.
The short Chat response also differed from the DSpark control. The short Code response matched.
The capture interposer recorded 0 standard and 0 conditional stream captures after readiness across the live matrix.

## Reproduction

Target checkpoint `models/Inkling-2-Mini` and draft checkpoint `models/Inkling-2-Mini-DFlash2` under the shared project root. CLI and serve used dense FP8, FP8 KV, NVFP4 target head, speculation and fine2 widths. DSpark used the packed NVFP4 drafter at short, 2K and 128K contexts. The 32K context used the FP8 DSpark configuration. DSpark thresholds were 0.65, 0.75 and 0.80 at 2K, 32K and 128K respectively. Standalone DFlash 2 used M=8 unless labeled adaptive, in which case it used threshold 0.50 and FP8 weights.

Long inputs came from `bench/agentic-2048.jsonl`, `bench/agentic-32768.jsonl` and `bench/agentic-131072.jsonl`. Each item continued for 256 tokens. The CLI used `--forced-free 0` and `--forced-trace` to retain exact continuation IDs.

Raw CLI traces, live response fixtures and per-process NVML samples are under `/tmp/frankengine-tp1-results` on rank0. Each GPU invocation held `/tmp/frankengine_gpu.lock` and checked both hosts for compute processes at exit. The HTTP clients completed before the server received SIGTERM.

Final GPU compute process inventory was empty on rank0 and rank1.
