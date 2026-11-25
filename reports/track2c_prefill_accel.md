# Track 2c — Prefill roofline and acceleration

The stock CUDA 13.0.88 build completed for GB10 sm_121a with tests enabled in `/tmp/track2c-prefill-build` on rank0. The communications worktree started at `7bc250ac693adbe15013fa3744904d27cd78bf69`. The engine worktree started at `a62aa5ab3783bb15c01adb13964aabd0e13769e1`. TP=2 used two RoCE rails and FP32 reductions.

## Stock Nsight Systems traces

Times are summed rank-0 CUDA kernel durations in milliseconds, from the initial prefill widen through the GPU head work. Elapsed GPU span includes gaps. Each run used one cold agentic prompt and 16 teacher-forced decode steps. The decode steps are excluded from this table. The additional rank-0 candidate merge on TP=2 sets the end boundary. Initial library setup and profiling overhead affect the elapsed span, so use the unprofiled runs for throughput comparisons.

| Stage, ms | TP=1 2K | TP=1 32K | TP=2 2K | TP=2 32K |
|---|---:|---:|---:|---:|
| Elapsed GPU span | 1192.502 | 12523.878 | 1168.952 | 12026.316 |
| Summed kernels | 701.773 | 12011.195 | 687.918 | 11495.604 |
| Grouped MoE GEMMs | 317.282 | 4824.607 | 180.050 | 2884.134 |
| ↳ NVFP4 gate/up | 180.695 | 2768.395 | 89.954 | 1445.211 |
| ↳ NVFP4 down or down partial | 104.705 | 1609.776 | 72.771 | 1162.933 |
| ↳ MXFP8 gate/up | 20.470 | 286.781 | 10.046 | 158.894 |
| ↳ MXFP8 down or down partial | 11.412 | 159.655 | 7.279 | 117.097 |
| Dense FP8 to BF16 dequant | 15.396 | 170.872 | 7.064 | 111.017 |
| cuBLAS kernel launches | 95.374 | 1575.696 | 72.534 | 1178.334 |
| Attention | 82.186 | 2363.889 | 42.264 | 1181.773 |
| Relative logits | 26.470 | 427.705 | 13.227 | 206.268 |
| Attention preparation | 10.206 | 162.147 | 3.828 | 61.554 |
| Sliding ring gather and update, bounds | 2.034 | 23.406 | 1.151 | 15.326 |
| Mix | 61.995 | 996.186 | 61.713 | 972.550 |
| Static convolution and state update | 9.019 | 173.215 | 9.484 | 151.912 |
| MoE top-k and routing | 1.602 | 22.487 | 1.532 | 25.212 |
| MoE combine | 16.847 | 274.030 | 34.563 | 550.815 |
| Gate inputs and apply | 62.062 | 982.103 | 61.641 | 980.613 |
| RoCE FP32 GPU pack | — | — | 24.372 | 379.772 |
| RoCE FP32 GPU publish and wait | — | — | 151.203 | 2446.146 |
| RoCE FP32 GPU reduce | — | — | 16.350 | 253.516 |
| RoCE FP32 GPU release | — | — | 0.240 | 3.735 |
| Other kernels | 1.300 | 14.853 | 6.703 | 92.925 |

TP=2 ran 96 bulk reductions at 2K and 1,536 at 32K. The trace's GPU publish and wait time includes proxy polling, verbs posting, completion handling and peer synchronization. Grouped MoE GEMMs were the largest compute stage in all four traces. At 32K, the sum of bulk communication kernels was 3,083.169 ms on TP=2. Stock traces and per-kernel counts are on rank0 in `/tmp/track2c-profile`.

## TP=2 host proxy instrumentation

A separate trace build added NVTX ranges around verbs posting and completion polling. It used a scratch copy of the communication source.

| Rank-0 elapsed sum, ms | 2K, 96 calls | 32K, 1,536 calls |
|---|---:|---:|
| GPU pack | 24.345 | 381.382 |
| GPU publish and wait | 139.767 | 2427.808 |
| GPU reduce | 16.226 | 265.519 |
| GPU release | 0.226 | 3.916 |
| Publish-kernel entry to proxy post start | 0.901 | 13.504 |
| Host verbs posting | 0.262 | 1.298 |
| Host CQ polling after posting | 138.939 | 2418.817 |

Kernel entry to proxy post includes GPU publication and proxy pickup. The proxy busy-polls. The CQ range includes payload transfer, polling, and any difference in peer readiness. These traces do not resolve a wire-only duration or a standalone GPU-publication-to-proxy-pickup duration. The cross-clock CQ-end to GPU-exit residual was negative by about 4 μs per call, so it is unsuitable for estimating the return-side latency.

## Isolated bulk FP32 control

The isolated test used a graph-captured reduction of 18,874,368 bytes per rank, 30 warmups, and 200 timed replays. It checked every output element in all 230 replays against round-to-nearest FP32 addition. Bandwidth is the logical payload per rank divided by median CUDA event time.

| Rank | Median ms | P95 ms | GB/s | Bit-exact replays |
|---|---:|---:|---:|---:|
| 0 | 1.255616 | 1.270880 | 15.031958 | 230/230 |
| 1 | 1.239008 | 1.247232 | 15.233452 | 230/230 |

The 32K model trace averages 2.007 ms per bulk reduction from pack through release, using the sum of kernel durations. The isolated test removes interleaved GPU compute and measures the collective directly.

At 32K the model trace moved 28.99 GB of logical FP32 payload across the 1,536 operations. This corresponds to 9.40 GB/s from pack through release and 11.85 GB/s if only the combined publish-and-wait interval is counted. These rates use one rank's logical payload. Full-duplex wire byte counters and negotiated line rate use different denominators.

## Four-slice bulk FP32 probe

The probe forked four disjoint 4,718,592-byte slices across CUDA streams. Each stream used an independent dual-rail communicator and a separate pinned proxy core. A CUDA graph joined all four streams before reporting completion. This tests overlap through independent communicators. A single proxy with multiple in-flight sends on each existing rail was not implemented.

| Variant | Rank 0 median ms / GB/s | Rank 1 median ms / GB/s | Exact replays per rank |
|---|---:|---:|---:|
| Single operation, control 1 | 1.269440 / 14.868262 | 1.255088 / 15.038283 | 230/230 |
| Four concurrent slices | 1.447664 / 13.037810 | 1.434832 / 13.154410 | 230/230 |
| Single operation, control 2 | 1.260544 / 14.973193 | 1.248288 / 15.120202 | 230/230 |

The probe reduced bandwidth by about 13% on each rank and was rejected. An earlier launch using different rendezvous ports timed out. The measured launch reused the established port for sequential communicator bootstrap.

## Interleaved engine bracket

The experimental `wgemm.cu` build reduced activation staging from three stages to two while keeping the same weight formats and BF16 MMA arithmetic. Runs used one cold prompt per context, free greedy generation, and 16 decode tokens after the first token. TTFT is the engine's prefill timing. Each cell shows TTFT in seconds, prefill tok/s, and decode tok/s, in that order. Matching continuations also matched the first token.

| TP | Prompt | Control 1 | Candidate | Control 2 | Exact continuation |
|---|---:|---:|---:|---:|---:|
| 1 | 2K | 0.957 / 2,140.8 / 60.2 | 0.975 / 2,099.9 / 57.4 | 0.958 / 2,136.7 / 59.7 | 16/16 |
| 1 | 8K | 3.061 / 2,676.3 / 58.4 | 3.091 / 2,650.3 / 58.3 | 3.050 / 2,685.6 / 55.6 | 16/16 |
| 1 | 32K | 12.006 / 2,729.3 / 53.2 | 12.127 / 2,702.0 / 53.0 | 11.997 / 2,731.3 / 52.7 | 16/16 |
| 1 | 128K | 56.198 / 2,332.3 / 38.7 | 56.790 / 2,308.0 / 38.4 | 56.561 / 2,317.4 / 38.3 | 16/16 |
| 2 | 2K | 0.989 / 2,071.4 / 82.4 | 1.025 / 1,998.4 / 83.4 | 1.020 / 2,007.5 / 81.6 | 16/16 |
| 2 | 8K | 3.105 / 2,638.2 / 80.3 | 3.124 / 2,622.7 / 77.1 | 3.105 / 2,638.1 / 79.6 | 16/16 |
| 2 | 32K | 11.951 / 2,741.9 / 74.7 | 11.977 / 2,735.9 / 76.0 | 11.880 / 2,758.3 / 72.5 | 16/16 |
| 2 | 128K | 53.028 / 2,471.8 / 58.6 | 53.088 / 2,469.0 / 54.8 | 53.155 / 2,465.9 / 55.8 | 16/16 |

The candidate trailed both control runs in seven of eight brackets. At TP=2 128K its prefill rate fell between controls and its decode rate was below both. These 16-token decode samples are sanity checks. The data establish no performance win or zero-regression qualification for the candidate.

The TP=1 candidate achieved 2,650.3 tok/s at 8K and 2,702.0 tok/s at 32K. The supplied TensorFold reference rates are 3,395 tok/s at 8K and 3,162 tok/s at 31K. That leaves a 21.9% gap at 8K and a 14.5% gap using the nearby 31K reference. The 31K and 32K prompt lengths differ, and the TensorFold workload was not rerun on these hosts.

## Quality and disposition

| Validation | Golden chat | Top-1 | Mean KL |
|---|---:|---:|---:|
| TP=1 stock | 33/33 | 65/66 | 0.00390126280368 |
| TP=1 two-stage candidate | 33/33 | 65/66 | 0.00390126280368 |
| TP=2 two-stage candidate | 33/33 | 65/66 | 0.00192697476039 |

TP=1 matched stock on the reported quality metrics. TP=2 passed the 0.0020 mean KL gate against the golden reference. All eight short continuations matched. The extended 768-token and 512-token continuation gates were not run after the candidate failed the performance screen. No native Tensor Core quantization variant was tested.

The full `cuobjdump --dump-sass` output for the stock and two-stage candidate builds' communications libraries compared byte-for-byte equal. Both dumps had SHA-256 `7b3f6b86aa27abae462e402e3bd1bd842d11f16f571c1d54fe405204660da803`. Resource usage for the cache-bypass collective was 40 registers, zero stack bytes and zero local bytes. The probes did not modify the protected decode sources.

No optimization was retained. The communication branch remains at `7bc250ac693adbe15013fa3744904d27cd78bf69`. The engine code remains at `a62aa5ab3783bb15c01adb13964aabd0e13769e1`, with a report-only commit on the experiment branch. The requested build directory was rebuilt from restored source. Binaries, rejected experiment sources, trace exports, exact token logs, and launcher logs are under `/tmp/track2c-profile` on rank0.

Each Track 2c launch acquired the Spark lock for its GPU invocation, and each completed or failed GPU launch reported zero compute PIDs on both hosts afterward. TP=2 worker executables were staged to per-run directories below `/tmp/frankengine_track2c`, checked by SHA-256 before launch, and cleaned up afterward. Track 2b's suite delayed the qualification queue while it held the shared lock.
