# ARCH-06a engineering report

**Status.** Native DFlash2 is available as an opt-in greedy TP=2 drafter. Fixed and adaptive DFlash2 failed strict continuation parity against the matched M5 controls. It remains an experimental branch.

Use `--spec --draft-type dflash2 --dflash2-dir <checkpoint>` on both ranks. `INKLING_DFLASH2_DIR` supplies the directory on rank 0 when the flag is omitted. `--draft-nvfp4` packs the 35 Q/K/V/O and MLP matrices. The default verifies width 8. `--dflash2-adaptive --spec-widths fine2 --spec-threshold 0.35` enables confidence-based width selection. DFlash2 currently uses cold prefill and accepts greedy requests. The stock DSpark fast path remains the default.

### Native integration

The SGLang compatibility patch forwards the DFlash2 feature IDs to the DSpark capture adapter and preserves the final-norm slot at ID 48. Native prefill stores the BF16 residual stream entering zero-based visits 5, 23 and 41, after the preceding residual write, followed by the final RMSNorm output. Decode captures the same mixed partial stream and final norm. The drafter uses the target embedding kernel for the anchor and mask token `200056`. The checkpoint sets `is_causal=false`, so every draft query can read the full proposal block within its sliding window.

Rank 0 commits projected K/V into five 2,048-row rings. It commits only the prefix kept by the verifier and holds proposal-block K/V in separate scratch. A terminal round needs no subsequent draft or commit. Each new request resets this context and prefills independently on both ranks.

The implementation uses option B, a rank-zero copy of the complete packed target head. It adds 259,275,168 bytes of codes and scales and reuses the existing target checkpoint packer. The projection emits BF16-rounded raw logits for only valid target vocabulary rows. The selector applies the correction and then the softcap. This path uses the same existing proposal broadcast as DSpark. Option A would reuse the sharded head with no full-head allocation and reduce the per-rank head weight read, but needs a hidden broadcast and an exact cross-rank top-16 merge before that proposal broadcast. I did not measure option A on hardware.

### Qualification

| Check | Observed result |
|---|---|
| Native CUDA build | Bench, validator and server built with CUDA 13.0.88 for `sm_121a` |
| CUDA selector | 1,792 lattice values and seven greedy tokens matched the unit fixture under the GPU lock |
| Official golden chat, BF16 | **33/33** exactly matched |
| Official golden chat, NVFP4 | **33/33** exactly matched |
| Official width-1 logit fixture | **65/66** top-1, mean KL **0.00192697476039**, maximum KL **0.0226913602959** |
| Official 2,100-token fixture, free generation | BF16 DFlash2 matched 32/33. The width-1 logit gate above is a separate forced-prefix fixture. |
| Real-tap parity | See the capture results below |
| 2K and 32K strict continuation parity | See the per-request comparisons below |

The real-tap capture is enabled with `INKLING_DFLASH2_DUMP=<output-dir>`, and `tools/arch06a/compare_dump.py <checkpoint> <output-dir>` evaluates it with the committed PyTorch reference. The default path performs no dump I/O.
The first built-in chat draft used 30 captured target rows. The BF16 CUDA hidden-state RMSE against PyTorch was 0.06366743 with maximum absolute error 0.578125. Decoding the captured NVFP4 head on CPU and projecting the same native hidden state reproduced 99.986076% of unary values exactly, with maximum absolute error 0.125. Using the independent PyTorch hidden state gave unary RMSE 0.07763824 and maximum absolute error 0.5. The native top-16 matched a stable sort of its logits in all seven slots. Overlap with the independent backbone and head was 16, 16, 16, 16, 16, 16 and 15. The native and independent CPU path both selected `200008 371 150409 171960 149887 161043 149879`. This establishes greedy-token agreement on the capture, with measurable differences in the intermediate tensors.

Strict parity requires the exact adaptive M5 continuations. Archived M5 runs already show that changing DSpark's 2K NVFP4 confidence threshold from 0.65 to 0.70 changes requests 2 and 3 at tokens 106 and 103. On 32K, changing the FP8 threshold from 0.75 to 0.80 changes requests 2 and 3 at tokens 155 and 23. This demonstrates that the existing target arithmetic can produce different greedy continuations across verify schedules. It does not isolate the cause of every DFlash2 divergence.

The 2K M5 width-8 diagnostic used DSpark threshold 0 and reproduced all 256 tokens in request 1 against adaptive M5. Its first divergences for requests 2 and 3 were at tokens 106 and 95. Comparing DFlash2 directly against that width-8 diagnostic still found first divergences at 71, 13 and 94 for BF16, and exact, 13 and 94 for NVFP4. The two drafters can take different accepted-block boundaries even at the same verify width. No tested DFlash2 policy qualified for the strict continuation gate.

## Checkpoint and math

Rank 0 has revision `4f9088ea66c21f2b99ed657d1941c306b03e84d2` in `~/code/inkling-2-mini-engine/models/Inkling-2-Mini-DFlash2`. The safetensors file contains 81 BF16 tensors and 989,966,848 payload bytes. Its SHA-256 is `217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee`. [Pinned checkpoint and compatibility patch](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/tree/4f9088ea66c21f2b99ed657d1941c306b03e84d2)

| Tensor or family | Shape |
|---|---|
| Tap projection and context RMSNorm | `[2304,9216]`, `[2304]` |
| Per-layer Q, K, V, O weights | `[4096,2304]`, `[1024,2304]` twice, `[2304,4096]` |
| Per-layer Q and K norm | `[128]` each |
| Per-layer MLP gate, up, down | `[6912,2304]` twice, `[2304,6912]` |
| Attention and MLP convolution, each | `[2,2,2304]` base, `[576,2304]` projection |
| Selector projection and codebooks | `[256,2304]`, two `[201024,256]` tables |
| Final norm | `[2304]` |

The reference concatenates taps for visits 5, 23, 41 and final norm, projects them once, and uses the resulting context for K/V projections in every layer. The proposal block starts with the anchor embedding and seven mask embeddings. Each layer applies pre-norm attention with Q/K RMSNorm and RoPE, and a pre-norm SwiGLU MLP. Each sublayer uses the same coefficient projection to generate its input and output convolution kernels. A tap at offset one reads the preceding row within the block. The compatibility patch adapts SGLang's capture hooks, valid vocabulary, and softcap placement. The model and worker math lives in SGLang, and the standalone reference follows the inspected source with an unverified sliding-attention boundary and BF16 operation ordering. [SGLang model source](https://github.com/sgl-project/sglang/blob/2070487eccdda795a6a4a00ecb0b9b560612cfee/python/sglang/srt/models/dflash.py) · [Worker](https://github.com/sgl-project/sglang/blob/2070487eccdda795a6a4a00ecb0b9b560612cfee/python/sglang/srt/speculative/dflash_worker_v2.py)

The selector takes raw top-16 target-head logits over the valid vocabulary. It adds a dot product of the successor code and the elementwise product of the predecessor code and projected hidden state, then applies `30 × tanh(score / 30)`. Greedy selection walks one locally best edge per proposal, conditioned on the anchor and then the preceding proposal. The CUDA component consumes precomputed top-16 candidates and projected hidden states, emits the lattice and seven token IDs, and enqueues both launches on the caller's stream.

## GB10 bandwidth model

These are optimistic weight-read latency floors using the advertised 273 GB/s per Spark. They exclude kernel overhead, reductions, collectives, KV reads, activation traffic, and reuse across commit and draft. The two hosts report NVIDIA GB10 with drivers 580.178.04 and 580.159.03. I did not measure sustained bandwidth. [NVIDIA DGX Spark hardware guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)

| Per-round component | Bytes | Single GB10 floor |
|---|---:|---:|
| Five BF16 decoder layers | 740,462,080 | 2.71 ms |
| Same layers with FP8 Q/K/V/O and MLP, before scale bytes | 383,618,560 | 1.41 ms |
| Same layers with NVFP4 Q/K/V/O and MLP, including one byte per 16 weights for scales | 227,499,520 | 0.83 ms |
| BF16 commit K/V matrices and tap projection if read separately | 89,653,248 | 0.33 ms |
| NVFP4 `200058 × 2304` target head, including one byte per 16 weights for scales | 259,275,168 | 0.95 ms |
| Five BF16 K/V windows at 2048 cached rows | 41,943,040 | 0.15 ms |

The two resident selector tables total 205,848,576 bytes. A one-path walk needs at most 119 distinct codebook row reads across the two tables when looking up each predecessor selected by the walk and the 16 successors per slot. The current full-lattice kernel can reload the same rows across CTAs. A rank-0 implementation avoids per-layer draft collectives. Ideal two-way sharding would halve shardable weight traffic, with a floor near 0.42 ms for the NVFP4 backbone if fully shardable. Actual TP=2 savings depend on replicated convolution weights and synchronization on the draft critical path. Head sharding requires exact global top-16 across the two vocabulary shards.

## Interleaved dual-Spark results

The four primary runs used M5 control 1, BF16 DFlash2, NVFP4 DFlash2, and M5 control 2 in that order. Controls use the existing M5 executable at checkout `8306c64`. Runs used FP32 transport, FP8 target dense and KV paths, and NVFP4 target heads. Prefill reuse was disabled. The long workloads use the first three requests from each M5 agentic corpus with 256 generated tokens per request. Rates are aggregated as total tokens over total time. Decode excludes the first output token and prefill. End-to-end includes prefill and all emitted tokens. Draft times for DFlash2 include accepted-context projection and cache updates, with no commit before the first round. M5 did not export round timings in free generation. Additional instrumented DSpark timings use the same draft and verify algorithms in the candidate build. Adaptive NVFP4 uses selector softmax confidence over 16 edges and a threshold of 0.35.

| Workload | Mode | Rounds | Accepted drafts | α+1 | Draft ms | Verify ms | Decode tok/s | End-to-end tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Built-in chat | ctl1 | 113 | 142 | 2.2566 | — | — | 120.761 | — |
| Built-in code | ctl1 | 110 | 148 | 2.3455 | — | — | 125.478 | — |
| Built-in chat | bf16 | 58 | 201 | 4.4655 | 6.218 | 26.449 | 134.444 | — |
| Built-in code | bf16 | 67 | 188 | 3.8060 | 6.138 | 26.072 | 118.046 | — |
| Built-in chat | nvfp4 | 75 | 187 | 3.4933 | 3.914 | 26.397 | 112.049 | — |
| Built-in code | nvfp4 | 75 | 180 | 3.4000 | 3.835 | 25.709 | 114.956 | — |
| Built-in chat | ctl2 | 113 | 142 | 2.2566 | — | — | 120.238 | — |
| Built-in code | ctl2 | 110 | 148 | 2.3455 | — | — | 125.098 | — |
| Built-in chat | adaptive-fine2 | 83 | 175 | 3.1084 | 3.867 | 24.148 | 109.535 | — |
| Built-in code | adaptive-fine2 | 74 | 182 | 3.4595 | 3.785 | 23.823 | 124.673 | — |
| Built-in chat | adaptive-coarse | 83 | 175 | 3.1084 | 3.863 | 24.735 | 107.313 | — |
| Built-in code | adaptive-coarse | 74 | 182 | 3.4595 | 3.782 | 24.187 | 123.073 | — |
| Built-in chat | timed-control | 113 | 142 | 2.2566 | 1.409 | 17.464 | 119.293 | — |
| Built-in code | timed-control | 110 | 148 | 2.3455 | 1.399 | 17.092 | 125.070 | — |
| 2048 | ctl1 | 217 | 551 | 3.5392 | — | — | 166.040 | 94.944 |
| 2048 | bf16 | 179 | 595 | 4.3240 | 10.817 | 27.176 | 112.392 | 74.228 |
| 2048 | nvfp4 | 180 | 590 | 4.2778 | 8.469 | 27.184 | 119.104 | 76.420 |
| 2048 | ctl2 | 217 | 551 | 3.5392 | — | — | 166.512 | 94.910 |
| 2048 | adaptive-fine2 | 186 | 584 | 4.1398 | 8.435 | 24.352 | 125.318 | 79.336 |
| 2048 | adaptive-coarse | 181 | 589 | 4.2541 | 8.486 | 24.967 | 126.211 | 79.799 |
| 2048 | timed-control | 217 | 551 | 3.5392 | 1.475 | 19.758 | 165.610 | 94.848 |
| 32768 | ctl1 | 222 | 551 | 3.4820 | — | — | 154.621 | 13.119 |
| 32768 | bf16 | 228 | 541 | 3.3728 | 10.756 | 28.060 | 86.367 | 12.134 |
| 32768 | nvfp4 | 231 | 544 | 3.3550 | 8.449 | 28.005 | 90.767 | 12.209 |
| 32768 | ctl2 | 222 | 551 | 3.4820 | — | — | 153.766 | 13.114 |
| 32768 | adaptive-fine2 | 235 | 539 | 3.2936 | 8.449 | 26.308 | 93.568 | 12.220 |
| 32768 | adaptive-coarse | 235 | 540 | 3.2979 | 8.465 | 26.608 | 92.728 | 12.219 |
| 32768 | timed-control | 222 | 551 | 3.4820 | 2.013 | 20.330 | 153.808 | 13.231 |

### Acceptance distribution

The bins below count rounds accepting zero through seven draft tokens. Each round also emits one target token.

| Context | Mode | Accepted-draft bins 0–7 |
|---|---|---|
| 2048 | bf16 | 33, 27, 26, 17, 14, 7, 11, 44 |
| 2048 | nvfp4 | 32, 35, 17, 19, 13, 13, 10, 41 |
| 2048 | adaptive-fine2 | 37, 38, 17, 20, 11, 11, 11, 41 |
| 2048 | adaptive-coarse | 33, 36, 16, 20, 13, 11, 10, 42 |
| 2048 | timed-control | 32, 76, 28, 18, 11, 16, 11, 25 |
| 32768 | bf16 | 53, 58, 35, 20, 14, 15, 9, 24 |
| 32768 | nvfp4 | 58, 51, 40, 20, 16, 12, 9, 25 |
| 32768 | adaptive-fine2 | 55, 58, 40, 24, 15, 13, 6, 24 |
| 32768 | adaptive-coarse | 59, 59, 34, 22, 14, 15, 8, 24 |
| 32768 | timed-control | 35, 77, 24, 26, 17, 6, 9, 28 |

### Strict continuation parity

First divergent generated token versus matched M5 control 1, indexed from 1. An exact result means all 256 tokens match for that request. Golden chat and the width-1 fixture are separate gates.

| Context | Mode | Request 1 | Request 2 | Request 3 |
|---|---|---:|---:|---:|
| 2048 | bf16 | 71 | 13 | 94 |
| 2048 | nvfp4 | Exact | 13 | 94 |
| 2048 | ctl2 | Exact | Exact | Exact |
| 2048 | adaptive-fine2 | Exact | 13 | 94 |
| 2048 | adaptive-coarse | Exact | 13 | 94 |
| 2048 | timed-control | Exact | Exact | Exact |
| 2048 | m5-width8 | Exact | 106 | 95 |
| 32768 | bf16 | 7 | 159 | 23 |
| 32768 | nvfp4 | 7 | 159 | 23 |
| 32768 | ctl2 | Exact | Exact | Exact |
| 32768 | adaptive-fine2 | 7 | 155 | 23 |
| 32768 | adaptive-coarse | 7 | 159 | 23 |
| 32768 | timed-control | Exact | Exact | Exact |

### Performance and run inventory

BF16 width 8 gave the highest built-in chat throughput at 134.444 decode tok/s. NVFP4 adaptive fine2 came closest to M5 on built-in code at 124.673 decode tok/s. Adaptive coarse gave the highest DFlash2 throughput on 2K at 126.211 decode tok/s, and fine2 led on 32K at 93.568 decode tok/s. Both remained below their matched M5 controls. NVFP4 mean draft time was 3.8–3.9 ms for the built-ins and 8.4–8.5 ms on the long contexts, above the 1.7–2.2 ms goal. The built-ins exported decode throughput only. The optional 128K run was omitted.

The fixed built-in interleave and both golden chat gates were repeated with the final build. The initial alternate rendezvous port timed out. The retry and interleaved suite used the established M5 port 29620. Launcher traps tracked the exact two rank processes. The final query completed under the lock and returned zero compute processes on each Spark. Candidate bench, server and validator binaries were staged to rank1 under `/tmp/inkling_tp_arch06a` and verified against the rank0 SHA-256 hashes. The production repositories and operational scripts were used read-only. Qualification logs and captures are under `/tmp/arch06a-suite`, `/tmp/arch06a-suite.log` and `/tmp/arch06a-real-taps-final` on rank0.

## Cross-architecture transferability

- **GB10.** Bandwidth and inter-host collective latency both matter at concurrency one. Rank-0 quantization reduces draft weight traffic with no new collective in each decoder layer.
- **B200 and GB200 NVL72.** Blackwell supports FP4, and B200 has much higher per-GPU HBM bandwidth. NVLink in NVL72 changes the economics of head sharding and draft TP. Rebuild for the data-center architecture and remeasure launch and collective costs. [NVIDIA HGX components](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html)
- **H100.** Use BF16 or FP8 Tensor Core kernels and requalify acceptance. Hopper needs a different implementation for FP4-packed weights. [NVIDIA HGX components](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html)
- **RTX 5090.** The native project exposes a separate `120a` build. Re-evaluate single-card model and KV capacity and inter-GPU communication before transferring the dual-Spark configuration.
- **Apple Silicon.** The CPU reference provides a starting point for math fixtures. A production backend needs Metal or MLX implementations of projection, attention, weight packing, and selector operations, with independent rounding and acceptance checks.

## Follow-up on the split-KV and precision experiments

The branch remains experimental. The fastest tested policies reached **151.4 tok/s on built-in Chat** and **137.9 tok/s on built-in Code**, ahead of their matched M5 controls. At **2K** and **32K**, the best policies reached **144.6** and **117.7 tok/s** against M5 at **166.3** and **153.6 tok/s**. The requested 170–200+ tok/s and 1.7–2.2 ms total-draft targets remain open.

### Attention, window geometry, and embedding

The long-window kernel uses 128-key splits, coalesced BF16 pair loads, four warp-local FP32 online softmax accumulators, and a stable merge across splits. Short windows retain the prototype kernel. The CUDA-event microbenchmark repeatedly ran each kernel on synthetic BF16 K/V on one GB10 and scaled per-layer times by five. It excludes the other draft stages and benefits from reusing the same buffers.

| Absolute position | Prototype attention, ms | Split-KV attention, ms | Max absolute BF16 output difference |
|---:|---:|---:|---:|
| 64 | 0.102 | 0.072 | 0.000244 |
| 2,048 | 2.438 | 0.695 | 0.000244 |
| 3,000 | 2.450 | 0.678 | 0.000122 |
| 32,768 | 2.462 | 0.676 | 0.000244 |

For absolute query position `a + t`, cached positions are visible from `max(0, a + t - 2047)` to `a - 1`, and all eight proposal positions are visible. The ring stores committed rows at `p % 2048` and applies RoPE using absolute `p`. Proposal K/V use separate scratch and positions `a ... a + 7`. TensorFold uses a larger ring to fit proposal K/V alongside committed K/V, with the same visibility and absolute position rules. SGLang translates the 2,048-token checkpoint setting to a 2,047-position left window and constructs proposal positions from prefix length plus block offsets. The target embedding call applies the target's `embed_norm(embed_tokens(ids))` to both anchor and mask, consistent with SGLang's Inkling embedding wrapper and TensorFold's reference. [SGLang drafter model](https://github.com/sgl-project/sglang/blob/2070487eccdda795a6a4a00ecb0b9b560612cfee/python/sglang/srt/models/dflash.py) · [SGLang worker](https://github.com/sgl-project/sglang/blob/2070487eccdda795a6a4a00ecb0b9b560612cfee/python/sglang/srt/speculative/dflash_worker_v2.py) · [SGLang window conversion](https://github.com/sgl-project/sglang/blob/2070487eccdda795a6a4a00ecb0b9b560612cfee/python/sglang/srt/speculative/dflash_utils.py) · [SGLang Inkling embedding](https://github.com/sgl-project/sglang/blob/2070487eccdda795a6a4a00ecb0b9b560612cfee/python/sglang/srt/models/inkling.py#L772) · [TensorFold patch 0018](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark/blob/2fe0eb57429013c0ccb1c6535339ae61a9e44d9a/tensorfold/patches/0018-mini-dflash2-drafter.patch) · [TensorFold PyTorch reference](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark/blob/2fe0eb57429013c0ccb1c6535339ae61a9e44d9a/tensorfold/tools/inkling2_dflash2_ref.py)

The in-tree PyTorch oracle applied the 2,048-token window to actual target taps and matched **7/7** native draft tokens at both 30 and 3,000 rows. Hidden RMSE was 0.06367 and 0.06819 respectively, and the 3,000-row capture exercised physical ring wraparound. The BF16 split-KV run still had 3.3728 tokens per verify at 32K, the same aggregate as the prototype. These captures show correct window behavior on the tested prompt and leave the 32K acceptance drop unexplained. At fixed width FP8 increased 32K acceptance only to 3.418, still below M5 at 3.482.

The opt-in `INKLING_DFLASH2_GQA=1` path stages 128-bit K/V loads once per KV group and uses BF16 Tensor Cores for QK and the weighted value sum, with FP32 score and merge accumulators. It achieved **0.312 ms** across five layer-equivalents at 2,048, 3,000 and 32,768 positions in the isolated benchmark, with maximum BF16 output difference 0.000977 at each long position. The instrumented 2K FP8 run measured **0.556 ms** across five attention layers. Its 3,000-row BF16 real-tap run also matched **7/7** draft tokens, with hidden RMSE 0.08336. With graph replay and BF16 projections, this attention path reduced mean draft latency at both long contexts and reduced 32K acceptance to 3.139, so it remains opt-in.

### Token 13 at 2K

The diagnostic used Request 2, fixed width 8 and the same target precision on both paths. It projected both NVFP4 target-head shards from each verify step's actual final hidden row after that step completed. M5 emitted token 180433 at verify row 5 with post-softcap logits **28.48004** for 180433 and **28.37809** for 149878. DFlash2 emitted token 149878 at verify row 6, where both logits were **28.45515** and the lower token ID won the tie. The pre-softcap BF16 values were 54.75 and 53.75 for M5, and 54.50 for both tokens for DFlash2. The winning M5 logit moved by one BF16 ULP and the other moved by three. The different block boundaries and tie explain the immediate token choice. A second diagnostic let each real drafter run, then supplied identical width-8 proposals to both verifiers for the first two rounds, at output offsets 1 and 7 with five accepted drafts each. Both paths emitted token 180433 at token 13 and produced identical hashes of the complete token-13 logit shards on both ranks. This isolates the observed divergence to the different verify schedules for this replay. The results do not characterize all rollback or tap states.

### Qualification and operating modes

The BF16 split-KV build passed the official **33/33** golden Chat generation fixture and the width-1 logit fixture at **65/66** top-1 and **0.00192697476039** mean KL. FP8, hybrid, graph BF16 and GQA BF16 each passed the golden Chat fixture in separate runs. The two best short-workload FP8 settings, fine2 0.35 and 0.55 with graph replay, each matched **33/33** on the official Chat prompt and expected tokens through the benchmark runner. The validator CLI does not expose the adaptive flag. The final build reran the official width-1 fixture. The precision ablations are opt-in on rank 0 with `INKLING_DFLASH2_PRECISION=fp8` or `INKLING_DFLASH2_PRECISION=hybrid`. FP8 uses per-row E4M3 weights for Q/K/V/O and gate/up/down with BF16 activations and outputs. Hybrid uses the same FP8 attention and rowwise NVFP4 MLP. BF16 mirrors of quantized K/V support larger context commits. Precision overrides and `--draft-nvfp4` are mutually exclusive. Graph replay is opt-in via `INKLING_DFLASH2_GRAPH=1` and applies Q/K RoPE at the current absolute position, supplied on device. Commit remains eager and precedes replay. Stage profiling uses `INKLING_DFLASH2_PROFILE=1` and disables graph replay while it synchronizes timing events.

An isolated TP=2 head trial compared the same packed synthetic NVFP4 weights and seven BF16 rows. The full head plus top-16 took **1.434 ms** on rank 0. The sharded path, including the 32,256-byte hidden broadcast, 896-byte top-16 exchange per rank and global merge, took **0.754 ms** and reproduced **112/112** score/token pairs on both ranks. Generation still uses the rank-zero head. The isolated trial measures the head-stage opportunity, with end-to-end integration and acceptance qualification open. At 4.47 tokens per verify, a 26.45 ms verifier alone caps fixed-width throughput at about 169 tok/s. Reaching 170–200 tok/s with a 2.2 ms draft calls for about 20.2–24.1 ms of verification and control overhead, depending on the throughput target.

### Interleaved and adaptive dual-Spark results

These cold-prefill runs use M5 `8306c64` controls, FP32 transport, FP8 target dense and KV, and NVFP4 target heads. The long rows aggregate three 256-token requests per workload. The short runs generate 256 tokens each. Width sweeps are specific to the tested prompts. Timed controls and candidate graphs used the same default rendezvous port. M5 did not report per-stage times.

| Run set | Workload | Policy | α+1 | Draft ms | Verify ms | Decode tok/s |
|---|---|---|---:|---:|---:|---:|
| Split-KV | Chat | M5 control 1 | 2.257 | — | — | 120.743 |
| Split-KV | Code | M5 control 1 | 2.345 | — | — | 125.405 |
| Split-KV | Chat | BF16 split-KV | 4.390 | 6.041 | 26.350 | 133.283 |
| Split-KV | Code | BF16 split-KV | 3.806 | 5.923 | 26.031 | 118.991 |
| Split-KV | Chat | M5 control 2 | 2.257 | — | — | 120.133 |
| Split-KV | Code | M5 control 2 | 2.345 | — | — | 125.441 |
| Split-KV | 2048 | M5 control 1 | 3.539 | — | — | 166.200 |
| Split-KV | 2048 | BF16 split-KV | 4.356 | 6.732 | 27.175 | 127.341 |
| Split-KV | 2048 | M5 control 2 | 3.539 | — | — | 166.247 |
| Split-KV | 32768 | M5 control 1 | 3.482 | — | — | 154.429 |
| Split-KV | 32768 | BF16 split-KV | 3.373 | 6.659 | 28.102 | 96.435 |
| Split-KV | 32768 | M5 control 2 | 3.482 | — | — | 153.574 |
| Precision | Chat | M5 control 1 | 2.257 | — | — | 119.411 |
| Precision | Code | M5 control 1 | 2.345 | — | — | 125.236 |
| Precision | Chat | FP8 | 4.466 | 4.321 | 26.228 | 143.751 |
| Precision | Code | FP8 | 3.446 | 4.152 | 25.947 | 114.371 |
| Precision | Chat | Hybrid | 3.493 | 4.032 | 26.534 | 111.114 |
| Precision | Code | Hybrid | 3.148 | 3.899 | 25.933 | 105.418 |
| Precision | Chat | M5 control 2 | 2.257 | — | — | 119.916 |
| Precision | Code | M5 control 2 | 2.345 | — | — | 124.723 |
| Graph | Chat | Eager BF16 | 4.390 | 6.070 | 26.412 | 132.917 |
| Graph | Code | Eager BF16 | 3.806 | 5.902 | 26.056 | 118.975 |
| Graph | Chat | Graph BF16 | 4.390 | 5.833 | 26.263 | 134.512 |
| Graph | Code | Graph BF16 | 3.806 | 5.684 | 25.908 | 120.351 |
| Graph | Chat | Graph FP8 | 4.466 | 4.065 | 26.393 | 144.182 |
| Graph | Code | Graph FP8 | 3.446 | 3.905 | 26.025 | 115.008 |
| Graph | Chat | Graph hybrid | 3.493 | 3.722 | 26.329 | 113.015 |
| Graph | Code | Graph hybrid | 3.148 | 3.573 | 25.767 | 107.191 |
| Graph | Chat | BF16 fine2 0.10 | 4.390 | 5.894 | 26.425 | 133.576 |
| Graph | Code | BF16 fine2 0.10 | 3.806 | 5.745 | 26.068 | 119.508 |
| Graph | Chat | BF16 fine2 0.20 | 4.390 | 5.866 | 26.319 | 134.131 |
| Graph | Code | BF16 fine2 0.20 | 3.806 | 5.691 | 25.968 | 120.090 |
| Graph | 2048 | BF16 fine2 0.20 | 4.356 | 6.419 | 27.009 | 129.162 |
| Graph | Chat | BF16 fine2 0.35 | 4.390 | 5.736 | 24.189 | 144.267 |
| Graph | Code | BF16 fine2 0.35 | 3.413 | 5.622 | 23.994 | 114.678 |
| Graph | 2048 | BF16 fine2 0.35 | 4.173 | 6.380 | 25.083 | 131.296 |
| Graph | Chat | BF16 fine2 0.50 | 4.047 | 5.870 | 21.336 | 146.267 |
| Graph | Code | BF16 fine2 0.50 | 3.542 | 5.729 | 20.774 | 133.470 |
| Graph | 2048 | BF16 fine2 0.50 | 4.037 | 6.438 | 21.815 | 141.599 |
| Graph | Chat | BF16 fine2 0.65 | 2.848 | 5.830 | 18.378 | 114.338 |
| Graph | Code | BF16 fine2 0.65 | 3.084 | 5.681 | 18.157 | 128.719 |
| FP8 sweep | Chat | M5 control 1 | 2.257 | — | — | 119.860 |
| FP8 sweep | Code | M5 control 1 | 2.345 | — | — | 124.888 |
| FP8 sweep | Chat | Graph FP8 fine2 0.25 | 4.466 | 4.085 | 26.106 | 145.447 |
| FP8 sweep | Code | Graph FP8 fine2 0.25 | 3.446 | 3.913 | 25.614 | 116.571 |
| FP8 sweep | Chat | Graph FP8 fine2 0.35 | 4.390 | 4.004 | 24.505 | 151.415 |
| FP8 sweep | Code | Graph FP8 fine2 0.35 | 3.400 | 3.874 | 23.591 | 123.645 |
| FP8 sweep | Chat | Graph FP8 fine2 0.45 | 3.316 | 4.022 | 22.961 | 119.478 |
| FP8 sweep | Code | Graph FP8 fine2 0.45 | 3.377 | 3.864 | 20.875 | 133.699 |
| FP8 sweep | Chat | Graph FP8 fine2 0.55 | 3.493 | 4.055 | 19.501 | 148.063 |
| FP8 sweep | Code | Graph FP8 fine2 0.55 | 3.263 | 3.910 | 19.175 | 137.886 |
| FP8 sweep | Chat | M5 control 2 | 2.257 | — | — | 119.851 |
| FP8 sweep | Code | M5 control 2 | 2.345 | — | — | 125.244 |
| FP8 sweep | 2048 | M5 control 1 | 3.539 | — | — | 166.358 |
| FP8 sweep | 2048 | FP8 width 8 | 4.356 | 4.850 | 27.335 | 134.143 |
| FP8 sweep | 2048 | FP8 fine2 0.20 | 4.356 | 4.905 | 27.070 | 135.022 |
| FP8 sweep | 2048 | FP8 fine2 0.35 | 4.360 | 4.833 | 25.005 | 143.869 |
| FP8 sweep | 2048 | FP8 fine2 0.50 | 3.980 | 4.813 | 22.141 | 144.625 |
| FP8 sweep | 2048 | M5 control 2 | 3.539 | — | — | 166.226 |
| FP8 sweep | 32768 | M5 control 1 | 3.482 | — | — | 153.413 |
| FP8 sweep | 32768 | FP8 width 8 | 3.418 | 4.853 | 27.950 | 103.549 |
| FP8 sweep | 32768 | FP8 fine2 0.20 | 3.418 | 4.857 | 27.853 | 103.837 |
| FP8 sweep | 32768 | FP8 fine2 0.35 | 3.339 | 4.823 | 26.349 | 106.593 |
| FP8 sweep | 32768 | FP8 fine2 0.50 | 3.275 | 4.854 | 22.643 | 117.740 |
| FP8 sweep | 32768 | M5 control 2 | 3.482 | — | — | 153.760 |
| GQA ablation | 2048 | Graph BF16, width 8 | 4.348 | 6.117 | 27.267 | 128.616 |
| GQA ablation | 32768 | Graph BF16, width 8 | 3.139 | 6.125 | 28.141 | 91.414 |

### Draft stage profiling

CUDA-event means at the 64- and 2,048-token starting contexts. Each run decoded eight total tokens from a single request and recorded four to six rounds. The first round has no accepted-context commit. Planning and event insertion can increase the measured stage durations, so the regular decode totals above provide the wall-time comparison. The graph ablation above measures the effect of replay with these profiling events disabled.

| Context | Precision | Commit ms | Embed and final norm ms | Five-layer projection ms | Five-layer attention ms | Five-layer conv and norm ms | Head and top-k ms | Selector ms | Stage sum ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | bf16 | 1.7955 | 0.0338 | 3.4981 | 0.2036 | 0.4395 | 1.2695 | 0.0378 | 7.2778 |
| 64 | nvfp4 | 1.7131 | 0.0294 | 1.2750 | 0.1939 | 0.4374 | 1.2736 | 0.0393 | 4.9617 |
| 64 | fp8 | 1.8452 | 0.0254 | 1.7700 | 0.2096 | 0.3955 | 1.2797 | 0.0388 | 5.5642 |
| 64 | hybrid | 1.8662 | 0.0265 | 1.4392 | 0.1972 | 0.4063 | 1.2830 | 0.0440 | 5.2625 |
| 2048 | bf16 | 2.5437 | 0.0334 | 3.7965 | 0.9331 | 2.5548 | 1.6508 | 0.0422 | 11.5545 |
| 2048 | nvfp4 | 2.4118 | 0.0368 | 1.5002 | 0.7883 | 2.3736 | 1.3040 | 0.0420 | 8.4568 |
| 2048 | fp8 | 2.4180 | 0.0316 | 1.7714 | 0.7830 | 2.2503 | 1.3004 | 0.0509 | 8.6056 |
| 2048 | hybrid | 2.5484 | 0.0325 | 1.6136 | 0.7709 | 2.3463 | 1.2885 | 0.0410 | 8.6411 |
| 2048 | FP8 with GQA | 2.9074 | 0.0347 | 2.0307 | 0.5560 | 2.9176 | 1.4743 | 0.0623 | 9.9831 |

### Build and GPU inventory

CUDA 13.0.88 built the bench, validator and server for `sm_121a`. The staged executables under `/tmp/inkling_tp_arch06a` on rank1 matched the rank0 SHA-256 digests. The launcher tracked both rank PIDs and all GPU runs were enclosed by `flock -w 1200 /tmp/frankengine_gpu.lock`. The final locked inventory reported zero compute processes on both Sparks. No changes were made to the production directories or operational scripts.

Raw traces and capture logs are on rank0 under `/tmp/arch06a-split`, `/tmp/arch06a-precision`, `/tmp/arch06a-graph`, `/tmp/arch06a-fp8-long`, `/tmp/arch06a-gqa`, `/tmp/arch06a-head`, `/tmp/arch06a-token13`, `/tmp/arch06a-same-schedule`, and `/tmp/arch06a-final-follow-up`. The 3,000-token fixture uses the first 3,000 prompt IDs from the first 32K case and captures only the first draft. The optional 128K benchmark was omitted. The branch retains the original greedy TP=2 production default and does not qualify a replacement for M5.

## Part 4 — FP8 commit, sharded NVFP4 head and 32K ring parity

Part 4 starts from `942d6e1` in the isolated `arch/06a-dflash2-drafter` worktree. Production code and operational scripts were read-only.

### Decode commit

The eager Part 3 path used shape-specific cuBLASLt BF16 plans for the FC and, for accepted widths below eight, the ten K/V projections. Those K/V projections read the BF16 mirrors of the FP8 weights. Each new accepted width could also trigger cuBLASLt heuristic planning inside the first profiled commit of that shape. The decode path now packs FC weights per row to E4M3 at initialization, reuses the packed K/V weights and dispatches the vectorized eight-row FP8 MMA kernels. Short commits zero-pad the working input. Normalization and RoPE write only the accepted K rows into the ring, and the V copy handles wraparound. Bulk prefill retains its established BF16 FC and K/V mirror path. The decode commit remains stream-ordered and eager ahead of the proposal graph.

The full-head candidate's matched eager stage profiles measured **0.257 ms** at starting context 64 and **0.227 ms** at 2,048, averaged across proposals including the zero-commit first proposal. Excluding that first proposal yields approximately **0.308 ms** and **0.302 ms** per actual commit, respectively. These timings satisfy the 0.35 ms gate without batching K/V with the proposal. Combining the accepted rows with eight proposal rows would need a kernel that can reuse a weight tile across more than eight activation rows. It would also put the ring update inside each layer's proposal graph.

### 32K request 1, round 20

`INKLING_DFLASH2_DUMP_ROUND=20` retains the original dump format and adds chronological K/V ring snapshots for all five layers. The existing `compare_dump.py` reconstructs the visible context from raw committed taps and independently compares both the proposal and the ring. For request 1, both modes reached absolute position **32,825**, after **57** decode commits by token count and evictions of 57 old rows. The oracle checked all **2,048** live rows in each layer.

| Precision | Proposal token parity | Hidden RMSE | Ring key RMSE across layers | Ring value RMSE across layers |
|---|---:|---:|---:|---:|
| BF16 | 7/7 | 0.05219 | 0.00148–0.00153 | 0.00176–0.00204 |
| FP8 | 7/7 | 0.23235 | 0.02035–0.02212 | 0.04853–0.06837 |

Both native runs and the PyTorch oracle proposed `[13, 68014, 2918, 149991, 154766, 149876, 152118]`. BF16 worst-row RMSE stayed below 0.00688 for keys and 0.01504 for values. FP8 was compared with the BF16 checkpoint oracle and includes quantization error. The ring contents and proposal parity support correct commit and rollback at this capture. The 2,048-token window remains a plausible source of the long-context acceptance gap. Attributing the entire gap to the window would require a matched global-context ablation.

### Live TP=2 head and qualification

The experimental `INKLING_DFLASH2_SHARDED_HEAD=1` path requires `--head-nvfp4` and must be enabled on both ranks. It reuses each rank's existing verifier NVFP4 matrix and checkpoint-wide quantization scale. Rank 0 broadcasts seven BF16 hidden rows (32,256 bytes), the ranks exchange local top-16 records (896 bytes per rank), and rank 0 merges by descending score and ascending token ID before running the selector. The proposal graph captures the backbone. Communication, head and selector are stream-ordered after graph replay. Both ranks complete the draft head collectives before the existing proposal broadcast can signal cancellation. Removing the duplicate head eliminates 259.3 MB of packed weights from rank 0 before workspace differences. Oracle dumps use the full-head path to retain dense logits.

The official golden Chat fixture passed **33/33**. Separate runs using adaptive `fine2` at 0.35 and 0.55 also matched all 33 tokens. The official width-1 logit fixture reproduced **65/66** top-1 and **0.00192697476039** mean KL. In the matched short-workload full-head and sharded-head runs, Chat and Code each retained the same α+1 at both thresholds. Across Part 3 and Part 4, changing commit precision also changed Code acceptance. At per-position threshold 0.55 the full-head Part 4 run already dropped from 3.263 to 2.899, before head sharding.

### Stage profiles

CUDA-event measurements include profiling overhead. The commit column averages over proposals. The per-commit column excludes the initial proposal. Production-speed runs in the next section disable the stage profiler.

| Context and mode | Commit ms | Per-commit ms | Embed and final norm ms | 5L proj ms | 5L attn ms | Conv and norm ms | Head and top-k ms | Selector ms | Stage sum ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64, eager | 0.247 | 0.296 | 0.023 | 1.720 | 0.193 | 0.392 | 0.810 | 0.037 | 3.421 |
| 64, graph | 0.245 | 0.294 | 0.021 | 1.687 | 0.198 | 0.356 | 0.790 | 0.037 | 3.333 |
| 2K, eager | 0.226 | 0.301 | 0.023 | 1.793 | 0.803 | 2.487 | 0.863 | 0.042 | 6.237 |
| 2K, graph | 0.223 | 0.298 | 0.027 | 1.752 | 0.814 | 2.138 | 0.846 | 0.045 | 5.846 |
| Built-in Chat, graph | 0.285 | 0.290 | 0.021 | 1.704 | 0.197 | 0.337 | 0.774 | 0.033 | 3.351 |
| Built-in Code, graph | 0.289 | 0.293 | 0.020 | 1.700 | 0.187 | 0.335 | 0.771 | 0.034 | 3.336 |
| 32K request 1, graph | 0.288 | 0.292 | 0.019 | 1.639 | 0.765 | 0.417 | 0.778 | 0.035 | 3.942 |

The four-round 2K fixture still attributes about 2.1–2.5 ms to conv and norm in the event profiler. A separate Nsight Systems capture of graph kernel nodes with those events disabled measured **0.286 ms** at context 64 and **0.297 ms** at 2K for the same norm, cuBLASLt coefficient projection and convolution spans. The convolution kernels averaged **1.28 µs** and **1.26 µs** respectively. Even the first traced 2K replay's summed spans were **0.341 ms**. The sustained 32K request profiled conv and norm at **0.417 ms**. The short fixture's multi-millisecond jump is a profiling effect. Attention is the measured context-dependent stage, rising from roughly 0.20 ms at short context to 0.77–0.81 ms with a full ring.

The remaining five-layer FP8 projections account for about **1.7 ms**. Taking the proposed **1.05 ms** sharded projection estimate at face value would put the measured short draft around **2.53 ms** and the 2K draft above **3 ms**, before accounting for distributing the committed context and actual transport latency. A head-group split for Q/K/V and column splits for gate/up would still require an output reduction after O and down at every layer. Backbone sharding was evaluated analytically. Reaching 1.8–2.3 ms would also require gains outside those projections.

### Interleaved throughput

Each sequence ran M5 `8306c64` control 1, the listed FP8 graph and sharded-head candidates, and M5 control 2 with matched target settings. The controls use the original uninstrumented M5 runner, which reports acceptance and throughput. Short runs use the two built-in prompts and 256 output tokens each. Long runs aggregate three cold-prefill requests per workload. Numeric candidate labels use per-position `fine2`. Product labels use cumulative `fine2`. GQA uses per-position threshold 0.50.

| Workload | Run | α+1 | Total draft ms | Verify ms | Decode tok/s |
|---|---|---:|---:|---:|---:|
| Chat | ctl1 | 2.257 | — | — | 119.69 |
| Code | ctl1 | 2.345 | — | — | 124.00 |
| Chat | 0.35 | 4.390 | 3.230 | 24.163 | 157.58 |
| Code | 0.35 | 3.467 | 3.225 | 23.576 | 126.71 |
| Chat | 0.55 | 3.446 | 3.236 | 19.623 | 150.53 |
| Code | 0.55 | 2.899 | 3.180 | 18.395 | 132.61 |
| Chat | ctl2 | 2.257 | — | — | 119.03 |
| Code | ctl2 | 2.345 | — | — | 124.42 |
| 2048 | ctl1 | 3.539 | — | — | 166.05 |
| 2048 | 0.35 | 4.118 | 3.777 | 24.635 | 143.81 |
| 2048 | 0.55 | 3.929 | 3.818 | 20.997 | 156.29 |
| 2048 | gqa | 3.980 | 3.473 | 21.954 | 153.30 |
| 2048 | product-0.20 | 4.272 | 3.811 | 23.758 | 153.98 |
| 2048 | product-0.35 | 4.151 | 3.832 | 21.667 | 161.10 |
| 2048 | product-0.55 | 3.880 | 3.774 | 20.145 | 159.71 |
| 2048 | ctl2 | 3.539 | — | — | 165.38 |
| 32768 | ctl1 | 3.482 | — | — | 153.22 |
| 32768 | 0.35 | 3.305 | 3.799 | 26.181 | 109.40 |
| 32768 | 0.55 | 3.236 | 3.769 | 21.532 | 127.42 |
| 32768 | ctl2 | 3.482 | — | — | 153.86 |

### Adaptive width at 2K

The per-position policy extends the verified prefix while each local probability exceeds the threshold. The cumulative policy uses the product of the conditional probabilities along the selected path. The emitted proposal tokens themselves are independent of the policy. Acceptance can still change across successive rounds when a shorter verification moves the next anchor.

| Policy | α+1 | Mean verify width | Mean unused verify rows | Verify ms | Decode tok/s |
|---|---:|---:|---:|---:|---:|
| Per-position 0.35 | 4.118 | 6.797 | 2.679 | 24.635 | 143.81 |
| Per-position 0.55 | 3.929 | 4.924 | 0.995 | 20.997 | 156.29 |
| Product 0.20 | 4.272 | 6.144 | 1.872 | 23.758 | 153.98 |
| Product 0.35 | 4.151 | 5.215 | 1.065 | 21.667 | 161.10 |
| Product 0.55 | 3.880 | 4.495 | 0.615 | 20.145 | 159.71 |

The product threshold of 0.35 was the fastest 2K policy. It saved **1.61 unused rows** per verify versus the same per-position threshold and finished within about **2.8%** of the mean interleaved M5 throughput. GQA at per-position threshold 0.50 reached **153.30 tok/s**.

### Follow-up threshold sweep

These runs followed the interleave. They were used to tune the operating point.

| Workload | Per-position threshold | α+1 | Draft ms | Verify ms | Decode tok/s |
|---|---:|---:|---:|---:|---:|
| Chat | 0.45 | 3.316 | 3.256 | 22.711 | 124.15 |
| Code | 0.45 | 3.333 | 3.200 | 20.779 | 136.18 |
| Chat | 0.50 | 3.821 | 3.261 | 20.496 | 159.98 |
| Code | 0.50 | 3.000 | 3.183 | 18.942 | 135.41 |
| 2K | 0.50 | 3.939 | 3.769 | 21.971 | 149.91 |

Code at threshold 0.45 retained more acceptance than Part 3's best-throughput Code setting. Chat at 0.35 retained α+1 of 4.390 and improved to 157.58 tok/s in the interleave. The lowest measured end-to-end draft time was **3.18 ms**, above the **2.5 ms** goal. The 2K and 32K candidates remain behind matched M5.

### Build and GPU inventory

CUDA built `ingine-bench`, `ingine-validate` and `ingine-serve`. Final worker copies under `/tmp/inkling_tp_arch06a` on `worker-node` matched all three rank0 SHA-256 hashes. All GPU runs used `flock -w 1200 /tmp/frankengine_gpu.lock`. The launchers tracked the exact rank PIDs and the final locked inventory reported **zero compute processes on both Sparks**. An initial sharded interleave stopped after a temporary launcher was edited during a run. The complete interleave was repeated under the lock.

Raw timings, oracle outputs and ring dumps are on rank0 under `/tmp/arch06a-part4`. Sharded qualification, final interleaved results, Nsight Systems captures and the threshold sweep are under `/tmp/arch06a-part4-shard`.
