# Phase C TP=2 unified qualification

## Scope

Final candidate tested on two NVIDIA DGX Sparks with GB10, CUDA 13.0.88, `sm_121a`, direct visibility for the performance runs and an FP32 wire. M5 CLI and M6 HTTP controls use engine `8306c6423b8b74870cb9b3eaec26fc6c096f5318` and stock communication. The CLI control includes the same prefill reporting instrumentation as the candidate. Raw final evidence is under `/tmp/phase-c-fix-v2` on rank0. The first follow-up iteration is under `/tmp/phase-c-fix-qualification`.

The tested communication commit is `166a53642e88feec18a157d0ed0047d3ed49bd6a`, based on `923b0a8414d8f42e713e7799cd4f9349b68ded50`. The tested engine commit is `4844f999d1fc1d7958575c33c484d12d19f3c40c`, based on `524b1dad65692e50f53b2777737c28a1706dcda1`.

## Fixes and diagnostics

- DFlash2 now commits zeroed dummy taps and starts each graph variant before serving. The warmup covers unsplit attention and each split count, or the shared split graph in GQA mode. Startup also runs the sharded head across both ranks and the rank-0 selector, then resets the draft length and proposal state.
- The RoCE backend restores the stock M5 `pack`, `finish` and `collective<bypass>` bodies and parameter layouts. Host dispatch sends row gather to a dedicated `publish_gather` kernel. Gather and large FP32 AllReduce use staged publication outside the stock collective. Stock `459366d` itself contains row-gather branches. Those original branches are present in the restored SASS and the gather API bypasses them entirely. The fixed stock path has no staged flag or secondary epoch write.
- For both visibility variants, `nvcc --resource-usage` and `cuobjdump --dump-sass` showed 40 registers, one barrier, nine bytes of shared memory, zero spills and byte-identical instruction encodings against `459366d`. The pre-fix collective also used 40 registers, but its instruction encodings differed.
- DFlash2 had shared a variable-stride GEMV epilogue with the target's fixed-stride convolution path. DFlash2 now uses a separate epilogue. All 28 fixed-stride target GEMV entry points have byte-identical `sm_121a` instruction encodings against `8306c64`.
- The 160-token SSE case uses the hash-map question and low reasoning effort from built-in CLI Chat. Both reasoning and answer deltas contribute to response parity and the first-token timestamp. The 12-token probe and coding settings use the original request settings.

## CLI performance

Cells contain cold prefill seconds, prefill tok/s, decode tok/s and tokens per verify (α+1). Chat and Code show the median of six decode runs, mean prefill timing and α+1 from the last run. The three long-context rows show mean cold prefill time, aggregate throughput and 3, 3 and 2 items, respectively. Each final run used the order M5 control 1 → fixed hybrid → M5 control 2.

### Before

| Workload | M5 control 1 | Unified hybrid | M5 control 2 | Exact continuation vs controls |
|---|---:|---:|---:|---|
| Chat | 0.079 / 380.71 / 120.30 / 2.257 | 0.078 / 384.72 / 158.68 / 3.821 | 0.079 / 381.88 / 120.30 / 2.257 | — |
| Code | 0.087 / 458.80 / 124.65 / 2.345 | 0.087 / 459.52 / 135.21 / 3.000 | 0.087 / 457.35 / 124.80 / 2.345 | — |
| 2K | 1.162 / 1762.2 / 166.30 / 3.539 | 0.788 / 2597.9 / 165.77 / 3.539 | 1.170 / 1750.1 / 166.28 / 3.539 | 768/768 and 768/768 |
| 32K | 17.805 / 1840.4 / 152.97 / 3.482 | 11.872 / 2760.2 / 150.27 / 3.482 | 17.779 / 1843.1 / 153.78 / 3.482 | 768/768 and 768/768 |
| 128K | 77.309 / 1695.4 / 102.31 / 2.646 | 53.186 / 2464.4 / 101.90 / 2.646 | 77.366 / 1694.2 / 102.67 / 2.646 | 512/512 and 512/512 |

### After

| Workload | M5 control 1 | Fixed hybrid | M5 control 2 | Exact IDs vs controls |
|---|---:|---:|---:|---|
| Chat | 0.078 / 383.58 / 120.44 / 2.257 | 0.078 / 384.75 / 158.56 / 3.821 | 0.078 / 384.30 / 120.29 / 2.257 | — |
| Code | 0.087 / 458.74 / 124.84 / 2.345 | 0.087 / 460.06 / 135.08 / 3.000 | 0.088 / 455.33 / 124.90 / 2.345 | — |
| 2K | 1.165 / 1757.9 / 165.32 / 3.539 | 0.784 / 2613.4 / 165.29 / 3.539 | 1.168 / 1752.9 / 165.43 / 3.539 | 768/768 and 768/768 |
| 32K | 17.802 / 1840.7 / 153.34 / 3.482 | 11.697 / 2801.5 / 153.11 / 3.482 | 17.796 / 1841.3 / 153.89 / 3.482 | 768/768 and 768/768 |
| 128K | 77.211 / 1697.6 / 102.52 / 2.646 | 53.035 / 2471.4 / 102.30 / 2.646 | 77.512 / 1691.0 / 102.92 / 2.646 | 512/512 and 512/512 |

## Live OpenAI HTTP/SSE

Cells contain TTFT seconds and decode tok/s. Parity refers to the concatenated bytes of each decoded response channel. The interleave used M6 control 1 → fixed hybrid → M6 control 2. Coding prompts used 1,784 greedy or 1,786 sampled tokens. The 160-token Chat case is new.

### Before

| Request | M6 control 1 | Unified hybrid | M6 control 2 | Exact response vs controls |
|---|---:|---:|---:|---|
| short | 0.395 / 84.39 | 0.395 / 63.39 | 0.408 / 85.96 | PASS / PASS |
| 0/0 cold | 0.934 / 121.94 | 0.609 / 120.96 | 0.944 / 121.53 | PASS / PASS |
| 0/0 warm | 0.003 / 122.78 | 0.002 / 120.64 | 0.003 / 122.52 | PASS / PASS |
| 0.7/40 cold | 0.933 / 113.80 | 0.608 / 115.82 | 0.947 / 112.79 | PASS / PASS |
| 0.7/40 warm | 0.002 / 114.09 | 0.003 / 117.38 | 0.002 / 114.45 | PASS / PASS |
| 0.7/0 cold | 0.935 / 105.18 | 0.619 / 107.03 | 0.946 / 104.75 | PASS / PASS |
| 0.7/0 warm | 0.003 / 103.82 | 0.003 / 106.09 | 0.003 / 103.62 | PASS / PASS |

### After

| Request | M6 control 1 | Fixed hybrid | M6 control 2 | Exact text vs controls |
|---|---:|---:|---:|---|
| short-12tok | 0.378 / 87.47 | 0.396 / 70.64 | 0.390 / 87.77 | PASS / PASS |
| short-chat-160tok | 0.107 / 116.88 | 0.109 / 142.83 | 0.108 / 117.40 | FAIL / FAIL |
| 0/0 cold | 0.941 / 122.90 | 0.617 / 122.41 | 0.937 / 122.04 | PASS / PASS |
| 0/0 warm | 0.003 / 122.83 | 0.003 / 123.14 | 0.003 / 120.80 | PASS / PASS |
| 0.7/40 cold | 0.940 / 112.79 | 0.605 / 116.21 | 0.935 / 113.98 | PASS / PASS |
| 0.7/40 warm | 0.002 / 114.30 | 0.003 / 117.87 | 0.003 / 113.79 | PASS / PASS |
| 0.7/0 cold | 0.941 / 104.46 | 0.616 / 107.25 | 0.933 / 103.37 | PASS / PASS |
| 0.7/0 warm | 0.002 / 104.00 | 0.003 / 106.79 | 0.003 / 104.71 | PASS / PASS |

## Correctness and cache reuse

- All four CTests passed (`test_tp2_kernels`, `spec_accept`, `serve_scheduler`, and `serve_server`).
- The two-rank smoke passed all 64 mixed collectives in direct and cache-bypass visibility. This includes the in-place 2048 × 2304 AllReduce over eight graph replays. Gather was bit-exact at M = 1, 2, 4 and 8 in both visibility modes.
- The fixed and stock raw FP32 target logits are byte-identical (52,815,312 bytes, SHA-256 `6278bcace51713b9de25a44fbd4e1c5d460c5549ffe84e9a88e87b4017cc5dac`).
- Eager and lazy diagnostic runs produced identical 11,424-byte proposal and FP32 confidence dumps across two Chat and two Code requests. SHA-256 `54abf06d4c9dd7ab4f8df990d9980db5e0c905d89290decb3c15a30070224d8b`.
- `ingine-validate` reproduced 65/66 top-1 and mean KL 0.00192697476039. DSpark, DFlash2 and hybrid each matched all 33 Golden Chat tokens.
- Cold and warm coding responses remained byte-identical to M6. Warm greedy, top-k 40 and top-k 0 requests reported 1,784, 1,786 and 1,786 cached tokens. The intervening DFlash2 request returned `6` and the next greedy coding request restored its parked 1,784-token prefix with identical response bytes.
- Live retrieval returned `CORMORANT-78416-QUARTZ` exactly at 65,495 prompt tokens and `MANGROVE-29573-COBALT` exactly at 121,559 prompt tokens. TTFT was 23.707 s and 47.378 s, respectively.

## First-request capture and routing diagnostics

CUDA API interception on the candidate counted 48 capture calls before the server began listening and zero during the live requests. That places zero graph capture work in the first 12-token request. A diagnostic binary with lazy proposal capture recorded two calls after listening. The traced eager run measured 69.21 tok/s for the first arithmetic request and 69.19 tok/s for its repeat. The lazy diagnostic measured 70.78 tok/s on its first request. The prior 44 ms estimate remains unconfirmed in this controlled comparison.

Both hybrid and unified DSpark took six rounds and accepted six draft tokens on the 12-token probe. The instrumented hybrid spent 18.12 ms drafting and 137.90 ms in target verification. DFlash2 added draft work on this probe without improving acceptance. On the full 160-token turn it reduced the number of rounds from 77 to 51, but the emitted sequences diverged.

| DSpark workload | Stock control 1 | Unified DSpark only | Unified hybrid | Stock control 2 |
|---|---:|---:|---:|---:|
| 32K CLI | 153.34 | 153.11 | 153.11 | 153.89 |
| Live coding greedy cold | 122.90 | 120.64 | 122.41 | 122.04 |
| Live coding greedy warm | 122.83 | 122.66 | 123.14 | 120.80 |

Rates are decode tok/s. The DSpark-only diagnostic ran separately from the promotion interleaves using the fixed binaries and the same workloads. Hybrid was flat at 32K and faster in the two live greedy cases. The comparison found no slowdown above 0.5% from loading DFlash2 on these DSpark requests.

## Performance and promotion

- Built-in Chat and Code decode improved 31.7% and 8.2% versus the average of the final controls. Long-context prefill throughput improved 48.9%, 52.2% and 45.9% at 2K, 32K and 128K. Long-context decode came in 0.02%, 0.15% and 0.21% below the slower controls, respectively. Each gap was smaller than the spread within its control pair. The measurements recover most of the earlier 32K loss and leave the exact position of the long-context decode floor unresolved.
- Cold coding TTFT fell about 34–35%. Greedy coding decode was inside or above its controls. Sampled coding improved 2.3–3.4% versus the average of the corresponding controls.
- The 160-token hybrid response ran 21.9% faster than the mean M6 control. Unified DSpark reproduced both M6 responses. Hybrid diverged after 312 shared reasoning bytes and 11 shared answer bytes. The underlying verifier decision at the divergence was unverified.
- **NO-GO.** The first 12-token stream remains 19.4% slower than M6, and the 160-token Chat response fails byte parity. Both block strict promotion.

## Process hygiene

Both ranks ran under their GPU locks for each GPU command. The launchers checked SHA-256 after staging each worker executable under `/tmp/inkling_tp_unified` and stopped the live servers after the SSE clients drained. Per-run inventories showed zero GPU compute processes after exit. A separate final locked inventory also showed zero compute PIDs on rank0 and rank1.
