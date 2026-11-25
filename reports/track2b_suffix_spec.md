# Track 2b — Zero-weight suffix-copy speculation

**Disposition.** Experimental and off by default. The candidate failed the exact continuation and suite-wide throughput gates on both configurations. Repeated JSON tool calls were the clearest positive workload.

The implementation is opt-in via `--spec --suffix-spec`. `--suffix-min-match` defaults to 4 and accepts 1–6. The evaluation uses `fine2`, so a seven-token continuation selects width 8. The index checks the longest exact token key first and uses its most recent committed occurrence with a continuation. It builds once per generation request and inserts only newly committed rows after that. It stores CPU metadata and loads no model weights. TP=1 DSpark uses the existing host-verification machinery when the option is on, including its small device proposal buffer.

Rank 0 publishes the proposal through the existing TP transport. Both ranks use their identical committed token history to decide whether the DFlash2 head exchange is needed. Greedy acceptance and target rollback use the existing kernels. DSpark consumes preceding verifier taps via the existing fused commit-and-draft on misses and the same small-commit arithmetic without drafting on hits. It commits only the retained rows at the terminal rollback. DFlash2 keeps its per-round decode commit and skips the proposal and head exchange on hits. The tap consumer runs before the next target verify can reuse the tap buffer. A multi-round deferred catch-up would require tap storage or a copy to retain overwritten rows. It would also have to preserve the original grouping and decode projection path, particularly for DFlash2. The implementation uses a one-round delay with no tap copy.

Sampled and forced requests bypass suffix lookup. The current batch benchmark only accepts greedy generation at TP=2. A sampled deterministic-proposal performance gate requires a separate implementation and experiment.

## Build and reproducibility

- Base `a62aa5ab3783bb15c01adb13964aabd0e13769e1`, branch `feat/track2b-suffix-spec`, implementation commit `6115af1c90c01ee5dbb4ffd27cb3fae35cf4dc9b`.
- Release CUDA 13.0.88, `sm_121a`, tests enabled, existing `tp_comm` source. The build is in `/tmp/track2b-suffix-build` and bench, validate and serve worker executables are staged in `/tmp/frankengine_track2b` on rank1 with matching SHA-256 checks.
- The four dense and two MoE CUDA object disassemblies matched the prior base build after removing the source path and compiler-generated internal symbol prefix. SHA-256 of normalized dense GEMV disassembly is `db3fb253318a0a7d217a1cdae5c2a34c131df2b5164fa1e7fabd044f73b12cbe`. The target GEMV and `tp_comm` sources were untouched.
- The CPU suffix test covered exact keys, longest match, recency, the unprocessed-anchor guard, incremental updates and a 256K-token history. Measured steady-state lookup plus insertion was 2.445–5.520 µs per step over seven executions. Prompt indexing time is outside this microbenchmark.
- Benchmark and validator commands each acquire the rank0 shared lock for the run and cleanup check. The full CTest command uses the same lock. Raw bench and validator results are under `/tmp/track2b-results` on rank0.
- Interleaved control / suffix / control runs use hybrid drafting, `fine2`, dense FP8, FP8 target KV, NVFP4 target head, cold prefill, temperature 0 and DSpark threshold 0.75. The default hybrid cutoff selects DFlash2 for the built-ins and DSpark for the 2K and longer prompts. The built-in runs report the median of six decodes and the last-run acceptance and GPU timing. The 2K and 32K agentic inputs are the first three fixed-length prompts in the existing `bench/agentic-*.jsonl` files, with 256 free generated IDs per prompt. The new coding fixtures come from `python tools/track2b/make_suite.py MODEL/tokenizer.json`, with three tasks at each of 2K and 16K. Their 256-token budget is fixed to support exact ID comparisons. The fixed-budget free path can continue past an end-of-turn token.
- The `α+1` column measures accepted tokens plus the target token per verify round. GPU draft timing includes the transition in the reported row and is averaged over speculative rounds. Built-in output previews are limited to 120 IDs and their exact full continuations were not captured.

## Interleaved measurements

| TP | Workload | tok/s ctl1 / cand / ctl2 | α+1 ctl1 / cand / ctl2 | Draft ms ctl1 / cand / ctl2 | Candidate suffix rounds | Candidate vs ctl1 / ctl2 IDs |
|---:|---|---|---|---|---|---|
| 1 | Built-in chat | 87.074 / 86.858 / 86.870 | 3.2785 / 3.2785 / 3.2785 | 3.695572 / 3.654012 / 3.720548 | 1.27% | — |
| 1 | Built-in code | 95.691 / 96.724 / 95.605 | 3.4133 / 3.4932 / 3.4133 | 3.684180 / 3.475545 / 3.685030 | 5.48% | — |
| 1 | 2048 context | 104.442 / 100.670 / 105.162 | 3.3377 / 3.2797 / 3.3377 | 1.73191 / 1.43971 / 1.68683 | 17.80% | 495/768 / 495/768 |
| 1 | 32768 context | 94.949 / 89.124 / 94.751 | 3.2821 / 3.1867 / 3.2821 | 2.06292 / 1.63577 / 2.07628 | 21.58% | 446/768 / 446/768 |
| 1 | edit-2048 | 85.523 / 86.652 / 85.400 | 2.4057 / 2.5859 / 2.4057 | 1.75900 / 1.53009 / 1.76988 | 14.14% | 67/256 / 67/256 |
| 1 | refactor-diff-2048 | 113.083 / 102.736 / 113.737 | 3.8657 / 3.6250 / 3.8657 | 1.76890 / 1.31639 / 1.76577 | 26.39% | 256/256 / 256/256 |
| 1 | tool-json-2048 | 141.340 / 139.216 / 141.334 | 6.5500 / 6.4500 / 6.5500 | 1.76866 / 0.45339 / 1.77501 | 77.50% | 256/256 / 256/256 |
| 1 | edit-16384 | 76.409 / 77.064 / 76.045 | 2.0726 / 2.1333 / 2.0726 | 1.88377 / 1.83182 / 1.89623 | 3.33% | 223/256 / 223/256 |
| 1 | refactor-diff-16384 | 109.995 / 107.590 / 109.597 | 4.4407 / 4.0635 / 4.4407 | 1.90522 / 1.09391 / 1.91140 | 44.44% | 256/256 / 256/256 |
| 1 | tool-json-16384 | 141.335 / 136.340 / 141.195 | 6.7179 / 6.3750 / 6.7179 | 1.91934 / 0.48760 / 1.92742 | 77.50% | 256/256 / 256/256 |
| 2 | Built-in chat | 159.177 / 157.736 / 158.816 | 3.8209 / 3.8209 / 3.8209 | 3.207470 / 3.141775 / 3.210823 | 1.49% | — |
| 2 | Built-in code | 135.508 / 147.771 / 134.895 | 3.0000 / 3.6056 / 3.0000 | 3.202280 / 3.040621 / 3.190690 | 5.63% | — |
| 2 | 2048 context | 162.706 / 158.357 / 161.792 | 3.3450 / 3.3506 / 3.3450 | 1.70680 / 1.41716 / 1.71315 | 18.61% | 409/768 / 409/768 |
| 2 | 32768 context | 153.190 / 143.741 / 152.992 | 3.4820 / 3.3077 / 3.4820 | 2.04995 / 1.63438 / 2.04352 | 22.22% | 533/768 / 533/768 |
| 2 | edit-2048 | 118.550 / 116.224 / 119.467 | 2.1966 / 2.2069 / 2.1966 | 1.77167 / 1.66167 / 1.74390 | 6.03% | 24/256 / 24/256 |
| 2 | refactor-diff-2048 | 178.285 / 167.324 / 180.918 | 4.0952 / 3.8088 / 4.0952 | 1.77573 / 1.26349 / 1.74475 | 30.88% | 208/256 / 208/256 |
| 2 | tool-json-2048 | 224.162 / 233.552 / 226.650 | 6.7179 / 6.8919 / 6.7179 | 1.80942 / 0.50055 / 1.77301 | 75.68% | 256/256 / 256/256 |
| 2 | edit-16384 | 124.815 / 116.766 / 124.560 | 2.5243 / 2.2174 / 2.5243 | 1.90675 / 1.78832 / 1.88815 | 6.09% | 173/256 / 173/256 |
| 2 | refactor-diff-16384 | 177.952 / 169.266 / 175.903 | 4.2500 / 4.1129 / 4.2500 | 1.91585 / 1.20509 / 1.88726 | 38.71% | 196/256 / 196/256 |
| 2 | tool-json-16384 | 219.321 / 216.848 / 219.103 | 6.7179 / 6.3750 / 6.7179 | 1.90372 / 0.53260 / 1.89754 | 75.00% | 256/256 / 256/256 |

## DFlash2 and sampled fallback checks

| TP | Mode | tok/s ctl1 / cand / ctl2 | α+1 ctl1 / cand / ctl2 | Draft ms ctl1 / cand / ctl2 | Candidate suffix rounds | Candidate vs ctl1 / ctl2 IDs |
|---:|---|---|---|---|---|---|
| 1 | DFlash2 greedy edit-2048 | 79.297 / 69.862 / 79.141 | 2.7128 / 2.3611 / 2.7128 | 4.30959 / 4.19499 / 4.30229 | 1.85% | 48/256 / 48/256 |
| 1 | DFlash2 greedy refactor-diff-2048 | 105.739 / 98.421 / 105.456 | 4.2097 / 3.8806 / 4.2097 | 4.26670 / 3.09117 / 4.29093 | 29.85% | 256/256 / 256/256 |
| 1 | DFlash2 greedy tool-json-2048 | 150.223 / 155.699 / 149.275 | 7.9394 / 7.5588 / 7.9394 | 4.22399 / 1.11766 / 4.31439 | 79.41% | 256/256 / 256/256 |
| 2 | DFlash2 greedy edit-2048 | 124.793 / 134.356 / 125.280 | 2.8333 / 3.1446 / 2.8333 | 3.75019 / 3.17707 / 3.75439 | 18.07% | 33/256 / 33/256 |
| 2 | DFlash2 greedy refactor-diff-2048 | 162.051 / 156.257 / 164.017 | 4.2500 / 4.0469 / 4.2500 | 3.76520 / 2.51871 / 3.74608 | 35.94% | 153/256 / 153/256 |
| 2 | DFlash2 greedy tool-json-2048 | 226.729 / 242.853 / 227.733 | 7.6765 / 7.3429 / 7.6765 | 3.74724 / 1.02544 / 3.74811 | 80.00% | 256/256 / 256/256 |
| 1 | DSpark T=0.7 | 101.886 / 102.255 / 102.317 | 3.2375 / 3.2375 / 3.2375 | — / — / — | 0.00% | 768/768 / 768/768 |

## Qualification

| Gate | TP=1 | TP=2 |
|---|---|---|
| Golden chat, DSpark | 33/33 | 33/33 |
| Golden chat, DFlash2 | 33/33 | 33/33 |
| Golden chat, hybrid | 33/33 | 33/33 |

CTest enumerated all 46 tests. 40 passed, five returned the configured skip code, and one failed. The base build reproduced the BF16 prefill failure with the same top-1 IDs and both KL values. The skipped tests were the four opt-in prefill tools and the external text trace fixture test. The CPU suffix index, sampled-fidelity, forced-speculation, TP kernels, generator reuse and speculative path tests passed.

In the greedy benchmark the controls reproduced their token sequences. The candidate changed tokens on edit prompts and the 2K/32K set. Existing verify widths can use different target arithmetic and the repository's speculative path gate allows small logit gaps. Width-dependent numerical behavior is a possible cause of the observed mismatches. The contribution of width-dependent arithmetic versus other effects of the changed verify schedule remains unisolated. Exact parity is therefore a failed gate. The measured throughput gains on some repetitive tool calls are workload-specific. The candidate missed the suite-wide performance gate.

The tap synchronization design uses the existing commit-only or fused operations and avoids retaining old target taps. This experiment measured per-round draft time. Isolated commit-only timing and the multi-round buffered catch-up comparison remain unmeasured. Sampled mode stayed on neural drafting, so the T=0.7 row tests the fallback rather than deterministic-proposal sampled performance.

CPU and target kernel sources, golden cases and checkpoint weights were held fixed during the A/B runs. For a reproduction, run each variant with `ingine-bench --dense-fp8 --head-nvfp4 --kv-fp8 --spec --spec-widths fine2 --temperature 0 --draft-type hybrid --spec-threshold .75 --cold-prefill --spec-timing` and add `--suffix-spec` only for the middle run. Use `--max-ctx 4096 --runs 6 --tokens 256` for the built-ins or `--forced INPUT.jsonl --forced-free 0 --forced-trace TRACE.jsonl` with a context limit at least eight tokens above prompt plus output. For the extra DFlash2 rows use `--draft-type dflash2 --dflash2-adaptive --spec-threshold .5` on the first three coding fixtures.

The 33 benchmark commands, six successful golden commands, CTest and the base-test retry all recorded empty compute-PID lists on both hosts at release. A TP=2 rendezvous attempt on the alternate port timed out and cleared both GPUs. Its retry used the established port. The TP=2 coding closing control was rerun after a repeated candidate overwrote the first trace path, and its IDs matched the opening control.
