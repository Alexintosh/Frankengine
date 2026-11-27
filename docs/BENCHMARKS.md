# Frankengine — Verified Benchmarks

All benchmarks below were measured end-to-end on live hardware running `thinkingmachines/Inkling-2-Mini` (`30B-A3.6B` Looped MoE) with `incoai/Inkling-2-Mini-DFlash2` (`GAMMA=7`, `PREFILL_CHUNK=8192`, Vision ON unless noted).

---

## 1. `1× NVIDIA RTX PRO 6000 Blackwell (96 GB, sm_120a)` vs `1× DGX Spark (GB10, sm_121a)`

Same-harness measurement using [`knapcio/inkling-2-mini-spark-bench`](https://github.com/knapcio/inkling-2-mini-spark-bench) (`cmpbench.sh`, `MAX_CTX=262144`, `MAX_BATCH=32`, `BATCH_CTX=32768`, `BATCH_SPEC=7`, Vision ON, `0` failed requests across every step):

### A. Fast FP8 Preset (`--dense-fp8 --kv-fp8 --head-nvfp4 --draft-nvfp4`)

| Metric (`spark-bench`, Vision ON) | `1× DGX Spark` (`231 GB/s`) | **`1× RTX PRO 6000 96 GB` (`1,792 GB/s`)** | Speedup |
|---|---:|---:|---:|
| **Startup (`Vision ON`)** | `8 s` | **`3 s`** | **`2.7×`** |
| **VRAM Used (`256K` + `32×32K` batch)** | `~55 GiB` (`56.8 GiB` free) | **`54.6 / 97.9 GiB`** | — |
| **`speedbench full` ALL 73 (`c1`, reasoning on)** | `82.1 tok/s` *(geo `86.0`, TTFT `49.7 ms`)* | **`398.4 tok/s`** *(geo `410.4`, TTFT `9.7 ms`)* | **`4.85×`** |
| **Prose / Code (`c1`)** | `78.8 / 96.3 tok/s` | **`379.2 / 416.2 tok/s`** | **`4.81× / 4.32×`** |
| **Thinking `c1` (`T=0.8`, high)** | `88.7 tok/s` (`90.6/req`, TTFT `0.16 s`) | **`412.2 tok/s`** (`422.5/req`, TTFT `0.04 s`) | **`4.65×`** |
| **Thinking `c8` (`16×4096`, `T=0.8`)** | `269.9 tok/s` (`40.1/req`, TTFT `0.36 s`) | **`1,020.1 tok/s`** (`152.8/req`, TTFT `0.27 s`) | **`3.78×`** |
| **Thinking `c24` (`48×4096`, `T=0.8`)** | `522.7 tok/s` (`26.6/req`, TTFT `0.69 s`) | **`2,339.9 tok/s`** (`119.3/req`, TTFT `0.45 s`) | **`4.48×`** |
| **Decode Table Prose `c8 / c24`** | `191.3 / 380.9 tok/s` | **`750.5 / 1,586.2 tok/s`** | **`3.92× / 4.16×`** |
| **Decode Table Code `c8 / c24`** | `195.8 / 391.2 tok/s` | **`761.5 / 1,501.7 tok/s`** | **`3.89× / 3.84×`** |
| **Cold Prefill `8K / 32K / 128K`** | `4,787 / 4,811 / 3,930 tok/s` | **`25,041 / 24,559 / 18,743 tok/s`** | **`5.23× / 5.10× / 4.77×`** |
| **Warm Decode `@32K / @128K`** | `89.1 / 60.6 tok/s` | **`438.1 / 317.6 tok/s`** | **`4.92× / 5.24×`** |
| **Needle `128K` / `240K`** | `PASS` (`34.2 s` / `74.1 s`) | **`PASS` (`7.0 s` / `16.0 s`)** | **`4.89× / 4.63×`** |
| **Quality Guard `30/30` / `inkqc` (`3×295`)** | `PASS` / `285.7 / 295` (`[283, 286, 288]`) | **`PASS`** (`417.6 tok/s`, drift `0.014%`) | ✓ |

### B. 100% Bit-Exact Lossless Preset (`--dense-pico12 --cert-head --draft-nvfp4`, BF16 KV)

| Metric (`spark-bench`, Vision ON) | `1× DGX Spark` (`231 GB/s`) | **`1× RTX PRO 6000 96 GB` (`1,792 GB/s`)** | Speedup |
|---|---:|---:|---:|
| **Startup (`Vision ON`)** | `10 s` | **`4 s`** | **`2.5×`** |
| **VRAM Used (`256K` + `32×32K` batch)** | `~76 GiB` (`34.1 GiB` free) | **`76.2 / 97.9 GiB`** | — |
| **`speedbench full` ALL 73 (`c1`, reasoning on)** | `76.7 tok/s` *(geo `77.1`, TTFT `53.3 ms`)* | **`365.9 tok/s`** *(geo `385.1`, TTFT `10.3 ms`)* | **`4.77×`** |
| **Prose / Code (`c1`)** | `69.7 / 84.8 tok/s` | **`341.8 / 444.9 tok/s`** | **`4.90× / 5.25×`** |
| **Thinking `c1` (`T=0.8`, high)** | `73.2 tok/s` (`71.2/req`, TTFT `0.20 s`) | **`357.6 tok/s`** (`362.1/req`, TTFT `0.04 s`) | **`4.89×`** |
| **Thinking `c8` (`16×4096`, `T=0.8`)** | `218.6 tok/s` (`33.1/req`, TTFT `0.55 s`) | **`809.4 tok/s`** (`113.1/req`, TTFT `0.25 s`) | **`3.70×`** |
| **Thinking `c24` (`48×4096`, `T=0.8`)** | `449.3 tok/s` (`22.9/req`, TTFT `0.76 s`) | **`1,764.0 tok/s`** (`93.1/req`, TTFT `0.48 s`) | **`3.93×`** |
| **Decode Table Prose `c8 / c24`** | `159.3 / 334.9 tok/s` | **`573.2 / 1,257.5 tok/s`** | **`3.60× / 3.75×`** |
| **Decode Table Code `c8 / c24`** | `161.3 / 320.2 tok/s` | **`644.3 / 1,310.0 tok/s`** | **`3.99× / 4.09×`** |
| **Cold Prefill `8K / 32K / 128K`** | `4,442 / 4,295 / 3,146 tok/s` | **`23,033 / 21,573 / 14,512 tok/s`** | **`5.19× / 5.02× / 4.61×`** |
| **Warm Decode `@32K / @128K`** | `70.2 / 42.8 tok/s` | **`397.5 / 241.0 tok/s`** | **`5.66× / 5.63×`** |
| **Needle `128K` / `240K`** | `PASS` (`39.5 s` / `100.4 s`) | **`PASS` (`8.5 s` / `22.5 s`)** | **`4.65× / 4.46×`** |
| **Quality Guard `30/30` / `inkqc` (`3×295`)** | `PASS` / `286.0 / 295` (`[287, 287, 284]`) | **`PASS`** (`379.9 tok/s`, drift `-0.022%`) | ✓ |

---

## 2. `1× DGX Spark` (`TP=1`) Public Leaderboard Comparison

Same-harness comparison against public `1× DGX Spark` recipes (`knapcio/inkling-2-mini-spark-bench`):

| Recipe (`1× DGX Spark`) | Lossy? | ALL 73 (`c1`) | Prose (`c1`) | Code (`c1`) | Thinking (`c24`) | Prefill `8K` | Prefill `128K` | Startup | Quality (`/295`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Frankengine TP=1 (FP8, Vision ON)** | Yes (`FP8 KV/dense`) | **`82.1 tok/s`** *(geo `86.0`)* | **`78.8`** | **`96.3`** | **`523`** *(c8 `270`)* | `4,787` | **`3,930`** | **`8 s`** | `285.7` |
| **Frankengine TP=1 exact (`PICO12 + CERT_HEAD`, Vision ON)** | **No (bit-exact)** | **`76.7 tok/s`** *(geo `77.1`)* | **`69.7`** | **`84.8`** | **`449`** *(c8 `219`)* | `4,442` | `3,146` | **`10 s`** | **`286.0`** |
| `knapcio v2.9.7` (SGLang + overlays, Vision OFF) | No | `73.5 tok/s` | `65.6` | `82.2` | `538` | `5,720` | `3,860` | `128 s` | `287.3` |

On steady-state 512-token full-24 bursts (no early-stop tail), FP8 reaches **`552.3 tok/s`** (`26.9 tok/s` per stream).

---

## 3. Single-Stream (`c = 1`) & Long-Context Scaling (`1×` vs `2× DGX Spark TP=2`)

Each DGX Spark (`GB10`) has **128 GB LPDDR5X unified memory** (`231.38 GB/s` streaming DRAM bandwidth), linked via dual-rail 200 GbE RoCEv2 (`bf16-moe` wire, `BATCH_SPEC=7`, `BATCH_SPEC_THRESHOLD=0.30`).

| Workload (`256K` Context, `DRAFT_TYPE=hybrid`) | `1× DGX Spark` (`TP=1`) | **`2× DGX Spark` (`TP=2` Live Prod)** | Notes / Speedup |
|---|---:|---:|---|
| **Short Code Greedy (`36t`, `DFlash 2`)** | `155.4 tok/s` (`207 ms`) | **`257.5 tok/s`** warm / **`247.9`** cold (`72 ms`) | **`1.66×`** `TP=2/TP=1` · **`2.21×`** vs TensorFold TP=1 (`116.5`) |
| **File-Edit / Refactor (`1.5K`, `320` out)** | — | **`263.9 tok/s`** (`405 ms` TTFT) | Single-stream peak on live OpenAI API |
| **8-Category Diversity Mean (`256t`, `T=0`)** | — | **`159.6 tok/s`** (`math 204`, `html 200`, `count 218`, `code 174`) | Full `bench_matrix.py` 8-domain average |
| **Short Code Sampled (`T=0.7`, think ON / OFF)** | — | **`163.0 / 170.7 tok/s`** (`94 / 74 ms` TTFT) | Exact TP=2 sampled head (LRU warm peak **`191.9 tok/s`**, `3.3 ms`) |
| **CLI Built-in Chat / Code** (`ingine-bench`) | `86.8 / 95.5 tok/s` | **`148.3 / 184.4 tok/s`** *(peak **`185.3`**)* | `50/50` sharded `DFlash 2` (`-22.9%` draft ms, `-490 MB` VRAM) |
| **Cold Prefill (`2.8K / 4K / 11K / 16K / 32K`)** | `3,619` (`2K` FP8) · `2,770` (`32K`) | **`3,637 / 3,761 / 3,732 / 3,606 / 3,579 tok/s`** | Live HTTP SSE (`4K TTFT 1.08s`, `11K 2.94s`, `32K 9.22s`) |
| **Multi-Turn Coding Agent (`~5.6K`, `T=0.7`)** | — | **`132.8 → 221.6 / 189.8 / 171.6 tok/s`** | Turns 2–4 prefix-cache hit (`80–86 ms` TTFT) |
| **Warm `16K` / `32K` Decode (`T=0` / `T=0.7`)** | `94.7 tok/s` (`32K`) | **`144.7 / 170.2`** (`16K`) · **`136.9 / 143.6`** (`32K`) | `88–99 ms` warm TTFT (`199.9 tok/s` code peak @ `32K`) |
| **Needle-in-a-Haystack (`16K` / `65.5K` / `121.6K`)** | `PASS` (`25.1 / 51.5 s`, `26.1 GiB`) | **`PASS`** (`156.7 / 181.6 / 129.2 tok/s`, `31.3 GiB`) | **100% exact passkey retrieval** across full `256K` window |

---

## 4. Batched Speculative Decoding & Persistent Prefix Cache (`2× DGX Spark TP=2`)

### A0. `spark-bench` Thinking & Decode Table (`2× DGX Spark TP=2`, `fp32` wire, 80 B/row Exact Rejection Sampling)

Measured live on `2× DGX Spark (GB10)` over dual-rail ConnectX-7 RoCEv2 (`8 s` startup, `0` failed requests, `30/30` quality guard `PASS` at `136.4 tok/s`, `30.6 ms` TTFT, drift `-0.06%`):

| Workload (`2× DGX Spark TP=2`, `spark-bench`) | Before (`REJECT=0`) | **After (`Track 4 + 5`, `REJECT=1`)** | Gain |
|---|---:|---:|---:|
| **Thinking `c = 1` (`3×2048`, `T=0.8`, high)** | `137.3 tok/s` (`141.2/req`, TTFT `0.12 s`) | **`142.3 tok/s`** (`142.9/req`, TTFT `0.12 s`) | **`+3.6%`** |
| **Thinking `c = 8` (`16×4096`, `T=0.8`, high)** | `325.1 tok/s` (`47.0/req`, TTFT `0.42 s`) | **`365.1 tok/s`** (`53.6/req`, TTFT `0.41 s`) | **`+12.3%`** (`+14.0%/req`) |
| **Thinking `c = 24` (`48×4096`, `T=0.8`, high)** | `599.2 tok/s` (`31.2/req`, TTFT `0.56 s`) | **`698.8 tok/s`** (`33.9/req`, TTFT `0.64 s`) | **`+16.6%`** |
| **Thinking `c = 32` (`32×2048`, `T=0.8`, high)** | `717.8 tok/s` (`26.4/req`) | **`787.0–798.5 tok/s`** (`29.9/req`) | **`+10–11%`** (`+13.3%/req`) |
| **Decode Table Code `c = 8`** | `274.0 tok/s` (`TTFT 0.566 s`) | **`280.3 tok/s`** (`TTFT 0.247 s`) | **`+2.3%`** (`-56%` TTFT) |

### A. Live OpenAI HTTP/SSE (`--max-batch 32 --batch-ctx 32768 --batch-spec 7 --tp-wire bf16-moe`)

| Live HTTP/SSE Concurrency Workload | Previous (`spec=4`) | **Promoted (`BatchDFlash2` `spec=7`)** | Gain |
|---|---:|---:|---:|
| **Short Prompts (`~40t`, `256` out) `c = 8`** | `200.2 tok/s` wall (`TTFT 0.70s`) | **`246.1` wall / `340.1` sum** (`TTFT 0.34s`) | **`+22.9%` wall · `2.1×` TTFT** |
| **Short Prompts (`~40t`, `256` out) `c = 16`** | `322.1 tok/s` wall (`TTFT 1.36s`) | **`363.2` wall / `508.4` sum** (`TTFT 0.65s`) | **`+12.8%` wall · `2.1×` TTFT** |
| **Short Prompts (`~40t`, `256` out) `c = 32`** | `479.7 tok/s` wall (`TTFT 2.44s`) | **`585.1` wall / `830.2` sum** (`TTFT 1.25s`) | **`+22.0%` wall · `2.0×` TTFT** |
| **Shared-Prefix Code (`3K–4.5K`) `c = 4 / c = 8`** | `132.4 / 178.3 tok/s` sum | **`264.5 / 417.7 tok/s` sum** (`190.6 / 301.5` wall) | **`2.00× / 2.34×`** |
| **Shared-Prefix Code (`3K–4.5K`) `c = 16 / c = 32`** | `195.9 / 293.8 tok/s` sum | **`633.0 / 846.4 tok/s` sum** (`307.7 / 499.8` wall) | **`3.23× / 2.88×`** |
| **Single-Session Fan-Out `32 × 4K`** (`160t`, `T=0.65`) | — | **`378.9` wall / `633.5` sum** (`31/32` cloned, `95.4%` cached) | `5,120` tok in `13.51s` |
| **Single-Session Fan-Out `64 × 2K`** (`512t`, `T=0.65`) | `742.4 tok/s` sum (`64×4K`) | **`994.2 tok/s` sum** (`401.2` wall, `31,367` tok) | **`+33.9%` peak decode** |

### B. Standalone Batched Engine Scaling (`ingine-batch-bench`, `TP=1` vs `TP=2`)

| Concurrency & Mode | `1× Spark` `spec=4` (`DSpark`) | **`1× Spark` `spec=7` (`BatchDFlash2`)** | **`2× Spark` (`TP=2`) `spec=4` Wall / Steady-State** | VRAM / Node (`TP=2`) |
|---|---:|---:|---:|---:|
| **`c = 1`** (`prompts-short`, `1K` ctx) | `55.0 tok/s` (`α+1=2.54`) | **`80.6 tok/s`** (`α+1=4.82`, **`+46.5%`**) | — | — |
| **`c = 8`** (`640`–`1K` ctx) | `156.5 tok/s` | **`198.3 tok/s`** (`thr=0.30`, **`+26.7%`**) | `223.6 / 271.2 tok/s` | `12.4 GB` |
| **`c = 16`** (`640`–`1K` ctx) | `240.8 tok/s` | **`305.0 tok/s`** (`thr=0.20`, **`+26.7%`**) | `359.1 / 424.5 tok/s` | `13.0 GB` |
| **`c = 32`** (`640`–`1K` ctx) | `389.0 tok/s` | **`493.1 tok/s`** (`α+1=3.31`, **`+26.8%`**) | `521.9 / 667.7 tok/s` | `14.2 GB` |
| **`c = 64`** (`640`, `T=0` / `T=1.0`) | `600.8 / 561.8 tok/s` | — | **`820.5 / 755.7`** (`1,131.6` SS) | `16.7 GB` |
| **`c = 128`** (`640` / **`32K` ctx, `4.19M` KV**) | — | — | **`1,187.0 / 1,227.6`** (**`1,614 / 1,504` SS**) | `21.8 / 69.0 GB` |

### C. Persistent Prefix Anchor Cache (`6K` System Prompt, Zero Extra VRAM, `2× Spark TP=2`)

| Cross-Wave Workload (`5,993`-tok system prompt) | Before | **After (Persistent Prefix Cache)** | Speedup |
|---|---:|---:|---:|
| **Wave 2 (`c = 8`, new user suffixes)** | `TTFT 2.035s` · `wall 7.42s` (`103.5 tok/s`) | **`TTFT 0.465s` · `wall 3.61s`** (`8/8` cached `5,968t`, `212.5 tok/s`) | **`4.38×` TTFT · `2.06×` wall** |
| **Wave 3 (`c = 8`, new user suffixes)** | `TTFT 2.028s` · `wall 7.66s` (`100.2 tok/s`) | **`TTFT 0.457s` · `wall 3.24s`** (`8/8` cached `5,968t`, `237.3 tok/s`) | **`4.44×` TTFT · `2.36×` wall** |
| **Solo Follow-Up (`c = 1`, same system prompt)** | `TTFT 1.710s` · `0` cached · `wall 2.50s` | **`TTFT 0.086s` · `5,948` cached (`99.4%`) · `wall 1.23s`** | **`19.9×` TTFT · `2.03×` wall** |
