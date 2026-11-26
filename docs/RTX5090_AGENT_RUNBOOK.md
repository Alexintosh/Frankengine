# Frankengine — Autonomous Agent Runbook (RTX 5090 / RTX PRO 6000 / Single-GPU Blackwell)

**Purpose:** Self-contained instructions for an AI coding agent (or engineer) to clone **Frankengine**, download the pinned `Inkling-2-Mini` checkpoint and `DFlash 2` block-diffusion drafter, compile for single-GPU Blackwell (`sm_120a` RTX 5090 / RTX PRO 6000, or `sm_100a` B200), launch the live OpenAI-compatible `TP=1` server with the exact same configuration used on our DGX Spark qualification, and run the complete apples-to-apples benchmark suite (`scripts/bench_matrix.py`).

**Target Repository:** `https://github.com/Alexintosh/Frankengine.git`

---

## 0. Preflight Checks (Run First)

Execute these checks and record the output:

```bash
nvidia-smi
nvcc --version
cmake --version
g++ --version
df -h .
```

**Requirements:**
- **OS:** Linux (`x86_64` or `aarch64`)
- **GPU:** NVIDIA Blackwell (`sm_120a`: RTX 5090 32 GB, RTX PRO 6000 96 GB; `sm_121a`: DGX Spark GB10; `sm_100a`: B200 / GB200)
- **Driver / CUDA:** NVIDIA driver ≥ 580 + CUDA Toolkit **13.0+** (`nvcc` supporting `-gencode arch=compute_120a,code=sm_120a` for RTX 5090)
- **Compiler & Build Tools:** GCC 12+ (`g++`), `cmake` ≥ 3.24, `ninja` (optional), `python3` ≥ 3.10, `git`
- **Free Disk:** ≥ 28 GB (`~20.2 GB` base model + `~0.95 GB` DFlash 2 drafter + `~1.5 GB` build)
- **Free VRAM (auto-selected in Step 4):**
  - **`≥ 60 GiB` free (e.g., RTX PRO 6000 96 GB / B200 / DGX Spark 128 GB):** full **`MAX_CTX=262144` (256K solo `DFlash 2`) + `MAX_BATCH=32, BATCH_CTX=32768, BATCH_SPEC=7, BATCH_SPEC_THRESHOLD=0.30`** (`BatchDFlash2` `hybrid` auto-selected when `--dflash2-dir` & `--batch-spec >= 5`; ~`52 GiB` peak VRAM on `TP=1`).
  - **`27.5–59 GiB` free (e.g., dedicated 32 GB RTX 5090):**
    - For single-stream `256K` qualification (`scripts/bench_matrix.py`): **`MAX_CTX=262144 MAX_BATCH=1 BATCH_SPEC=0 PARK_MB=2048`** (`26.1 GiB` peak NVML).
    - For live multi-agent batched speculative serving on 32 GB (with `BatchDFlash2`, `~20 MiB/slot`, plus `clone_slot` & Persistent Prefix Anchors): **`MAX_CTX=131072 MAX_BATCH=16 BATCH_CTX=4096 BATCH_SPEC=7 BATCH_SPEC_THRESHOLD=0.30 PARK_MB=0`** (or `MAX_BATCH=8 BATCH_CTX=8192`, ~`28–29 GiB`).
  - **`23–27.4 GiB` free** (e.g., 32 GB RTX 5090 also driving a desktop display): **`MAX_CTX=131072 MAX_BATCH=1 BATCH_SPEC=0 PARK_MB=512`** (~`22–23 GiB` peak VRAM). `scripts/bench_matrix.py` auto-detects `max_model_len` from `/v1/models` and adjusts its maximum context test automatically.

---

## 1. Clone Frankengine

```bash
git clone https://github.com/Alexintosh/Frankengine.git
cd Frankengine
git rev-parse --short HEAD
```

---

## 2. Download Model Checkpoint & DFlash 2 Drafter

Install Hugging Face CLI if not already available:
```bash
command -v hf >/dev/null 2>&1 || pip install -U "huggingface_hub[cli]"
```

### 2a. Base Model (`thinkingmachines/Inkling-2-Mini` — ~20.2 GB, Apache-2.0)
Contains `model-00001-of-00001.safetensors` (`19.5 GB`, NVFP4/MXFP8), `mtp.safetensors` (`607 MB`, DSpark drafter), tokenizer, and chat template.

```bash
mkdir -p models
hf download thinkingmachines/Inkling-2-Mini \
  --local-dir models/Inkling-2-Mini
```

Verify required files exist:
```bash
ls -lh models/Inkling-2-Mini/model-00001-of-00001.safetensors \
       models/Inkling-2-Mini/mtp.safetensors \
       models/Inkling-2-Mini/config.json \
       models/Inkling-2-Mini/tokenizer.json \
       models/Inkling-2-Mini/chat_template.jinja
```

### 2b. DFlash 2 Block-Diffusion Drafter (`incoai/Inkling-2-Mini-DFlash2` — ~945 MB, Apache-2.0)
Download at the exact pinned revision used in our qualification:

```bash
hf download incoai/Inkling-2-Mini-DFlash2 \
  --revision 4f9088ea66c21f2b99ed657d1941c306b03e84d2 \
  --local-dir models/Inkling-2-Mini-DFlash2
```

Verify exact SHA-256 of the drafter weights:
```bash
echo "217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee  models/Inkling-2-Mini-DFlash2/model.safetensors" | sha256sum -c -
```

---

## 3. Build Frankengine for Single-GPU Blackwell (`TP=1`)

> **Important:** Passing `-DINKLING_COMM_SOURCE_DIR=""` disables the multi-node RoCE RDMA (`libibverbs`) transport cleanly — no InfiniBand/RoCE headers are required for single-GPU builds.

Pick the architecture flag matching your GPU:
- **RTX 5090 / RTX PRO 6000:** `ARCH=120a`
- **NVIDIA DGX Spark (GB10):** `ARCH=121a`
- **NVIDIA B200 / GB200:** `ARCH=100a`

```bash
export ARCH=120a
export NVCC_BIN="$(command -v nvcc)"

cmake -S engine/native -B "engine/native/build-${ARCH}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$NVCC_BIN" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DINKLING_COMM_SOURCE_DIR="" \
  -DINGINE_BUILD_TESTS=OFF

cmake --build "engine/native/build-${ARCH}" -j "$(nproc)" \
  --target ingine-serve ingine-bench ingine-batch-bench ingine-validate
```

Verify the compiled binaries exist:
```bash
ls -lh "engine/native/build-${ARCH}/apps/serve/ingine-serve" \
       "engine/native/build-${ARCH}/apps/bench/ingine-bench" \
       "engine/native/build-${ARCH}/apps/batch/ingine-batch-bench" \
       "engine/native/build-${ARCH}/apps/validate/ingine-validate"
```

---

## 4. Launch the Live `TP=1` OpenAI Server

Frankengine's production single-GPU settings:
- **Precision & Blackwell Kernels:** `--dense-fp8 --kv-fp8 --head-nvfp4 --draft-nvfp4` (native `mxf4nvf4`/`mxf8f6f4` prefill MoE, custom FP8 dense prefill `OUTER_VEC_32F` MMA, BF16 MMA `rel_logits`, and zero-copy NVFP4 LM head reused by DFlash 2)
- **Speculative Decoding:** `--spec --spec-widths fine2 --spec-threshold 0.65 --draft-type hybrid --dflash2-adaptive --dflash2-threshold 0.50 --dflash2-max-ctx 6144`
  - Uses eager-captured `FP8` CUDA-graph `DFlash 2` on short/medium prompts (`≤6,144` tokens) and global `DSpark` on longer contexts and sampled (`temperature > 0`) requests.
- **Batched Speculation & Prefix Caching:** when `BATCH_SPEC=7` (`BATCH_SPEC_THRESHOLD=0.30`, `BATCH_SPEC_MIN=2`) is active, `ingine-serve` auto-selects `BatchDFlash2` (`hybrid`) + `c=32` CUDA Graphs + automatic GPU `clone_slot` + zero-VRAM Persistent Prefix Anchors (`<|end_message|>` / `<|im_end|>` boundary detection) and exposes live `GET /v1/telemetry` & `GET /metrics`.
- **Context:** `MAX_CTX=262144` (256K) if `≥27.5 GiB` VRAM is free; otherwise set `MAX_CTX=131072 PARK_MB=512`.

Start the server in the background and log to `/tmp/frankengine_serve.log`:

```bash
export BUILD_DIR="$(pwd)/engine/native/build-${ARCH}"
export HEAD_MODEL_DIR="$(pwd)/models/Inkling-2-Mini"
export DFLASH2_DIR="$(pwd)/models/Inkling-2-Mini-DFlash2"
export HOST=127.0.0.1
export PORT=8834
export SERVED_NAME=inkling-2-mini

# Auto-size solo context & BatchDFlash2 speculative pool for free VRAM on GPU 0:
# (Note: scripts/config.sh defaults to Dual-Spark/96GB 32x32K spec=7; override on 32 GB GPUs to avoid OOM!)
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | head -n 1 | tr -d ' ')
if [ "${FREE_MIB:-0}" -ge 60000 ]; then
  echo "Free VRAM is ${FREE_MIB} MiB (>=60 GiB, e.g. RTX PRO 6000 / B200 / Spark): enabling 256K solo + 32x32K BatchDFlash2 (spec=7)."
  export MAX_CTX=262144
  export MAX_BATCH=32
  export BATCH_CTX=32768
  export BATCH_SPEC=7
  export BATCH_SPEC_THRESHOLD=0.30
  export BATCH_SPEC_MIN=2
  export MAX_QUEUE=64
  export PARK_MB=2048
elif [ "${FREE_MIB:-0}" -ge 28000 ]; then
  echo "Free VRAM is ${FREE_MIB} MiB (32 GB RTX 5090 class): using 256K single-stream preset for bench_matrix.py."
  export MAX_CTX=262144
  export MAX_BATCH=1
  export BATCH_SPEC=0
  export MAX_QUEUE=16
  export PARK_MB=2048
else
  echo "Free VRAM is ${FREE_MIB} MiB (<28000 MiB); using 128K single-stream preset."
  export MAX_CTX=131072
  export MAX_BATCH=1
  export BATCH_SPEC=0
  export MAX_QUEUE=16
  export PARK_MB=512
fi

./scripts/serve_tp1.sh > /tmp/frankengine_serve.log 2>&1 &
SERVER_PID=$!
echo "Started ingine-serve PID=$SERVER_PID (MAX_CTX=$MAX_CTX, MAX_BATCH=$MAX_BATCH, BATCH_CTX=${BATCH_CTX:-0}, BATCH_SPEC=$BATCH_SPEC)"
```

Wait until `/v1/models` returns HTTP 200 (startup + eager CUDA-graph capture typically takes `15–45 s`):

```bash
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "Server ready!"
    curl -sS "http://127.0.0.1:${PORT}/v1/models"
    echo ""
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: Server exited early. Tail of /tmp/frankengine_serve.log:"
    tail -n 60 /tmp/frankengine_serve.log
    exit 1
  fi
  sleep 2
done
```

Quick 1-shot sanity check + live telemetry check using the bundled `scripts/llmstat`:
```bash
curl -sSN "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": "Write a concise Python LRU cache class using OrderedDict with get and put methods."}],
    "temperature": 0,
    "max_tokens": 160,
    "chat_template_kwargs": {"enable_thinking": false},
    "stream": true,
    "stream_options": {"include_usage": true}
  }' | ./scripts/llmstat

curl -sS "http://127.0.0.1:${PORT}/v1/telemetry" && echo ""
```

---

## 5. Run the Complete Benchmark Suite (`--full` Default)

Run `scripts/bench_matrix.py` (zero external dependencies; defaults to `--full`). Pass a descriptive `--tag` (e.g., `rtx5090-tp1` or `rtxpro6000-tp1`) and `--server-log /tmp/frankengine_serve.log`:

```bash
python3 scripts/bench_matrix.py \
  --url "http://127.0.0.1:${PORT}/v1" \
  --tag "rtx5090-tp1" \
  --server-log /tmp/frankengine_serve.log
```

### What the `--full` Suite Measures Automatically
1. **Hardware & Build Fingerprint:** GPU name, driver/CUDA, clocks/power/temp, peak NVML VRAM residency, Frankengine git SHA, `max_model_len`.
2. **Core Headline Prompts:**
   - Short Code greedy (`LRU cache`, 160 tok, cold + warm) + **Greedy Repeatability Gate** (`PASS/DIFF`)
   - Short Chat greedy (`160` tok)
   - Sampled Code (`temp=0.7, top_p=0.9`, **`thinking=off`** AND **`thinking=on`**, `256` tok)
3. **8-Category Prompt Diversity Suite** (`code`, `math`, `json`, `html`, `prose`, `narrative`, `summary`, `counting` — `temp=0`, `256` tok, apples-to-apples multi-domain average + ITL `p50/p95/p99`).
4. **Full Context Scaling Curve (`4K`, `16K`, `32K`, `64K`, `128K`, and `240K` if `max_model_len ≥ 262144`):**
   - **Cold Prefill:** TTFT (`s`) and prefill `tok/s`
   - **Warm Greedy Decode (`256` tok)** at that exact context (prefix-cache hit, warm TTFT `ms` + decode `tok/s`)
   - **Warm Sampled Decode (`temp=0.7`, `256` tok)** at that exact context
5. **Multi-Turn Coding Agent Simulation (4 chained turns, `temp=0.7`):** verifies prefix-cache `extend` reuse (`cached_tokens`), inter-turn TTFT, and JSON tool-call generation.
6. **File-Edit / Refactor Workload (`temp=0`, `320` tok):** tests speculative acceptance on code modification.
7. **Needle-in-a-Haystack Exact Passkey Retrieval (`65K`, `122K`, and `240K` if `256K` enabled):** automated exact string match (`PASS/FAIL`) + TTFT + decode throughput.

*(Optional: pass `--quick` for a ~3-minute subset capped at 32K.)*

---

## 5b. Batched Speculative Concurrency Roofline (`ingine-batch-bench`, Optional)

Stop `ingine-serve` first (Step 6) so the GPU has full free VRAM, then run the standalone batched speculative benchmark with the fused `--head-nvfp4` unembed (`-922 MB` VRAM) — comparing `BatchDFlash2` (`--spec 7`) and `BatchDrafter` (`--spec 4`):

```bash
BATCH_BIN="engine/native/build-${ARCH}/apps/batch/ingine-batch-bench"
PROMPTS="engine/bench/prompts-short.json"

# 1. BatchDFlash2 (GAMMA=7, MXFP8 + FP8 KV rings, +27%..+46% over DSpark; fits 32 GB RTX 5090 up to c=32):
for C in 1 8 16 32; do
  "$BATCH_BIN" --model-dir models/Inkling-2-Mini --prompts "$PROMPTS" \
    --seqs "$C" --ctx 1024 --tokens 256 --kv-fp8 --head-nvfp4 --temperature 0 \
    --spec 7 --spec-threshold 0.30 --spec-min 2 \
    --batch-draft-type dflash2 --dflash2-dir models/Inkling-2-Mini-DFlash2
done

# 2. High-concurrency BatchDrafter DSpark (fits 32 GB RTX 5090 up to c=64 @ ~28.2 GB with --head-nvfp4):
for C in 8 16 32 64; do
  "$BATCH_BIN" --model-dir models/Inkling-2-Mini --prompts "$PROMPTS" \
    --seqs "$C" --ctx 640 --tokens 512 --kv-fp8 --head-nvfp4 --temperature 0 \
    --spec 4 --spec-threshold 0.50 --spec-min 2
done

# 3. 96 GB RTX PRO 6000 / B200 / Spark only (high concurrency & long-context pool):
if [ "${FREE_MIB:-0}" -ge 60000 ]; then
  "$BATCH_BIN" --model-dir models/Inkling-2-Mini --prompts "$PROMPTS" \
    --seqs 32 --ctx 8192 --tokens 256 --kv-fp8 --head-nvfp4 --temperature 0 \
    --spec 4 --spec-threshold 0.50 --spec-min 2
  "$BATCH_BIN" --model-dir models/Inkling-2-Mini --prompts "$PROMPTS" \
    --seqs 128 --ctx 640 --tokens 512 --kv-fp8 --head-nvfp4 --temperature 0 \
    --spec 4 --spec-threshold 0.50 --spec-min 2
fi
```

---

## 6. Clean Shutdown & Share Results Back

Stop the server cleanly once the benchmark completes and verify zero lingering GPU processes:

```bash
kill -TERM "$SERVER_PID"
wait "$SERVER_PID" || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

The benchmark writes a self-contained shareable folder:
```text
bench_results_rtx5090_tp1_<YYYYMMDDTHHMMSSZ>/
├── summary.md      # Full Markdown report (tables + PASS/FAIL gates)
├── results.json    # Structured raw metrics for direct cross-GPU comparison & plotting
└── bench_run.log   # Chronological execution log + sanitized server metric excerpt
```

Package and share back **all three files** (e.g., `tar -czf bench_results_rtx5090.tar.gz bench_results_rtx5090_tp1_*` or paste `summary.md` + attach `results.json` and `bench_run.log`).

---

## Reference Baselines for Comparison (NVIDIA DGX Spark GB10, `231 GB/s` DRAM BW, Promoted `main`)

| Workload | `1× DGX Spark` (`TP=1`) | `2× DGX Spark` (`TP=2` Promoted) | Expected `1× RTX 5090` (`1,792 GB/s`) |
|---|---:|---:|---:|
| **Short Code Greedy (`36t`, `DFlash 2`)** | `155.4 tok/s` | **`257.5 tok/s`** warm (`247.9` cold) | `~850–1,100 tok/s` |
| **File-Edit / Refactor (`1.5K`, `320t`)** | — | **`263.9 tok/s`** | `~900–1,150 tok/s` |
| **8-Category Mean Decode (`256t`, `T=0`)** | — | **`159.6 tok/s`** | `~600–750 tok/s` |
| **Cold Prefill (`2K–32K`, native FP8/NVFP4)** | `3,619` (`2K` FP8) / `2,770` (`32K`) | **`3,761` (`4K`) / `3,732` (`11K`) / `3,579` (`32K`)** | `~18,000–28,000 tok/s` |
| **32K Context Sampled Code (`T=0.7`, warm)** | `~77–95 tok/s` | **`143.6–199.9 tok/s`** (`9–99 ms` TTFT) | `~550–720 tok/s` |
| **Peak Memory @ `256K` Context (`c=1`)** | `26.1 GiB` (`1` GPU) | `31.3 GiB` (`32×32K` + `256K` solo) | `~26–27 GiB` (`c=1`) |
| **Batched `BatchDFlash2` `c=32` (`spec=7`)** | **`493.1 tok/s`** (`1K` ctx) | **`585.1` wall / `846.4` sum** (`live SSE`) | `~2,800–3,600 tok/s` |
| **Batched Spec `c=64` (`ctx=640`, `spec=4`)** | `600.8 tok/s` | `820.5 tok/s` (`1,132` SS) | `~4,100–4,500 tok/s` |
| **Batched Spec `c=128` (`ctx=640` / `32K`)** | — | `1,187 / 1,228 tok/s` (`1,614 / 1,504` SS) | `~5,000–6,500 tok/s` (PRO 6000 96 GB) |
| **6K Persistent Prefix Follow-Up (`c=1`)** | — | **`86 ms` TTFT** (`19.9×` vs `1.71s`) | `~15–25 ms` TTFT |
| **`16K` / `65K` / `122K` Exact Needles** | `PASS` | **`PASS`** (`157 / 182 / 129 tok/s`) | `PASS` |
