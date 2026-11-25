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
- **Free VRAM:**
  - **`≥ 27.5 GiB` free:** use full **`MAX_CTX=262144` (256K)** + `PARK_MB=2048` (exact match to our DGX Spark qualification; measured peak NVML residency is `26.1 GiB` / `26,733 MiB`).
  - **`23–27 GiB` free** (e.g., 32 GB RTX 5090 also driving a desktop display): use **`MAX_CTX=131072` (128K)** + `PARK_MB=512` (~`22–23 GiB` peak VRAM). `scripts/bench_matrix.py` auto-detects `max_model_len` from `/v1/models` and adjusts its maximum context test automatically.

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
  --target ingine-serve ingine-bench ingine-validate
```

Verify the compiled binaries exist:
```bash
ls -lh "engine/native/build-${ARCH}/apps/serve/ingine-serve" \
       "engine/native/build-${ARCH}/apps/bench/ingine-bench" \
       "engine/native/build-${ARCH}/apps/validate/ingine-validate"
```

---

## 4. Launch the Live `TP=1` OpenAI Server

Frankengine's production single-GPU settings:
- **Precision:** `--dense-fp8 --kv-fp8 --head-nvfp4 --draft-nvfp4` (zero-copy NVFP4 LM head reused by DFlash 2)
- **Speculative Decoding:** `--spec --spec-widths fine2 --spec-threshold 0.65 --draft-type hybrid --dflash2-adaptive --dflash2-threshold 0.50 --dflash2-max-ctx 1536`
  - Uses eager-captured `FP8` CUDA-graph `DFlash 2` on short greedy prompts (`≤1,536` tokens) and global `DSpark` on longer contexts and sampled (`temperature > 0`) requests.
- **Context:** `MAX_CTX=262144` (256K) if `≥27.5 GiB` VRAM is free; otherwise set `MAX_CTX=131072 PARK_MB=512`.

Start the server in the background and log to `/tmp/frankengine_serve.log`:

```bash
export BUILD_DIR="$(pwd)/engine/native/build-${ARCH}"
export HEAD_MODEL_DIR="$(pwd)/models/Inkling-2-Mini"
export DFLASH2_DIR="$(pwd)/models/Inkling-2-Mini-DFlash2"
export HOST=127.0.0.1
export PORT=8834
export SERVED_NAME=inkling-2-mini

# Check free VRAM on GPU 0 and choose 256K (preferred) or 128K fallback:
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | head -n 1 | tr -d ' ')
if [ "${FREE_MIB:-0}" -ge 28000 ]; then
  export MAX_CTX=262144
  export PARK_MB=2048
else
  echo "Free VRAM is ${FREE_MIB} MiB (<28000 MiB); using 128K context preset."
  export MAX_CTX=131072
  export PARK_MB=512
fi

./scripts/serve_tp1.sh > /tmp/frankengine_serve.log 2>&1 &
SERVER_PID=$!
echo "Started ingine-serve PID=$SERVER_PID (MAX_CTX=$MAX_CTX, PARK_MB=$PARK_MB)"
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

Quick 1-shot sanity check using the bundled `scripts/llmstat`:
```bash
curl -sSN "http://127.0.0.1:${PORT}/v1/chat/completions" \
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

## Reference Baselines for Comparison (NVIDIA DGX Spark GB10, `231 GB/s` DRAM BW)

| Workload | `1× DGX Spark` (`TP=1`) | `2× DGX Spark` (`TP=2`) | Expected `1× RTX 5090` (`1,792 GB/s`) |
|---|---:|---:|---:|
| **Short Code Greedy (`LRU cache`, 160 tok)** | `155.4 tok/s` | `243.1 tok/s` | `~800–1,050 tok/s` |
| **32K Context Sampled Code (`T=0.7`, warm)** | `~77–95 tok/s` | `199.9 tok/s` (`9 ms` TTFT) | `~550–700 tok/s` |
| **Cold Prefill (`4K–32K`)** | `2,770–2,889 tok/s` | `2,802–3,200 tok/s` | `~15,000–25,000 tok/s` |
| **Peak Memory @ `256K` Context** | `26.1 GiB` (`1` GPU) | `27.2 GiB` (`2` GPUs) | `~26–27 GiB` |
| **`65K` / `122K` Exact Needles** | `PASS` / `PASS` | `PASS` / `PASS` | `PASS` / `PASS` |
