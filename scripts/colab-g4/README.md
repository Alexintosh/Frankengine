# Frankengine on Google Colab G4 (RTX PRO 6000 Blackwell 96 GB)

One-command launcher and benchmark harness for running **Frankengine TP=1** (`Inkling-2-Mini` + `DFlash 2`, `sm_120a`) on **Google Colab G4** (`1× NVIDIA RTX PRO 6000 Blackwell 96 GB`, 48 vCPU, 176 GiB RAM, `~10.05 CU/hr`).

## Prerequisites

1. A Google Colab account with Compute Units (Colab Pro / Pro+ / Google AI Pro / Ultra).
2. `google-colab-cli` (`colab`) installed and authenticated locally:
   ```bash
   pipx install google-colab-cli   # or: uv tool install google-colab-cli
   colab auth
   ```

## Quickstart

```bash
cd scripts/colab-g4

# 1. Provision Colab G4, build Frankengine (sm_120a, CUDA 13+), download weights, start server + SSH tunnel
./start.sh fp8                    # Fast FP8 preset (default, 256K solo + 32x32K BatchDFlash2, Vision ON)
# or:
./start.sh exact                  # 100% bit-exact lossless (PICO12 + CERT_HEAD, BF16 KV, Vision ON)

# 2. Provision without opening a foreground tunnel (ready for automated benchmarking)
./start.sh fp8 --no-tunnel

# 3. Run full spark-bench (cmpbench.sh) inside the G4 VM, pull results locally, and auto-stop billing
./bench.sh --suite spark-bench --preset fp8 --auto-stop

# Run BOTH FP8 and 100% Lossless presets back-to-back (+ optional 3x295 quality inkqc)
./bench.sh --suite spark-bench --preset both --steps "sbf think dt lc needle guard quality" --auto-stop

# Run Frankengine's bench_matrix.py --full
./bench.sh --suite matrix --preset fp8 --auto-stop

# 4. Optional: launch MiaAI-Lab/sparkDash inside the G4 VM (zero-latency Decode/Quality/Tool Bench + live nvidia-smi)
./sparkdash.sh --session frank-g4 --ui-port 13000
# Open http://127.0.0.1:13000 in your browser

# 5. Always verify billing is stopped when done
./stop.sh
```

## Notes

- **CUDA 13 auto-check:** `start.sh` checks `nvcc --version` inside the Colab G4 VM and automatically installs `cuda-toolkit-13-*` from NVIDIA's Ubuntu repository if the image ships with CUDA 12.x.
- **Cold start:** ~3–4 minutes total (~1–2 min to download ~21.2 GB of weights + <60 s Ninja build on 48 vCPUs + ~8 s model load).
- **Results:** `bench.sh` downloads all JSONs, logs, `nvidia-smi`, and `/metrics` snapshots into `colab-g4-results/<timestamp>-rtxpro6000-<preset>/`.
