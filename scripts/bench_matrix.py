#!/usr/bin/env python3
"""Frankengine End-to-End Live Benchmark & Qualification Suite.

Zero external Python dependencies (stdlib only). Runs against any live OpenAI-compatible
Frankengine / ingine-serve endpoint (TP=1 or TP=2, RTX 5090, RTX PRO 6000, DGX Spark, B200).

Default mode is --full (complete context curve 0K..240K, 8-category diversity suite,
sampled thinking on/off, multi-turn coding agent, file-edit/refactor, and 65K/122K/240K needles).
Pass --quick for a fast ~3-minute run up to 32K.

Outputs a self-contained shareable folder:
  bench_results_<gpu>_<timestamp>/
    summary.md       # Human-readable Markdown tables + PASS/FAIL gates
    results.json     # Full structured raw metrics for cross-GPU comparison & plotting
    bench_run.log    # Chronological run log + sanitized server metric excerpt (if --server-log given)
"""

import argparse
import datetime
import json
import math
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def now_utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return s[:48] or "gpu"


def run_cmd(cmd: list[str], timeout: int = 10) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return r.stdout.strip()
    except Exception:
        return ""


def query_nvidia_smi() -> dict:
    info = {"gpus": [], "driver": "", "cuda": ""}
    hdr = run_cmd(["nvidia-smi"])
    if hdr:
        m = re.search(r"Driver Version:\s*([\d.]+)", hdr)
        if m:
            info["driver"] = m.group(1)
        m = re.search(r"CUDA Version:\s*([\d.]+)", hdr)
        if m:
            info["cuda"] = m.group(1)
    rows = run_cmd([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,power.draw,clocks.gr,clocks.mem",
        "--format=csv,noheader,nounits",
    ])
    for line in rows.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 9:
            info["gpus"].append({
                "index": parts[0],
                "name": parts[1],
                "vram_total_mib": parts[2],
                "vram_used_mib": parts[3],
                "vram_free_mib": parts[4],
                "temp_c": parts[5],
                "power_w": parts[6],
                "clock_gr_mhz": parts[7],
                "clock_mem_mhz": parts[8],
            })
    return info


def query_nvml_peak_mib() -> int:
    rows = run_cmd([
        "nvidia-smi",
        "--query-compute-apps=used_memory",
        "--format=csv,noheader,nounits",
    ])
    vals = []
    for line in rows.splitlines():
        line = line.strip()
        if line.isdigit():
            vals.append(int(line))
    if vals:
        return sum(vals)
    # Fallback to GPU total memory.used (e.g. unified memory systems where compute-apps may be N/A)
    gpu_rows = run_cmd([
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ])
    gvals = [int(x.strip()) for x in gpu_rows.splitlines() if x.strip().isdigit()]
    return sum(gvals) if gvals else 0


def collect_system_info(repo_dir: Path, base_url: str) -> dict:
    smi = query_nvidia_smi()
    git_sha = run_cmd(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"])
    git_branch = run_cmd(["git", "-C", str(repo_dir), "rev-parse", "--abbrev-ref", "HEAD"])
    mem_total_kb = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
                break
    except Exception:
        pass
    cpu_model = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if "model name" in line.lower():
                cpu_model = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass

    models_payload = {}
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=15) as r:
            models_payload = json.load(r)
    except Exception as exc:
        models_payload = {"error": str(exc)}

    model_id = "inkling-2-mini"
    max_model_len = 0
    data = models_payload.get("data") or []
    if data and isinstance(data, list):
        model_id = data[0].get("id") or model_id
        max_model_len = int(data[0].get("max_model_len") or 0)

    return {
        "timestamp_utc": now_utc_iso(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": cpu_model,
        "host_ram_gib": round(mem_total_kb / (1024 * 1024), 1) if mem_total_kb else None,
        "nvidia": smi,
        "frankengine_git_sha": git_sha or "unknown",
        "frankengine_git_branch": git_branch or "unknown",
        "model_id": model_id,
        "max_model_len": max_model_len,
    }


def percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def stream_chat(
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 160,
    enable_thinking: bool = False,
    seed: int | None = 42,
    timeout: int = 900,
    api_key: str | None = None,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None:
        payload["seed"] = seed

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    t0 = time.monotonic()
    t_first = None
    t_prev = None
    chunk_itls_ms: list[float] = []
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict = {}
    finish_reason = None
    chunk_count = 0

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if "error" in obj:
                    raise RuntimeError(f"API error: {obj['error']}")
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or ():
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    r_txt = delta.get("reasoning_content")
                    c_txt = delta.get("content")
                    if r_txt or c_txt:
                        now = time.monotonic()
                        if t_first is None:
                            t_first = now
                        else:
                            chunk_itls_ms.append((now - t_prev) * 1000.0)
                        t_prev = now
                        chunk_count += 1
                        if r_txt:
                            reasoning_parts.append(r_txt)
                        if c_txt:
                            content_parts.append(c_txt)
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "total_s": round(time.monotonic() - t0, 4),
        }

    t_end = time.monotonic()
    total_s = max(1e-6, t_end - t0)
    prompt_tok = int(usage.get("prompt_tokens") or 0)
    comp_tok = int(usage.get("completion_tokens") or chunk_count)
    cached_tok = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    uncached_tok = max(0, prompt_tok - cached_tok)

    ttft_s = max(1e-6, (t_first - t0)) if t_first is not None else total_s
    decode_s = max(1e-6, (t_end - t_first)) if t_first is not None else None
    decode_tok_s = (
        (max(comp_tok - 1, 0) / decode_s) if (decode_s and comp_tok > 1) else (comp_tok / decode_s if decode_s else 0.0)
    )
    prefill_tok_s = (uncached_tok / ttft_s) if uncached_tok > 0 else None
    e2e_tok_s = comp_tok / total_s if comp_tok else 0.0

    # Normalize chunk ITL to per-token average when speculative chunks carry >1 token
    avg_itl_ms = (decode_s * 1000.0 / max(comp_tok - 1, 1)) if (decode_s and comp_tok > 1) else None

    return {
        "ok": True,
        "prompt_tokens": prompt_tok,
        "cached_tokens": cached_tok,
        "uncached_tokens": uncached_tok,
        "completion_tokens": comp_tok,
        "finish_reason": finish_reason,
        "ttft_s": round(ttft_s, 4),
        "decode_s": round(decode_s, 4) if decode_s is not None else None,
        "total_s": round(total_s, 4),
        "prefill_tok_s": round(prefill_tok_s, 1) if prefill_tok_s is not None else None,
        "decode_tok_s": round(decode_tok_s, 1),
        "e2e_tok_s": round(e2e_tok_s, 1),
        "avg_itl_ms": round(avg_itl_ms, 2) if avg_itl_ms is not None else None,
        "chunk_itl_p50_ms": round(percentile(chunk_itls_ms, 50), 2) if chunk_itls_ms else None,
        "chunk_itl_p95_ms": round(percentile(chunk_itls_ms, 95), 2) if chunk_itls_ms else None,
        "chunk_itl_p99_ms": round(percentile(chunk_itls_ms, 99), 2) if chunk_itls_ms else None,
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "nvml_mib": query_nvml_peak_mib(),
    }


def make_code_haystack(target_tokens: int, *, needle: str | None = None, depth_frac: float = 0.65) -> str:
    """Deterministic realistic Python codebase haystack (~target_tokens tokens).
    Avoids trivial repetition so speculative acceptance and attention stay realistic.
    """
    # Calibrated on Inkling-2-Mini tokenizer: each generated module block is ~152 tokens.
    n_blocks = max(4, int(target_tokens / 152))
    needle_idx = max(1, min(n_blocks - 1, int(n_blocks * depth_frac))) if needle else -1
    subsystems = [
        "scheduler", "allocator", "router", "tokenizer", "dispatcher",
        "kv_cache", "transport", "watchdog", "profiler", "verifier",
    ]
    lines = [
        "Below is an archival snapshot of an internal distributed runtime repository.",
        "Read all modules carefully; you will be asked a concise engineering question at the end.",
        "",
    ]
    for i in range(n_blocks):
        sub = subsystems[i % len(subsystems)]
        port = 7000 + (i * 37) % 2000
        cap = 64 + (i * 13) % 512
        retries = 1 + (i % 5)
        lines.append(
            f"# module_{i:04d}_{sub}.py\n"
            f"class {sub.title()}Worker{i:04d}:\n"
            f"    def __init__(self, endpoint='node-{i % 16}:{port}', capacity={cap}, retries={retries}):\n"
            f"        self.endpoint = endpoint\n"
            f"        self.capacity = capacity\n"
            f"        self.retries = retries\n"
            f"        self.watermark = {((i * 19) % 97) / 100.0:.2f}\n"
            f"    def dispatch(self, batch_id: int, weight: float) -> float:\n"
            f"        budget = max(1, self.capacity - (batch_id % 17))\n"
            f"        return round((weight * self.watermark) / budget + {(i % 11) * 0.03:.2f}, 6)\n"
        )
        if i == needle_idx and needle:
            lines.append(
                f"# AUDIT_NOTE: The verified release override passkey for cluster deployment is {needle}.\n"
                f"# Record this exact passkey verbatim if requested.\n"
            )
    return "\n".join(lines)


def make_refactor_source(n_funcs: int) -> str:
    """Realistic Python file for edit/refactor benchmark (tests copy + edit acceptance)."""
    parts = [
        "from dataclasses import dataclass",
        "from typing import Dict, List, Optional",
        "",
        "@dataclass",
        "class MetricSample:",
        "    name: str",
        "    latency_ms: float",
        "    tokens: int",
        "    cached: bool = False",
        "",
        "class MetricsAggregator:",
        "    def __init__(self, window_size: int = 128):",
        "        self.window_size = window_size",
        "        self.samples: Dict[str, List[MetricSample]] = {}",
        "",
    ]
    for i in range(n_funcs):
        parts.append(
            f"    def record_stage_{i:02d}(self, latency_ms: float, tokens: int, cached: bool = False) -> float:\n"
            f"        key = 'stage_{i:02d}'\n"
            f"        bucket = self.samples.setdefault(key, [])\n"
            f"        bucket.append(MetricSample(name=key, latency_ms=latency_ms, tokens=tokens, cached=cached))\n"
            f"        if len(bucket) > self.window_size:\n"
            f"            del bucket[0]\n"
            f"        return sum(s.latency_ms for s in bucket) / len(bucket)\n"
        )
    return "\n".join(parts)


DIVERSITY_PROMPTS = [
    ("code", "Write a clean Python implementation of a thread-safe token bucket rate limiter with burst support."),
    ("math", "Prove step by step that the sum of the first n odd positive integers equals n^2, then compute it for n=47."),
    ("json", "Return a valid JSON object describing a 6-step CI/CD pipeline for a CUDA C++ engine with fields name, command, timeout_sec, and artifacts."),
    ("html", "Write a minimal self-contained HTML snippet with inline CSS for a dark-mode latency telemetry card showing TTFT, prefill, and decode metrics."),
    ("prose", "Explain in clear technical prose why low-batch LLM decoding is memory-bandwidth bound while long-prompt prefill is compute bound."),
    ("narrative", "Write a concise two-paragraph story about a systems engineer debugging an intermittent race condition in an RDMA cluster at 3 AM."),
    ("summary", "Summarize the key architectural differences between dense transformers, Mixture-of-Experts models, and looped-layer transformer architectures in 5 bullet points."),
    ("counting", "List the numbers from 1 to 80 separated by commas, marking every multiple of 7 in square brackets like [7]."),
]


def sanitize_server_log(path: Path, max_lines: int = 120) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    kept = []
    for ln in lines:
        if any(k in ln for k in ("ingine.server:", "drafter=", "reuse=", "decode=", "prefill=", "serving ")):
            # Strip IPv4 addresses or user home paths for clean sharing
            ln = re.sub(r"/home/[^/\s]+", "/home/user", ln)
            ln = re.sub(r"\b100\.\d+\.\d+\.\d+\b", "<tailnet-ip>", ln)
            kept.append(ln)
    return "\n".join(kept)


def main():
    p = argparse.ArgumentParser(description="Frankengine complete live OpenAI benchmark & qualification suite.")
    p.add_argument("--url", default="http://127.0.0.1:8834/v1", help="OpenAI-compatible base URL (default: http://127.0.0.1:8834/v1)")
    p.add_argument("--model", default="", help="Model ID (auto-detected from /v1/models if omitted)")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="Optional Bearer API key")
    p.add_argument("--quick", action="store_true", help="Fast ~3-min run capped at 32K context (default is --full)")
    p.add_argument("--full", action="store_true", help="Run full suite including 64K/128K/240K & needles (default)")
    p.add_argument("--out-dir", default="", help="Output directory (auto-generated if omitted)")
    p.add_argument("--server-log", default="", help="Optional path to ingine-serve log file to include sanitized excerpt")
    p.add_argument("--tag", default="", help="Optional human tag (e.g. 'rtx5090-tp1', 'dual-spark-tp2')")
    args = p.parse_args()

    is_full = not args.quick
    repo_dir = Path(__file__).resolve().parent.parent
    sysinfo = collect_system_info(repo_dir, args.url)
    model = args.model or sysinfo["model_id"]
    max_ctx = sysinfo["max_model_len"] or 131072

    gpu_name = sysinfo["nvidia"]["gpus"][0]["name"] if sysinfo["nvidia"]["gpus"] else "unknown_gpu"
    gpu_slug = slugify(args.tag or gpu_name)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else (Path.cwd() / f"bench_results_{gpu_slug}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "bench_run.log"

    log_lines: list[str] = []

    def log(msg: str):
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"Frankengine Benchmark Suite ({'FULL' if is_full else 'QUICK'})")
    log(f"Endpoint: {args.url} | Model: {model} | max_model_len: {max_ctx}")
    log(f"GPU: {gpu_name} | Driver: {sysinfo['nvidia']['driver']} | CUDA: {sysinfo['nvidia']['cuda']}")

    results: dict = {
        "meta": {**sysinfo, "mode": "full" if is_full else "quick", "tag": args.tag or gpu_slug, "model_used": model},
        "benchmarks": {},
        "gates": {},
    }
    peak_nvml_mib = query_nvml_peak_mib()

    def track(res: dict) -> dict:
        nonlocal peak_nvml_mib
        if res.get("nvml_mib", 0) > peak_nvml_mib:
            peak_nvml_mib = res["nvml_mib"]
        return res

    # Warmup (not scored)
    log("Warming up server...")
    track(stream_chat(args.url, model, [{"role": "user", "content": "Reply with READY."}], max_tokens=8, api_key=args.api_key))

    # ------------------------------------------------------------------
    # 1. Core Headline Short Prompts + Greedy Repeatability Gate
    # ------------------------------------------------------------------
    log("=== Section 1: Core Short Prompts (Greedy & Sampled) ===")
    lru_prompt = "Write a concise Python LRU cache class using OrderedDict with get and put methods."
    chat_prompt = "Explain how hash maps handle collisions in clear, concise prose."

    short_code_1 = track(stream_chat(args.url, model, [{"role": "user", "content": lru_prompt}], temperature=0.0, max_tokens=160, enable_thinking=False, api_key=args.api_key))
    short_code_2 = track(stream_chat(args.url, model, [{"role": "user", "content": lru_prompt}], temperature=0.0, max_tokens=160, enable_thinking=False, api_key=args.api_key))
    short_chat = track(stream_chat(args.url, model, [{"role": "user", "content": chat_prompt}], temperature=0.0, max_tokens=160, enable_thinking=False, api_key=args.api_key))
    sampled_code_nothink = track(stream_chat(args.url, model, [{"role": "user", "content": lru_prompt}], temperature=0.7, top_p=0.9, max_tokens=256, enable_thinking=False, seed=42, api_key=args.api_key))
    sampled_code_think = track(stream_chat(args.url, model, [{"role": "user", "content": lru_prompt}], temperature=0.7, top_p=0.9, max_tokens=256, enable_thinking=True, seed=42, api_key=args.api_key))

    greedy_repeat_ok = (short_code_1.get("ok") and short_code_2.get("ok") and short_code_1.get("content") == short_code_2.get("content"))
    results["gates"]["greedy_repeatability"] = bool(greedy_repeat_ok)
    results["benchmarks"]["core_short"] = {
        "short_code_greedy_cold": {k: v for k, v in short_code_1.items() if k not in ("content", "reasoning")},
        "short_code_greedy_warm": {k: v for k, v in short_code_2.items() if k not in ("content", "reasoning")},
        "short_chat_greedy": {k: v for k, v in short_chat.items() if k not in ("content", "reasoning")},
        "sampled_code_t07_thinking_off": {k: v for k, v in sampled_code_nothink.items() if k not in ("content", "reasoning")},
        "sampled_code_t07_thinking_on": {k: v for k, v in sampled_code_think.items() if k not in ("content", "reasoning")},
    }
    for name, r in results["benchmarks"]["core_short"].items():
        log(f"  {name}: ok={r.get('ok')} prompt={r.get('prompt_tokens')} cached={r.get('cached_tokens')} comp={r.get('completion_tokens')} TTFT={r.get('ttft_s')}s decode={r.get('decode_tok_s')} tok/s")
    log(f"  Gate greedy_repeatability: {'PASS' if greedy_repeat_ok else 'DIFF'}")

    # ------------------------------------------------------------------
    # 2. 8-Category Prompt Diversity Suite (apples-to-apples vs TensorFold/vLLM)
    # ------------------------------------------------------------------
    log("=== Section 2: 8-Category Diversity Suite (temp=0, thinking=off, 256 tok) ===")
    div_rows = []
    for cat, prompt_txt in DIVERSITY_PROMPTS:
        r = track(stream_chat(args.url, model, [{"role": "user", "content": prompt_txt}], temperature=0.0, max_tokens=256, enable_thinking=False, api_key=args.api_key))
        row = {"category": cat, **{k: v for k, v in r.items() if k not in ("content", "reasoning")}}
        div_rows.append(row)
        log(f"  [{cat:9s}] comp={r.get('completion_tokens')} TTFT={r.get('ttft_s')}s decode={r.get('decode_tok_s')} tok/s")
    valid_div = [x["decode_tok_s"] for x in div_rows if x.get("ok") and x.get("decode_tok_s")]
    div_mean = round(statistics.mean(valid_div), 1) if valid_div else 0.0
    results["benchmarks"]["diversity_8cat"] = {"mean_decode_tok_s": div_mean, "items": div_rows}
    log(f"  => 8-Category Mean Decode: {div_mean} tok/s")

    # ------------------------------------------------------------------
    # 3. Context Scaling Curve: Cold Prefill + Warm Decode (greedy & sampled)
    # ------------------------------------------------------------------
    log("=== Section 3: Context Scaling Curve (Cold Prefill + Warm Decode) ===")
    ctx_targets = [4096, 16384, 32768]
    if is_full:
        if max_ctx >= 70000:
            ctx_targets.append(65536)
        if max_ctx >= 131072:
            ctx_targets.append(128000)
        if max_ctx >= 262144:
            ctx_targets.append(240000)

    ctx_curve = []
    for target_ctx in ctx_targets:
        if target_ctx + 1024 > max_ctx:
            log(f"  Skipping {target_ctx} (exceeds max_model_len={max_ctx})")
            continue
        haystack = make_code_haystack(target_ctx)
        # Step A: Cold prefill (16 tokens, thinking=off -> clean TTFT & prefill tok/s, populates prefix cache)
        turn1_user = {"role": "user", "content": haystack + "\n\nConfirm you have read the repository snapshot in one short sentence."}
        cold = track(stream_chat(args.url, model, [turn1_user], temperature=0.0, max_tokens=16, enable_thinking=False, api_key=args.api_key))
        turn1_asst = {"role": "assistant", "content": cold.get("content") or "I have read the repository snapshot."}
        # Step B: Warm greedy decode at this context (256 tokens, reuses turn 1 prefix cache!)
        warm_g_msgs = [
            turn1_user,
            turn1_asst,
            {"role": "user", "content": "Write a complete, production-ready Python module (at least 45 lines) implementing a ClusterPoolManager class that tracks worker capacities, validates watermarks, and routes batches across these workers."},
        ]
        warm_g = track(stream_chat(args.url, model, warm_g_msgs, temperature=0.0, max_tokens=256, enable_thinking=False, api_key=args.api_key))
        # Step C: Warm sampled decode (temp=0.7) at this context (256 tokens, Turn 3 extends Turn 2 prefix cache!)
        warm_s_msgs = warm_g_msgs + [
            {"role": "assistant", "content": warm_g.get("content") or "class ClusterPoolManager:\n    pass"},
            {"role": "user", "content": "Now write a complete 45-line Python async health-check and retry controller class with exponential backoff and circuit breaker metrics for failed worker dispatches."},
        ]
        warm_s = track(stream_chat(args.url, model, warm_s_msgs, temperature=0.7, top_p=0.9, max_tokens=256, enable_thinking=False, seed=42, api_key=args.api_key))

        entry = {
            "target_ctx": target_ctx,
            "cold_prefill": {k: v for k, v in cold.items() if k not in ("content", "reasoning")},
            "warm_greedy_decode": {k: v for k, v in warm_g.items() if k not in ("content", "reasoning")},
            "warm_sampled_decode": {k: v for k, v in warm_s.items() if k not in ("content", "reasoning")},
        }
        ctx_curve.append(entry)
        log(
            f"  @{target_ctx//1024:>3d}K | cold: prompt={cold.get('prompt_tokens')} TTFT={cold.get('ttft_s')}s "
            f"prefill={cold.get('prefill_tok_s')} tok/s | "
            f"warm_g: cached={warm_g.get('cached_tokens')} TTFT={warm_g.get('ttft_s')}s decode={warm_g.get('decode_tok_s')} tok/s | "
            f"warm_s(0.7): cached={warm_s.get('cached_tokens')} TTFT={warm_s.get('ttft_s')}s decode={warm_s.get('decode_tok_s')} tok/s"
        )
    results["benchmarks"]["context_scaling"] = ctx_curve

    # ------------------------------------------------------------------
    # 4. Realistic Multi-Turn Coding Agent Simulation (prefix extend)
    # ------------------------------------------------------------------
    log("=== Section 4: Multi-Turn Coding Agent Simulation (4 turns, temp=0.7) ===")
    src_4k = make_refactor_source(40)
    msgs = [
        {"role": "system", "content": "You are a concise, precise systems coding agent. Output clean Python code or JSON tool calls."},
        {"role": "user", "content": f"Here is metrics.py:\n```python\n{src_4k}\n```\nSummarize the class structure in 3 bullet points."},
    ]
    agent_turns = []
    followups = [
        "Add a `reset(self, key: Optional[str] = None)` method to MetricsAggregator. Show only the new method.",
        "Emit a JSON tool call `{\"tool\": \"patch_file\", \"path\": \"metrics.py\", \"changes\": [...]}` describing a fix for empty bucket division.",
        "Now write a 20-line pytest unit test covering `record_stage_00`, window eviction, and `reset`.",
    ]
    for turn_idx in range(4):
        r = track(stream_chat(args.url, model, msgs, temperature=0.7, top_p=0.9, max_tokens=200, enable_thinking=False, seed=42 + turn_idx, api_key=args.api_key))
        agent_turns.append({"turn": turn_idx + 1, **{k: v for k, v in r.items() if k not in ("content", "reasoning")}})
        log(
            f"  turn {turn_idx+1}: prompt={r.get('prompt_tokens')} cached={r.get('cached_tokens')} "
            f"TTFT={r.get('ttft_s')}s decode={r.get('decode_tok_s')} tok/s"
        )
        msgs.append({"role": "assistant", "content": r.get("content") or "OK"})
        if turn_idx < len(followups):
            msgs.append({"role": "user", "content": followups[turn_idx]})
    results["benchmarks"]["multi_turn_agent"] = agent_turns

    # ------------------------------------------------------------------
    # 5. File-Edit / Refactor Workload (tests speculative copy acceptance)
    # ------------------------------------------------------------------
    log("=== Section 5: File-Edit / Refactor Workload (temp=0, 320 tok) ===")
    small_src = make_refactor_source(10)
    edit_prompt = (
        "Refactor the following Python class by adding a `min_latency_ms` attribute to `MetricSample` "
        "(default 0.0) while preserving all existing method names, signatures, and logic verbatim. "
        "Output the updated code starting from `from dataclasses import dataclass`.\n\n```python\n"
        + small_src + "\n```"
    )
    edit_res = track(stream_chat(args.url, model, [{"role": "user", "content": edit_prompt}], temperature=0.0, max_tokens=320, enable_thinking=False, api_key=args.api_key))
    results["benchmarks"]["file_edit_refactor"] = {k: v for k, v in edit_res.items() if k not in ("content", "reasoning")}
    log(f"  refactor: prompt={edit_res.get('prompt_tokens')} comp={edit_res.get('completion_tokens')} TTFT={edit_res.get('ttft_s')}s decode={edit_res.get('decode_tok_s')} tok/s")

    # ------------------------------------------------------------------
    # 6. Needle-in-a-Haystack Exact Passkey Retrieval (65K / 122K / 240K)
    # ------------------------------------------------------------------
    log("=== Section 6: Needle-in-a-Haystack Exact Passkey Retrieval ===")
    needle_specs = []
    if is_full:
        if max_ctx >= 70000:
            needle_specs.append(("65K", 65000, 0.40, "CORMORANT-78416-QUARTZ"))
        if max_ctx >= 131072:
            needle_specs.append(("122K", 121000, 0.84, "MANGROVE-29573-COBALT"))
        if max_ctx >= 262144:
            needle_specs.append(("240K", 238000, 0.52, "PEREGRINE-64209-BASALT"))
    else:
        needle_specs.append(("16K", 16000, 0.65, "KESTREL-41902-GRANITE"))

    needles_out = []
    all_needles_pass = True
    for label, n_tok, depth, passkey in needle_specs:
        hay = make_code_haystack(n_tok, needle=passkey, depth_frac=depth)
        prompt = hay + "\n\nWhat is the exact release override passkey mentioned in AUDIT_NOTE? Reply with ONLY the passkey."
        r = track(stream_chat(args.url, model, [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=32, enable_thinking=False, api_key=args.api_key))
        found = bool(r.get("ok") and passkey in (r.get("content") or ""))
        if not found:
            all_needles_pass = False
        needles_out.append({
            "label": label,
            "expected": passkey,
            "passed": found,
            "response_excerpt": (r.get("content") or "")[:80],
            **{k: v for k, v in r.items() if k not in ("content", "reasoning")},
        })
        log(f"  needle {label}: prompt={r.get('prompt_tokens')} TTFT={r.get('ttft_s')}s prefill={r.get('prefill_tok_s')} tok/s decode={r.get('decode_tok_s')} tok/s -> {'PASS' if found else 'FAIL'} ({(r.get('content') or '')[:40]!r})")

    results["benchmarks"]["needles"] = needles_out
    results["gates"]["needles_all_pass"] = all_needles_pass
    results["meta"]["peak_nvml_mib"] = peak_nvml_mib

    # ------------------------------------------------------------------
    # Write Shareable Artifacts: results.json, summary.md, bench_run.log
    # ------------------------------------------------------------------
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.server_log:
        excerpt = sanitize_server_log(Path(args.server_log))
        if excerpt:
            log_lines.append("\n=== SANITIZED SERVER LOG EXCERPT ===")
            log_lines.extend(excerpt.splitlines())
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    # Build Markdown summary
    md = [
        f"# Frankengine Benchmark Report — `{results['meta']['tag']}`",
        "",
        "## System & Build Fingerprint",
        f"- **Timestamp (UTC):** `{sysinfo['timestamp_utc']}`",
        f"- **GPU:** `{gpu_name}` (Driver `{sysinfo['nvidia']['driver']}`, CUDA `{sysinfo['nvidia']['cuda']}`)",
        f"- **Peak GPU Memory (NVML):** **`{peak_nvml_mib} MiB`** (`{peak_nvml_mib / 1024.0:.2f} GiB`)",
        f"- **Host:** `{sysinfo['os']}` · CPU `{sysinfo['cpu']}` · RAM `{sysinfo['host_ram_gib']} GiB`",
        f"- **Frankengine:** `{sysinfo['frankengine_git_branch']}` @ `{sysinfo['frankengine_git_sha']}`",
        f"- **Served Model:** `{model}` · **`max_model_len = {max_ctx}`**",
        f"- **Gates:** Greedy Repeatability = **`{'PASS' if greedy_repeat_ok else 'DIFF'}`** · Needles = **`{'PASS' if all_needles_pass else 'FAIL'}`**",
        "",
        "## 1. Core Headline Prompts",
        "| Workload | Prompt (cached) | Comp | TTFT (s) | Decode (tok/s) | E2E (tok/s) | Avg ITL (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results["benchmarks"]["core_short"].items():
        md.append(
            f"| `{name}` | {r.get('prompt_tokens')} ({r.get('cached_tokens')}) | {r.get('completion_tokens')} | "
            f"{r.get('ttft_s')} | **{r.get('decode_tok_s')}** | {r.get('e2e_tok_s')} | {r.get('avg_itl_ms')} |"
        )

    md += [
        "",
        f"## 2. 8-Category Prompt Diversity Suite (Mean: **`{div_mean} tok/s`**)",
        "| Category | Prompt | Comp | TTFT (s) | Decode (tok/s) | ITL p50/p95 (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in div_rows:
        md.append(
            f"| `{row['category']}` | {row.get('prompt_tokens')} | {row.get('completion_tokens')} | "
            f"{row.get('ttft_s')} | **{row.get('decode_tok_s')}** | {row.get('chunk_itl_p50_ms')} / {row.get('chunk_itl_p95_ms')} |"
        )

    md += [
        "",
        "## 3. Context Scaling Curve (Cold Prefill + Warm Decode)",
        "| Context | Cold Prompt | Cold TTFT (s) | Cold Prefill (tok/s) | Warm Greedy TTFT (s) | Warm Greedy Decode (tok/s) | Warm Sampled `T=0.7` TTFT (s) | Warm Sampled Decode (tok/s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in ctx_curve:
        cp = c["cold_prefill"]
        wg = c["warm_greedy_decode"]
        ws = c["warm_sampled_decode"]
        md.append(
            f"| **{c['target_ctx']//1024}K** | {cp.get('prompt_tokens')} | {cp.get('ttft_s')} | **{cp.get('prefill_tok_s')}** | "
            f"{wg.get('ttft_s')} ({wg.get('cached_tokens')} cached) | **{wg.get('decode_tok_s')}** | "
            f"{ws.get('ttft_s')} ({ws.get('cached_tokens')} cached) | **{ws.get('decode_tok_s')}** |"
        )

    md += [
        "",
        "## 4. Multi-Turn Coding Agent (`temp=0.7`) & File-Edit Refactor (`temp=0`)",
        "| Test | Prompt (cached) | Comp | TTFT (s) | Decode (tok/s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for t in agent_turns:
        md.append(
            f"| `agent_turn_{t['turn']}` | {t.get('prompt_tokens')} ({t.get('cached_tokens')}) | {t.get('completion_tokens')} | "
            f"{t.get('ttft_s')} | **{t.get('decode_tok_s')}** |"
        )
    er = results["benchmarks"]["file_edit_refactor"]
    md.append(
        f"| `file_edit_refactor` | {er.get('prompt_tokens')} ({er.get('cached_tokens')}) | {er.get('completion_tokens')} | "
        f"{er.get('ttft_s')} | **{er.get('decode_tok_s')}** |"
    )

    md += [
        "",
        "## 5. Needle-in-a-Haystack Exact Passkey Retrieval",
        "| Target | Prompt Tokens | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Result |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for n in needles_out:
        md.append(
            f"| **{n['label']}** | {n.get('prompt_tokens')} | {n.get('ttft_s')} | {n.get('prefill_tok_s')} | "
            f"{n.get('decode_tok_s')} | **`{'PASS' if n['passed'] else 'FAIL'}`** |"
        )

    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"Done! Shareable results written to: {out_dir}")
    log(f"  - {out_dir / 'summary.md'}")
    log(f"  - {out_dir / 'results.json'}")
    log(f"  - {out_dir / 'bench_run.log'}")


if __name__ == "__main__":
    main()
