#!/usr/bin/env python3
"""Single-Session Shared-Prefix Parallel Fan-Out Demo & Benchmark for Frankengine.

Demonstrates N parallel lanes (e.g., 16, 32, 48, 64) all working simultaneously on the
EXACT SAME long session context (e.g., a 4K-16K token codebase / system specification):
  - Shared prefix is prefilled ONCE by the leader lane.
  - Remaining N-1 lanes fork GPU KV + Conv + BatchDrafter state in <1 ms/lane via clone_slot.
  - All N lanes then decode in parallel at full batched speculative aggregate throughput.

Zero external dependencies (Python stdlib only). Works with both http:// and https:// endpoints.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import ssl
import sys
import time
import urllib.parse
from pathlib import Path


def build_shared_session_context(target_tokens: int) -> str:
    """Build a realistic multi-file C++/CUDA/Python systems architecture session prompt."""
    header = (
        "You are an expert systems & ML engineer working inside a shared collaborative session "
        "on the Frankengine Blackwell inference engine codebase.\n"
        "Below is the current session context (architecture docs, kernel contracts, scheduler code, "
        "and benchmark logs). Every specialist lane in this session shares this exact context and "
        "handles one focused sub-task in parallel.\n\n"
    )
    module_template = (
        "### Module {idx:02d}: frankengine_stage_{idx:02d}\n"
        "```cpp\n"
        "// Stage {idx:02d} contract: tiled BF16/FP8/NVFP4 execution with CUDA stream ordering.\n"
        "struct Stage{idx:02d}Config {{\n"
        "  int hidden_dim = 2304;\n"
        "  int kv_heads_local = 4;\n"
        "  int head_dim = 128;\n"
        "  int window_rows = 1024;\n"
        "  int max_slots = 64;\n"
        "  float spec_threshold = 0.50f;\n"
        "  bool kv_fp8 = true;\n"
        "}};\n"
        "// Invariants for Stage {idx:02d}:\n"
        "// 1. All device pointers are 16-byte aligned for vectorized uint4 loads.\n"
        "// 2. Sliding-window ring indices wrap modulo ring_rows (never hardcoded 1024).\n"
        "// 3. TP=2 collectives run in strict lockstep between Rank 0 and Rank 1.\n"
        "// 4. Sampling uses 23-bit midpoint uniform u in (0, 1) to prevent FP32 rounding to 1.0f.\n"
        "// 5. Prefix clone copies k_ring, v_ring, ring_pos, conv_*, and active len rows of k_full/v_full.\n"
        "// Benchmark telemetry for Stage {idx:02d}: step_ms={ms:.2f}, bw_gbps={bw:.1f}, accept={acc:.2f}\n"
        "```\n\n"
    )
    parts = [header]
    # Roughly ~310 tokens per module block under Inkling-2-Mini BPE
    blocks = max(1, (target_tokens - 120) // 310)
    for i in range(blocks):
        parts.append(module_template.format(idx=i + 1, ms=1.85 + (i % 7) * 0.12, bw=248.0 + (i % 5) * 4.5, acc=2.45 + (i % 4) * 0.15))
    return "".join(parts)


LANE_TASKS = [
    ("Audit ring-buffer wrap invariants", "List 3 concrete edge cases in sliding-window ring wrapping and give a 6-line C++ guard."),
    ("Design shared-prefix clone benchmark", "Propose a 5-point benchmark matrix comparing cold prefill vs GPU D2D clone_slot across 2K-16K contexts."),
    ("Check FP8 KV quantization error", "Explain where FP8 E4M3 KV cache error accumulates across 8 full-attention visits and how to measure KL drift."),
    ("Verify TP=2 lockstep frame ordering", "Write a concise state-machine checklist ensuring Rank 0 and Rank 1 never deadlock on clone + prefill + decode_multi."),
    ("Optimize D2D clone memory layout", "Calculate exact D2D copy bytes for cloning an 8K FP8 KV slot on TP=2 and estimate copy time at 273 GB/s."),
    ("Write Python unit test for prefix fork", "Write a short pytest function verifying two concurrent requests sharing a 4K prefix both report cached_tokens > 0."),
    ("Analyze speculative acceptance at c=48", "Explain how adaptive threshold theta * (active / max_batch) balances verify cost vs acceptance at 48 lanes."),
    ("Audit Gumbel-max numerical stability", "Explain why 24-bit (16777215 + 0.5f)/16777216.f overflows to +inf in FP32 and why 23-bit midpoint is exact."),
    ("Design hierarchical prefix tree fork", "Sketch a 10-line algorithm for forking a batch with 2-level shared prefixes (system prompt + repo context)."),
    ("Compare solo DFlash2 vs batched DSpark", "Summarize when a request should run solo with DFlash2 vs promote to batched DSpark on Dual-Spark."),
    ("Estimate VRAM budget for 48x16K", "Break down per-rank VRAM usage on Dual-Spark for 48 slots x 16K FP8 KV + BF16 head + MXFP8 drafter."),
    ("Draft OpenAI client fan-out helper", "Write a 15-line Python asyncio snippet launching 32 parallel streaming requests against one session."),
    ("Review conv state rollback semantics", "Explain why batched short-conv states are written once after acceptance rather than rolled back."),
    ("Analyze tail latency in 48-lane bursts", "Identify the 2 main causes of tail latency when 48 parallel lanes decode together and how to mitigate them."),
    ("Propose verifier scoring for code swarm", "Design a 3-stage filter (syntax, unit tests, majority vote) for selecting the winning lane out of 32 candidates."),
    ("Inspect NVFP4 vs BF16 head tradeoffs", "Compare memory footprint and GEMM throughput of BF16 sharded head vs NVFP4 head at batch size 48."),
]


def run_lane(
    url: str,
    model: str,
    shared_prefix: str,
    lane_id: int,
    mode: str,
    max_tokens: int,
    temperature: float,
    effort: str,
    timeout: int,
) -> dict:
    title, task_prompt = LANE_TASKS[lane_id % len(LANE_TASKS)]
    if mode == "identical":
        user_msg = shared_prefix + "\n\nTask: Provide a concise 4-bullet engineering summary of the key invariants and bottlenecks in this session."
        lane_label = f"Lane {lane_id:02d} (Same Prompt, T={temperature:.2f})"
    else:
        user_msg = (
            shared_prefix
            + f"\n\n--- LANE {lane_id:02d} SUB-TASK: {title} ---\n"
            + f"{task_prompt}\nKeep your answer crisp, technical, and under 120 words."
        )
        lane_label = f"Lane {lane_id:02d}: {title}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_msg}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    parsed = urllib.parse.urlparse(url.rstrip("/") + "/chat/completions")
    body = json.dumps(payload).encode()
    t_send = time.perf_counter()
    t_first = None
    chunks = []
    reasoning_chunks = []
    prompt_tokens = 0
    cached_tokens = 0
    completion_tokens = 0

    if parsed.scheme == "https":
        conn = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=timeout, context=ssl.create_default_context())
    else:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)
    try:
        conn.request("POST", parsed.path + (f"?{parsed.query}" if parsed.query else ""), body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"lane {lane_id}: HTTP {resp.status}: {resp.read()[:200]!r}")
        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            ev = json.loads(data)
            for ch in ev.get("choices") or []:
                delta = ch.get("delta") or {}
                c = delta.get("content") or ""
                r = delta.get("reasoning_content") or ""
                if (c or r) and t_first is None:
                    t_first = time.perf_counter()
                if c:
                    chunks.append(c)
                if r:
                    reasoning_chunks.append(r)
            usage = ev.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                details = usage.get("prompt_tokens_details") or {}
                cached_tokens = details.get("cached_tokens", 0)
    finally:
        conn.close()

    t_end = time.perf_counter()
    ttft = (t_first - t_send) if t_first else (t_end - t_send)
    decode_s = max(1e-6, t_end - (t_first or t_send))
    return {
        "lane": lane_id,
        "label": lane_label,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": round(ttft, 3),
        "total_s": round(t_end - t_send, 3),
        "decode_tps": round(max(0, completion_tokens - 1) / decode_s, 1),
        "preview": "".join(chunks)[:220].replace("\n", " "),
        "non_ascii_count": sum(1 for ch in ("".join(chunks) + "".join(reasoning_chunks)) if ord(ch) > 127),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Single-Session Shared-Prefix Parallel Fan-Out Demo")
    ap.add_argument("--url", default="http://127.0.0.1:8834/v1")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--prefix-tokens", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--mode", choices=["subtasks", "identical"], default="subtasks")
    ap.add_argument("--temperature", type=float, default=0.65)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    shared_prefix = build_shared_session_context(args.prefix_tokens)
    print(
        f"Launching {args.lanes} parallel lanes on ONE shared ~{args.prefix_tokens}-tok session "
        f"(mode={args.mode}, max_tokens={args.max_tokens}, T={args.temperature})..."
    )
    t0 = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.lanes) as ex:
        futs = [
            ex.submit(
                run_lane,
                args.url,
                args.model,
                shared_prefix,
                i,
                args.mode,
                args.max_tokens,
                args.temperature if args.mode == "subtasks" else (0.0 if i == 0 else args.temperature),
                args.effort,
                args.timeout,
            )
            for i in range(args.lanes)
        ]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
    wall = time.perf_counter() - t0
    results.sort(key=lambda r: r["lane"])

    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_cached = sum(r["cached_tokens"] for r in results)
    total_out = sum(r["completion_tokens"] for r in results)
    ttfts = [r["ttft_s"] for r in results]
    sum_decode_tps = sum(r["decode_tps"] for r in results)
    wall_agg_tps = total_out / max(1e-6, wall)
    cloned_lanes = sum(1 for r in results if r["cached_tokens"] >= 64)

    summary = {
        "lanes": args.lanes,
        "mode": args.mode,
        "prefix_tokens_target": args.prefix_tokens,
        "avg_prompt_tokens": round(total_prompt / max(1, len(results)), 1),
        "total_prompt_tokens": total_prompt,
        "total_cached_tokens": total_cached,
        "cache_hit_pct": round(100.0 * total_cached / max(1, total_prompt), 1),
        "cloned_lanes": cloned_lanes,
        "total_completion_tokens": total_out,
        "wall_s": round(wall, 2),
        "ttft_min_s": round(min(ttfts), 2),
        "ttft_median_s": round(sorted(ttfts)[len(ttfts) // 2], 2),
        "ttft_max_s": round(max(ttfts), 2),
        "wall_aggregate_tps": round(wall_agg_tps, 1),
        "sum_decode_tps": round(sum_decode_tps, 1),
        "lanes_data": results,
    }

    print(
        f"\n=== SINGLE-SESSION {args.lanes}-LANE FAN-OUT SUMMARY ===\n"
        f"  Prompt/lane   : ~{summary['avg_prompt_tokens']:.0f} tok ({total_prompt:,} logical prompt tok)\n"
        f"  Shared-prefix : {cloned_lanes}/{args.lanes} lanes cloned | {total_cached:,} cached tok ({summary['cache_hit_pct']}%)\n"
        f"  TTFT (min/med/max): {summary['ttft_min_s']}s / {summary['ttft_median_s']}s / {summary['ttft_max_s']}s\n"
        f"  Output tokens : {total_out:,} tok in {wall:.2f}s wall\n"
        f"  Throughput    : {wall_agg_tps:.1f} tok/s wall-clock | {sum_decode_tps:.1f} tok/s summed decode\n"
    )
    for r in results[:5]:
        print(f"  [{r['lane']:02d}] cached={r['cached_tokens']} ttft={r['ttft_s']}s out={r['completion_tokens']}t ({r['decode_tps']} t/s) | {r['preview'][:90]}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2))
        print(f"Saved telemetry -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
