#!/usr/bin/env python3
"""inkling_swarm.py — Zero-dependency parallel Agent Swarm orchestrator for Frankengine.

Drives N concurrent Inkling-2-Mini agents (up to 32+ parallel streams on a 32x32K
Frankengine server) with diverse personas/temperatures, automated verification,
multi-round traceback repair, or parallel critic/synthesizer consensus.

Modes:
  code   - Parallel diverse coding agents + syntax/test verification + Round-2 repair
  reason - Parallel diverse reasoning agents + parallel critic/synthesizer consensus
  batch  - Run a JSONL list of tasks concurrently using up to --concurrency streams

Examples:
  # Verifiable coding swarm (8 parallel agents, automated pytest check, 2 rounds):
  python3 scripts/inkling_swarm.py code \
    --prompt "Implement a thread-safe LRU cache with TTL in Python" \
    --agents 8 --rounds 2 --test-cmd "python3 -m pytest test_lru.py" --write-out lru.py

  # General reasoning / architecture / math swarm (8 solvers + synthesizer):
  python3 scripts/inkling_swarm.py reason \
    --prompt "Design a lock-free ring buffer for dual-rail RDMA collectives..." \
    --agents 8

  # Custom endpoint (or set INKLING_API_URL):
  python3 scripts/inkling_swarm.py reason --url https://your-host/v1 --prompt "..."
"""
import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

CODE_PERSONAS = [
    ("Direct", 0.65, "Write a clean, idiomatic, bug-free implementation."),
    ("EdgeCase", 0.75, "Focus carefully on tricky edge cases, empty inputs, boundaries, and error handling."),
    ("Algorithm", 0.70, "Analyze algorithmic invariants and complexity first, then write the optimal solution."),
    ("Defensive", 0.80, "Check every single specification detail word-for-word (types, ordering, corner cases)."),
    ("Trace", 0.75, "Mentally trace concrete examples step by step before emitting the final code."),
    ("Minimal", 0.85, "Prefer a simple, direct, unmistakable implementation without unnecessary abstraction."),
    ("Counterexample", 0.80, "Think of counterexamples that break naive solutions and ensure your code passes all of them."),
    ("StrictSpec", 0.70, "Follow every requirement with mathematical precision and clean structure."),
]

REASON_PERSONAS = [
    ("FirstPrinciples", 0.60, "Reason from first principles with rigorous step-by-step derivations."),
    ("Skeptic", 0.75, "Actively look for hidden traps, unstated assumptions, and counterexamples."),
    ("Pragmatist", 0.70, "Prioritize concrete, practical, production-proven engineering trade-offs."),
    ("Systems", 0.70, "Analyze from a systems/performance/failure-mode perspective."),
    ("Formal", 0.65, "Structure the argument cleanly with explicit invariants and edge-case checks."),
    ("Creative", 0.85, "Explore non-obvious alternative approaches and compare them critically."),
    ("Concise", 0.65, "Cut through noise and focus on the highest-leverage core insight."),
    ("DeepDive", 0.75, "Go deep on subtle details that standard answers gloss over."),
]


def call_chat(
    base_url: str,
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    effort: str,
    api_key: str = "",
    timeout: int = 240,
) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if effort and effort != "default":
        if effort == "off":
            payload["enable_thinking"] = False
        else:
            payload["enable_thinking"] = True
            payload["reasoning_effort"] = effort
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            obj = json.loads(resp.read().decode("utf-8", "ignore"))
        msg = obj["choices"][0]["message"]
        usage = obj.get("usage") or {}
        return {
            "ok": True,
            "content": (msg.get("content") or "").strip(),
            "reasoning": (msg.get("reasoning_content") or "").strip(),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "dt": time.perf_counter() - t0,
        }
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", "ignore")[:240]
        return {"ok": False, "error": f"HTTP {e.code}: {err}", "completion_tokens": 0, "dt": time.perf_counter() - t0}
    except Exception as e:
        return {"ok": False, "error": str(e), "completion_tokens": 0, "dt": time.perf_counter() - t0}


def extract_code_block(text: str) -> str:
    blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)```", text, flags=re.S)
    return blocks[-1].strip() if blocks else text.strip()


def run_verify_cmd(cmd: str, code: str, write_path: Path | None, timeout: int = 30) -> tuple[bool, str]:
    if write_path is not None:
        write_path.write_text(code + "\n")
    try:
        p = subprocess.run(cmd, shell=True, timeout=timeout, capture_output=True, text=True)
        out = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
        return p.returncode == 0, out[-1200:]
    except subprocess.TimeoutExpired:
        return False, f"Verification timed out after {timeout}s"
    except Exception as e:
        return False, str(e)


def run_code_swarm(args, prompt: str) -> dict:
    t0 = time.perf_counter()
    n = args.agents
    tot_tokens = 0
    best_code = ""
    best_meta = {}
    all_candidates = []
    last_error = ""
    write_path = Path(args.write_out) if args.write_out else None

    for rnd in range(1, args.rounds + 1):
        round_n = n if rnd == 1 else max(2, n // 2)
        sys.stderr.write(f"[swarm:code] Round {rnd}/{args.rounds} — launching {round_n} parallel agents...\n")
        sys.stderr.flush()

        def worker(idx: int):
            pname, temp, hint = CODE_PERSONAS[idx % len(CODE_PERSONAS)]
            if rnd == 1:
                user_msg = (
                    f"{prompt}\n\nGuidance ({pname}): {hint}\n"
                    "Return the complete solution inside a single fenced code block."
                )
            else:
                user_msg = (
                    f"{prompt}\n\nA previous candidate failed verification with this error:\n"
                    f"```\n{last_error}\n```\n"
                    f"Guidance ({pname}): Diagnose the exact root cause and return a fixed complete solution "
                    "inside a single fenced code block."
                )
            res = call_chat(
                args.url, args.model, [{"role": "user", "content": user_msg}],
                temperature=temp if args.temperature < 0 else args.temperature,
                max_tokens=args.max_tokens, effort=args.effort, api_key=args.api_key,
            )
            res["agent"] = f"r{rnd}-{pname}-{idx}"
            return res

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(round_n, args.concurrency)) as ex:
            results = list(ex.map(worker, range(round_n)))

        tot_tokens += sum(r["completion_tokens"] for r in results)
        ok_results = [r for r in results if r["ok"] and r["content"]]
        sys.stderr.write(
            f"[swarm:code] Round {rnd} generated {len(ok_results)}/{round_n} candidates "
            f"({sum(r['completion_tokens'] for r in results)} tok, {time.perf_counter()-t0:.1f}s elapsed)\n"
        )
        sys.stderr.flush()

        # Verify sequentially if --test-cmd writes to a shared file, else syntax/consensus check
        passing = []
        for r in ok_results:
            code = extract_code_block(r["content"])
            norm = re.sub(r"\s+", " ", code).strip()
            entry = {"agent": r["agent"], "code": code, "norm": norm, "verified": False, "log": ""}
            if args.test_cmd:
                passed, log = run_verify_cmd(args.test_cmd, code, write_path, timeout=args.test_timeout)
                entry["verified"] = passed
                entry["log"] = log
                if passed:
                    passing.append(entry)
                elif log:
                    last_error = log
            else:
                # Default Python syntax check if code looks like Python
                try:
                    compile(code, "<candidate>", "exec")
                    entry["verified"] = True
                    passing.append(entry)
                except SyntaxError as se:
                    entry["log"] = str(se)
                    last_error = str(se)
            all_candidates.append(entry)

        if passing:
            counts = Counter(c["norm"] for c in passing)
            passing.sort(key=lambda c: counts[c["norm"]], reverse=True)
            winner = passing[0]
            best_code = winner["code"]
            best_meta = {
                "winner": winner["agent"],
                "round": rnd,
                "verified_count": len(passing),
                "total_candidates": len(all_candidates),
                "consensus_votes": counts[winner["norm"]],
            }
            if write_path is not None:
                write_path.write_text(best_code + "\n")
            break

    wall = time.perf_counter() - t0
    if not best_code and all_candidates:
        counts = Counter(c["norm"] for c in all_candidates)
        all_candidates.sort(key=lambda c: counts[c["norm"]], reverse=True)
        best_code = all_candidates[0]["code"]
        best_meta = {
            "winner": all_candidates[0]["agent"],
            "round": args.rounds,
            "verified_count": 0,
            "total_candidates": len(all_candidates),
            "consensus_votes": counts[all_candidates[0]["norm"]],
        }
        if write_path is not None:
            write_path.write_text(best_code + "\n")

    return {
        "mode": "code",
        "wall_s": round(wall, 2),
        "total_tokens": tot_tokens,
        "aggregate_tps": round(tot_tokens / max(wall, 1e-6), 1),
        **best_meta,
        "output": best_code,
    }


def run_reason_swarm(args, prompt: str) -> dict:
    t0 = time.perf_counter()
    n = args.agents
    sys.stderr.write(f"[swarm:reason] Stage 1 — launching {n} diverse parallel agents...\n")
    sys.stderr.flush()

    def worker(idx: int):
        pname, temp, hint = REASON_PERSONAS[idx % len(REASON_PERSONAS)]
        user_msg = f"{prompt}\n\nApproach ({pname}): {hint}"
        res = call_chat(
            args.url, args.model, [{"role": "user", "content": user_msg}],
            temperature=temp if args.temperature < 0 else args.temperature,
            max_tokens=args.max_tokens, effort=args.effort, api_key=args.api_key,
        )
        res["agent"] = f"{pname}-{idx}"
        return res

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(n, args.concurrency)) as ex:
        results = list(ex.map(worker, range(n)))

    ok = [r for r in results if r["ok"] and r["content"]]
    s1_toks = sum(r["completion_tokens"] for r in results)
    sys.stderr.write(
        f"[swarm:reason] Stage 1 done: {len(ok)}/{n} answers ({s1_toks} tok in {time.perf_counter()-t0:.1f}s). "
        "Stage 2 — synthesizing consensus...\n"
    )
    sys.stderr.flush()

    dossier = []
    for i, r in enumerate(ok):
        snippet = r["content"][:3000]
        dossier.append(f"### Agent {i+1} ({r['agent']})\n{snippet}")
    synth_prompt = (
        "You are the Lead Synthesizer of a multi-agent swarm. Below is the original user problem followed by "
        f"{len(ok)} independent solutions produced in parallel by diverse agents.\n\n"
        "Compare them critically: identify agreements, catch subtle errors or blind spots in individual answers, "
        "and produce ONE definitive, clear, accurate final answer.\n\n"
        f"## Original Problem\n{prompt}\n\n## Agent Solutions\n" + "\n\n".join(dossier)
    )
    synth = call_chat(
        args.url, args.model, [{"role": "user", "content": synth_prompt}],
        temperature=0.0, max_tokens=max(args.max_tokens, 1600), effort=args.effort, api_key=args.api_key,
    )
    tot_tokens = s1_toks + synth["completion_tokens"]
    wall = time.perf_counter() - t0
    return {
        "mode": "reason",
        "agents_ok": len(ok),
        "agents_total": n,
        "wall_s": round(wall, 2),
        "total_tokens": tot_tokens,
        "aggregate_tps": round(tot_tokens / max(wall, 1e-6), 1),
        "output": synth["content"] if synth["ok"] else (ok[0]["content"] if ok else ""),
    }


def main():
    ap = argparse.ArgumentParser(description="Parallel Inkling-2-Mini Agent Swarm CLI for Frankengine.")
    ap.add_argument("mode", choices=["code", "reason"], help="Swarm orchestration mode")
    ap.add_argument("--prompt", default="", help="Problem prompt text")
    ap.add_argument("--file", default="", help="Read prompt from file")
    ap.add_argument("--url", default=os.environ.get("INKLING_API_URL", "http://127.0.0.1:8834/v1"),
                    help="OpenAI-compatible /v1 endpoint URL")
    ap.add_argument("--api-key", default=os.environ.get("INKLING_API_KEY", ""), help="Optional bearer token")
    ap.add_argument("--model", default=os.environ.get("INKLING_MODEL", "inkling-2-mini"))
    ap.add_argument("--agents", type=int, default=8, help="Parallel agents per round (default: 8, up to 32)")
    ap.add_argument("--concurrency", type=int, default=32, help="Max concurrent HTTP requests (default: 32)")
    ap.add_argument("--rounds", type=int, default=2, help="Max rounds in code mode (default: 2)")
    ap.add_argument("--max-tokens", type=int, default=1400, help="Max completion tokens per agent")
    ap.add_argument("--effort", default="low", choices=["off", "minimal", "low", "medium", "high", "default"],
                    help="Reasoning effort (default: low)")
    ap.add_argument("--temperature", type=float, default=-1.0, help="Override persona temperatures (<0 uses diverse defaults)")
    ap.add_argument("--test-cmd", default="", help="[code] Shell command to verify each candidate (exit 0 = pass)")
    ap.add_argument("--test-timeout", type=int, default=30, help="[code] Timeout per verification command (seconds)")
    ap.add_argument("--write-out", default="", help="[code] Write winning code to this file path")
    ap.add_argument("--json", action="store_true", help="Print full JSON result instead of plain output")
    args = ap.parse_args()

    prompt = Path(args.file).read_text().strip() if args.file else args.prompt.strip()
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        ap.error("Provide --prompt, --file, or stdin")

    res = run_code_swarm(args, prompt) if args.mode == "code" else run_reason_swarm(args, prompt)
    sys.stderr.write(
        f"[swarm] DONE in {res['wall_s']}s | {res['total_tokens']} tok | {res['aggregate_tps']} tok/s aggregate\n"
    )
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(res["output"])


if __name__ == "__main__":
    main()
