#!/usr/bin/env python3
"""Frankengine End-to-End Real Coding-Agent Regression Suite.

Zero external Python dependencies (stdlib only). Evaluates a live OpenAI-compatible
Frankengine / ingine-serve endpoint across 5 real-world suites rather than isolated
micro-benchmarks:

  Suite 1 — Interactive Chat (c=1): short & medium prompts across DFLASH2_MAX_CTX,
            T=0 vs T=0.7, thinking OFF/ON, code AST, raw JSON, IT/EN prose.
  Suite 2 — Real Multi-Turn Coding Agent (`omp` in `tmux`):
            (2.1) debug+fix thinking=OFF, (2.2) arch+atomic flush thinking=ON,
            (2.3) 7-file refactor, (2.4) huge prompt (>40K, > BATCH_CTX).
  Suite 3 — Edge Case c=2 Concurrency:
            (A) omp + micro title-gen, (B) omp + 13K sub-call, (C) 2x concurrent omp.
  Suite 4 — Parallel Agent Fan-Out (c=4 omp, c=8 cold/warm waves, c=16 wave):
            BatchDFlash2 GAMMA=7 & BatchDrafter quality, clone_slot, Persistent Prefix
            Anchors, and isolated c=1 straggler tail tok/s.
  Suite 5 — Transitions & Recovery:
            Post-wave immediate c=1 Solo recovery, 4-way mid-stream cancel/disconnect
            (Solo/Batch x prefill/decode), and 16K/65K Needle-in-a-Haystack recall.

Privacy: contains NO hardcoded hostnames, IPs, or private URLs.
Cleanup: all temporary repos and tmux sessions are automatically removed on exit.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import datetime
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


def utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_sh(cmd: str, cwd: Path | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True, check=check)


class ServerForensics:
    """Optional rank0.log & /v1/telemetry reader (local path or user-supplied command)."""

    def __init__(self, base_url: str, log_path: str | None = None, log_cmd: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.log_path = Path(log_path) if log_path else None
        self.log_cmd = log_cmd

    def telemetry(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/telemetry", timeout=10) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            return {"error": str(exc)}

    def read_log_lines(self) -> list[str]:
        if self.log_path and self.log_path.exists():
            return self.log_path.read_text(errors="replace").splitlines()
        if self.log_cmd:
            res = run_sh(self.log_cmd)
            if res.returncode == 0:
                return res.stdout.splitlines()
        return []

    def mark(self) -> int:
        return len(self.read_log_lines())

    def parse_since(self, start_line: int) -> dict[str, Any]:
        lines = self.read_log_lines()[start_line:]
        by_id: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        forks: list[str] = []
        anchors_saved: list[str] = []
        anchor_reuses: list[str] = []
        cancels: list[str] = []
        errors: list[str] = []

        for ln in lines:
            if "shared-prefix fork:" in ln:
                forks.append(ln)
            if "prefix-anchor saved:" in ln:
                anchors_saved.append(ln)
            if "prefix-anchor reuse:" in ln:
                anchor_reuses.append(ln)
            if "cancelled" in ln.lower() or "disconnect" in ln.lower():
                cancels.append(ln)
            if " ERROR " in ln or "failed:" in ln.lower():
                errors.append(ln)
            m1 = re.search(
                r"(chatcmpl-[0-9a-f]+): reuse=(\S+) reused=(\d+) spec_rounds=(\d+) drafter=(\S+) interrupted=(\S+)",
                ln,
            )
            if m1:
                cid = m1.group(1)
                if cid not in by_id:
                    by_id[cid] = {"id": cid}
                    order.append(cid)
                by_id[cid].update(
                    reuse=m1.group(2),
                    reused=int(m1.group(3)),
                    spec_rounds=int(m1.group(4)),
                    drafter=m1.group(5),
                    interrupted=(m1.group(6) == "true"),
                )
            m2 = re.search(
                r"(chatcmpl-[0-9a-f]+): prompt=(\d+) completion=(\d+) finish=(\S+) queued=([\d.]+)s ttft=([\d.]+)s decode=([\d.]+) tok/s",
                ln,
            )
            if m2:
                cid = m2.group(1)
                if cid not in by_id:
                    by_id[cid] = {"id": cid}
                    order.append(cid)
                by_id[cid].update(
                    prompt=int(m2.group(2)),
                    completion=int(m2.group(3)),
                    finish=m2.group(4),
                    queued_s=float(m2.group(5)),
                    ttft_s=float(m2.group(6)),
                    decode_tps=float(m2.group(7)),
                )

        turns: list[dict[str, Any]] = []
        for cid in order:
            d = by_id[cid]
            sr = d.get("spec_rounds", 0)
            comp = d.get("completion", 0)
            dec = d.get("decode_tps", 0.0)
            d["alpha_plus_1"] = round(comp / sr, 2) if sr > 0 else 0.0
            d["ms_per_round"] = round((comp / dec) * 1000.0 / sr, 2) if (sr > 0 and dec > 0) else 0.0
            turns.append(d)

        return {
            "turns": turns,
            "by_id": by_id,
            "forks": forks,
            "anchors_saved": anchors_saved,
            "anchor_reuses": anchor_reuses,
            "cancels": cancels,
            "errors": errors,
        }


def open_conn(base_url: str, timeout: int = 180) -> tuple[http.client.HTTPConnection, str]:
    parsed = urllib.parse.urlparse(base_url.rstrip("/"))
    path_prefix = parsed.path.rstrip("/")
    if parsed.scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            parsed.hostname, parsed.port or 443, timeout=timeout
        )
    else:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)
    return conn, path_prefix


def sse_chat(
    base_url: str,
    model: str,
    prompt: str,
    *,
    sys_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
    enable_thinking: bool = False,
    reasoning_effort: str | None = None,
    abort_after_s: float | None = None,
    abort_after_chunks: int | None = None,
    timeout: int = 180,
) -> dict[str, Any]:
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    body = json.dumps(payload).encode("utf-8")
    conn, prefix = open_conn(base_url, timeout=timeout)
    t0 = time.perf_counter()
    ft: float | None = None
    cid: str | None = None
    content: list[str] = []
    reasoning: list[str] = []
    usage: dict[str, Any] | None = None
    finish: str | None = None
    chunk_times: list[float] = []
    aborted = False

    try:
        conn.request("POST", f"{prefix}/chat/completions", body=body, headers={"Content-Type": "application/json"})
        if abort_after_s is not None:
            if conn.sock is not None:
                conn.sock.settimeout(abort_after_s)
            try:
                resp = conn.getresponse()
                resp.read(16)
            except Exception:
                pass
            aborted = True
            conn.close()
            return {"aborted": True, "elapsed_s": round(time.perf_counter() - t0, 3)}

        resp = conn.getresponse()
        if resp.status != 200:
            err_body = resp.read().decode("utf-8", errors="replace")
            return {"error": f"HTTP {resp.status}: {err_body[:200]}", "wall_s": round(time.perf_counter() - t0, 3)}

        for raw in resp:
            ln = raw.decode("utf-8", errors="replace").strip()
            if not ln.startswith("data: "):
                continue
            data = ln[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if cid is None and obj.get("id"):
                cid = obj["id"]
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                c0 = choices[0]
                if c0.get("finish_reason"):
                    finish = c0["finish_reason"]
                delta = c0.get("delta") or {}
                if delta.get("content"):
                    now = time.perf_counter()
                    if ft is None:
                        ft = now
                    content.append(delta["content"])
                    chunk_times.append(now)
                if delta.get("reasoning_content"):
                    now = time.perf_counter()
                    if ft is None:
                        ft = now
                    reasoning.append(delta["reasoning_content"])
                    chunk_times.append(now)
                if abort_after_chunks and len(chunk_times) >= abort_after_chunks:
                    aborted = True
                    conn.close()
                    break
    finally:
        try:
            conn.close()
        except Exception:
            pass

    t1 = time.perf_counter()
    c_str = "".join(content)
    r_str = "".join(reasoning)
    comp = (usage or {}).get("completion_tokens", 0)
    prompt_t = (usage or {}).get("prompt_tokens", 0)
    cached_t = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    ttft = (ft - t0) if ft is not None else (t1 - t0)
    dec_s = max(t1 - (ft if ft is not None else t0), 1e-6)
    return {
        "id": cid,
        "aborted": aborted,
        "prompt_tokens": prompt_t,
        "cached_tokens": cached_t,
        "completion_tokens": comp,
        "finish": finish,
        "ttft_s": round(ttft, 4),
        "wall_s": round(t1 - t0, 4),
        "t_end": t1,
        "client_dec_tps": round(max(comp - 1, 0) / dec_s, 1) if comp > 1 else 0.0,
        "content": c_str,
        "reasoning": r_str,
        "special_token_leak": bool(SPECIAL_TOKEN_RE.search(c_str + r_str)),
        "chunk_times": chunk_times,
    }


def build_template_repo(repo: Path) -> None:
    (repo / "taskflow").mkdir(parents=True, exist_ok=True)
    (repo / "tests").mkdir(parents=True, exist_ok=True)
    (repo / "tests" / "__init__.py").write_text("")
    (repo / "taskflow" / "__init__.py").write_text(
        "from .rate_limit import TokenBucket\n"
        "from .scheduler import Task, TaskScheduler, CycleError\n"
        "from .storage import LedgerStore\n"
    )
    (repo / "taskflow" / "rate_limit.py").write_text(
        'from __future__ import annotations\n\n'
        'class TokenBucket:\n'
        '    def __init__(self, rate_per_sec: float, capacity: int):\n'
        '        if rate_per_sec <= 0 or capacity <= 0:\n'
        '            raise ValueError("rate_per_sec and capacity must be positive")\n'
        '        self.rate = float(rate_per_sec)\n'
        '        self.capacity = float(capacity)\n'
        '        self.tokens = float(capacity)\n'
        '        self.last_ts: float = 0.0\n\n'
        '    def allow(self, now: float, cost: float = 1.0) -> bool:\n'
        '        if cost < 0:\n'
        '            raise ValueError("cost must be non-negative")\n'
        '        elapsed = now - self.last_ts\n'
        '        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)\n'
        '        self.last_ts = now\n'
        '        if self.tokens >= cost:\n'
        '            self.tokens -= cost\n'
        '            return True\n'
        '        return False\n'
    )
    (repo / "taskflow" / "storage.py").write_text(
        'from __future__ import annotations\n'
        'import json\n'
        'from pathlib import Path\n'
        'from typing import Any\n\n'
        'class LedgerStore:\n'
        '    def __init__(self, path: str | Path):\n'
        '        self.path = Path(path)\n'
        '        self._records: dict[str, dict[str, Any]] = {}\n'
        '        if self.path.exists():\n'
        '            self._records = json.loads(self.path.read_text())\n\n'
        '    def put(self, key: str, value: dict[str, Any]) -> None:\n'
        '        self._records[key] = dict(value)\n\n'
        '    def get(self, key: str) -> dict[str, Any] | None:\n'
        '        val = self._records.get(key)\n'
        '        return dict(val) if val is not None else None\n\n'
        '    def flush(self) -> None:\n'
        '        self.path.parent.mkdir(parents=True, exist_ok=True)\n'
        '        self.path.write_text(json.dumps(self._records, indent=2, sort_keys=True))\n'
    )
    (repo / "taskflow" / "scheduler.py").write_text(
        'from __future__ import annotations\n'
        'from dataclasses import dataclass, field\n'
        'from typing import Iterable\n\n'
        'class CycleError(Exception):\n'
        '    pass\n\n'
        '@dataclass\n'
        'class Task:\n'
        '    task_id: str\n'
        '    priority: int = 0\n'
        '    deps: list[str] = field(default_factory=list)\n\n'
        'class TaskScheduler:\n'
        '    def __init__(self, tasks: Iterable[Task] = ()):\n'
        '        self._tasks: dict[str, Task] = {}\n'
        '        for t in tasks:\n'
        '            self.add(t)\n\n'
        '    def add(self, task: Task) -> None:\n'
        '        if task.task_id in self._tasks:\n'
        '            raise ValueError(f"duplicate task: {task.task_id}")\n'
        '        self._tasks[task.task_id] = task\n\n'
        '    def execution_order(self) -> list[str]:\n'
        '        for t in self._tasks.values():\n'
        '            for dep in t.deps:\n'
        '                if dep not in self._tasks:\n'
        '                    raise KeyError(f"missing dep {dep!r}")\n'
        '        indeg = {tid: len(t.deps) for tid, t in self._tasks.items()}\n'
        '        deps = {tid: [] for tid in self._tasks}\n'
        '        for tid, t in self._tasks.items():\n'
        '            for d in t.deps:\n'
        '                deps[d].append(tid)\n'
        '        ready = [tid for tid, d in indeg.items() if d == 0]\n'
        '        order = []\n'
        '        while ready:\n'
        '            ready.sort(key=lambda tid: (self._tasks[tid].priority, tid))\n'
        '            cur = ready.pop(0)\n'
        '            order.append(cur)\n'
        '            for nxt in deps[cur]:\n'
        '                indeg[nxt] -= 1\n'
        '                if indeg[nxt] == 0:\n'
        '                    ready.append(nxt)\n'
        '        return order\n'
    )
    (repo / "tests" / "test_taskflow.py").write_text(
        'import tempfile, unittest\n'
        'from pathlib import Path\n'
        'from taskflow import TokenBucket, Task, TaskScheduler, CycleError, LedgerStore\n\n'
        'class TestTokenBucket(unittest.TestCase):\n'
        '    def test_clock_skew_does_not_drain(self):\n'
        '        tb = TokenBucket(rate_per_sec=2.0, capacity=5)\n'
        '        self.assertTrue(tb.allow(now=100.0, cost=2.0))\n'
        '        self.assertTrue(tb.allow(now=90.0, cost=2.0))\n'
        '        self.assertAlmostEqual(tb.tokens, 1.0, places=5)\n\n'
        'class TestScheduler(unittest.TestCase):\n'
        '    def test_priority_order(self):\n'
        '        s = TaskScheduler([Task("low", 1), Task("high", 10), Task("mid", 5)])\n'
        '        self.assertEqual(s.execution_order(), ["high", "mid", "low"])\n\n'
        '    def test_cycle_raises(self):\n'
        '        s = TaskScheduler([Task("a", deps=["b"]), Task("b", deps=["a"])])\n'
        '        with self.assertRaises(CycleError):\n'
        '            s.execution_order()\n\n'
        '    def test_dag_deps_and_priority(self):\n'
        '        s = TaskScheduler([Task("root", 1), Task("b", 5, ["root"]), Task("c", 10, ["root"])])\n'
        '        self.assertEqual(s.execution_order(), ["root", "c", "b"])\n\n'
        'class TestLedgerStore(unittest.TestCase):\n'
        '    def test_roundtrip(self):\n'
        '        with tempfile.TemporaryDirectory() as td:\n'
        '            p = Path(td) / "ledger.json"\n'
        '            store = LedgerStore(p)\n'
        '            store.put("job1", {"status": "done"})\n'
        '            store.flush()\n'
        '            self.assertEqual(LedgerStore(p).get("job1"), {"status": "done"})\n'
    )


def run_omp_tmux(
    omp_bin: str,
    omp_model: str,
    session_name: str,
    workdir: Path,
    thinking: str,
    prompt_text: str,
    *,
    extra_args: str = "",
    timeout_s: int = 240,
    active_sessions: list[str] | None = None,
) -> dict[str, Any]:
    run_sh(f"tmux kill-session -t {session_name} 2>/dev/null || true")
    if active_sessions is not None:
        active_sessions.append(session_name)
    pf = workdir / ".eval_prompt.txt"
    df = workdir / ".eval_done"
    ol = workdir / ".eval_out.txt"
    pf.write_text(prompt_text)
    if df.exists():
        df.unlink()
    runner = workdir / ".eval_run.sh"
    runner.write_text(
        f"#!/usr/bin/env bash\nset -o pipefail\n"
        f"cd {shlex_quote(str(workdir))}\n"
        f"PROMPT=\"$(cat {shlex_quote(str(pf))})\"\n"
        f"{shlex_quote(omp_bin)} --model {shlex_quote(omp_model)} --thinking {shlex_quote(thinking)} "
        f"--no-session --no-lsp -p {extra_args} \"$PROMPT\" < /dev/null 2>&1 | tee {shlex_quote(str(ol))}\n"
        f"echo \"${{PIPESTATUS[0]}}\" > {shlex_quote(str(df))}\n"
    )
    os.chmod(runner, 0o755)
    t0 = time.perf_counter()
    run_sh(f"tmux new-session -d -s {session_name} -x 200 -y 60 {shlex_quote(str(runner))}", check=True)
    timed_out = False
    while not df.exists():
        if time.perf_counter() - t0 > timeout_s:
            timed_out = True
            break
        time.sleep(0.4)
    wall = round(time.perf_counter() - t0, 2)
    run_sh(f"tmux kill-session -t {session_name} 2>/dev/null || true")
    ec = int(df.read_text().strip()) if df.exists() else -1
    txt = ol.read_text(errors="replace") if ol.exists() else ""
    return {
        "session": session_name,
        "wall_s": wall,
        "exit_code": ec,
        "timed_out": timed_out,
        "special_token_leak": bool(SPECIAL_TOKEN_RE.search(txt)),
    }


def shlex_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def check_quality(kind: str, content: str, finish: str | None) -> tuple[bool, str]:
    if not content.strip():
        return False, "empty_content"
    if kind == "qa_math":
        return ("16:15" in content), ("ok" if "16:15" in content else "wrong_time")
    if kind == "python":
        m = re.search(r"```python\s*(.*?)```", content, re.S)
        try:
            ast.parse(m.group(1) if m else content)
            return True, "ast_ok"
        except SyntaxError as exc:
            return False, f"ast_err:{exc.msg}"
    if kind == "json":
        try:
            obj = json.loads(content.strip())
            return isinstance(obj, dict), "json_ok"
        except Exception as exc:
            return False, f"json_err:{exc}"
    return True, f"len={len(content)}"


def run_suite1(args: argparse.Namespace, forensics: ServerForensics) -> dict[str, Any]:
    print("\n=== Suite 1: Fast Interactive Chat (c=1) ===", flush=True)
    ctx_block = (
        "class RateLimiter:\n"
        "    def __init__(self, rate: float, burst: int): self.rate, self.burst, self.tokens, self.last = rate, burst, float(burst), 0.0\n"
        "    def allow(self, now: float, cost: float = 1.0) -> bool:\n"
        "        if now > self.last: self.tokens = min(float(self.burst), self.tokens + (now - self.last) * self.rate); self.last = now\n"
        "        if self.tokens >= cost: self.tokens -= cost; return True\n"
        "        return False\n"
    )
    med_5k = "".join(ctx_block.replace("RateLimiter", f"RL_{i}") for i in range(60))
    med_7k = "".join(ctx_block.replace("RateLimiter", f"RL7_{i}") for i in range(110))

    cases = [
        ("short_qa_T0_noThink", 0.0, False, None, 180, "text", "Explain TCP vs UDP in 4 crisp bullet points."),
        ("short_qa_T07_noThink", 0.7, False, None, 180, "text", "Explain TCP vs UDP in 4 concise bullets."),
        ("short_qa_T0_think", 0.0, True, "low", 1024, "qa_math", "A train leaves at 14:15 at 90 km/h and another at 14:45 at 120 km/h. Exact catch-up time?"),
        ("short_qa_T07_think_low", 0.7, True, "low", 2048, "qa_math", "A train departs at 14:15 at 90 km/h; another departs at 14:45 at 120 km/h. Exact overtake time?"),
        ("short_code_T0_noThink", 0.0, False, None, 400, "python", "Write a compact Python 3 class `TTLCache` with `get`/`set` and LRU eviction. Return only one ```python block."),
        ("short_code_T07_noThink", 0.7, False, None, 400, "python", "Write a compact Python 3 class `BoundedTTLCache` with `get`/`put` and LRU eviction. Return only one ```python block."),
        ("json_T0_noThink", 0.0, False, None, 260, "json", "Return ONLY valid raw JSON (no fences) with keys: service, replicas (3), ports ([80,443]), tls (true)."),
        ("json_T07_noThink", 0.7, False, None, 260, "json", "Output ONLY valid raw JSON (no fences) with keys: app, workers (4), ports ([8080,8443]), tls (true)."),
        ("prose_it_T0_noThink", 0.0, False, None, 260, "text", "Scrivi un breve paragrafo tecnico in italiano sul compromesso tra batching e latenza a bassa concorrenza."),
        ("prose_it_T07_noThink", 0.7, False, None, 260, "text", "Spiega in un breve paragrafo italiano perché il batching può penalizzare c=1 su GPU limitate da banda."),
        ("med_5k_T0_noThink", 0.0, False, None, 260, "text", med_5k + "\nWhat bug happens when `now` moves backwards? Give a 3-line fix."),
        ("med_7k_T0_noThink", 0.0, False, None, 260, "text", med_7k + "\nHow do you support dynamic rate updates safely?"),
    ]
    if args.quick:
        cases = cases[:8]

    mark = forensics.mark()
    rows = []
    for name, temp, think, effort, mt, kind, prompt in cases:
        r = sse_chat(
            args.base_url, args.model, prompt,
            temperature=temp, max_tokens=mt, enable_thinking=think, reasoning_effort=effort,
        )
        ok, note = check_quality(kind, r.get("content", ""), r.get("finish"))
        ok = ok and not r.get("special_token_leak", False) and "error" not in r
        row = {
            "name": name, "id": r.get("id"), "prompt": r.get("prompt_tokens", 0),
            "comp": r.get("completion_tokens", 0), "finish": r.get("finish"),
            "ttft_s": r.get("ttft_s", 0), "client_dec_tps": r.get("client_dec_tps", 0),
            "ok": ok, "note": note, "leak": r.get("special_token_leak", False),
        }
        rows.append(row)
        print(f"  {name:24s} p={row['prompt']:5d} out={row['comp']:4d} ttft={row['ttft_s']:.3f}s dec={row['client_dec_tps']:5.1f} ok={ok} ({note})", flush=True)

    time.sleep(0.3)
    log_data = forensics.parse_since(mark)
    for row in rows:
        lg = log_data["by_id"].get(row["id"] or "", {})
        row.update({k: lg[k] for k in ("drafter", "reuse", "interrupted", "decode_tps", "alpha_plus_1", "ms_per_round") if k in lg})

    passed = all(r["ok"] for r in rows)
    return {"passed": passed, "cases": rows, "telemetry": forensics.telemetry()}


def run_suite2(args: argparse.Namespace, forensics: ServerForensics, tmp: Path, tm_sessions: list[str]) -> dict[str, Any]:
    print("\n=== Suite 2: Multi-Turn Real Coding Agent (`omp` in `tmux`) ===", flush=True)
    if args.skip_omp or not shutil.which(args.omp_bin) or not shutil.which("tmux"):
        print("  SKIPPED (omp/tmux unavailable or --skip-omp set)", flush=True)
        return {"skipped": True, "passed": True}

    template = tmp / "template"
    build_template_repo(template)

    def copy_repo(name: str) -> Path:
        dst = tmp / name
        shutil.copytree(template, dst)
        return dst

    tasks: list[tuple[str, str, str, str, str]] = [
        ("t1_debug_off", "off",
         "Run `python3 -m unittest discover -s tests -t . -v`, fix all bugs in `taskflow/`, and re-run tests to verify all pass.",
         "", "unittest"),
        ("t2_arch_med", "medium",
         "Fix all bugs in `taskflow/`, make `LedgerStore.flush()` atomic (temp file + os.replace) with a test, add `TaskScheduler.ready_batch(completed: set[str]) -> list[str]` with a test, and verify all tests pass.",
         "", "unittest"),
    ]
    if not args.quick:
        tasks.append((
            "t4_huge_ctx", "low",
            "Read attached `docs/LEGACY_DUMP.py`, find `LEGACY_FAILOVER_QUORUM_TOKEN` and `LEGACY_RETRY_BACKOFF_MS`, create `taskflow/quorum.py` exporting both and `compute_quorum_threshold(nodes)`, and verify with python3.",
            "@docs/LEGACY_DUMP.py", "quorum",
        ))

    out: dict[str, Any] = {}
    all_ok = True
    for key, thinking, prompt, extra, check_kind in tasks:
        wd = copy_repo(f"s2_{key}")
        if check_kind == "quorum":
            (wd / "docs").mkdir(exist_ok=True)
            parts = ["# Archive\n"]
            for idx in range(700):
                if idx == 455:
                    parts.append('LEGACY_FAILOVER_QUORUM_TOKEN = "SPARK-QUORUM-7942-OMEGA"\nLEGACY_RETRY_BACKOFF_MS = 375\ndef compute_quorum_threshold(n: int) -> int: return (n // 2) + 1\n')
                else:
                    parts.append(f"class Shard_{idx}:\n    ID = {idx}\n    def route(self, k: str) -> int: return (hash(k) + {idx}) % 1024\n")
            (wd / "docs" / "LEGACY_DUMP.py").write_text("".join(parts))

        mark = forensics.mark()
        res = run_omp_tmux(args.omp_bin, args.omp_model, f"fe_reg_{key}", wd, thinking, prompt, extra_args=extra, active_sessions=tm_sessions)
        time.sleep(0.4)
        log_data = forensics.parse_since(mark)
        if check_kind == "unittest":
            ok = run_sh("python3 -m unittest discover -s tests -t . -v", cwd=wd).returncode == 0
        else:
            ok = run_sh("python3 -c 'import taskflow.quorum as q; assert q.LEGACY_FAILOVER_QUORUM_TOKEN==\"SPARK-QUORUM-7942-OMEGA\" and q.LEGACY_RETRY_BACKOFF_MS==375'", cwd=wd).returncode == 0
        any_intr = any(t.get("interrupted", False) for t in log_data["turns"])
        res.update(functional_ok=ok, any_interrupted=any_intr, turns=log_data["turns"])
        out[key] = res
        all_ok = all_ok and ok and (not res["special_token_leak"]) and (not any_intr)
        print(f"  {key:16s} wall={res['wall_s']:5.1f}s turns={len(log_data['turns']):2d} ok={ok} intr={any_intr} leak={res['special_token_leak']}", flush=True)

    return {"passed": all_ok, "tasks": out}


def run_suite3(args: argparse.Namespace, forensics: ServerForensics) -> dict[str, Any]:
    print("\n=== Suite 3: Edge Case c=2 Concurrency Guard ===", flush=True)
    ctx_13k = "# Context\n" + "".join(f"def fn_{i}(x: int) -> int:\n    return (x * {i+3}) % 997\n" for i in range(420))
    mark = forensics.mark()
    main_box: dict[str, Any] = {}
    bg_box: dict[str, Any] = {}

    def run_main() -> None:
        main_box["r"] = sse_chat(
            args.base_url, args.model,
            ctx_13k + "\nWrite a detailed Python test suite testing fn_0..fn_15 with docstrings and edge cases.",
            temperature=0.7, max_tokens=500, enable_thinking=False,
        )

    def run_bg() -> None:
        time.sleep(1.5)
        bg_box["r"] = sse_chat(
            args.base_url, args.model,
            "Generate a 5-word title for a Python refactoring session.",
            temperature=0.7, max_tokens=48, enable_thinking=False,
        )

    t1 = threading.Thread(target=run_main)
    t2 = threading.Thread(target=run_bg)
    t1.start(); t2.start(); t1.join(); t2.join()
    time.sleep(0.4)
    log_data = forensics.parse_since(mark)
    any_intr = any(t.get("interrupted", False) for t in log_data["turns"])
    main_dec = main_box["r"].get("client_dec_tps", 0.0)
    ok = (not any_intr) and ("error" not in main_box["r"]) and ("error" not in bg_box["r"]) and main_dec >= 80.0
    print(f"  c=2 guard: main_dec={main_dec} tok/s bg_ok={'error' not in bg_box['r']} intr={any_intr} ok={ok}", flush=True)
    return {"passed": ok, "any_interrupted": any_intr, "main_dec_tps": main_dec, "turns": log_data["turns"]}


def run_suite4(args: argparse.Namespace, forensics: ServerForensics) -> dict[str, Any]:
    print("\n=== Suite 4: Parallel Fan-Out (c=8 Cold/Warm + Straggler) ===", flush=True)
    sys_3k = "You are a senior Python engineer. Return clean valid code or raw JSON.\n" + "".join(
        f"### Spec {i:02d}\nclass S_{i}:\n    def ok(self, x: int) -> bool: return 0 <= x <= {64+i}\n" for i in range(60)
    )
    tasks = [
        ("lru_ttl", "Write compact Python `LruTtlCache` (max 30 lines). One ```python block.", 480, "python"),
        ("topo_sort", "Write compact Python `topo_sort(edges)` raising ValueError on cycles. One ```python block.", 480, "python"),
        ("bucket", "Write compact Python `MonotonicBucket` handling backwards clock skew. One ```python block.", 480, "python"),
        ("intervals", "Write compact Python `merge_intervals(spans)`. One ```python block.", 480, "python"),
        ("json_plan", "Return ONLY valid raw JSON (no fences) with keys `plan_id`, `steps` (3 objects), `rollback` (bool).", 480, "json"),
        ("window", "Write compact Python `SlidingWindowAvg`. One ```python block.", 480, "python"),
        ("wal", "Write compact Python `MiniWal` with `put`/`delete`/`replay`. One ```python block.", 480, "python"),
        ("straggler", "Write a complete Python retry + circuit breaker module with docstrings and 3 self-tests. One ```python block.", 760, "python"),
    ]
    waves = {}
    all_ok = True
    for tag in (["W1_cold"] if args.quick else ["W1_cold", "W2_warm"]):
        mark = forensics.mark()
        tel0 = forensics.telemetry()
        tw0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            futs = [
                ex.submit(
                    sse_chat, args.base_url, args.model, f"[{tag} L{i}] {p}",
                    sys_prompt=sys_3k, temperature=0.65, max_tokens=mt, enable_thinking=False,
                )
                for i, (_, p, mt, _) in enumerate(tasks)
            ]
            lanes = [f.result() for f in futs]
        wall = round(time.perf_counter() - tw0, 3)
        time.sleep(0.4)
        tel1 = forensics.telemetry()
        log_data = forensics.parse_since(mark)
        passed_lanes = 0
        for (name, _, _, kind), r in zip(tasks, lanes):
            ok, note = check_quality(kind, r.get("content", ""), r.get("finish"))
            if r.get("finish") == "length":
                ok = not r.get("special_token_leak", False)
            r["lane_name"] = name
            r["ok"] = ok and not r.get("special_token_leak", False)
            r["note"] = note
            if r["ok"]:
                passed_lanes += 1
        ends = sorted(l["t_end"] for l in lanes if "t_end" in l)
        last = max(lanes, key=lambda x: x.get("t_end", 0))
        tail_s = last["t_end"] - ends[-2] if len(ends) >= 2 else 0.0
        tail_ch = sum(1 for ts in last.get("chunk_times", []) if ts >= ends[-2]) if len(ends) >= 2 else 0
        est_tok = last.get("completion_tokens", 0) * (tail_ch / max(1, len(last.get("chunk_times", []))))
        straggler_tps = round(est_tok / tail_s, 1) if tail_s > 0.4 else 0.0
        tot_out = sum(l.get("completion_tokens", 0) for l in lanes)
        sum_dec = round(sum(l.get("client_dec_tps", 0) for l in lanes), 1)
        ttft_p50 = round(sorted(l.get("ttft_s", 0) for l in lanes)[len(lanes) // 2], 3)
        cached_cnt = sum(1 for l in lanes if l.get("cached_tokens", 0) > 0)
        for l in lanes:
            l.pop("chunk_times", None)
            l.pop("content", None)
            l.pop("reasoning", None)
        waves[tag] = {
            "wall_s": wall, "tot_out": tot_out, "wall_agg_tps": round(tot_out / wall, 1),
            "sum_dec_tps": sum_dec, "ttft_p50_s": ttft_p50, "passed_lanes": passed_lanes,
            "cached_lanes": cached_cnt, "straggler_c1_tps": straggler_tps,
            "avg_accept_len": round(tel1.get("avg_accept_len", 0.0), 2),
            "clone_saved_delta": tel1.get("clone_slot_saved_tokens", 0) - tel0.get("clone_slot_saved_tokens", 0),
            "anchor_reuses": len(log_data["anchor_reuses"]),
        }
        all_ok = all_ok and (passed_lanes == len(tasks))
        print(f"  {tag:8s} wall={wall:5.2f}s agg={waves[tag]['wall_agg_tps']:5.1f} sum_dec={sum_dec:5.1f} ttft_p50={ttft_p50:.3f}s cached={cached_cnt}/8 pass={passed_lanes}/8 straggler={straggler_tps}t/s", flush=True)
    return {"passed": all_ok, "waves": waves}


def run_suite5(args: argparse.Namespace, forensics: ServerForensics) -> dict[str, Any]:
    print("\n=== Suite 5: Recovery, Cancel Mid-Stream & Needle Recall ===", flush=True)
    # 5A: Post-wave Solo recovery (immediately after Suite 4!)
    mark = forensics.mark()
    rec = sse_chat(args.base_url, args.model, "Write a Python function `quicksort(arr)`.", temperature=0.0, max_tokens=200)
    time.sleep(0.3)
    rec_lg = forensics.parse_since(mark)["by_id"].get(rec.get("id") or "", {})
    rec_ok = rec.get("client_dec_tps", 0) >= 100.0 and not rec_lg.get("interrupted", False)
    print(f"  5A post-wave Solo recovery: dec={rec.get('client_dec_tps')} tok/s drafter={rec_lg.get('drafter')} ok={rec_ok}", flush=True)

    # 5B: Cancel mid-prefill & mid-decode (Solo + Batch) with benign follow-ups
    big = "# Big\n" + "".join(f"class M_{i}: x={i}\n" for i in range(1000))
    sse_chat(args.base_url, args.model, big + "\nSummarize.", temperature=0.7, max_tokens=100, abort_after_s=1.0)
    time.sleep(1.5)
    f1 = sse_chat(args.base_url, args.model, "What is 2 + 2? Reply with just the digit.", temperature=0.0, max_tokens=16)
    sse_chat(args.base_url, args.model, "Write a 500-word essay about databases.", temperature=0.7, max_tokens=500, abort_after_chunks=25)
    time.sleep(1.0)
    f2 = sse_chat(args.base_url, args.model, "What is 3 + 4? Reply with just the digit.", temperature=0.0, max_tokens=16)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        f_ab = [ex.submit(sse_chat, args.base_url, args.model, f"Essay {i} about distributed systems.", temperature=0.7, max_tokens=400, abort_after_chunks=20) for i in range(2)]
        f_ok = [ex.submit(sse_chat, args.base_url, args.model, f"Write Python `add_{i}(a, b)`.", temperature=0.7, max_tokens=120) for i in range(2)]
        for f in f_ab: f.result()
        survivors = [f.result() for f in f_ok]
    time.sleep(1.0)
    tel_cancel = forensics.telemetry()
    cancel_ok = ("4" in f1.get("content", "")) and ("7" in f2.get("content", "")) and all(bool(s.get("content", "").strip()) for s in survivors) and tel_cancel.get("active", 0) == 0 and tel_cancel.get("decoding", 0) == 0
    print(f"  5B cancels (Solo+Batch): f1={f1.get('content','').strip()!r} f2={f2.get('content','').strip()!r} active={tel_cancel.get('active')} dec={tel_cancel.get('decoding')} ok={cancel_ok}", flush=True)

    # 5C: Needles (16K, and 65K if not --quick)
    needle_specs = [("needle_16k", 600, 340, "ALPHA-16K-48291")]
    if not args.quick:
        needle_specs.append(("needle_65k", 2400, 1500, "GAMMA-65K-71604"))
    needles = []
    for name, blocks, pos, key in needle_specs:
        parts = ["Below is an archive of telemetry logs containing OVERRIDE_PASSKEY.\n"]
        for i in range(blocks):
            if i == pos:
                parts.append(f"\n[CRITICAL RECORD] OVERRIDE_PASSKEY is: {key}\n")
            else:
                parts.append(f"LOG {i:05d}: shard={i%32} latency_us={120+(i*17)%400} status=OK\n")
        parts.append("\nWhat is the exact value of OVERRIDE_PASSKEY? Reply with ONLY the passkey string.")
        r = sse_chat(args.base_url, args.model, "".join(parts), temperature=0.0, max_tokens=48)
        found = key in r.get("content", "")
        needles.append({"name": name, "prompt": r.get("prompt_tokens", 0), "ttft_s": r.get("ttft_s", 0), "dec_tps": r.get("client_dec_tps", 0), "found": found})
        print(f"  5C {name:12s} p={r.get('prompt_tokens',0):5d} ttft={r.get('ttft_s',0):5.2f}s dec={r.get('client_dec_tps',0):5.1f}t/s found={found}", flush=True)

    passed = rec_ok and cancel_ok and all(n["found"] for n in needles)
    return {"passed": passed, "recovery_ok": rec_ok, "cancel_ok": cancel_ok, "needles": needles, "telemetry": forensics.telemetry()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Frankengine real coding-agent end-to-end regression suite")
    ap.add_argument("--base-url", default="http://127.0.0.1:8834/v1", help="OpenAI-compatible API base URL")
    ap.add_argument("--model", default="inkling-2-mini", help="Served model ID")
    ap.add_argument("--suites", default="1,2,3,4,5", help="Comma-separated suite numbers to run")
    ap.add_argument("--quick", action="store_true", help="Shorter smoke variant")
    ap.add_argument("--skip-omp", action="store_true", help="Skip Suite 2 tmux/omp tests")
    ap.add_argument("--omp-bin", default="omp", help="Path to omp binary")
    ap.add_argument("--omp-model", default="inkling-2-mini", help="Model identifier passed to omp --model")
    ap.add_argument("--server-log", default=None, help="Local path to rank0.log for forensic parsing")
    ap.add_argument("--server-log-cmd", default=None, help="Optional shell command to read rank0.log")
    ap.add_argument("--out-dir", default=None, help="Directory to write results.json and summary.md")
    ap.add_argument("--keep-tmp", action="store_true", help="Preserve temporary sandbox directory")
    args = ap.parse_args()

    selected = {s.strip() for s in args.suites.split(",") if s.strip()}
    forensics = ServerForensics(args.base_url, args.server_log, args.server_log_cmd)
    tmp_dir = Path(tempfile.mkdtemp(prefix="frankengine_agent_eval_"))
    tm_sessions: list[str] = []
    report: dict[str, Any] = {
        "timestamp_utc": utc_stamp(),
        "base_url": args.base_url,
        "model": args.model,
        "quick": args.quick,
        "suites": {},
    }

    try:
        if "1" in selected:
            report["suites"]["suite1"] = run_suite1(args, forensics)
        if "2" in selected:
            report["suites"]["suite2"] = run_suite2(args, forensics, tmp_dir, tm_sessions)
        if "3" in selected:
            report["suites"]["suite3"] = run_suite3(args, forensics)
        if "4" in selected:
            report["suites"]["suite4"] = run_suite4(args, forensics)
        if "5" in selected:
            report["suites"]["suite5"] = run_suite5(args, forensics)
    finally:
        for sess in tm_sessions:
            run_sh(f"tmux kill-session -t {sess} 2>/dev/null || true")
        if not args.keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    all_pass = all(v.get("passed", False) for v in report["suites"].values())
    report["overall_passed"] = all_pass

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"real_agent_eval_{report['timestamp_utc']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    md_lines = [
        f"# Frankengine Real Coding-Agent Eval ({report['timestamp_utc']})",
        f"- **Model:** `{args.model}` | **Overall:** `{'PASS' if all_pass else 'FAIL'}`",
        "",
        "| Suite | Status |",
        "|---|---|",
    ]
    for k, v in report["suites"].items():
        st = "SKIPPED" if v.get("skipped") else ("PASS" if v.get("passed") else "FAIL")
        md_lines.append(f"| `{k}` | **{st}** |")
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n")

    print(f"\n=== OVERALL: {'PASS' if all_pass else 'FAIL'} (saved to {out_dir}) ===")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
