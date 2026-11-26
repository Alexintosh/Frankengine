---
name: inkling-swarm
description: >-
  Launch a parallel Inkling-2-Mini Agent Swarm (8-32 concurrent agents) against a Frankengine
  OpenAI-compatible server to solve hard coding, debugging, math, or architecture problems faster
  and with higher solve-rate than a single agent. Use when the user asks to run a swarm on a
  problem ("fai partire la swarm su questo problema", "run inkling swarm", "solve with swarm").
---

# Inkling Agent Swarm (`inkling-swarm`)

Run **8 to 32 parallel `Inkling-2-Mini` agents** concurrently on a Frankengine batched-speculative server (`32 × 32K`) to explore diverse solution strategies simultaneously, verify candidates against real tests (in `code` mode) or synthesize a consensus answer via critic review (in `reason` mode).

## Modes & When to Use Them

1. **`code` mode** — Use for coding, bug-fixing, algorithms, refactoring, or any task with a runnable check (`pytest`, `cargo test`, `python3 test.py`, compiler, etc.):
   - Launches `N` parallel agents (`--agents 8` or `16`) with diverse engineering personas and temperatures (`0.65–0.85`).
   - Runs `--test-cmd` (or syntax check) on candidates; if Round 1 fails, Round 2 feeds the exact failure traceback to parallel repair agents.
   - Selects the consensus verified winner and optionally writes it to `--write-out`.

2. **`reason` mode** — Use for architecture design, hard math, root-cause debugging without a test harness, or deep technical analysis:
   - Stage 1 launches `N` diverse reasoning agents in parallel (`FirstPrinciples`, `Skeptic`, `Systems`, `Formal`, `Creative`, etc.).
   - Stage 2 synthesizes all independent solutions into one verified consensus answer, catching blind spots from individual runs.

## Commands

Set `INKLING_API_URL` (defaults to `http://127.0.0.1:8834/v1`):

```bash
# Coding problem with automated verification + Round-2 repair:
python3 scripts/inkling_swarm.py code \
  --prompt "Solve <problem>..." \
  --agents 8 --rounds 2 --effort low \
  --write-out solution.py \
  --test-cmd "python3 -m pytest test_solution.py"

# General reasoning / math / architecture swarm:
python3 scripts/inkling_swarm.py reason \
  --prompt "Analyze <problem>..." \
  --agents 8 --effort medium
```

## Practical Guidelines

- **Concurrency sweet spot:** `--agents 8` (fast, ~30 tok/s per agent on Dual-Spark) or `--agents 16` (high diversity on very hard problems, ~26 tok/s per agent). Keep `--concurrency <= 32` for a `32 × 32K` server.
- **Effort level:** Use `--effort low` for fast coding/bug-fix loops, `--effort medium` for deep math or architectural reasoning, and `--effort off` for simple scaffolding.
- **Report to user:** Share the winning solution, wall-clock time, total tokens, aggregate `tok/s`, and how many agents passed verification.
