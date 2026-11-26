---
name: parallel-fanout
description: Build a multi-module app, refactor, or large codebase feature using Frankengine's Skeleton-of-Thought + N-lane parallel subagent fan-out (16-32 simultaneous lanes sharing one identical Blueprint prefix so GPU `clone_slot` triggers automatically). Use whenever the user asks for parallel build, fan-out, swarm build, or multi-lane generation in omp.
---

# Parallel Fan-Out Builder (Frankengine `clone_slot`)

This workflow exploits Frankengine's batched speculative decoder (`32 × 32K`) and automatic GPU shared-prefix slot cloning (`clone_slot`). Instead of writing a large app or feature linearly in a single stream (`c=1`), split it into `16–32` independent modules and generate them **all simultaneously in ONE single `task` call using the `lane-writer` agent**.

## Critical Rules for Maximum Speed & `clone_slot`

1. **Always use `agent: "lane-writer"` for parallel file-writing bursts.** `lane-writer` has ONLY the `write` tool and `thinkingLevel: minimal`, eliminating wasted `ls`/`cat`/`read` exploration rounds and keeping the system prompt tiny so ALL lanes execute in **1 synchronized wave**.
2. **Never trickle 1–2 subagents at a time.** Emit **16 to 32 tasks in ONE single `task` tool call** so all sibling lanes hit Frankengine together and saturate the batch decoder.
3. **Put the ENTIRE shared contract & Master Blueprint in the `context` parameter of the `task` tool.** Because `omp` prepends `context` verbatim to every subagent in the batch, all `N` lanes share a byte-identical prefix → Frankengine prefills it **once** and clones KV + conv + drafter state to all `N-1` siblings in `< 1 ms/lane`.
4. **Zero cross-lane write conflicts.** Every task owns one disjoint file path (e.g. `modules/01_terrain.js` .. `modules/16_sky_wildlife.js`).

## 3-Stage Procedure

### Stage 1 — Entrypoint & Shared Contract (Coordinator, 1 turn)
1. Write the entrypoint / harness file (e.g., `index.html`) directly using the `write` tool (do NOT inspect an empty directory first if unnecessary).
2. Design a crisp **Master Blueprint** dividing the deliverable into `16–32` disjoint module files with an exact shared contract.

### Stage 2 — Single-Turn N-Lane Parallel Burst (`task` tool with `agent: "lane-writer"`)
In **ONE single `task` tool call**, set `agent: "lane-writer"`, pass the full shared contract + blueprint in `context`, and pass all `N` module tasks in `tasks` (each specifying only its exact file path and visual/functional spec).

### Stage 3 — Verify & Report
1. Once all `N` `lane-writer` lanes return, run one quick verification command (e.g. `for f in modules/*.js; do node --check "$f"; done`).
2. Fix any syntax/integration issue if found, then report total modules built and how to open/run the output.
