import * as fs from "fs";
import * as os from "os";
import * as path from "path";

/**
 * Frankengine Parallel Fan-Out & Aggregate Token Counter Extension for omp.
 *
 * Features:
 * - `/fanout [on|off|N|reset]` (alias `/parallel`): toggles automatic parallel fan-out mode
 *   (injects Skeleton-of-Thought + `lane-writer` single-turn batch rules via `before_agent_start`
 *   so any build/refactor prompt automatically triggers N parallel lanes + GPU `clone_slot`).
 * - Real-time aggregate token counter & aggregate tok/s across the parent session AND all
 *   parallel subagent lanes (via `task:subagent:progress`, `task:subagent:lifecycle`,
 *   `message_update`, `message_end`, and `tool_result`), rendered directly in omp's status bar.
 */

interface FanoutConfig {
  enabled: boolean;
  lanes: number;
}

interface LaneState {
  id: string;
  agent: string;
  status: "running" | "completed" | "failed" | "aborted";
  startedAt: number;
  finishedAt?: number;
  /** Exact tokens reported by omp subagent progress/result (input+output or total) */
  reportedTokens: number;
  /** Exact output tokens if available from result usage */
  outputTokens: number;
  /** Exact cached tokens if available from result usage */
  cacheReadTokens: number;
  /** Live streaming character estimate while a turn is still in flight */
  liveEstOutputTokens: number;
  requests: number;
}

const CONFIG_PATH = path.join(os.homedir(), ".omp", "agent", "frankengine-parallel.json");

function loadConfig(): FanoutConfig {
  try {
    const raw = fs.readFileSync(CONFIG_PATH, "utf8");
    const parsed = JSON.parse(raw);
    const lanes = Number(parsed?.lanes);
    return {
      enabled: Boolean(parsed?.enabled),
      lanes: Number.isFinite(lanes) && lanes >= 4 && lanes <= 64 ? Math.round(lanes) : 16,
    };
  } catch {
    return { enabled: false, lanes: 16 };
  }
}

function saveConfig(cfg: FanoutConfig): void {
  try {
    fs.mkdirSync(path.dirname(CONFIG_PATH), { recursive: true });
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(cfg, null, 2) + "\n", "utf8");
  } catch {
    // Non-fatal if home dir is read-only
  }
}

function fmtNum(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 10_000) return `${(n / 1_000).toFixed(1)}K`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(2)}K`;
  return `${Math.round(n)}`;
}

function extractUsageTokens(usage: any): { input: number; output: number; cacheRead: number; total: number } {
  if (!usage || typeof usage !== "object") return { input: 0, output: 0, cacheRead: 0, total: 0 };
  const input = Number(usage.input ?? usage.prompt_tokens ?? usage.inputTokens ?? 0) || 0;
  const output = Number(usage.output ?? usage.completion_tokens ?? usage.outputTokens ?? 0) || 0;
  const cacheRead = Number(usage.cacheRead ?? usage.cached_tokens ?? usage.cache_read_input_tokens ?? 0) || 0;
  const total = Number(usage.totalTokens ?? usage.total_tokens ?? input + output + cacheRead) || input + output + cacheRead;
  return { input, output, cacheRead, total };
}

function buildFanoutDirective(lanes: number): string {
  return [
    "",
    "<frankengine_parallel_fanout_mode>",
    `Frankengine Parallel Fan-Out Mode is ACTIVE (target: ${lanes} parallel lanes).`,
    "When the user asks to build an app, scene, multi-module feature, refactor, or any multi-component deliverable:",
    `1. STAGE 1 (1 turn): Write the entrypoint / shared harness file first via \`write\` (skip unnecessary \`ls\`/\`cat\` inspection if the directory is new/empty). Design a clean shared runtime contract dividing the project into ${lanes} disjoint module files.`,
    `2. STAGE 2 (ONE single turn): Call the \`task\` tool ONCE with \`agent: "lane-writer"\`, passing the FULL shared runtime contract and module blueprint in the shared \`context\` parameter (byte-identical across all lanes so GPU \`clone_slot\` clones KV state in <1 ms/lane), and emit ALL ${lanes} disjoint module tasks in \`tasks\` simultaneously.`,
    "3. STAGE 3 (1 turn): Once all parallel lanes finish, run one quick verification check (e.g. `node --check`, `tsc`, or `pytest`) and summarize the result.",
    "NOTE: For simple single-turn questions, explanations, or tiny 1-file edits, answer directly without spawning subagents.",
    "</frankengine_parallel_fanout_mode>",
  ].join("\n");
}

export default function frankengineParallelExtension(pi: any) {
  if (typeof pi.setLabel === "function") {
    pi.setLabel("Frankengine Parallel");
  }

  pi.registerFlag?.("fanout", {
    description: "Start omp with Frankengine Parallel Fan-Out mode enabled",
    type: "boolean",
    default: false,
  });

  const cfg: FanoutConfig = loadConfig();

  // Session-wide aggregate telemetry (parent + all parallel subagents)
  let parentOutputTokens = 0;
  let parentInputTokens = 0;
  let parentCacheReadTokens = 0;
  let parentLiveEstOutput = 0;
  let parentTurnStartMs = 0;

  const lanes = new Map<string, LaneState>();
  let burstStartMs = 0;
  let burstLastActivityMs = 0;
  let burstBaselineSubagentOut = 0;

  let lastUi: any = null;
  let refreshTimer: ReturnType<typeof setInterval> | null = null;

  function computeAggregate() {
    let subTotal = 0;
    let subOutput = 0;
    let subCache = 0;
    let running = 0;
    let done = 0;
    let failed = 0;

    for (const lane of lanes.values()) {
      if (lane.status === "running") running++;
      else if (lane.status === "completed") done++;
      else failed++;

      // Prefer exact output tokens when known; otherwise live streaming estimate;
      // for reportedTokens (total tokens across turns), track separately.
      const estOut = lane.outputTokens > 0 ? lane.outputTokens : lane.liveEstOutputTokens;
      subOutput += estOut;
      subTotal += Math.max(lane.reportedTokens, estOut);
      subCache += lane.cacheReadTokens;
    }

    const aggOutput = parentOutputTokens + parentLiveEstOutput + subOutput;
    const aggTotal = parentInputTokens + parentOutputTokens + parentCacheReadTokens + parentLiveEstOutput + subTotal;
    const aggCache = parentCacheReadTokens + subCache;

    // Aggregate decode rate during active burst or active parent turn
    let aggRate = 0;
    const now = Date.now();
    if (burstStartMs > 0 && (running > 0 || now - burstLastActivityMs < 4000)) {
      const elapsedS = Math.max(0.25, ((running > 0 ? now : burstLastActivityMs) - burstStartMs) / 1000);
      const burstOut = Math.max(0, subOutput - burstBaselineSubagentOut);
      // If exact subagent output tokens haven't landed yet, fall back to live estimate or reported delta
      const effectiveTok = burstOut > 0 ? burstOut : subTotal;
      aggRate = effectiveTok / elapsedS;
    } else if (parentTurnStartMs > 0 && parentLiveEstOutput > 0) {
      const elapsedS = Math.max(0.25, (now - parentTurnStartMs) / 1000);
      aggRate = parentLiveEstOutput / elapsedS;
    }

    return {
      running,
      done,
      failed,
      totalLanes: lanes.size,
      aggOutput,
      aggTotal,
      aggCache,
      subOutput,
      subTotal,
      aggRate,
    };
  }

  function renderStatus(ui?: any) {
    const targetUi = ui ?? lastUi;
    if (!targetUi || typeof targetUi.setStatus !== "function") return;
    lastUi = targetUi;

    const s = computeAggregate();
    const modeTag = cfg.enabled ? `⚡ FANOUT ${cfg.lanes}L` : `FANOUT OFF`;

    const parts: string[] = [modeTag];
    if (s.running > 0) {
      parts.push(`${s.done}/${s.totalLanes} lanes`);
    } else if (s.totalLanes > 0) {
      parts.push(`${s.done}/${s.totalLanes} done`);
    }

    if (s.aggOutput > 0 || s.aggTotal > 0) {
      // Show aggregate generated tokens (parent + all subagent lanes) and total processed
      if (s.aggOutput > 0) {
        parts.push(`Σout ${fmtNum(s.aggOutput)}`);
      }
      parts.push(`Σtot ${fmtNum(s.aggTotal)}`);
    }

    if (s.aggRate >= 1) {
      parts.push(`${Math.round(s.aggRate)} tok/s`);
    }

    targetUi.setStatus("frankengine-fanout", parts.join(" · "));

    if (typeof targetUi.setWidget === "function") {
      if (s.running > 0) {
        const elapsedS = burstStartMs > 0 ? ((Date.now() - burstStartMs) / 1000).toFixed(1) : "0.0";
        const barWidth = 16;
        const filled = s.totalLanes > 0 ? Math.round((s.done / s.totalLanes) * barWidth) : 0;
        const bar = "█".repeat(filled) + "░".repeat(Math.max(0, barWidth - filled));
        targetUi.setWidget("frankengine-fanout-live", [
          `⚡ Frankengine Parallel [${bar}] ${s.done}/${s.totalLanes} lanes (${s.running} active · ${elapsedS}s) · Σout ${fmtNum(s.aggOutput)} tok · Σtot ${fmtNum(s.aggTotal)} tok${s.aggRate >= 1 ? ` · ~${Math.round(s.aggRate)} tok/s agg` : ""}`,
        ]);
      } else {
        targetUi.setWidget("frankengine-fanout-live", undefined);
      }
    }
  }

  // Helper attached inline to avoid extra allocations
  function resetStats() {
    parentOutputTokens = 0;
    parentInputTokens = 0;
    parentCacheReadTokens = 0;
    parentLiveEstOutput = 0;
    parentTurnStartMs = 0;
    lanes.clear();
    burstStartMs = 0;
    burstLastActivityMs = 0;
    burstBaselineSubagentOut = 0;
  }

  function ensureTicker() {
    if (refreshTimer) return;
    refreshTimer = setInterval(() => {
      const s = computeAggregate();
      if (s.running > 0) {
        renderStatus();
      } else if (refreshTimer) {
        clearInterval(refreshTimer);
        refreshTimer = null;
        renderStatus();
      }
    }, 300);
  }

  // ---- Subscribe to omp's live subagent eventBus channels ----
  if (pi.events && typeof pi.events.on === "function") {
    pi.events.on("task:subagent:lifecycle", (ev: any) => {
      if (!ev || !ev.id) return;
      const now = Date.now();
      let lane = lanes.get(ev.id);
      if (!lane) {
        if (burstStartMs === 0 || [...lanes.values()].every((l) => l.status !== "running")) {
          burstStartMs = now;
          burstBaselineSubagentOut = [...lanes.values()].reduce(
            (acc, l) => acc + (l.outputTokens > 0 ? l.outputTokens : l.liveEstOutputTokens),
            0,
          );
        }
        lane = {
          id: ev.id,
          agent: String(ev.agent ?? "task"),
          status: "running",
          startedAt: now,
          reportedTokens: 0,
          outputTokens: 0,
          cacheReadTokens: 0,
          liveEstOutputTokens: 0,
          requests: 0,
        };
        lanes.set(ev.id, lane);
      }
      if (ev.status === "started") {
        lane.status = "running";
      } else if (ev.status === "completed" || ev.status === "failed" || ev.status === "aborted") {
        lane.status = ev.status;
        lane.finishedAt = now;
      }
      burstLastActivityMs = now;
      ensureTicker();
      renderStatus();
    });

    pi.events.on("task:subagent:progress", (ev: any) => {
      const p = ev?.progress;
      const id = p?.id ?? ev?.id;
      if (!id || !p) return;
      const now = Date.now();
      let lane = lanes.get(id);
      if (!lane) {
        if (burstStartMs === 0 || [...lanes.values()].every((l) => l.status !== "running")) {
          burstStartMs = now;
          burstBaselineSubagentOut = [...lanes.values()].reduce(
            (acc, l) => acc + (l.outputTokens > 0 ? l.outputTokens : l.liveEstOutputTokens),
            0,
          );
        }
        lane = {
          id,
          agent: String(p.agent ?? ev.agent ?? "task"),
          status: "running",
          startedAt: now,
          reportedTokens: 0,
          outputTokens: 0,
          cacheReadTokens: 0,
          liveEstOutputTokens: 0,
          requests: 0,
        };
        lanes.set(id, lane);
      }
      if (p.status === "completed" || p.status === "failed" || p.status === "aborted") {
        lane.status = p.status;
        lane.finishedAt = now;
      } else {
        lane.status = "running";
      }
      if (typeof p.tokens === "number" && p.tokens > lane.reportedTokens) {
        lane.reportedTokens = p.tokens;
      }
      if (typeof p.requests === "number") {
        lane.requests = p.requests;
      }
      // Estimate live streaming output tokens from recentOutput or streaming tool args while turn is in flight
      let liveChars = 0;
      if (Array.isArray(p.recentOutput)) {
        for (const line of p.recentOutput) liveChars += String(line).length;
      }
      if (p.streaming?.args && typeof p.streaming.args === "string") {
        liveChars += p.streaming.args.length;
      }
      const est = Math.round(liveChars / 3.8);
      if (est > lane.liveEstOutputTokens) {
        lane.liveEstOutputTokens = est;
      }
      // Once progress.tokens lands on message_end, if exact outputTokens wasn't set yet,
      // estimate output portion cleanly (for lane-writer, prompt is ~3.8K so output = max(est, tokens - context))
      if (lane.reportedTokens > 0 && lane.outputTokens === 0) {
        const ctxTok = Number(p.contextTokens ?? 0) || 0;
        const inferredOut = ctxTok > 0 && lane.reportedTokens > ctxTok ? lane.reportedTokens - ctxTok : 0;
        if (inferredOut > lane.liveEstOutputTokens) {
          lane.liveEstOutputTokens = inferredOut;
        }
      }
      burstLastActivityMs = now;
      ensureTicker();
      renderStatus();
    });
  }

  // ---- Session & Turn Lifecycle Hooks ----
  pi.on("session_start", (_ev: any, ctx: any) => {
    if (pi.getFlag?.("fanout") === true) {
      cfg.enabled = true;
    }
    renderStatus(ctx?.ui);
  });

  pi.on("session_switch", (_ev: any, ctx: any) => {
    resetStats();
    renderStatus(ctx?.ui);
  });

  pi.on("before_agent_start", (ev: any, ctx: any) => {
    lastUi = ctx?.ui ?? lastUi;
    parentTurnStartMs = Date.now();
    parentLiveEstOutput = 0;
    renderStatus(ctx?.ui);

    if (!cfg.enabled) return;

    // Never inject fan-out recursion into subagent prompts or if already present
    const sysText = Array.isArray(ev?.systemPrompt)
      ? ev.systemPrompt.join("\n\n")
      : typeof ev?.systemPrompt === "string"
        ? ev.systemPrompt
        : "";
    if (sysText.includes("<frankengine_parallel_fanout_mode>")) return;

    const directive = buildFanoutDirective(cfg.lanes);
    const nextSystemPrompt = Array.isArray(ev?.systemPrompt)
      ? [...ev.systemPrompt, directive]
      : sysText + "\n" + directive;

    return { systemPrompt: nextSystemPrompt };
  });

  pi.on("message_update", (ev: any, ctx: any) => {
    const delta = ev?.assistantMessageEvent?.delta;
    if (typeof delta === "string" && delta.length > 0) {
      parentLiveEstOutput += delta.length / 3.8;
      renderStatus(ctx?.ui);
    }
  });

  pi.on("message_end", (ev: any, ctx: any) => {
    const msg = ev?.message;
    if (msg?.role === "assistant") {
      const u = extractUsageTokens(msg.usage ?? ev?.usage);
      parentInputTokens += u.input;
      parentOutputTokens += u.output;
      parentCacheReadTokens += u.cacheRead;
      parentLiveEstOutput = 0;
      parentTurnStartMs = 0;
      renderStatus(ctx?.ui);
    }
  });

  // Capture exact per-subagent usage when `task` tool returns its results
  pi.on("tool_result", (ev: any, ctx: any) => {
    const results = ev?.details?.results;
    if (Array.isArray(results)) {
      const now = Date.now();
      for (const r of results) {
        if (!r || !r.id) continue;
        const u = extractUsageTokens(r.usage);
        let lane = lanes.get(r.id);
        if (!lane) {
          lane = {
            id: r.id,
            agent: String(r.agent ?? "task"),
            status: r.exitCode === 0 ? "completed" : "failed",
            startedAt: now - (Number(r.durationMs) || 0),
            finishedAt: now,
            reportedTokens: Number(r.tokens) || u.total,
            outputTokens: u.output,
            cacheReadTokens: u.cacheRead,
            liveEstOutputTokens: u.output,
            requests: Number(r.requests) || 1,
          };
          lanes.set(r.id, lane);
        } else {
          lane.status = r.exitCode === 0 ? "completed" : "failed";
          lane.finishedAt = now;
          if (Number(r.tokens) > lane.reportedTokens) lane.reportedTokens = Number(r.tokens);
          if (u.total > lane.reportedTokens) lane.reportedTokens = u.total;
          if (u.output > lane.outputTokens) lane.outputTokens = u.output;
          if (u.cacheRead > lane.cacheReadTokens) lane.cacheReadTokens = u.cacheRead;
        }
      }
      burstLastActivityMs = now;
    }
    renderStatus(ctx?.ui);
  });

  // ---- Slash Commands: /fanout (and /parallel alias) ----
  const fanoutCommand = {
    description: "Toggle Frankengine Parallel Fan-Out mode & aggregate token counter (usage: /fanout [on|off|16|32|reset])",
    getArgumentCompletions(prefix: string) {
      const p = prefix.trim().toLowerCase();
      const opts = [
        { label: "on", value: "on", description: "Enable automatic parallel fan-out" },
        { label: "off", value: "off", description: "Disable automatic parallel fan-out" },
        { label: "16", value: "16", description: "Enable with 16 parallel lane-writer lanes" },
        { label: "32", value: "32", description: "Enable with 32 parallel lane-writer lanes" },
        { label: "reset", value: "reset", description: "Reset aggregate token counters" },
      ];
      const filtered = p ? opts.filter((o) => o.label.startsWith(p)) : opts;
      return filtered.length > 0 ? filtered : null;
    },
    async handler(args: string, ctx: any) {
      const tok = (args ?? "").trim().toLowerCase();
      if (tok === "reset") {
        resetStats();
        renderStatus(ctx?.ui);
        ctx?.ui?.notify?.("Frankengine aggregate token counters reset", "info");
        return;
      }
      if (tok === "on") {
        cfg.enabled = true;
      } else if (tok === "off") {
        cfg.enabled = false;
      } else if (/^\d+$/.test(tok)) {
        const n = Math.max(4, Math.min(64, Number(tok)));
        cfg.lanes = n;
        cfg.enabled = true;
      } else {
        cfg.enabled = !cfg.enabled;
      }
      saveConfig(cfg);
      renderStatus(ctx?.ui);
      const s = computeAggregate();
      ctx?.ui?.notify?.(
        cfg.enabled
          ? `⚡ Frankengine Fan-Out ON (${cfg.lanes} lanes, auto lane-writer + clone_slot) · Σout ${fmtNum(s.aggOutput)} tok · Σtot ${fmtNum(s.aggTotal)} tok`
          : `Frankengine Fan-Out OFF · Σout ${fmtNum(s.aggOutput)} tok · Σtot ${fmtNum(s.aggTotal)} tok`,
        "info",
      );
    },
  };

  pi.registerCommand("fanout", fanoutCommand);
  pi.registerCommand("parallel", fanoutCommand);
}
