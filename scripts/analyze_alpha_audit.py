#!/usr/bin/env python3
"""Summarize DFlash2 audit JSONL and replay width policies on full verification rounds."""

import argparse
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

WIDTHS = range(2, 9)
DEPTHS = range(1, 8)
THRESHOLDS = [struct.unpack("f", struct.pack("f", t / 10))[0] for t in range(3, 10)]
ARRAYS = ("conf", "in_support", "target_mass", "overlap", "picked_p", "picked_q")


def mean(values):
    return fmean(values) if values else None


def survival(rows):
    """At depth k, a full match censored at k-1 has already left the risk set."""
    risk, failures, curve = [], [], []
    probability = 1.0
    for k in DEPTHS:
        reached = sum(r["eligible"] >= k and r["accepted"] >= k - 1 for r in rows)
        failed = sum(r["eligible"] >= k and r["accepted"] == k - 1 for r in rows)
        risk.append(reached)
        failures.append(failed)
        if reached:
            probability *= (reached - failed) / reached
        elif probability != 0:
            probability = None  # tail unidentifiable beyond the last observed risk set
        curve.append(probability)
    return {"rounds": len(rows), "at_risk": risk, "failed": failures, "survive": curve,
            "survive_all_7": curve[-1], "gate_a_upper_bound": 8 * curve[-1] if curve[-1] is not None else None}


def greedy_misses(rows):
    result = []
    for i in range(7):
        hits = Counter(r["in_support"][i] for r in rows if r["accepted"] == i < r["eligible"])
        known = hits[0] + hits[1]
        result.append({"position": i, "misses": known, "selector_pct": 100 * hits[1] / known if known else None,
                       "head_pct": 100 * hits[0] / known if known else None})
    return result


def sampled_overlap(rows):
    result = []
    for i in range(7):
        reached = [r for r in rows if i <= r["accepted"] and i < r["eligible"]]
        exact = [r for r in reached if r["target_mass"][i] >= 0]
        mass = mean([r["target_mass"][i] for r in exact])
        overlap = mean([r["overlap"][i] for r in exact])
        ratios = [r["picked_p"][i] / r["picked_q"][i] for r in exact
                  if r["picked_q"][i] > 0 and r["picked_p"][i] >= 0]
        result.append({"position": i, "reached": len(reached), "exact_rows": len(exact), "target_mass": mass,
                       "overlap": overlap, "missing_candidate_gap": 1 - mass if exact else None,
                       "calibration_gap": mass - overlap if exact else None, "picked_ratio": mean(ratios),
                       "acceptance_probability": mean([min(1, ratio) for ratio in ratios])})
    return result


def threshold_width(conf, threshold):
    for i, value in enumerate(conf):
        if value <= threshold:  # engine choose_width uses a strict greater-than test
            return max(2, i + 1)
    return 8


def prefix_confidence(row):
    if row.get("cumulative_confidence", False):
        return row["conf"]
    product = 1.0
    prefix = []
    for value in row["conf"]:
        product *= value
        prefix.append(product)
    return prefix


def optimal_width(prefix, cost, rate):
    """Backward DP for max E[tokens] - rate * ms. Width two is mandatory."""
    gain, best = 0.0, 8
    for m in range(7, 1, -1):
        # Extending m to m+1 earns the m-th surviving draft.
        extend = prefix[m - 1] - rate * (cost[m + 1] - cost[m]) + gain
        if extend > 0:
            gain = extend
        else:
            gain, best = 0.0, m
    return best


def simulate(rows, cost):
    if not rows:
        return None

    def score(widths):
        tokens = sum(1 + min(r["accepted"], m - 1) for r, m in zip(rows, widths))
        ms = sum(cost[m] for m in widths)
        return {"tok_s": 1000 * tokens / ms, "tokens": tokens, "ms": ms, "widths": dict(Counter(widths))}

    fixed = {m: score([m] * len(rows)) for m in WIDTHS}
    thresholds = {f"{t:.2f}": score([threshold_width(r["conf"], t) for r in rows]) for t in THRESHOLDS}
    prefixes = [prefix_confidence(r) for r in rows]
    rate = 0.0
    # Fractional programming: each Bellman pass maximizes net predicted reward
    # at the common time price. The root gives the best aggregate predicted rate.
    for _ in range(64):
        widths = [optimal_width(prefix, cost, rate) for prefix in prefixes]
        reward = sum(1 + sum(prefix[:m - 1]) for prefix, m in zip(prefixes, widths))
        new_rate = reward / sum(cost[m] for m in widths)
        if math.isclose(new_rate, rate, rel_tol=1e-12):
            break
        rate = new_rate
    return {"rounds": len(rows), "fixed": fixed, "threshold": thresholds, "dp": score(widths)}


def summarize(temperature, prompt, rows, cost):
    full = [r for r in rows if r["width"] == 8]
    probed = [r for r in full if r["probed_m8"]]
    untrimmed = [r for r in rows if not r["trimmed"]]
    return {"temperature": temperature, "prompt": prompt, "rounds": len(rows),
            "emitted": sum(r["emitted"] for r in rows), "untrimmed_rounds": len(untrimmed),
            "alpha_plus_one": 1 + fmean(r["accepted"] for r in rows),
            "alpha_plus_one_untrimmed": mean([1 + r["accepted"] for r in untrimmed]),
            "widths": dict(sorted(Counter(r["width"] for r in rows).items())),
            "m8": survival(full), "probe_m8": survival(probed),
            "kaplan_meier": survival(rows),
            "misses": greedy_misses(rows) if temperature == 0 else None,
            "sampled": sampled_overlap(rows) if temperature > 0 else None,
            "simulator": simulate(full, cost),
            "simulator_probe_m8": simulate(probed, cost)}


def read_groups(path):
    groups = defaultdict(list)
    forced = 0
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get("teacher_forced", False):
                    forced += 1
                    continue
                t, prompt = r["temperature"], r["prompt"]
                m, a, n = r["width"], r["accepted"], r["emitted"]
                if not (isinstance(prompt, str) and math.isfinite(t) and t >= 0 and 1 <= m <= 8
                        and r["eligible"] == m - 1 and 0 <= a < m and 1 <= n <= a + 1
                        and r["trimmed"] == (n < a + 1) and (not r["probed_m8"] or m == 8)
                        and r["sampled"] == (t > 0)):
                    raise ValueError("invalid round metadata")
                for key in ARRAYS:
                    if len(r[key]) != 7 or not all(math.isfinite(x) for x in r[key]):
                        raise ValueError(f"invalid {key}")
                groups[t, None].append(r)
                groups[t, prompt].append(r)
            except (TypeError, ValueError, KeyError, AttributeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    if not groups:
        raise ValueError(f"no free-generation rounds ({forced} teacher-forced rows skipped)")
    return groups, forced


def fmt(x):
    return "—" if x is None else f"{x:.4g}"


def markdown(report):
    print("# DFlash2 prefix audit\n")
    print(f"Teacher-forced rows skipped: {report['teacher_forced_skipped']}.")
    for note in report["method"]:
        print(f"\n{note}")
    for group in report["groups"]:
        prompt = group["prompt"] if group["prompt"] is not None else "All prompts"
        print(f"\n## T={fmt(group['temperature'])} · {prompt}\n")
        print(f"{group['rounds']} rounds, {group['emitted']} emitted tokens. "
              f"α + 1 = {fmt(group['alpha_plus_one'])}, untrimmed = {fmt(group['alpha_plus_one_untrimmed'])} "
              f"({group['untrimmed_rounds']} rounds). Width counts {group['widths']}.")
        print("\n| Cohort | N | " + " | ".join(f"S({k})" for k in DEPTHS) + " | Γ=15 Gate A bound |")
        print("|---|" + "---:|" * 9)
        for key in ("m8", "probe_m8", "kaplan_meier"):
            curve = group[key]
            print(f"| {key} | {curve['rounds']} | " + " | ".join(map(fmt, curve['survive']))
                  + f" | {fmt(curve['gate_a_upper_bound'])} |")
        metrics = group["misses"] or group["sampled"]
        print("\n| " + " | ".join(metrics[0]) + " |")
        print("|" + "---:|" * len(metrics[0]))
        for row in metrics:
            print("| " + " | ".join(map(fmt, row.values())) + " |")
        for sim_key, label in (("simulator_probe_m8", "Unbiased probe_m8"), ("simulator", "Pooled M=8")):
            sim = group.get(sim_key)
            if not sim:
                continue
            bias_note = " (selection-biased on adaptive runs)" if sim_key == "simulator" and sim["rounds"] < group["rounds"] else ""
            print(f"\n**Simulator ({label}, N={sim['rounds']}){bias_note}:**")
            for key in ("fixed", "threshold"):
                print(f"- {key.capitalize()} tok/s: " + ", ".join(
                    f"{setting} → {fmt(value['tok_s'])}" for setting, value in sim[key].items()))
            print(f"- DP tok/s: {fmt(sim['dp']['tok_s'])}, width counts {sim['dp']['widths']}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--t-draft-ms", type=float, default=0.90)
    parser.add_argument("--t-verify-ms", default="4.7,5.2,5.7,6.2,6.8,7.4,8.0", metavar="M2,...,M8")
    parser.add_argument("--format", choices=("both", "markdown", "json"), default="both")
    args = parser.parse_args()
    try:
        verify = [float(v) for v in args.t_verify_ms.split(",")]
        if len(verify) != 7 or not all(math.isfinite(v) and v > 0 for v in verify):
            raise ValueError("--t-verify-ms requires seven positive, finite costs for M=2..8")
        if not math.isfinite(args.t_draft_ms) or args.t_draft_ms < 0:
            raise ValueError("--t-draft-ms must be finite and nonnegative")
        cost = {m: args.t_draft_ms + v for m, v in zip(WIDTHS, verify)}
        groups, forced = read_groups(args.audit)
        order = sorted(groups, key=lambda key: (key[0], key[1] is not None, key[1]))
        report = {"t_draft_ms": args.t_draft_ms, "t_verify_ms": verify, "teacher_forced_skipped": forced,
                  "method": [
                      "S(k) is prefix survival at depth k. M=8 pools natural and probe rounds. "
                      "Probe-only estimates avoid confidence selection. Kaplan–Meier assumes independent censoring. "
                      "The Γ=15 Gate A upper bound is 8 × S(7) additional tokens per round.",
                      "The simulator replays full-round yields on M=8 contexts, before stop or budget trimming. "
                      "Its rates use total replayed tokens divided by total modeled time.",
                      "DP uses selector confidence as a survival proxy. At positive temperature, policies using "
                      "current proposal confidence are offline diagnostics and would bias exact rejection if deployed."],
                  "groups": [summarize(t, p, groups[t, p], cost) for t, p in order]}
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    if args.format != "json":
        markdown(report)
    if args.format != "markdown":
        if args.format == "both":
            print("\n## JSON\n\n```json")
        print(json.dumps(report, allow_nan=False))
        if args.format == "both":
            print("```")


if __name__ == "__main__":
    main()
