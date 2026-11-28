"""Synthetic CLI and policy oracles for analyze_alpha_audit.py."""

import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import analyze_alpha_audit as audit


def row(width, accepted, *, emitted=None, prompt="code", temp=0, probe=False):
    eligible = width - 1
    emitted = accepted + 1 if emitted is None else emitted
    return dict(prompt=prompt, seed=42, temperature=temp, top_k=0, top_p=1,
                round_index=0, position=100, width=width, eligible=eligible, accepted=accepted,
                emitted=emitted, trimmed=emitted < accepted + 1, probed_m8=probe, sampled=temp > 0,
                conf=[0.5] * 7, in_support=[int(i % 2 == 0) if i < eligible and temp == 0 else -1 for i in range(7)],
                target_mass=[0.75 if i < eligible and temp else -1 for i in range(7)],
                overlap=[0.5 if i < eligible and temp else -1 for i in range(7)],
                picked_p=[0.125 if i < eligible and temp else -1 for i in range(7)],
                picked_q=[0.0625 if i < eligible and temp else -1 for i in range(7)])


class TestAudit(unittest.TestCase):
    def test_jsonl(self):
        rows = [row(8, 7, emitted=2, probe=True), row(8, 1, prompt="chat"), row(3, 2),
                row(2, 0), row(1, 0), row(2, 1, temp=0.8), row(8, 0, temp=0.8)]
        rows.append(dict(rows[0], teacher_forced=True))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            cli = [sys.executable, audit.__file__, str(path), "--t-draft-ms", "1", "--t-verify-ms", "1,2,3,4,5,6,7"]
            report = json.loads(subprocess.check_output([*cli, "--format", "json"]))
            md = subprocess.check_output(cli, text=True)
            self.assertIn("| Γ=15 Gate A bound |", md)
            self.assertIn("```json", md)
        self.assertEqual(report["teacher_forced_skipped"], 1)
        groups = {(g["temperature"], g["prompt"]): g for g in report["groups"]}
        g = groups[0, None]
        self.assertEqual((g["rounds"], g["emitted"], g["alpha_plus_one"]), (5, 9, 3))
        self.assertEqual(g["alpha_plus_one_untrimmed"], 1.75)
        self.assertEqual(g["m8"]["survive"], [1] + [0.5] * 6)
        self.assertEqual(g["probe_m8"]["survive_all_7"], 1)
        self.assertEqual(g["m8"]["gate_a_upper_bound"], 4)
        self.assertEqual(g["kaplan_meier"]["at_risk"], [4, 3, 1, 1, 1, 1, 1])
        self.assertEqual(g["kaplan_meier"]["survive"], [0.75] + [0.5] * 6)
        self.assertEqual(g["misses"][0]["selector_pct"], 100)
        self.assertEqual(g["misses"][1]["head_pct"], 100)
        self.assertEqual(g["simulator"]["fixed"]["8"]["tok_s"], 625)
        self.assertEqual(g["simulator"]["threshold"]["0.50"]["widths"], {"2": 2})
        self.assertEqual(g["simulator"]["threshold"]["0.40"]["widths"], {"8": 2})
        sampled = groups[0.8, None]["sampled"]
        self.assertEqual(sampled[0], dict(position=0, reached=2, exact_rows=2, target_mass=0.75, overlap=0.5,
                                          missing_candidate_gap=0.25, calibration_gap=0.25,
                                          picked_ratio=2, acceptance_probability=1))
        self.assertEqual(sampled[1]["reached"], 0)
        self.assertIsNone(sampled[1]["overlap"])

    def test_censored_tail(self):
        g = audit.survival([row(3, 2)])
        self.assertEqual(g["survive"], [1, 1, None, None, None, None, None])
        self.assertIsNone(g["gate_a_upper_bound"])
        self.assertEqual(audit.survival([row(2, 0)])["survive"], [0] * 7)
        self.assertEqual(audit.survival([])["survive"], [None] * 7)

    def test_bellman_against_exhaustive_policy(self):
        # Non-monotone verify costs exercise lookahead across an expensive intermediate width.
        cost = dict(zip(audit.WIDTHS, (2, 6, 3.5, 4, 4.1, 6, 5)))
        rows = [row(8, 2), dict(row(8, 7), conf=[0.97] * 7),
                dict(row(8, 1), conf=[0.8 ** k for k in audit.DEPTHS], cumulative_confidence=True)]
        prefixes = [audit.prefix_confidence(r) for r in rows]

        def rate(widths):
            return sum(1 + sum(p[:m-1]) for p, m in zip(prefixes, widths)) / sum(cost[m] for m in widths)

        best = max(itertools.product(audit.WIDTHS, repeat=len(rows)), key=rate)
        actual = audit.simulate(rows, cost)["dp"]
        self.assertEqual(actual["widths"], dict(Counter(best)))
        self.assertAlmostEqual(actual["tok_s"], 1000 * sum(1 + min(r["accepted"], m - 1)
                                                           for r, m in zip(rows, best)) / sum(cost[m] for m in best))


if __name__ == "__main__":
    unittest.main()
