#!/usr/bin/env python3
"""Summarize one paired capture measurement, preserving the slow-rank tails."""

import argparse
import csv
import math
from pathlib import Path

BARRIERS = 96
SAMPLES = 200


def read_rank(path, expected_rank):
    out = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            assert int(row["rank"]) == expected_rank, path
            group = int(row["rails"]), row["mode"], int(row["m"])
            replay = int(row["replay"])
            samples = out.setdefault(group, {})
            assert replay not in samples, (path, group, replay)
            samples[replay] = tuple(float(row[key]) for key in ("wall_us", "sum_us", "head_us"))
    for group, samples in out.items():
        assert samples.keys() == set(range(SAMPLES)), (path, group)
    return out


def percentiles(values, divisor=1):
    ordered = sorted(values)
    return tuple(ordered[math.ceil(q * len(ordered)) - 1] / divisor for q in (0.50, 0.95, 0.99))


def summarize(prefix):
    first, second = (read_rank(Path(f"{prefix}.rank{rank}.out"), rank) for rank in range(2))
    assert first.keys() == second.keys(), prefix
    for group in sorted(first):
        ranks = first[group], second[group]
        wall, reductions, gather = (
            [max(rank[i][metric] for rank in ranks) for i in range(SAMPLES)] for metric in range(3)
        )
        yield group, (
            percentiles(wall, 1000), percentiles(reductions, 1000),
            percentiles(reductions, BARRIERS), percentiles(gather),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", nargs="+", help="log prefix(es) passed to run_pair.sh")
    args = parser.parse_args()
    print("| Rails | Visibility | M | Graph wall ms | 96 SUMs ms | µs/SUM | Head µs |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    data = [item for prefix in args.prefix for item in summarize(prefix)]
    for (rails, mode, width), metrics in sorted(data):
        cells = (" / ".join(f"{x:.3f}" for x in values) for values in metrics)
        print(f"| {rails} | {mode} | {width} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
