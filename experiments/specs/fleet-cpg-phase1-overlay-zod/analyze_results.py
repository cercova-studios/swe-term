#!/usr/bin/env python3
"""Evaluator for fleet-cpg-phase1-overlay-zod.

Reads results.jsonl from every official run-*/ directory (pilot-* excluded)
and reports overlay extract time per bucket vs. the full-repo baseline.
"""
import glob
import json
import statistics
import sys

RUNS_DIR = "experiments/runs/fleet-cpg-phase1-overlay-zod"

# Full-repo baseline wall times, measured separately (see README) — our own
# extractor run with --all over all 324 files in packages/zod/src, 3 reps.
FULL_REPO_BASELINE_MS = [422.659, 576.325, 385.810]


def main():
    rows = []
    for path in sorted(glob.glob(f"{RUNS_DIR}/run-*/results.jsonl")):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    if not rows:
        print(json.dumps({"error": "no results found"}))
        return 1

    by_bucket = {}
    for r in rows:
        by_bucket.setdefault(r["bucket"], []).append(r)

    defs_calls_stable = {}
    summary = {"buckets": {}, "full_repo_baseline_median_ms": statistics.median(FULL_REPO_BASELINE_MS)}
    for bucket, items in by_bucket.items():
        times = [i["extract_wall_time_ms"] for i in items]
        defs = {i["defs"] for i in items}
        calls = {i["calls"] for i in items}
        defs_calls_stable[bucket] = len(defs) == 1 and len(calls) == 1
        summary["buckets"][bucket] = {
            "files_changed": items[0]["files_changed"],
            "lines_changed": items[0]["lines_changed"],
            "n_reps": len(times),
            "median_ms": statistics.median(times),
            "all_ms": times,
            "defs_stable": len(defs) == 1,
            "calls_stable": len(calls) == 1,
            "order_of_magnitude_below_baseline": statistics.median(times) * 10 <= summary["full_repo_baseline_median_ms"],
        }

    summary["all_buckets_pass_10x_rule"] = all(
        b["order_of_magnitude_below_baseline"] for b in summary["buckets"].values()
    )
    summary["all_defs_calls_stable"] = all(defs_calls_stable.values())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
