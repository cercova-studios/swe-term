#!/usr/bin/env python3
"""Evaluator for fleet-cpg-phase3-differential-zod.

Reads results.jsonl from every official run-*/ directory (pilot-* excluded)
and reports, per step across the chain: incremental update cost, and
whether L1 facts and the L2 blast-radius relation stayed correct against a
from-scratch build at that same commit. Zero exceptions across every step
of every repetition is the exit-gate bar -- one failure anywhere fails the
whole run, not just that step, because a real differential base doesn't
self-heal either.
"""
import glob
import json
import statistics
import sys

RUNS_DIR = "experiments/runs/fleet-cpg-phase3-differential-zod"


def main():
    reps = []
    for path in sorted(glob.glob(f"{RUNS_DIR}/run-*/results.jsonl")):
        rows = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if rows:
            reps.append((path, rows))

    if not reps:
        print(json.dumps({"error": "no results found"}))
        return 1

    all_ok = True
    per_bucket_step_update_ms = {}
    step_failures = []
    for path, rows in reps:
        for r in rows:
            bucket = r.get("bucket", "unknown")
            step = r["step"]
            per_bucket_step_update_ms.setdefault(bucket, {}).setdefault(step, []).append(
                r["incremental_update_wall_time_ms"])
            ok = r.get("defs_match_truth") and r.get("calls_match_truth") and r.get("blast_radius_match_truth")
            if not ok:
                all_ok = False
                step_failures.append({
                    "run": path, "bucket": bucket, "step": step, "to_sha": r["to_sha"],
                    "defs_match_truth": r.get("defs_match_truth"),
                    "calls_match_truth": r.get("calls_match_truth"),
                    "blast_radius_match_truth": r.get("blast_radius_match_truth"),
                    "defs_missing_from_merge": r.get("defs_missing_from_merge"),
                    "defs_extra_in_merge": r.get("defs_extra_in_merge"),
                })

    buckets_summary = {}
    for bucket, steps in per_bucket_step_update_ms.items():
        buckets_summary[bucket] = {
            "n_steps": len(steps),
            "median_incremental_update_ms_by_step": {
                step: statistics.median(times) for step, times in sorted(steps.items())
            },
        }

    summary = {
        "n_repetitions": len(reps),
        "buckets": buckets_summary,
        "all_steps_correct_all_reps": all_ok,
        "step_failures": step_failures,
    }
    print(json.dumps(summary, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
