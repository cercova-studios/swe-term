#!/usr/bin/env python3
"""Evaluator for fleet-cpg-phase1-commit-zod.

Reads results.jsonl from every official run-*/ directory (pilot-* excluded)
and reports, per bucket: commit write cost, merge/read cost, storage ratio
of overlay-to-base, and correctness against a from-scratch ground-truth
extraction at the PR's head commit.
"""
import glob
import json
import statistics
import sys

RUNS_DIR = "experiments/runs/fleet-cpg-phase1-commit-zod"


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

    summary = {"buckets": {}}
    all_correct = True
    for bucket, items in by_bucket.items():
        commit_ms = [i["commit_write_wall_time_ms"] for i in items]
        merge_ms = [i["merge_query_wall_time_ms"] for i in items]
        ratios = [i["overlay_to_base_bytes_ratio"] for i in items]
        defs_ok = all(i.get("defs_match_truth") for i in items)
        calls_ok = all(i.get("calls_match_truth") for i in items)
        bucket_correct = defs_ok and calls_ok
        all_correct = all_correct and bucket_correct
        summary["buckets"][bucket] = {
            "files_changed": items[0]["files_changed"],
            "n_reps": len(items),
            "median_commit_write_ms": statistics.median(commit_ms),
            "median_merge_query_ms": statistics.median(merge_ms),
            "median_overlay_to_base_bytes_ratio": statistics.median(ratios),
            "defs_match_truth_all_reps": defs_ok,
            "calls_match_truth_all_reps": calls_ok,
            "defs_missing_from_merge": [i.get("defs_missing_from_merge") for i in items],
            "defs_extra_in_merge": [i.get("defs_extra_in_merge") for i in items],
            "calls_missing_from_merge": [i.get("calls_missing_from_merge") for i in items],
            "calls_extra_in_merge": [i.get("calls_extra_in_merge") for i in items],
        }

    summary["all_buckets_correct"] = all_correct
    summary["all_storage_ratios_small"] = all(
        b["median_overlay_to_base_bytes_ratio"] < 0.1 for b in summary["buckets"].values()
    )
    print(json.dumps(summary, indent=2))
    return 0 if all_correct else 1


if __name__ == "__main__":
    sys.exit(main())
