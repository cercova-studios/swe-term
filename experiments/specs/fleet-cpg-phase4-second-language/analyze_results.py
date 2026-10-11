#!/usr/bin/env python3
"""Evaluator for fleet-cpg-phase4-second-language.

Reads results.jsonl from every official run-*/ directory (pilot-* excluded)
and checks the agnosticism gate mechanically: core.py's recorded sha256
must be identical across every run regardless of language, AND must match
the actual core.py file on disk right now -- not just agree with itself,
which would pass even if core.py had been edited after every recorded run.
"""
import glob
import hashlib
import json
import sys
from pathlib import Path

RUNS_DIR = "experiments/runs/fleet-cpg-phase4-second-language"
CORE_PY = Path(__file__).with_name("core.py")


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

    current_core_hash = hashlib.sha256(CORE_PY.read_bytes()).hexdigest()
    recorded_hashes = {r["core_py_sha256"] for r in rows}

    by_bucket = {}
    for r in rows:
        by_bucket.setdefault(r["bucket"], []).append(r)

    buckets_summary = {}
    all_correct = True
    for bucket, items in by_bucket.items():
        defs_ok = all(i.get("defs_match_truth") for i in items)
        calls_ok = all(i.get("calls_match_truth") for i in items)
        all_correct = all_correct and defs_ok and calls_ok
        buckets_summary[bucket] = {
            "n_reps": len(items),
            "language": items[0]["language"],
            "merged_defs": [i["merged_defs"] for i in items],
            "merged_calls": [i["merged_calls"] for i in items],
            "impact_set_size": [i["impact_set_size"] for i in items],
            "defs_match_truth_all_reps": defs_ok,
            "calls_match_truth_all_reps": calls_ok,
            "median_end_to_end_ms": sorted(i["end_to_end_wall_time_ms"] for i in items)[len(items) // 2],
        }

    core_unchanged_across_runs = len(recorded_hashes) == 1
    core_matches_disk = len(recorded_hashes) == 1 and current_core_hash in recorded_hashes

    summary = {
        "buckets": buckets_summary,
        "all_correctness_checks_pass": all_correct,
        "core_py_hash_identical_across_all_runs": core_unchanged_across_runs,
        "core_py_hash_matches_current_file_on_disk": core_matches_disk,
        "recorded_core_py_hashes": sorted(recorded_hashes),
        "current_core_py_hash": current_core_hash,
        "agnosticism_gate_passed": core_unchanged_across_runs and core_matches_disk and all_correct,
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["agnosticism_gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
