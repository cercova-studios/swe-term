#!/usr/bin/env python3
"""Evaluator for fleet-cpg-phase2-blast-radius-zod.

Reads results.jsonl from every official run-*/ directory (pilot-* excluded)
and reports, per bucket: end-to-end wall time (with per-stage breakdown),
impact-set size, and packet round-trip integrity.
"""
import glob
import json
import statistics
import sys

RUNS_DIR = "experiments/runs/fleet-cpg-phase2-blast-radius-zod"

STAGES = [
    "overlay_extract_wall_time_ms",
    "commit_write_wall_time_ms",
    "merge_query_wall_time_ms",
    "blast_radius_compute_wall_time_ms",
    "packet_materialize_wall_time_ms",
    "packet_consume_wall_time_ms",
]


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
    all_round_trip = True
    for bucket, items in by_bucket.items():
        e2e = [i["end_to_end_wall_time_ms"] for i in items]
        round_trip_ok = all(i["packet_round_trips"] for i in items)
        all_round_trip = all_round_trip and round_trip_ok
        stage_medians = {
            s: statistics.median(i[s] for i in items) for s in STAGES
        }
        summary["buckets"][bucket] = {
            "files_changed": items[0]["files_changed"],
            "n_reps": len(items),
            "median_end_to_end_ms": statistics.median(e2e),
            "stage_medians_ms": stage_medians,
            "impact_set_size": [i["impact_set_size"] for i in items],
            "max_bfs_depth": [i["max_bfs_depth"] for i in items],
            "packet_round_trips_all_reps": round_trip_ok,
        }

    summary["all_packets_round_trip"] = all_round_trip
    print(json.dumps(summary, indent=2))
    return 0 if all_round_trip else 1


if __name__ == "__main__":
    sys.exit(main())
