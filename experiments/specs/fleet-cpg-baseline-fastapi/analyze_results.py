#!/usr/bin/env python3
"""Evaluator for fleet-cpg-baseline-fastapi.

Reads every results.jsonl under experiments/runs/fleet-cpg-baseline-fastapi/run-*/
(pilot-* directories are excluded — they predate preregistration and are
exploratory data, not evidence) and reports the primary/secondary metrics
plus the rubric's pass/fail checks. Prints a JSON summary; does not decide
promotion.
"""
import glob
import json
import statistics
import re
import sys

REPO_ROOT_RUNS = "experiments/runs/fleet-cpg-baseline-fastapi"


def load_rows():
    rows = []
    for path in sorted(glob.glob(f"{REPO_ROOT_RUNS}/run-*/results.jsonl")):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def load_cold_build_stats():
    stats = []
    for path in sorted(glob.glob(f"{REPO_ROOT_RUNS}/run-*/artifacts/cold_build.log")):
        text = open(path).read()
        real = re.search(r"([\d.]+)\s+real", text)
        rss = re.search(r"(\d+)\s+maximum resident set size", text)
        if real and rss:
            stats.append({"run": path, "wall_time_s": float(real.group(1)), "peak_rss_bytes": int(rss.group(1))})
    return stats


def main():
    rows = load_rows()
    if not rows:
        print(json.dumps({"error": "no results found under " + REPO_ROOT_RUNS}))
        return 1

    errors = [r for r in rows if r["status"] != "ok"]
    result_counts = {}
    for r in rows:
        if r["query_id"] == "_process_wall_time":
            continue
        key = (r["query_id"], r["variant"])
        result_counts.setdefault(key, set()).add(r["result_count"])
    unstable = {f"{k[0]}/{k[1]}": list(v) for k, v in result_counts.items() if len(v) > 1}

    wall = {"cold": [], "warm": []}
    for r in rows:
        if r["query_id"] == "_process_wall_time":
            wall[r["variant"]].append(r["latency_ms"])

    cold_med = statistics.median(wall["cold"]) if wall["cold"] else None
    warm_med = statistics.median(wall["warm"]) if wall["warm"] else None
    accepted = None
    if cold_med and warm_med:
        accepted = warm_med <= cold_med * 0.8

    summary = {
        "n_rows": len(rows),
        "errors": errors,
        "unstable_result_counts": unstable,
        "process_wall_time_ms": {"cold_median": cold_med, "warm_median": warm_med, "cold_all": wall["cold"], "warm_all": wall["warm"]},
        "acceptance_rule_met": accepted,
        "cold_build": load_cold_build_stats(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
