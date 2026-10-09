#!/usr/bin/env python3
"""Phase 4 driver: run the full Phase 2 pipeline (extract -> commit ->
merge -> blast radius -> packet) through core.py, selecting the lowering
module by name on the command line. core.py is imported unmodified
regardless of which language is selected -- that's the entire point.

Usage:
  phase4_run.py <lowering-module> <src_root> <touched_files...>
      --store DIR --base-tag TAG --out-packet DIR --pr-id ID
      [--truth-root ROOT]
"""
import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import core


def core_sha256():
    return hashlib.sha256(Path(__file__).with_name("core.py").read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lowering_module")
    ap.add_argument("src_root")
    ap.add_argument("touched_files", nargs="+")
    ap.add_argument("--store", required=True)
    ap.add_argument("--base-tag", required=True)
    ap.add_argument("--out-packet", required=True)
    ap.add_argument("--pr-id", default="unknown")
    ap.add_argument("--truth-root", default=None)
    args = ap.parse_args()

    lowering = importlib.import_module(args.lowering_module)
    root = Path(args.src_root)
    touched = args.touched_files
    base_dir = Path(args.store) / "base" / args.base_tag

    t_start = time.perf_counter()

    if not base_dir.exists():
        t0 = time.perf_counter()
        all_files = [str(p.relative_to(root)) for p in root.rglob(lowering.FILE_GLOB)]
        base_defs, base_calls = lowering.extract_many(root, all_files)
        base_dir.mkdir(parents=True, exist_ok=True)
        core.write_parquet(base_defs, base_dir / "defs.parquet", core.DEFS_COLS)
        core.write_parquet(base_calls, base_dir / "calls.parquet", core.CALLS_COLS)
        base_build_ms = (time.perf_counter() - t0) * 1000
    else:
        base_build_ms = 0.0

    t0 = time.perf_counter()
    ov_defs, ov_calls = lowering.extract_many(root, touched)
    extract_ms = (time.perf_counter() - t0) * 1000

    ov_dir = Path(args.store) / "overlays" / args.pr_id
    ov_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    core.write_parquet(ov_defs, ov_dir / "defs.parquet", core.DEFS_COLS)
    core.write_parquet(ov_calls, ov_dir / "calls.parquet", core.CALLS_COLS)
    commit_ms = (time.perf_counter() - t0) * 1000

    layers = [{"tag": args.pr_id, "dir": str(ov_dir), "files": touched}]

    import duckdb
    con = duckdb.connect()
    t0 = time.perf_counter()
    merged_defs_rows = con.execute(core.build_merge_query(base_dir, layers, "defs")).fetchall()
    merged_calls_rows = con.execute(core.build_merge_query(base_dir, layers, "calls")).fetchall()
    con.close()
    merge_ms = (time.perf_counter() - t0) * 1000

    merged_defs = [
        {"symbol_id": r[0], "file": r[1], "kind": r[2], "name": r[3], "start_line": r[4], "end_line": r[5]}
        for r in merged_defs_rows
    ]
    merged_calls = [
        {"caller_symbol_id": r[0], "callee_name": r[1], "file": r[2], "call_line": r[3]}
        for r in merged_calls_rows
    ]

    t0 = time.perf_counter()
    impact_rows, max_depth, touched_symbols = core.compute_blast_radius(merged_defs, merged_calls, set(touched))
    blast_ms = (time.perf_counter() - t0) * 1000

    out_packet = Path(args.out_packet)
    t0 = time.perf_counter()
    core.materialize_packet(out_packet, args.pr_id, touched, merged_defs, impact_rows, max_depth, lowering.LANGUAGE_NAME)
    packet_ms = (time.perf_counter() - t0) * 1000

    end_to_end_ms = (time.perf_counter() - t_start) * 1000

    result = {
        "language": lowering.LANGUAGE_NAME,
        "pr_id": args.pr_id,
        "files_changed": len(touched),
        "touched_defs": len(touched_symbols),
        "merged_defs": len(merged_defs),
        "merged_calls": len(merged_calls),
        "impact_set_size": len(impact_rows),
        "max_bfs_depth": max_depth,
        "base_build_wall_time_ms": base_build_ms,
        "overlay_extract_wall_time_ms": extract_ms,
        "commit_write_wall_time_ms": commit_ms,
        "merge_query_wall_time_ms": merge_ms,
        "blast_radius_compute_wall_time_ms": blast_ms,
        "packet_materialize_wall_time_ms": packet_ms,
        "end_to_end_wall_time_ms": end_to_end_ms,
        "core_py_sha256": core_sha256(),
    }

    if args.truth_root:
        truth_root = Path(args.truth_root)
        truth_files = [str(p.relative_to(truth_root)) for p in truth_root.rglob(lowering.FILE_GLOB)]
        truth_defs, truth_calls = lowering.extract_many(truth_root, truth_files)
        merged_def_keys = {(d["symbol_id"], d["file"], d["kind"], d["name"]) for d in merged_defs}
        truth_def_keys = {(d["symbol_id"], d["file"], d["kind"], d["name"]) for d in truth_defs}
        merged_call_keys = {(c["caller_symbol_id"], c["callee_name"], c["file"], c["call_line"]) for c in merged_calls}
        truth_call_keys = {(c["caller_symbol_id"], c["callee_name"], c["file"], c["call_line"]) for c in truth_calls}
        result.update({
            "defs_match_truth": merged_def_keys == truth_def_keys,
            "calls_match_truth": merged_call_keys == truth_call_keys,
        })

    print(json.dumps(result))


if __name__ == "__main__":
    main()
