#!/usr/bin/env python3
"""Phase 1 overlay-COMMIT spike: base snapshot + copy-on-write overlay layer,
merged at query time, verified against a from-scratch ground-truth extraction.

fleet-cpg-phase1-overlay-zod measured only the extraction/lowering half of
the O(diff) claim (re-extracting facts for touched files). This script
measures the other half the design doc requires: committing that overlay
into a snapshot store without copying the base, and reading a merged view
back out, without silently diverging from what a full rebuild would produce.

Store layout (content-addressed, not owned segment format -- DuckDB/Parquet
scaffolding per the design doc's own "scaffolding is fine here" note):

  <store>/base/<tag>/{defs,calls}.parquet        -- one full-repo snapshot
  <store>/overlays/<run_id>_<bucket>/
      {defs,calls}.parquet                       -- ONLY touched files
      manifest.json                              -- {base_tag, touched_files}

Reading a commit's merged view is a query, not a copy:
  SELECT * FROM base_defs WHERE file NOT IN touched_files
  UNION ALL
  SELECT * FROM overlay_defs
(same shape for calls). This is the COW mechanism under test: write cost is
O(touched files), storage cost is O(touched files), and the base is never
duplicated or mutated.

Subcommands:
  base    <src_root> --store DIR --tag TAG
              Full-repo extraction, written as the base snapshot.
  overlay <src_root> <file...> --store DIR --base-tag TAG --run-id ID
          --bucket NAME --truth-root ROOT [--truth-files file...]
              Extracts only the given files as an overlay layer, commits it,
              merges it against the base via DuckDB, and (if --truth-root is
              given) verifies the merged view against a from-scratch
              extraction of the full repo at truth-root. Prints one JSON
              line with every timing + correctness metric.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser

TS_LANGUAGE = Language(tsts.language_typescript())

DEF_KINDS = {
    "function_declaration": "function",
    "class_declaration": "class",
    "method_definition": "method",
}


def node_text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def find_name(node, src: bytes):
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return node_text(src, name_node)
    return None


def extract_file(parser: Parser, root: Path, rel_path: str):
    src = (root / rel_path).read_bytes()
    tree = parser.parse(src)
    defs, calls = [], []

    def enclosing_symbol_id(stack):
        return stack[-1] if stack else f"{rel_path}::<module>"

    def walk(node, scope_stack):
        kind = node.type
        if kind in DEF_KINDS:
            name = find_name(node, src) or "<anonymous>"
            symbol_id = f"{rel_path}:{node.start_point[0]+1}:{name}"
            defs.append({
                "symbol_id": symbol_id, "file": rel_path,
                "kind": DEF_KINDS[kind], "name": name,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
            })
            scope_stack = scope_stack + [symbol_id]
        elif kind == "call_expression":
            callee = node.child_by_field_name("function")
            callee_name = None
            if callee is not None:
                if callee.type == "identifier":
                    callee_name = node_text(src, callee)
                elif callee.type == "member_expression":
                    prop = callee.child_by_field_name("property")
                    if prop is not None:
                        callee_name = node_text(src, prop)
            if callee_name:
                calls.append({
                    "caller_symbol_id": enclosing_symbol_id(scope_stack),
                    "callee_name": callee_name, "file": rel_path,
                    "call_line": node.start_point[0] + 1,
                })
        for child in node.children:
            walk(child, scope_stack)

    walk(tree.root_node, [])
    return defs, calls


def extract_many(root: Path, files):
    parser = Parser(TS_LANGUAGE)
    all_defs, all_calls = [], []
    for f in files:
        try:
            defs, calls = extract_file(parser, root, f)
        except Exception as e:  # noqa: BLE001 - spike: record and move on
            print(f"error extracting {f}: {e}", file=sys.stderr)
            continue
        all_defs.extend(defs)
        all_calls.extend(calls)
    return all_defs, all_calls


def write_parquet(rows, columns, out_path: Path):
    con = duckdb.connect()
    cols_sql = ", ".join(f"{c} VARCHAR" if c not in ("start_line", "end_line", "call_line")
                          else f"{c} BIGINT" for c in columns)
    if rows:
        con.execute(f"CREATE TABLE t AS SELECT * FROM (SELECT unnest($1, max_depth:=2))", [rows])
    else:
        con.execute(f"CREATE TABLE t ({cols_sql})")
    con.execute(f"COPY t TO '{out_path}' (FORMAT PARQUET)")
    con.close()


def cmd_base(args):
    root = Path(args.src_root)
    files = [str(p.relative_to(root)) for p in root.rglob("*.ts")]
    start = time.perf_counter()
    defs, calls = extract_many(root, files)
    elapsed_ms = (time.perf_counter() - start) * 1000

    base_dir = Path(args.store) / "base" / args.tag
    base_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(defs, ["symbol_id", "file", "kind", "name", "start_line", "end_line"], base_dir / "defs.parquet")
    write_parquet(calls, ["caller_symbol_id", "callee_name", "file", "call_line"], base_dir / "calls.parquet")

    bytes_total = (base_dir / "defs.parquet").stat().st_size + (base_dir / "calls.parquet").stat().st_size
    print(json.dumps({
        "op": "base", "tag": args.tag, "files": len(files),
        "defs": len(defs), "calls": len(calls),
        "extract_wall_time_ms": elapsed_ms, "snapshot_bytes": bytes_total,
    }))


def cmd_overlay(args):
    root = Path(args.src_root)
    touched = args.files

    t0 = time.perf_counter()
    ov_defs, ov_calls = extract_many(root, touched)
    extract_ms = (time.perf_counter() - t0) * 1000

    ov_dir = Path(args.store) / "overlays" / f"{args.run_id}_{args.bucket}"
    ov_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    write_parquet(ov_defs, ["symbol_id", "file", "kind", "name", "start_line", "end_line"], ov_dir / "defs.parquet")
    write_parquet(ov_calls, ["caller_symbol_id", "callee_name", "file", "call_line"], ov_dir / "calls.parquet")
    (ov_dir / "manifest.json").write_text(json.dumps({
        "base_tag": args.base_tag, "touched_files": touched,
    }, indent=2))
    commit_ms = (time.perf_counter() - t0) * 1000

    base_dir = Path(args.store) / "base" / args.base_tag
    base_bytes = (base_dir / "defs.parquet").stat().st_size + (base_dir / "calls.parquet").stat().st_size
    overlay_bytes = (ov_dir / "defs.parquet").stat().st_size + (ov_dir / "calls.parquet").stat().st_size

    # Merge at query time: base rows for untouched files, union overlay rows.
    con = duckdb.connect()
    touched_list = "[" + ",".join(f"'{f}'" for f in touched) + "]"
    t0 = time.perf_counter()
    merged_defs = con.execute(f"""
        SELECT * FROM read_parquet('{base_dir}/defs.parquet') WHERE file NOT IN {touched_list}
        UNION ALL
        SELECT * FROM read_parquet('{ov_dir}/defs.parquet')
    """).fetchall()
    merged_calls = con.execute(f"""
        SELECT * FROM read_parquet('{base_dir}/calls.parquet') WHERE file NOT IN {touched_list}
        UNION ALL
        SELECT * FROM read_parquet('{ov_dir}/calls.parquet')
    """).fetchall()
    merge_ms = (time.perf_counter() - t0) * 1000
    con.close()

    result = {
        "op": "overlay", "run_id": args.run_id, "bucket": args.bucket,
        "files_changed": len(touched),
        "overlay_extract_wall_time_ms": extract_ms,
        "commit_write_wall_time_ms": commit_ms,
        "merge_query_wall_time_ms": merge_ms,
        "base_snapshot_bytes": base_bytes,
        "overlay_snapshot_bytes": overlay_bytes,
        "overlay_to_base_bytes_ratio": overlay_bytes / base_bytes if base_bytes else None,
        "merged_defs": len(merged_defs), "merged_calls": len(merged_calls),
    }

    if args.truth_root:
        truth_root = Path(args.truth_root)
        truth_files = args.truth_files or [str(p.relative_to(truth_root)) for p in truth_root.rglob("*.ts")]
        t0 = time.perf_counter()
        truth_defs, truth_calls = extract_many(truth_root, truth_files)
        verify_ms = (time.perf_counter() - t0) * 1000

        def def_key(row):
            return (row[0], row[1], row[2], row[3])  # symbol_id, file, kind, name

        def call_key(row):
            return (row[0], row[1], row[2], row[3])  # caller, callee, file, line

        merged_def_keys = {def_key(r) for r in merged_defs}
        truth_def_keys = {(d["symbol_id"], d["file"], d["kind"], d["name"]) for d in truth_defs}
        merged_call_keys = {call_key(r) for r in merged_calls}
        truth_call_keys = {(c["caller_symbol_id"], c["callee_name"], c["file"], c["call_line"]) for c in truth_calls}

        result.update({
            "verify_wall_time_ms": verify_ms,
            "truth_defs": len(truth_defs), "truth_calls": len(truth_calls),
            "defs_match_truth": merged_def_keys == truth_def_keys,
            "calls_match_truth": merged_call_keys == truth_call_keys,
            "defs_missing_from_merge": len(truth_def_keys - merged_def_keys),
            "defs_extra_in_merge": len(merged_def_keys - truth_def_keys),
            "calls_missing_from_merge": len(truth_call_keys - merged_call_keys),
            "calls_extra_in_merge": len(merged_call_keys - truth_call_keys),
        })

    print(json.dumps(result))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_base = sub.add_parser("base")
    p_base.add_argument("src_root")
    p_base.add_argument("--store", required=True)
    p_base.add_argument("--tag", required=True)
    p_base.set_defaults(func=cmd_base)

    p_overlay = sub.add_parser("overlay")
    p_overlay.add_argument("src_root")
    p_overlay.add_argument("files", nargs="+")
    p_overlay.add_argument("--store", required=True)
    p_overlay.add_argument("--base-tag", required=True)
    p_overlay.add_argument("--run-id", required=True)
    p_overlay.add_argument("--bucket", required=True)
    p_overlay.add_argument("--truth-root", default=None)
    p_overlay.add_argument("--truth-files", nargs="*", default=None)
    p_overlay.set_defaults(func=cmd_overlay)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
