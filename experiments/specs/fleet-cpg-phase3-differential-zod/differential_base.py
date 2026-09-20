#!/usr/bin/env python3
"""Phase 3 spike: differential base maintenance over a real push chain.

Phases 1 and 2 tested a base held FIXED while a single PR overlay was
built and merged against it. Phase 3's gate is different and harder:
"correctness under continuous update" — the base itself must advance by
small deltas across a whole chain of real pushes, never a full re-scan,
and every derived relation (L1 facts, L2 blast radius) computed on the
incrementally-maintained base must still match a from-scratch build at
every single step. A bug at step i silently corrupts every step after it
-- this is the differential-fixpoint idea from the design doc's "base
indexer" component, and the first time it's tested as a sequence rather
than a single update.

Subcommands:
  init  <src_root> --store DIR --tag TAG
            Full-repo extraction, written as the chain's starting base.
  step  <src_root> <changed_file...> --store DIR --from-tag TAG
            --to-tag TAG [--truth-root ROOT]
            Extracts a delta for only the files this push changed, merges
            it with the current base (untouched files unioned with the
            delta) to produce the NEXT base snapshot -- never re-scanning
            unchanged files -- and (if --truth-root given) verifies both
            the merged L1 facts and an L2 blast-radius relation computed
            on them against a from-scratch full extraction.
"""
import argparse
import json
import sys
import time
from collections import deque
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


def write_parquet(rows, out_path: Path, empty_cols_sql: str):
    con = duckdb.connect()
    if rows:
        con.execute("CREATE TABLE t AS SELECT * FROM (SELECT unnest($1, max_depth:=2))", [rows])
    else:
        con.execute(f"CREATE TABLE t ({empty_cols_sql})")
    con.execute(f"COPY t TO '{out_path}' (FORMAT PARQUET)")
    con.close()


DEFS_COLS = "symbol_id VARCHAR, file VARCHAR, kind VARCHAR, name VARCHAR, start_line BIGINT, end_line BIGINT"
CALLS_COLS = "caller_symbol_id VARCHAR, callee_name VARCHAR, file VARCHAR, call_line BIGINT"


def compute_blast_radius(defs, calls, seed_files):
    name_to_symbols = {}
    for d in defs:
        name_to_symbols.setdefault(d["name"], []).append(d["symbol_id"])
    symbol_by_id = {d["symbol_id"]: d for d in defs}

    reverse_edges = {}
    for c in calls:
        for callee_symbol in name_to_symbols.get(c["callee_name"], []):
            reverse_edges.setdefault(callee_symbol, set()).add(c["caller_symbol_id"])

    seed_symbols = {d["symbol_id"] for d in defs if d["file"] in seed_files}
    visited = {}
    q = deque()
    for s in seed_symbols:
        visited[s] = 0
        q.append(s)
    while q:
        cur = q.popleft()
        for caller in reverse_edges.get(cur, ()):
            if caller not in visited:
                visited[caller] = visited[cur] + 1
                q.append(caller)

    impact = set()
    for symbol_id in visited:
        if symbol_id in seed_symbols:
            continue
        if symbol_id in symbol_by_id:
            impact.add(symbol_id)
    return impact


def cmd_init(args):
    root = Path(args.src_root)
    files = [str(p.relative_to(root)) for p in root.rglob("*.ts")]
    start = time.perf_counter()
    defs, calls = extract_many(root, files)
    elapsed_ms = (time.perf_counter() - start) * 1000

    base_dir = Path(args.store) / "base" / args.tag
    base_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(defs, base_dir / "defs.parquet", DEFS_COLS)
    write_parquet(calls, base_dir / "calls.parquet", CALLS_COLS)

    print(json.dumps({
        "op": "init", "tag": args.tag, "files": len(files),
        "defs": len(defs), "calls": len(calls),
        "extract_wall_time_ms": elapsed_ms,
    }))


def load_layers_manifest(store: Path):
    path = store / "layers.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


def save_layers_manifest(store: Path, layers):
    (store / "layers.json").write_text(json.dumps(layers, indent=2))


def build_merge_query(base_dir: Path, delta_layers_oldest_first, table: str):
    """newest-wins LSM merge: iterate delta layers newest->oldest, each one
    excluding files already claimed by a newer layer; base is the
    catch-all at the bottom, excluding everything any delta ever touched.
    Write cost per step is just that step's own small delta (elsewhere);
    this query is the read-path cost of NOT compacting the layer chain --
    tracked as its own metric, not conflated with write cost."""
    claimed = set()
    parts = []
    for layer in reversed(delta_layers_oldest_first):
        excl = "[" + ",".join(f"'{f}'" for f in claimed) + "]"
        parts.append(f"SELECT * FROM read_parquet('{layer['dir']}/{table}.parquet') WHERE file NOT IN {excl}")
        claimed |= set(layer["files"])
    excl = "[" + ",".join(f"'{f}'" for f in claimed) + "]"
    parts.append(f"SELECT * FROM read_parquet('{base_dir}/{table}.parquet') WHERE file NOT IN {excl}")
    return " UNION ALL ".join(parts)


def cmd_step(args):
    root = Path(args.src_root)
    changed = args.files
    store = Path(args.store)
    base_dir = store / "base" / args.base_tag
    layer_dir = store / "layers" / args.to_tag

    t_start = time.perf_counter()

    t0 = time.perf_counter()
    delta_defs, delta_calls = extract_many(root, changed)
    extract_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    layer_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(delta_defs, layer_dir / "defs.parquet", DEFS_COLS)
    write_parquet(delta_calls, layer_dir / "calls.parquet", CALLS_COLS)
    write_ms = (time.perf_counter() - t0) * 1000

    incremental_update_ms = (time.perf_counter() - t_start) * 1000

    layers = load_layers_manifest(store)
    layers.append({"tag": args.to_tag, "dir": str(layer_dir), "files": changed})
    save_layers_manifest(store, layers)

    t0 = time.perf_counter()
    con = duckdb.connect()
    merged_defs_rows = con.execute(build_merge_query(base_dir, layers, "defs")).fetchall()
    merged_calls_rows = con.execute(build_merge_query(base_dir, layers, "calls")).fetchall()
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

    result = {
        "op": "step", "base_tag": args.base_tag, "to_tag": args.to_tag,
        "files_changed": len(changed),
        "n_layers": len(layers),
        "delta_extract_wall_time_ms": extract_ms,
        "layer_write_wall_time_ms": write_ms,
        "incremental_update_wall_time_ms": incremental_update_ms,
        "merge_query_wall_time_ms": merge_ms,
        "base_defs": len(merged_defs), "base_calls": len(merged_calls),
    }

    if args.truth_root:
        truth_root = Path(args.truth_root)
        truth_files = [str(p.relative_to(truth_root)) for p in truth_root.rglob("*.ts")]
        t0 = time.perf_counter()
        truth_defs, truth_calls = extract_many(truth_root, truth_files)
        verify_ms = (time.perf_counter() - t0) * 1000

        merged_def_keys = {(d["symbol_id"], d["file"], d["kind"], d["name"]) for d in merged_defs}
        truth_def_keys = {(d["symbol_id"], d["file"], d["kind"], d["name"]) for d in truth_defs}
        merged_call_keys = {(c["caller_symbol_id"], c["callee_name"], c["file"], c["call_line"]) for c in merged_calls}
        truth_call_keys = {(c["caller_symbol_id"], c["callee_name"], c["file"], c["call_line"]) for c in truth_calls}

        merged_impact = compute_blast_radius(merged_defs, merged_calls, set(changed))
        truth_impact = compute_blast_radius(truth_defs, truth_calls, set(changed))

        result.update({
            "verify_wall_time_ms": verify_ms,
            "truth_defs": len(truth_defs), "truth_calls": len(truth_calls),
            "defs_match_truth": merged_def_keys == truth_def_keys,
            "calls_match_truth": merged_call_keys == truth_call_keys,
            "defs_missing_from_merge": len(truth_def_keys - merged_def_keys),
            "defs_extra_in_merge": len(merged_def_keys - truth_def_keys),
            "calls_missing_from_merge": len(truth_call_keys - merged_call_keys),
            "calls_extra_in_merge": len(merged_call_keys - truth_call_keys),
            "blast_radius_match_truth": merged_impact == truth_impact,
            "blast_radius_size_merged": len(merged_impact),
            "blast_radius_size_truth": len(truth_impact),
        })

    print(json.dumps(result))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("src_root")
    p_init.add_argument("--store", required=True)
    p_init.add_argument("--tag", required=True)
    p_init.set_defaults(func=cmd_init)

    p_step = sub.add_parser("step")
    p_step.add_argument("src_root")
    p_step.add_argument("files", nargs="+")
    p_step.add_argument("--store", required=True)
    p_step.add_argument("--base-tag", required=True)
    p_step.add_argument("--to-tag", required=True)
    p_step.add_argument("--truth-root", default=None)
    p_step.set_defaults(func=cmd_step)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
