#!/usr/bin/env python3
"""Phase 2 spike: PR event -> overlay -> impact-set (L2 blast radius) ->
materialized context packet, end to end.

Phase 1's two spikes proved the extraction and commit/merge mechanisms in
isolation. Nothing has chained them, and nothing has built an L2 derived
relation (blast radius) on top of L1 facts (defs/calls), or materialized
the file-tree context packet the harness's progressive-disclosure doctrine
calls for. This script is the first thing that does all of that in one
pass, timing every stage plus one end-to-end wall-clock span.

Blast radius here means: which symbols (transitively) CALL a touched
symbol. Callee resolution is name-based only (same heuristic tier as the
L1 extractor -- a callee_name can match multiple defs sharing a name; all
matches are treated as edges, which is the honest behavior of this tier,
not a bug to hide). Reverse-BFS from the touched defs over the call graph
produces the impact set, ranked by BFS distance.

Packet layout (file tree, POSIX-drillable, per the harness's stated
progressive-disclosure doctrine -- not a query API):

  packet/README.md              -- index: PR summary, counts, freshness
  packet/touched/<file>.json     -- defs for each touched file
  packet/impact/impact-set.json  -- ranked list: symbol, file, distance
  packet/impact/by-file.json     -- impact aggregated per file

Usage:
  blast_radius.py <src_root> <touched_files...> --store DIR --base-tag TAG
      --out-packet DIR [--pr-id ID]
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


# ---- extraction (same as commit_store.py; copied per this framework's
# per-experiment self-containment convention) --------------------------

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


# ---- L2: blast radius (new -- not present in either Phase 1 spike) ----

def compute_blast_radius(defs, calls, touched_files):
    """Reverse-BFS over the name-resolved call graph, starting from every
    def in a touched file. Returns (impact_rows, max_depth) where each
    impact row is a caller reached transitively, tagged with its BFS
    distance from the nearest touched def."""
    name_to_symbols = {}
    for d in defs:
        name_to_symbols.setdefault(d["name"], []).append(d["symbol_id"])
    symbol_by_id = {d["symbol_id"]: d for d in defs}

    # reverse edges: callee_symbol_id -> set of caller_symbol_id
    reverse_edges = {}
    for c in calls:
        for callee_symbol in name_to_symbols.get(c["callee_name"], []):
            reverse_edges.setdefault(callee_symbol, set()).add(c["caller_symbol_id"])

    touched_symbols = {d["symbol_id"] for d in defs if d["file"] in touched_files}

    visited = dict()  # symbol_id -> distance
    q = deque()
    for s in touched_symbols:
        visited[s] = 0
        q.append(s)

    while q:
        cur = q.popleft()
        for caller in reverse_edges.get(cur, ()):
            if caller not in visited:
                visited[caller] = visited[cur] + 1
                q.append(caller)

    impact_rows = []
    for symbol_id, dist in visited.items():
        if symbol_id in touched_symbols:
            continue  # impact set = what's affected BY the change, not the change itself
        d = symbol_by_id.get(symbol_id)
        if d is None:
            continue  # module-level enclosing "symbol" with no def record
        impact_rows.append({
            "symbol_id": symbol_id, "file": d["file"], "name": d["name"],
            "kind": d["kind"], "distance": dist,
        })
    impact_rows.sort(key=lambda r: (r["distance"], r["file"], r["symbol_id"]))
    max_depth = max((r["distance"] for r in impact_rows), default=0)
    return impact_rows, max_depth, sorted(touched_symbols)


# ---- packet materialization (new) --------------------------------------

def materialize_packet(out_dir: Path, pr_id, touched_files, defs, impact_rows, max_depth, timings):
    out_dir.mkdir(parents=True, exist_ok=True)
    touched_dir = out_dir / "touched"
    impact_dir = out_dir / "impact"
    touched_dir.mkdir(exist_ok=True)
    impact_dir.mkdir(exist_ok=True)

    defs_by_file = {}
    for d in defs:
        defs_by_file.setdefault(d["file"], []).append(d)

    for f in touched_files:
        safe = f.replace("/", "__")
        (touched_dir / f"{safe}.json").write_text(json.dumps(defs_by_file.get(f, []), indent=2))

    (impact_dir / "impact-set.json").write_text(json.dumps(impact_rows, indent=2))

    by_file = {}
    for r in impact_rows:
        by_file.setdefault(r["file"], 0)
        by_file[r["file"]] += 1
    by_file_ranked = sorted(by_file.items(), key=lambda kv: -kv[1])
    (impact_dir / "by-file.json").write_text(json.dumps(
        [{"file": f, "impacted_symbols": n} for f, n in by_file_ranked], indent=2))

    readme = f"""# Context packet — PR {pr_id}

Generated by fleet-cpg-phase2-blast-radius-zod. This is a materialized
file tree, not a query API — drill down with cat/jq, per the harness's
progressive-disclosure doctrine.

## Touched files ({len(touched_files)})
{chr(10).join(f'- `touched/{f.replace("/", "__")}.json` — {f}' for f in touched_files)}

## Impact set ({len(impact_rows)} symbols, max BFS depth {max_depth})

Symbols that transitively call something in the touched files, ranked by
BFS distance (closer = more directly affected). See
`impact/impact-set.json` for the full ranked list, `impact/by-file.json`
for a per-file rollup.

## Freshness

Base snapshot + overlay committed in this same run; no staleness window
between fact extraction and packet materialization (single-process spike,
not the async production pipeline).

## Provenance

Callee resolution is name-based only (heuristic tier) — an edge exists
between a call site and every def sharing the callee's name, which is
honest ambiguity at this resolution tier, not resolved to a single
definite caller.
"""
    (out_dir / "README.md").write_text(readme)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_root")
    ap.add_argument("touched_files", nargs="+")
    ap.add_argument("--store", required=True)
    ap.add_argument("--base-tag", required=True)
    ap.add_argument("--out-packet", required=True)
    ap.add_argument("--pr-id", default="unknown")
    args = ap.parse_args()

    root = Path(args.src_root)
    touched = args.touched_files
    base_dir = Path(args.store) / "base" / args.base_tag

    t_start = time.perf_counter()

    t0 = time.perf_counter()
    ov_defs, ov_calls = extract_many(root, touched)
    extract_ms = (time.perf_counter() - t0) * 1000

    ov_dir = Path(args.store) / "overlays" / f"{args.pr_id}"
    ov_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    write_parquet(ov_defs, ov_dir / "defs.parquet",
                  "symbol_id VARCHAR, file VARCHAR, kind VARCHAR, name VARCHAR, start_line BIGINT, end_line BIGINT")
    write_parquet(ov_calls, ov_dir / "calls.parquet",
                  "caller_symbol_id VARCHAR, callee_name VARCHAR, file VARCHAR, call_line BIGINT")
    commit_ms = (time.perf_counter() - t0) * 1000

    con = duckdb.connect()
    touched_list = "[" + ",".join(f"'{f}'" for f in touched) + "]"
    t0 = time.perf_counter()
    merged_defs_rows = con.execute(f"""
        SELECT * FROM read_parquet('{base_dir}/defs.parquet') WHERE file NOT IN {touched_list}
        UNION ALL
        SELECT * FROM read_parquet('{ov_dir}/defs.parquet')
    """).fetchall()
    merged_calls_rows = con.execute(f"""
        SELECT * FROM read_parquet('{base_dir}/calls.parquet') WHERE file NOT IN {touched_list}
        UNION ALL
        SELECT * FROM read_parquet('{ov_dir}/calls.parquet')
    """).fetchall()
    merge_ms = (time.perf_counter() - t0) * 1000
    con.close()

    merged_defs = [
        {"symbol_id": r[0], "file": r[1], "kind": r[2], "name": r[3], "start_line": r[4], "end_line": r[5]}
        for r in merged_defs_rows
    ]
    merged_calls = [
        {"caller_symbol_id": r[0], "callee_name": r[1], "file": r[2], "call_line": r[3]}
        for r in merged_calls_rows
    ]

    t0 = time.perf_counter()
    impact_rows, max_depth, touched_symbols = compute_blast_radius(merged_defs, merged_calls, set(touched))
    blast_ms = (time.perf_counter() - t0) * 1000

    out_packet = Path(args.out_packet)
    t0 = time.perf_counter()
    materialize_packet(out_packet, args.pr_id, touched, merged_defs, impact_rows, max_depth, {})
    packet_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    # stub agent consumption: read the index + the impact set back, prove the
    # packet round-trips as a plain file tree with no special client needed
    readme_text = (out_packet / "README.md").read_text()
    impact_readback = json.loads((out_packet / "impact" / "impact-set.json").read_text())
    consume_ms = (time.perf_counter() - t0) * 1000

    end_to_end_ms = (time.perf_counter() - t_start) * 1000

    result = {
        "pr_id": args.pr_id,
        "files_changed": len(touched),
        "touched_defs": len(touched_symbols),
        "impact_set_size": len(impact_rows),
        "max_bfs_depth": max_depth,
        "overlay_extract_wall_time_ms": extract_ms,
        "commit_write_wall_time_ms": commit_ms,
        "merge_query_wall_time_ms": merge_ms,
        "blast_radius_compute_wall_time_ms": blast_ms,
        "packet_materialize_wall_time_ms": packet_ms,
        "packet_consume_wall_time_ms": consume_ms,
        "end_to_end_wall_time_ms": end_to_end_ms,
        "packet_readme_bytes": len(readme_text),
        "impact_set_readback_count": len(impact_readback),
        "packet_round_trips": len(impact_readback) == len(impact_rows),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
