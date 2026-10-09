"""Phase 4 language-agnostic core: store I/O, COW/LSM merge, L2 blast
radius, and context-packet materialization.

This is the layer under test. It knows the L1 fact schema
(defs: symbol_id, file, kind, name, start_line, end_line; calls:
caller_symbol_id, callee_name, file, call_line) and NOTHING about any
programming language, grammar, or AST node type. Every function here
operates on plain lists of dicts already in that shape.

The agnosticism test (Phase 4's gate): adding a second language must touch
only a new `lowering_<lang>.py` module (L0 grammar + L1 lowering rules,
per the design doc's layered IR). If this file needs a single line changed
to support Python after being finalized against TypeScript, the IR
boundary is wrong. `run_phase4.sh` asserts this mechanically by hashing
this file before and after the Python lowering module is added.

Ported verbatim (behavior-identical) from fleet-cpg-phase2-blast-radius-zod
and fleet-cpg-phase3-differential-zod -- this file is the shared spine
those two experiments each duplicated per this framework's
per-experiment self-containment convention. Extracting it once here is
exactly what "the engine, not the feature, changes" means in practice.
"""
import json
from collections import deque
from pathlib import Path

import duckdb

DEFS_COLS = "symbol_id VARCHAR, file VARCHAR, kind VARCHAR, name VARCHAR, start_line BIGINT, end_line BIGINT"
CALLS_COLS = "caller_symbol_id VARCHAR, callee_name VARCHAR, file VARCHAR, call_line BIGINT"


def write_parquet(rows, out_path: Path, empty_cols_sql: str):
    con = duckdb.connect()
    if rows:
        con.execute("CREATE TABLE t AS SELECT * FROM (SELECT unnest($1, max_depth:=2))", [rows])
    else:
        con.execute(f"CREATE TABLE t ({empty_cols_sql})")
    con.execute(f"COPY t TO '{out_path}' (FORMAT PARQUET)")
    con.close()


def load_layers_manifest(store: Path):
    path = store / "layers.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


def save_layers_manifest(store: Path, layers):
    (store / "layers.json").write_text(json.dumps(layers, indent=2))


def build_merge_query(base_dir: Path, delta_layers_oldest_first, table: str):
    """newest-wins LSM merge -- see fleet-cpg-phase3-differential-zod for
    the full rationale. Base is the catch-all at the bottom."""
    claimed = set()
    parts = []
    for layer in reversed(delta_layers_oldest_first):
        excl = "[" + ",".join(f"'{f}'" for f in claimed) + "]"
        parts.append(f"SELECT * FROM read_parquet('{layer['dir']}/{table}.parquet') WHERE file NOT IN {excl}")
        claimed |= set(layer["files"])
    excl = "[" + ",".join(f"'{f}'" for f in claimed) + "]"
    parts.append(f"SELECT * FROM read_parquet('{base_dir}/{table}.parquet') WHERE file NOT IN {excl}")
    return " UNION ALL ".join(parts)


def compute_blast_radius(defs, calls, seed_files):
    """Reverse-BFS over the name-resolved call graph -- see
    fleet-cpg-phase2-blast-radius-zod for the full rationale."""
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

    impact_rows = []
    for symbol_id, dist in visited.items():
        if symbol_id in seed_symbols:
            continue
        d = symbol_by_id.get(symbol_id)
        if d is None:
            continue
        impact_rows.append({
            "symbol_id": symbol_id, "file": d["file"], "name": d["name"],
            "kind": d["kind"], "distance": dist,
        })
    impact_rows.sort(key=lambda r: (r["distance"], r["file"], r["symbol_id"]))
    max_depth = max((r["distance"] for r in impact_rows), default=0)
    return impact_rows, max_depth, sorted(seed_symbols)


def materialize_packet(out_dir: Path, pr_id, touched_files, defs, impact_rows, max_depth, language):
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

    readme = f"""# Context packet — PR {pr_id} ({language})

Generated by fleet-cpg-phase4-second-language, using the same
language-agnostic core (`core.py`) as every other language.

## Touched files ({len(touched_files)})
{chr(10).join(f'- `touched/{f.replace("/", "__")}.json` — {f}' for f in touched_files)}

## Impact set ({len(impact_rows)} symbols, max BFS depth {max_depth})

See `impact/impact-set.json` for the full ranked list, `impact/by-file.json`
for a per-file rollup.
"""
    (out_dir / "README.md").write_text(readme)
