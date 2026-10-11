#!/usr/bin/env bash
# Phase 1 overlay-cost driver for fleet-cpg-phase1-overlay-zod.
#
# For a real merged zod PR (given as a merge-commit SHA), computes the files
# changed within packages/zod/src relative to the commit's parent, checks
# out the PR's head commit, and runs the extractor scoped to ONLY those
# changed files -- this is the "overlay build". Records files/lines changed
# alongside the overlay's extraction wall time so overlay cost can be
# checked against diff size rather than repo size.
#
# Usage: run_overlay.sh <run-id> <bucket-name> <merge-commit-sha>
set -euo pipefail

RUN_ID="${1:?usage: run_overlay.sh <run-id> <bucket> <sha>}"
BUCKET="${2:?usage: run_overlay.sh <run-id> <bucket> <sha>}"
SHA="${3:?usage: run_overlay.sh <run-id> <bucket> <sha>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SPEC_DIR="$REPO_ROOT/experiments/specs/fleet-cpg-phase1-overlay-zod"
SHARED_REPO="$REPO_ROOT/experiments/runs/fleet-cpg-phase1-overlay-zod/shared/repo"
RUN_DIR="$REPO_ROOT/experiments/runs/fleet-cpg-phase1-overlay-zod/$RUN_ID"
: "${PYTHON:?set PYTHON to the venv interpreter with tree-sitter/duckdb installed}"

mkdir -p "$RUN_DIR"
results="$RUN_DIR/results.jsonl"

cd "$SHARED_REPO"
diffstat=$(git diff --shortstat "$SHA^..$SHA" -- packages/zod/src)
files_changed=$(git diff --name-only "$SHA^..$SHA" -- packages/zod/src | wc -l | tr -d ' ')
changed_rel=$(git diff --name-only "$SHA^..$SHA" -- packages/zod/src | sed 's#^packages/zod/src/##')

lines_changed=$(echo "$diffstat" | grep -oE '[0-9]+ insertion' | grep -oE '[0-9]+' || echo 0)
del=$(echo "$diffstat" | grep -oE '[0-9]+ deletion' | grep -oE '[0-9]+' || echo 0)
total_lines=$((${lines_changed:-0} + ${del:-0}))

git checkout --quiet "$SHA"

extract_out=$("$PYTHON" "$SPEC_DIR/extractor.py" packages/zod/src $changed_rel \
  --out "$RUN_DIR/overlay_defs_${BUCKET}.parquet" "$RUN_DIR/overlay_calls_${BUCKET}.parquet")

python3 - "$extract_out" "$BUCKET" "$SHA" "$files_changed" "$total_lines" "$RUN_ID" <<'PYEOF' >> "$results"
import json, sys
extract_out, bucket, sha, files_changed, total_lines, run_id = sys.argv[1:7]
row = json.loads(extract_out)
row.update({
    "bucket": bucket,
    "sha": sha,
    "files_changed": int(files_changed),
    "lines_changed": int(total_lines),
    "run_id": run_id,
})
print(json.dumps(row))
PYEOF

git checkout --quiet main
echo "wrote $results ($BUCKET, $files_changed files, $total_lines lines)"
