#!/usr/bin/env bash
# Phase 2 end-to-end driver for fleet-cpg-phase2-blast-radius-zod.
#
# Simulates one PR event, start to finish: base snapshot at the PR's true
# parent commit (same freshness discipline fleet-cpg-phase1-commit-zod's
# pilot forced), overlay extraction + commit + merge, blast-radius
# computation (new L2 relation), and context-packet materialization +
# stub-agent readback. One JSON row appended to results.jsonl.
#
# Usage: run_phase2.sh <run-id> <bucket-name> <merge-commit-sha>
set -euo pipefail

RUN_ID="${1:?usage: run_phase2.sh <run-id> <bucket> <sha>}"
BUCKET="${2:?usage: run_phase2.sh <run-id> <bucket> <sha>}"
SHA="${3:?usage: run_phase2.sh <run-id> <bucket> <sha>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SPEC_DIR="$REPO_ROOT/experiments/specs/fleet-cpg-phase2-blast-radius-zod"
SHARED_REPO="$REPO_ROOT/experiments/runs/fleet-cpg-phase2-blast-radius-zod/shared/repo"
STORE="$REPO_ROOT/experiments/runs/fleet-cpg-phase2-blast-radius-zod/shared/store"
RUN_DIR="$REPO_ROOT/experiments/runs/fleet-cpg-phase2-blast-radius-zod/$RUN_ID"
: "${PYTHON:?set PYTHON to the venv interpreter with tree-sitter/duckdb installed}"

mkdir -p "$RUN_DIR"
results="$RUN_DIR/results.jsonl"

cd "$SHARED_REPO"

PARENT_SHA=$(git rev-parse "$SHA^")
BASE_TAG="$PARENT_SHA"
if [ ! -f "$STORE/base/$BASE_TAG/defs.parquet" ]; then
  git checkout --quiet "$PARENT_SHA"
  "$PYTHON" "$SPEC_DIR/../fleet-cpg-phase1-commit-zod/commit_store.py" base packages/zod/src --store "$STORE" --tag "$BASE_TAG" >/dev/null
fi

changed_rel=$(git diff --name-only "$SHA^..$SHA" -- packages/zod/src | sed 's#^packages/zod/src/##')

git checkout --quiet "$SHA"

pr_id="${RUN_ID}_${BUCKET}"
row=$("$PYTHON" "$SPEC_DIR/blast_radius.py" packages/zod/src $changed_rel \
  --store "$STORE" --base-tag "$BASE_TAG" \
  --out-packet "$RUN_DIR/packet" --pr-id "$pr_id")

python3 - "$row" "$RUN_ID" "$BUCKET" <<'PYEOF' >> "$results"
import json, sys
row, run_id, bucket = sys.argv[1:4]
d = json.loads(row)
d["run_id"] = run_id
d["bucket"] = bucket
print(json.dumps(d))
PYEOF

git checkout --quiet main
echo "wrote $results ($BUCKET) -- packet at $RUN_DIR/packet"
