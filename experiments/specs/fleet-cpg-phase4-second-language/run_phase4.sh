#!/usr/bin/env bash
# Phase 4 driver for fleet-cpg-phase4-second-language.
#
# Runs the same core.py pipeline through two different lowering modules --
# TypeScript (zod, reproducing fleet-cpg-phase2-blast-radius-zod's small
# bucket as a regression check) and Python (a real merged fastapi PR,
# genuinely new). The gate: core.py's sha256 must be IDENTICAL across both
# languages -- computed and recorded by phase4_run.py itself, and asserted
# again independently by the evaluator against the file on disk.
#
# Usage: run_phase4.sh <run-id> <bucket: typescript|python>
set -euo pipefail

RUN_ID="${1:?usage: run_phase4.sh <run-id> <bucket>}"
BUCKET="${2:?usage: run_phase4.sh <run-id> <bucket>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SPEC_DIR="$REPO_ROOT/experiments/specs/fleet-cpg-phase4-second-language"
RUN_DIR="$REPO_ROOT/experiments/runs/fleet-cpg-phase4-second-language/$RUN_ID"
: "${PYTHON:?set PYTHON to the venv interpreter with tree-sitter/duckdb installed}"

export PYTHONPATH="$SPEC_DIR"

mkdir -p "$RUN_DIR"
results="$RUN_DIR/results.jsonl"

case "$BUCKET" in
  typescript)
    REPO="$REPO_ROOT/experiments/runs/fleet-cpg-phase4-second-language/shared/zod"
    SRC_SUBDIR="packages/zod/src"
    LOWERING="lowering_typescript"
    PR_SHA="5ff9566508e6c95873d2648a5bdcc3a371f1b757"
    ;;
  python)
    REPO="$REPO_ROOT/experiments/runs/fleet-cpg-phase4-second-language/shared/fastapi"
    SRC_SUBDIR="fastapi"
    LOWERING="lowering_python"
    PR_SHA="d62354434b2e508fe89024213b220ca8e67dea5e"
    ;;
  *) echo "unknown bucket: $BUCKET (expected typescript|python)" >&2; exit 2 ;;
esac

cd "$REPO"
PARENT=$(git rev-parse "$PR_SHA^")
git checkout --quiet "$PARENT"
changed_rel=$(git diff --name-only "HEAD..$PR_SHA" -- "$SRC_SUBDIR" | sed "s#^$SRC_SUBDIR/##")
git checkout --quiet "$PR_SHA"

pr_id="${RUN_ID}_${BUCKET}"
row=$("$PYTHON" "$SPEC_DIR/phase4_run.py" "$LOWERING" "$SRC_SUBDIR" $changed_rel \
  --store "$RUN_DIR/store" --base-tag "$PARENT" \
  --out-packet "$RUN_DIR/packet" --pr-id "$pr_id" --truth-root "$SRC_SUBDIR")

python3 - "$row" "$RUN_ID" "$BUCKET" <<'PYEOF' >> "$results"
import json, sys
row, run_id, bucket = sys.argv[1:4]
d = json.loads(row)
d["run_id"] = run_id
d["bucket"] = bucket
print(json.dumps(d))
PYEOF

git checkout --quiet main 2>/dev/null || true
echo "wrote $results ($BUCKET)"
