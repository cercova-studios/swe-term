#!/usr/bin/env bash
# Phase 1 overlay-COMMIT driver for fleet-cpg-phase1-commit-zod.
#
# Builds (once, cached by tag) a base snapshot at the pinned revision, then
# for a given real merged PR: checks out its merge commit, extracts an
# overlay layer scoped to only the files it touched, commits that layer to
# the store (no base copy), merges base+overlay at query time, and verifies
# the merged view against a from-scratch full-repo extraction at the PR's
# head commit (ground truth). One JSON row is appended to results.jsonl.
#
# Usage: run_commit.sh <run-id> <bucket-name> <merge-commit-sha>
set -euo pipefail

RUN_ID="${1:?usage: run_commit.sh <run-id> <bucket> <sha>}"
BUCKET="${2:?usage: run_commit.sh <run-id> <bucket> <sha>}"
SHA="${3:?usage: run_commit.sh <run-id> <bucket> <sha>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SPEC_DIR="$REPO_ROOT/experiments/specs/fleet-cpg-phase1-commit-zod"
SHARED_REPO="$REPO_ROOT/experiments/runs/fleet-cpg-phase1-commit-zod/shared/repo"
STORE="$REPO_ROOT/experiments/runs/fleet-cpg-phase1-commit-zod/shared/store"
RUN_DIR="$REPO_ROOT/experiments/runs/fleet-cpg-phase1-commit-zod/$RUN_ID"
: "${PYTHON:?set PYTHON to the venv interpreter with tree-sitter/duckdb installed}"

mkdir -p "$RUN_DIR"
results="$RUN_DIR/results.jsonl"

cd "$SHARED_REPO"

# The base snapshot MUST be built at the PR's true parent commit, not some
# other pinned revision -- an unrelated file changed anywhere between the
# base's revision and the PR's real parent silently corrupts the merged
# view for that file, even though it was never "touched" by this PR. This
# is not a spike bug; it's the correctness constraint that motivates the
# design doc's <24h freshness SLO. Discovered via pilot-001 (see README).
# Cached per parent SHA so repetitions of the same bucket reuse it.
PARENT_SHA=$(git rev-parse "$SHA^")
BASE_TAG="$PARENT_SHA"
if [ ! -f "$STORE/base/$BASE_TAG/defs.parquet" ]; then
  git checkout --quiet "$PARENT_SHA"
  "$PYTHON" "$SPEC_DIR/commit_store.py" base packages/zod/src --store "$STORE" --tag "$BASE_TAG"
fi

changed_rel=$(git diff --name-only "$SHA^..$SHA" -- packages/zod/src | sed 's#^packages/zod/src/##')

git checkout --quiet "$SHA"

row=$("$PYTHON" "$SPEC_DIR/commit_store.py" overlay packages/zod/src $changed_rel \
  --store "$STORE" --base-tag "$BASE_TAG" \
  --run-id "$RUN_ID" --bucket "$BUCKET" \
  --truth-root packages/zod/src)

python3 - "$row" "$RUN_ID" <<'PYEOF' >> "$results"
import json, sys
row, run_id = sys.argv[1:3]
d = json.loads(row)
d["run_id"] = run_id
print(json.dumps(d))
PYEOF

git checkout --quiet main
echo "wrote $results ($BUCKET, $(echo "$changed_rel" | wc -l | tr -d ' ') files)"
