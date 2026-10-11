#!/usr/bin/env bash
# Phase 3 differential-base driver for fleet-cpg-phase3-differential-zod.
#
# Walks a real 7-commit chain on zod's own history (oldest to newest,
# every commit touching packages/zod/src), maintaining ONE base snapshot
# incrementally step by step -- never a full re-scan after the initial
# build. At every step, verifies the incrementally-updated L1 facts AND
# an L2 blast-radius relation computed on them against a from-scratch
# extraction at that commit. One JSON row per step appended to
# results.jsonl; a step's correctness failure does not stop the chain --
# a real differential base wouldn't self-heal either, and continuing is
# the honest simulation of that failure mode.
#
# Usage: run_phase3.sh <run-id> <bucket: short-chain|full-chain>
set -euo pipefail

RUN_ID="${1:?usage: run_phase3.sh <run-id> <bucket>}"
BUCKET="${2:?usage: run_phase3.sh <run-id> <bucket>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SPEC_DIR="$REPO_ROOT/experiments/specs/fleet-cpg-phase3-differential-zod"
SHARED_REPO="$REPO_ROOT/experiments/runs/fleet-cpg-phase3-differential-zod/shared/repo"
RUN_STORE="$REPO_ROOT/experiments/runs/fleet-cpg-phase3-differential-zod/$RUN_ID/store"
RUN_DIR="$REPO_ROOT/experiments/runs/fleet-cpg-phase3-differential-zod/$RUN_ID"
: "${PYTHON:?set PYTHON to the venv interpreter with tree-sitter/duckdb installed}"

# Real chain, oldest to newest, every commit touching packages/zod/src.
FULL_CHAIN=(0a69bcb3d9554c6ec382ea9ba6b43c2421f3fa78 8e03380510db36fa6fda979fc78a375fdea8021c 212b941791e7faae078e17645eb612824fd8f79a 9a193aa24b4efa3b315b91d4c56c8bc385b8513f eab51ff3592b2d11d863f4ee4d5452f31a3de1b6 1a16102a494b03ce1df7b80b663ae2df465f419e 84e416fbf4740527bbc8f319634f4e1b065bb42c)

case "$BUCKET" in
  short-chain) CHAIN=("${FULL_CHAIN[@]:0:3}") ;;   # 2 steps
  full-chain)  CHAIN=("${FULL_CHAIN[@]}") ;;        # 6 steps
  *) echo "unknown bucket: $BUCKET (expected short-chain|full-chain)" >&2; exit 2 ;;
esac

mkdir -p "$RUN_DIR"
results="$RUN_DIR/results.jsonl"
rm -f "$results"

cd "$SHARED_REPO"

C0="${CHAIN[0]}"
git checkout --quiet "$C0"
"$PYTHON" "$SPEC_DIR/differential_base.py" init packages/zod/src --store "$RUN_STORE" --tag "$C0" >/dev/null

prev="$C0"
step_n=0
for cur in "${CHAIN[@]:1}"; do
  step_n=$((step_n + 1))
  changed_rel=$(git diff --name-only "$prev..$cur" -- packages/zod/src | sed 's#^packages/zod/src/##')
  git checkout --quiet "$cur"

  row=$("$PYTHON" "$SPEC_DIR/differential_base.py" step packages/zod/src $changed_rel \
    --store "$RUN_STORE" --base-tag "$C0" --to-tag "$cur" \
    --truth-root packages/zod/src)

  python3 - "$row" "$RUN_ID" "$BUCKET" "$step_n" "$prev" "$cur" <<'PYEOF' >> "$results"
import json, sys
row, run_id, bucket, step_n, prev, cur = sys.argv[1:7]
d = json.loads(row)
d["run_id"] = run_id
d["bucket"] = bucket
d["step"] = int(step_n)
d["from_sha"] = prev
d["to_sha"] = cur
print(json.dumps(d))
PYEOF

  prev="$cur"
done

git checkout --quiet main 2>/dev/null || true
echo "wrote $results ($step_n steps)"
