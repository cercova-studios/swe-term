#!/usr/bin/env bash
# Experiment-specific prototype runner for fleet-cpg-baseline-fastapi.
# See fleet-cpg-baseline-zod/run_prototype.sh for the annotated original;
# this is the same procedure retargeted at fastapi's fastapi/ package via
# pysrc2cpg instead of jssrc2cpg.sh.
#
# Usage: run_prototype.sh <run-id>
set -euo pipefail

RUN_ID="${1:?usage: run_prototype.sh <run-id>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SPEC_DIR="$REPO_ROOT/experiments/specs/fleet-cpg-baseline-fastapi"
RUN_DIR="$REPO_ROOT/experiments/runs/fleet-cpg-baseline-fastapi/$RUN_ID"
REVISION="49033471594ea5d99a80abdf1043231b7791ee49"
REPO_URL="https://github.com/fastapi/fastapi.git"

: "${JOERN_CLI:?set JOERN_CLI to the joern-cli directory of a local Joern install}"

echo "== run $RUN_ID =="
mkdir -p "$RUN_DIR/materialized" "$RUN_DIR/artifacts"

if [ ! -d "$RUN_DIR/materialized/repo" ]; then
  git clone --quiet "$REPO_URL" "$RUN_DIR/materialized/repo"
fi
git -C "$RUN_DIR/materialized/repo" checkout --quiet "$REVISION"
actual_rev="$(git -C "$RUN_DIR/materialized/repo" rev-parse HEAD)"
if [ "$actual_rev" != "$REVISION" ]; then
  echo "revision mismatch: expected $REVISION, got $actual_rev" >&2
  exit 1
fi

/usr/bin/time -l "$JOERN_CLI/pysrc2cpg" \
  "$RUN_DIR/materialized/repo/fastapi" \
  -o "$RUN_DIR/artifacts/fastapi.cpg.bin" \
  > "$RUN_DIR/artifacts/cold_build.log" 2>&1

results="$RUN_DIR/results.jsonl"
: > "$results"

run_variant() {
  local variant="$1" log="$2"
  local start_ns end_ns wall_ms
  start_ns=$(date +%s%N)
  (cd "$RUN_DIR" && "$JOERN_CLI/joern" --script "$SPEC_DIR/queries.sc" artifacts/fastapi.cpg.bin >"$log" 2>&1) || true
  end_ns=$(date +%s%N)
  wall_ms=$(( (end_ns - start_ns) / 1000000 ))
  echo "{\"query_id\":\"_process_wall_time\",\"category\":\"process\",\"latency_ms\":$wall_ms,\"result_count\":-1,\"status\":\"ok\",\"variant\":\"$variant\",\"run_id\":\"$RUN_ID\"}" >> "$results"
  grep -E '^\{"query_id"' "$log" | jq -c '. + {variant:"'"$variant"'", run_id: "'"$RUN_ID"'"}' >> "$results"
}

rm -rf "$RUN_DIR/workspace"
run_variant cold "$RUN_DIR/artifacts/cold_query.log"
run_variant warm "$RUN_DIR/artifacts/warm_query.log"

echo "wrote $results"
