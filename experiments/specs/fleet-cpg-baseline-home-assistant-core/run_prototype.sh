#!/usr/bin/env bash
# Experiment-specific prototype runner for fleet-cpg-baseline-home-assistant-core.
# See fleet-cpg-baseline-zod/run_prototype.sh for the annotated original.
# Differences: shallow git clone (--depth 1 — this repo's full history is
# ~850MB and irrelevant to a single pinned-revision snapshot) and a capped
# JVM heap (_JAVA_OPTIONS=-Xmx24g) given the CPG's measured ~12.3GB peak RSS
# on cold build.
#
# Usage: run_prototype.sh <run-id>
set -euo pipefail

RUN_ID="${1:?usage: run_prototype.sh <run-id>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SPEC_DIR="$REPO_ROOT/experiments/specs/fleet-cpg-baseline-home-assistant-core"
RUN_DIR="$REPO_ROOT/experiments/runs/fleet-cpg-baseline-home-assistant-core/$RUN_ID"
REVISION="6ad726ba56d517e56536cdd6fa2ba9f358bbc0ef"
REPO_URL="https://github.com/home-assistant/core.git"

: "${JOERN_CLI:?set JOERN_CLI to the joern-cli directory of a local Joern install}"
export _JAVA_OPTIONS="${_JAVA_OPTIONS:--Xmx24g}"

echo "== run $RUN_ID =="
mkdir -p "$RUN_DIR/materialized" "$RUN_DIR/artifacts"

if [ ! -d "$RUN_DIR/materialized/repo" ]; then
  git clone --quiet --depth 1 --no-tags "$REPO_URL" "$RUN_DIR/materialized/repo"
fi
actual_rev="$(git -C "$RUN_DIR/materialized/repo" rev-parse HEAD)"
if [ "$actual_rev" != "$REVISION" ]; then
  echo "revision mismatch: expected $REVISION, got $actual_rev (shallow clone tracks whatever was HEAD at clone time — re-clone if this repo has since moved)" >&2
  exit 1
fi

/usr/bin/time -l "$JOERN_CLI/pysrc2cpg" \
  "$RUN_DIR/materialized/repo/homeassistant" \
  -o "$RUN_DIR/artifacts/home-assistant-core.cpg.bin" \
  > "$RUN_DIR/artifacts/cold_build.log" 2>&1

results="$RUN_DIR/results.jsonl"
: > "$results"

run_variant() {
  local variant="$1" log="$2"
  local start_ns end_ns wall_ms
  start_ns=$(date +%s%N)
  (cd "$RUN_DIR" && "$JOERN_CLI/joern" --script "$SPEC_DIR/queries.sc" artifacts/home-assistant-core.cpg.bin >"$log" 2>&1) || true
  end_ns=$(date +%s%N)
  wall_ms=$(( (end_ns - start_ns) / 1000000 ))
  echo "{\"query_id\":\"_process_wall_time\",\"category\":\"process\",\"latency_ms\":$wall_ms,\"result_count\":-1,\"status\":\"ok\",\"variant\":\"$variant\",\"run_id\":\"$RUN_ID\"}" >> "$results"
  grep -E '^\{"query_id"' "$log" | jq -c '. + {variant:"'"$variant"'", run_id: "'"$RUN_ID"'"}' >> "$results"
}

rm -rf "$RUN_DIR/workspace"
run_variant cold "$RUN_DIR/artifacts/cold_query.log"
run_variant warm "$RUN_DIR/artifacts/warm_query.log"

echo "wrote $results"
