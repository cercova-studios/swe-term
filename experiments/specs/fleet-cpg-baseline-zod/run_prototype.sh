#!/usr/bin/env bash
# Experiment-specific prototype runner for fleet-cpg-baseline-zod.
#
# The core experimentctl tool only validates and preregisters manifests
# (see experiments/README.md — "The execution runner is intentionally not
# part of this first scaffold"). This script is the reviewed procedure that
# materializes the frozen source, builds the CPG, and runs the golden query
# set for the `cold` and `warm` variants declared in manifest.json.
#
# Usage: run_prototype.sh <run-id>
# Writes: experiments/runs/fleet-cpg-baseline-zod/<run-id>/{materialized,artifacts,results.jsonl}
set -euo pipefail

RUN_ID="${1:?usage: run_prototype.sh <run-id>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SPEC_DIR="$REPO_ROOT/experiments/specs/fleet-cpg-baseline-zod"
RUN_DIR="$REPO_ROOT/experiments/runs/fleet-cpg-baseline-zod/$RUN_ID"
REVISION="b801439b5fb160d651084ff7d7c27a61e24a7334"
REPO_URL="https://github.com/colinhacks/zod.git"

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

# cold build: fresh jssrc2cpg parse, timed with peak RSS (BSD time -l).
/usr/bin/time -l "$JOERN_CLI/jssrc2cpg.sh" \
  "$RUN_DIR/materialized/repo/packages/zod/src" \
  -o "$RUN_DIR/artifacts/zod.cpg.bin" \
  > "$RUN_DIR/artifacts/cold_build.log" 2>&1

results="$RUN_DIR/results.jsonl"
: > "$results"

# variant: cold — first time this CPG is opened; no prior workspace project,
# no warmed page cache for the graph's on-disk representation.
run_variant() {
  local variant="$1" log="$2"
  local start_ns end_ns wall_ms
  start_ns=$(date +%s%N)
  (cd "$RUN_DIR" && "$JOERN_CLI/joern" --script "$SPEC_DIR/queries.sc" artifacts/zod.cpg.bin >"$log" 2>&1) || true
  end_ns=$(date +%s%N)
  wall_ms=$(( (end_ns - start_ns) / 1000000 ))
  # primary metric: full process wall time (JVM start, CPG load/overlay
  # recompute-or-reuse, all 8 queries, JVM teardown) — the per-query
  # System.nanoTime deltas below run entirely after the CPG is already
  # memory-resident, so they cannot see an OS-page-cache effect; process
  # wall time is the level at which cold vs warm should actually differ.
  echo "{\"query_id\":\"_process_wall_time\",\"category\":\"process\",\"latency_ms\":$wall_ms,\"result_count\":-1,\"status\":\"ok\",\"variant\":\"$variant\",\"run_id\":\"$RUN_ID\"}" >> "$results"
  grep -E '^\{"query_id"' "$log" | jq -c '. + {variant:"'"$variant"'", run_id: "'"$RUN_ID"'"}' >> "$results"
}

rm -rf "$RUN_DIR/workspace"
# variant: cold — first time this CPG is opened; no prior workspace project.
run_variant cold "$RUN_DIR/artifacts/cold_query.log"
# variant: warm — same workspace project reused; overlays already computed,
# OS page cache warm from the cold run immediately prior.
run_variant warm "$RUN_DIR/artifacts/warm_query.log"

echo "wrote $results"
