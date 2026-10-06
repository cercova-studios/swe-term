# Evidence: fleet-cpg-baseline-zod

Kind: benchmark · Status: complete · Promotion: proposed 2026-09-05, acceptance pending

## Preregistered manifest

- Spec: [`experiments/specs/fleet-cpg-baseline-zod/manifest.json`](../../specs/fleet-cpg-baseline-zod/manifest.json)
- Source: `https://github.com/colinhacks/zod.git` @ `b801439b5fb160d651084ff7d7c27a61e24a7334`,
  content digest `sha256:5c20f08b0bc5ce42fd1458c1940218cdbf767f53686ffdfe31a7a4bb409e4005`
- Rubric digest: `sha256:61bcb2cab58042894e754f0aa9818da0bc9acb71875a1919462f065cbefec0c4`
- Variants: `cold`, `warm` (OS page-cache state) · 3 repetitions · 54 terminal rows, 0 errors

## Component metrics

Mechanically derived by the spec's own evaluator: [`summary.json`](summary.json).

- Primary `process_wall_time_ms` (one `joern --script` invocation over 8 golden
  queries): cold median **7,575ms**, warm median **7,997ms**. Preregistered
  acceptance rule (cold measurably slower than warm) **not met → hypothesis rejected**.
- Cold build (`jssrc2cpg`, ~84k LOC): wall 2.65 / 3.41 / 3.69s, peak RSS 1.1–1.5GB.
- Per-query in-graph latency: sub-millisecond to low-ms regardless of cache state.

## Artifact references

Raw runs (`experiments/runs/fleet-cpg-baseline-zod/run-00{1,2,3}/`) are local
only; `results.jsonl` and gzipped `cold_build.log` retained, CPG binaries and
checkout removed after extraction.

## Discordant cases

None in the result counts (all 8 queries stable across 6 invocations). The
discordance is between the hypothesis and the mechanism: page-cache state
cannot affect query latency because the CPG is JVM-resident by query time.
The dominant, unhypothesized cost is per-invocation JVM/Ammonite startup and
load (~7.5s), which the design's resident-reader architecture exists to remove.

## Limitations (draft for human review)

- One machine, no pinned container image; numbers are relative, not portable.
- 8 golden queries, spot-checked by hand against tsserver, not a full golden set.
- Joern `--server` mode (its own answer to the startup tax) was not measured.

## Decision

Accept as baseline evidence. Proposed architecture effect: none on invariants;
supports the resident-sidecar shape in the implementation plan.
