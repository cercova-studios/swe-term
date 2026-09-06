# Evidence: fleet-cpg-baseline-fastapi

Kind: benchmark · Status: complete · Promotion: proposed 2026-09-05, acceptance pending

## Preregistered manifest

- Spec: [`experiments/specs/fleet-cpg-baseline-fastapi/manifest.json`](../../specs/fleet-cpg-baseline-fastapi/manifest.json)
- Source: `https://github.com/fastapi/fastapi.git` @ `49033471594ea5d99a80abdf1043231b7791ee49`,
  content digest `sha256:ef12eeab5598f196e46705ec0c840260b4ed436ea196aadf68f077581359db2b`
- Rubric digest: `sha256:61bcb2cab58042894e754f0aa9818da0bc9acb71875a1919462f065cbefec0c4`
- Variants: `cold`, `warm` · 3 repetitions · 54 terminal rows, 0 errors

## Component metrics

[`summary.json`](summary.json), from the spec's evaluator.

- `process_wall_time_ms`: cold median **7,254ms**, warm median **7,611ms** —
  acceptance rule not met → hypothesis rejected.
- Cold build (`pysrc2cpg`, ~21k LOC in `fastapi/`): wall 1.55–1.70s, peak RSS ~328MB.
- Smallest repo in the set; startup tax is ~82% of process wall time here.

## Artifact references

Raw runs local only (`experiments/runs/fleet-cpg-baseline-fastapi/run-00{1,2,3}/`);
`results.jsonl` and gzipped build log retained.

## Discordant cases

None. Result counts stable across all invocations.

## Limitations (draft for human review)

- Scope is the `fastapi/` package, not the whole repository (tests and docs excluded).
- Python resolution quality of `pysrc2cpg` was not graded here — only timing and stability.
- Same single-machine caveat as the other baselines.

## Decision

Accept as baseline evidence. No invariant affected.
