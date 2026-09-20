# Evidence: fleet-cpg-phase2-blast-radius-zod

Kind: benchmark · Status: complete · Promotion: proposed 2026-09-05, acceptance pending

## Preregistered manifest

- Spec: [`experiments/specs/fleet-cpg-phase2-blast-radius-zod/manifest.json`](../../specs/fleet-cpg-phase2-blast-radius-zod/manifest.json)
- Design reference: Fleet CPG Engine — Phase 2 (prove the value path)
- Source: zod @ `b801439b5fb160d651084ff7d7c27a61e24a7334`,
  content digest `sha256:5c20f08b0bc5ce42fd1458c1940218cdbf767f53686ffdfe31a7a4bb409e4005`
- Rubric digest: `sha256:0d7d04feaaa032686462f0c65f06e6eaf3e00873189b8c60534cc48b2c62b8af`
- Variants: `small`, `medium`, `large` · 3 reps each · preregistered before official runs

## Component metrics

[`summary.json`](summary.json), from the spec's evaluator.

| bucket | median end-to-end | impact set | max depth | packet round-trip |
|---|---|---|---|---|
| small | 457ms | 0 | 0 | 3/3 |
| medium | 460ms | 573 | 3 | 3/3 |
| large | 246ms | 729 | 5 | 3/3 |

Zero manual steps in 9/9 runs; every stage other than `commit_write`
(DuckDB connection overhead) under 25ms. **Hypothesis accepted.**

## Artifact references

One representative packet per bucket retained locally
(`experiments/runs/fleet-cpg-phase2-blast-radius-zod/run-00{1,4,7}/packet/`):
`README.md`, `touched/*.json`, `impact/impact-set.json`, `impact/by-file.json`.

## Discordant cases

**Concrete name-resolution false positive.** `v3/helpers/util.ts:joinValues`
entered the impact set at distance 1 because its `array.map(...)` call matched
a touched function literally named `map` in `v4/classic/schemas.ts:1992`.
Hand-verified. This is the heuristic tier behaving as specified, and the first
fixture for the compiler-integrated-resolution upgrade in the plan. The
`small` bucket's empty impact set was hand-verified as correct (barrel file +
test-local helpers), including a near-miss on `callSite` that turned out to be
independently defined same-named helpers.

## Limitations (draft for human review)

- No independent ground truth for blast radius at this scale; correctness is a two-bucket spot check.
- The 1s acceptance bar is a sanity threshold, not the harness's readiness SLO.
- Ranking is BFS distance only.

## Decision

Accept mechanism. Proposed architecture effect: the packet file tree plus
`Provenance` as the adapter's output contract (§5/§9); three-valued scope per
§10 inv. 11 is a consequence of the false-positive finding.
