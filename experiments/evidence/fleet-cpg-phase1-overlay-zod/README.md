# Evidence: fleet-cpg-phase1-overlay-zod

Kind: benchmark · Status: complete · Promotion: proposed 2026-09-05, acceptance pending

## Preregistered manifest

- Spec: [`experiments/specs/fleet-cpg-phase1-overlay-zod/manifest.json`](../../specs/fleet-cpg-phase1-overlay-zod/manifest.json)
- Design reference: Fleet CPG Engine — Phase 1 (fact-store spike, extraction half)
- Source: zod @ `b801439b5fb160d651084ff7d7c27a61e24a7334`,
  content digest `sha256:5c20f08b0bc5ce42fd1458c1940218cdbf767f53686ffdfe31a7a4bb409e4005`
- Rubric digest: `sha256:d53e423157b47702748e672dd5c5f6f6e7cd31fc025c7c4cfb36dbd081ff68ef`
- Variants: real merged PRs `small` (#6511), `medium` (#6488), `large` (#5913) · 3 reps each

## Component metrics

[`summary.json`](summary.json), from the spec's evaluator.

| bucket | files | lines | median overlay extract | vs. full-repo 422.7ms |
|---|---|---|---|---|
| small | 2 | 8 | 10.9ms | 39× lower |
| medium | 3 | 38 | 22.8ms | 18× lower |
| large | 6 | 241 | 9.6ms | 44× lower |

All buckets pass the preregistered 10× rule; `defs`/`calls` counts exactly
stable across every repetition. **Hypothesis accepted.**

## Artifact references

Raw runs local only; Parquet outputs and the shared checkout removed after
`results.jsonl` was captured.

## Discordant cases

The 241-line PR was not the slowest; the 38-line PR was, because its touched
files hold 318 defs versus 89. Overlay cost tracks **touched-file size**, not
diff line count, because whole files are re-extracted. Carried into the plan
as the whole-file-vs-hunk decision with a measured trigger.

## Limitations (draft for human review)

- Three small PRs (≤6 files); no genuinely large PR tested.
- Syntactic, name-based extraction only — complexity class, not fact correctness, is what was measured.
- Extraction half only; the commit/merge half is `fleet-cpg-phase1-commit-zod`.

## Decision

Accept mechanism. Proposed architecture effect: supports the O(diff) overlay
contract for the analyzer adapter (§9); no invariant added.
