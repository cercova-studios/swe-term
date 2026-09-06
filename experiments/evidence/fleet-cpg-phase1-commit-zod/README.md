# Evidence: fleet-cpg-phase1-commit-zod

Kind: benchmark · Status: complete · Promotion: proposed 2026-09-05 as
**corroborated** evidence (see qualification note), acceptance pending

## Preregistered manifest

- Spec: [`experiments/specs/fleet-cpg-phase1-commit-zod/manifest.json`](../../specs/fleet-cpg-phase1-commit-zod/manifest.json)
- Design reference: Fleet CPG Engine — Phase 1 (overlay commit/merge half)
- Source: zod @ `b801439b5fb160d651084ff7d7c27a61e24a7334`,
  content digest `sha256:5c20f08b0bc5ce42fd1458c1940218cdbf767f53686ffdfe31a7a4bb409e4005`
- Rubric digest: `sha256:4dc968f584351b2bd4ed955c486574376ecaa18cdf8d5fb1fca130bb870335c1`
- Variants: `small`, `medium`, `large` (same real PRs as the extraction spike) · 3 reps each

## Qualification note (promotion rule 1)

The manifest's `status` was set to `preregistered` **after** the nine official
repetitions ran and were inspected. Hypothesis, rubric, acceptance rule, and
driver were frozen before `run-001` and never altered afterward, but the
letter of rule 1 was not met. This bundle is therefore offered as
corroborated evidence: the identical merge mechanism is re-verified against
ground truth at every step, under proper preregistration, in
`fleet-cpg-phase3-differential-zod` (24 step-verifications) and
`fleet-cpg-phase4-second-language` (6 runs, two languages). The human
accepting promotion should weigh it accordingly.

## Component metrics

[`summary.json`](summary.json), from the spec's evaluator.

- Primary `defs_match_truth` / `calls_match_truth`: **9/9 exact**, 0 missing, 0 extra.
- Overlay-to-base storage ratio: 5.0% / 12.5% / 5.6%.
- Merge query: 12.4–14.2ms, flat across buckets. Commit write 146–390ms,
  dominated by DuckDB per-connection overhead.

## Discordant cases

1. **Base-freshness bug (pilot, excluded from evidence).** Merging against a
   base one unrelated commit later than the diff's parent produced 7/7 defs and
   15/15 calls missing-and-extra. Fixed by keying the base to `SHA^`. This is
   the origin of the freshness precondition proposed for the §9 adapter contract.
2. The `medium` bucket's storage ratio (12.5%) missed the preregistered <10%
   secondary bar — same touched-file-size effect as the extraction spike;
   recorded as a missed bar, not relaxed.

## Limitations (draft for human review)

- Same three small PRs; no large PR.
- Commit-write timings overstate a resident-process path.
- Correctness compares symbol/file/kind/name and caller/callee/file/line tuples, not def spans.

## Decision

Accept mechanism, as corroborated evidence. Proposed architecture effect:
the freshness precondition text in §9 (not a new invariant).
