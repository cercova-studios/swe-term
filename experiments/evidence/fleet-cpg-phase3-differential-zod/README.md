# Evidence: fleet-cpg-phase3-differential-zod

Kind: benchmark · Status: complete · Promotion: proposed 2026-09-05, acceptance pending

## Preregistered manifest

- Spec: [`experiments/specs/fleet-cpg-phase3-differential-zod/manifest.json`](../../specs/fleet-cpg-phase3-differential-zod/manifest.json)
- Design reference: Fleet CPG Engine — Phase 3 (steady-state operation)
- Source: zod, real 7-commit chain `0a69bcb3 → … → 84e416fbf4740527bbc8f319634f4e1b065bb42c`,
  content digest over the ordered chain `sha256:a40aad7dbfdb4fe7a387e1ba82235193579e0e146d15f11e37884fcff0594002`
- Rubric digest: `sha256:ce73511475bc1b3eb8723d62556ded12138afd36c27f09ea63f3953d07a22fb8`
- Variants: `short-chain` (2 steps), `full-chain` (6 steps) · 3 reps each · preregistered before official runs

## Component metrics

[`summary.json`](summary.json), from the spec's evaluator.

- Primary `all_steps_correct_all_reps`: **true** — 24 step-verifications, L1
  facts and L2 blast radius both equal to from-scratch at every step; `step_failures: []`.
- Incremental update per step: 1-file steps ≈8ms; 3-file steps 183–555ms
  (DuckDB write overhead). The same commit transition costs 301ms as the last
  step of `short-chain` and 305ms as step 2 of `full-chain`.
- Merge-query time 17.0–17.8ms from 1 to 6 layers.

## Discordant cases

**Mechanism corrected during the pilot (pilot excluded).** The first design
re-flattened the whole base each step (~8.5s/step, O(repo)). Replaced by a
layered store with newest-wins query-time merge — the design document's
"append small segments, compact in the background" pattern, implemented
rather than assumed.

## Limitations (draft for human review)

- Six layers is far too shallow to say anything about compaction cost at real depth; this is the plan's slice-3 experiment.
- No step deletes a file; deletion tombstones are unexercised.
- Fail-forward (no self-healing) is asserted by construction, never triggered.

## Decision

Accept mechanism. Proposed architecture effect: layered store with
compaction as a separate operation, in the plan's storage design; no invariant added.
