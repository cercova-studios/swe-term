# Evidence: fleet-cpg-phase4-second-language

Kind: benchmark · Status: complete · Promotion: proposed 2026-09-05, acceptance pending

## Preregistered manifest

- Spec: [`experiments/specs/fleet-cpg-phase4-second-language/manifest.json`](../../specs/fleet-cpg-phase4-second-language/manifest.json)
- Design reference: Fleet CPG Engine — Phase 4 (the agnosticism test)
- Sources: zod @ `5ff9566508e6c95873d2648a5bdcc3a371f1b757` (manifest `source`);
  fastapi @ `d62354434b2e508fe89024213b220ca8e67dea5e` (python variant `config`)
- Rubric digest: `sha256:52acc0362e339126c57d9a9080b88cc0a810e112becc78ae7e87cfda368186f2`
- Variants: `typescript`, `python` · 3 reps each · preregistered before official runs

## Component metrics

[`summary.json`](summary.json), from the spec's evaluator.

- Primary `core_py_sha256`: one value across all 6 runs,
  `f5c4c402b8fb17a2a9140d992975cd7407fce2fa594599ee2d58c7f27e862d87`, and it
  matches the file on disk at evaluation time. `agnosticism_gate_passed: true`.
- TypeScript regression: `merged_defs=1388`, `merged_calls=43888`,
  `impact_set_size=0` — identical to `fleet-cpg-phase2-blast-radius-zod`.
- Python on a real fastapi PR: 502 defs, 2,649 calls, 76-symbol impact set,
  3/3 exact match with from-scratch extraction.

## Artifact references

One packet per language retained locally
(`experiments/runs/fleet-cpg-phase4-second-language/run-00{1,4}/packet/`).
`core.py`, `lowering_typescript.py`, `lowering_python.py` in the spec directory
are the reviewed artifacts; the hash above is over `core.py` as committed.

## Discordant cases

None on the claim. Hand-verified Python behavior: `@functools.wraps(cmgr)` at
`fastapi/routing.py:226` captured as a call through the generic recursion;
method/function/class split (93/52/14) matches the source.

## Limitations (draft for human review)

- Two languages falsify a boundary violation; they do not prove the boundary survives N.
- The hash proves no code changed, not that the L1 schema survives a structurally different language.
- Both languages share the heuristic resolution tier.

## Decision

Accept mechanism. Proposed architecture effect: the L0/L1 ↔ engine boundary
becomes a dependency-direction test in CI (plan, package layout); no invariant added.
