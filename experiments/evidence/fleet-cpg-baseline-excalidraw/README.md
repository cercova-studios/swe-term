# Evidence: fleet-cpg-baseline-excalidraw

Kind: benchmark · Status: complete · Promotion: proposed 2026-09-05, acceptance pending

## Preregistered manifest

- Spec: [`experiments/specs/fleet-cpg-baseline-excalidraw/manifest.json`](../../specs/fleet-cpg-baseline-excalidraw/manifest.json)
- Source: `https://github.com/excalidraw/excalidraw.git` @ `e1bb9ff8f8931e783c11d104abb8967ac6605c9a`,
  content digest `sha256:b56b1ba712ba9960140a88bca7959097e6a7c4a899e701d23924f9f05d69f616`
- Rubric digest: `sha256:61bcb2cab58042894e754f0aa9818da0bc9acb71875a1919462f065cbefec0c4`
- Variants: `cold`, `warm` · 3 repetitions · 54 terminal rows, 0 errors

## Component metrics

[`summary.json`](summary.json), from the spec's evaluator.

- `process_wall_time_ms`: cold median **13,398ms**, warm median **13,420ms** —
  acceptance rule not met → hypothesis rejected.
- Cold build (`jssrc2cpg`, ~173k LOC in `packages/excalidraw`): wall 6.65–6.94s,
  peak RSS 2.0–2.3GB.

## Artifact references

Raw runs local only (`experiments/runs/fleet-cpg-baseline-excalidraw/run-00{1,2,3}/`).

## Discordant cases

**`typehier-all-typedecls` returned 5,915 in one repetition and 5,916 in the
other two**, in both cold and warm variants, on a pinned, unchanged revision.
Recorded by the evaluator as `unstable_result_counts`. This is a measured
counterexample to "incremental ≡ from-scratch, always" — in Joern, not in the
design under test — and did not recur on the 3.03M-LOC repo. Root cause not
bisected (file-enumeration order, pass parallelism, and type recovery are the
suspects). Consequence adopted in the plan: the differential-equality property
is a CI test, never an assumption.

## Limitations (draft for human review)

- Nondeterminism observed once in three; not enough repetitions to estimate a rate.
- No bisect; the suspects above are hypotheses, not findings.

## Decision

Accept as baseline evidence, with the nondeterminism case flagged as a
standing risk for any CPG builder including this one. No invariant affected.
