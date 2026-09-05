# Experiment: fleet-cpg-baseline-excalidraw

Status: complete

Kind: benchmark

## Design reference

[Fleet CPG Engine — Phase 0 (Baseline)](https://claude.ai/code/artifact/0c277fff-2382-485d-8316-31af8f4179e7).
Third of the four Phase 0 repos, the medium-scale TypeScript/React case.

## Hypothesis, null hypothesis, independent variable

Identical framing to `fleet-cpg-baseline-zod`, retargeted at
`packages/excalidraw`. This run checks whether a ~2× larger CPG (by LOC and by
peak RSS) changes the already-rejected cold/warm outcome.

## Frozen inputs

- Source: `https://github.com/excalidraw/excalidraw.git` @ `e1bb9ff8f8931e783c11d104abb8967ac6605c9a`
- Source content digest: `sha256:b56b1ba712ba9960140a88bca7959097e6a7c4a899e701d23924f9f05d69f616`
- Scope: `packages/excalidraw` only — ~173k lines of `.ts`/`.tsx`, matching the
  design doc's ~180k estimate closely (unlike zod's and fastapi's, which were
  both off)
- Evaluator rubric digest: `sha256:61bcb2cab58042894e754f0aa9818da0bc9acb71875a1919462f065cbefec0c4`

## Procedure

Same as `fleet-cpg-baseline-zod`, retargeted at `packages/excalidraw` with an
8-query set built from excalidraw's own vocabulary (`render`, `Scene`,
`Element`). Full procedure is `run_prototype.sh`.

## Results

Three repetitions, 8 queries × 2 variants, all `status: "ok"`.

| Metric | cold | warm |
|---|---|---|
| `process_wall_time_ms` (median of 3) | 13,398 | 13,420 |
| `process_wall_time_ms` (all 3) | 13,283 / 13,398 / 13,556 | 12,969 / 13,420 / 13,651 |

**Acceptance rule not met — hypothesis rejected**, same direction as zod and
fastapi.

Cold-build baseline:

| Metric | run-001 | run-002 | run-003 |
|---|---|---|---|
| wall time (s) | 6.65 | 6.74 | 6.94 |
| peak RSS (bytes) | 2,040,168,448 | 2,255,962,112 | 2,255,323,136 |

Cross-repo comparison against the two prior runs, cold-build only:

| Repo | LOC | wall time | peak RSS |
|---|---|---|---|
| fastapi | ~21k | ~1.6s | ~328MB |
| zod | ~84k | ~3.4s | ~1.2–1.5GB |
| excalidraw | ~173k | ~6.8s | ~2.0–2.3GB |

Wall time and RSS both scale roughly linearly with LOC across these three
points — no evidence yet of the superlinear blowup that would make the
"one huge repo" (home-assistant/core) a qualitatively different case rather
than an extrapolation of this line; that is precisely the question the fourth
repo is for.

## Discordant cases

`typehier-all-typedecls` returned **5,915 in one repetition and 5,916 in the
other two, in both the cold and the warm variant** — a `result_count` that is
not stable across repetitions of the same query on a pinned, unchanged source
revision. Per this experiment's own rubric, an unstable count "invalidates the
run" for that query. It does not appear to be caused by the cold/warm
distinction (both variants show the same split), and the other seven queries
were stable across all repetitions.

This is worth taking seriously rather than averaging away: at ~173k LOC,
`jssrc2cpg` is not perfectly deterministic across repeated builds of the same
pinned revision. That is a direct, measured counterexample to the "incremental
result ≡ from-scratch result, always" invariant the Fleet CPG Engine design
doc treats as a hard requirement borrowed from Glean — Joern itself doesn't
clear that bar at this scale, which is exactly the kind of gap the design
doc's differential golden-query tests (Build order, Phase 3) are meant to
catch before shipping.

Not investigated further here (root cause could be file-enumeration order,
a nondeterministic pass, or parallelism inside `jssrc2cpg` — no attempt was
made to bisect it): out of scope for a cost/latency benchmark, but flagged as
a candidate for its own follow-up experiment.

## Limitations

Same as the other two Phase 0 runs (3 repetitions, unpinned Joern release,
reduced 8-query set). The cross-repo LOC/wall-time/RSS comparison above is
3 points on a line — suggestive, not a fitted scaling law; home-assistant/core
sits far enough outside this range that linear extrapolation from these three
should not be trusted without measuring it directly.

## Decision

- [ ] accept mechanism for a bounded follow-up
- [x] reject hypothesis (consistent with zod and fastapi)
- [ ] revise and preregister a new experiment (candidate: bisect the
      `typehier-all-typedecls` nondeterminism — is it `jssrc2cpg` parallelism,
      file-enumeration order, or something in the type-recovery pass?)
- [ ] propose architecture promotion with human approval
