# Experiment: fleet-cpg-baseline-fastapi

Status: complete

Kind: benchmark

## Design reference

[Fleet CPG Engine — Phase 0 (Baseline)](https://claude.ai/code/artifact/0c277fff-2382-485d-8316-31af8f4179e7).
Second of the four Phase 0 repos, and the first Python one — Joern's
documented weak language for dynamic-typing analysis.

## Hypothesis, null hypothesis, independent variable

Identical framing to `fleet-cpg-baseline-zod` (see that experiment's README
for the full statement), retargeted at fastapi's `fastapi/` package. This run
exists to check whether the zod result (JVM startup tax swamps any cache-state
signal) was a TypeScript-specific artifact or holds for Joern's Python
frontend too.

## Frozen inputs

- Source: `https://github.com/fastapi/fastapi.git` @ `49033471594ea5d99a80abdf1043231b7791ee49`
- Source content digest: `sha256:ef12eeab5598f196e46705ec0c840260b4ed436ea196aadf68f077581359db2b`
- Scope: `fastapi/` package only (~21k lines of `.py` — the design doc's
  "~100k incl. tests" figure was for the whole repo, not this scope; corrected
  here rather than carried forward silently)
- Evaluator rubric digest: `sha256:61bcb2cab58042894e754f0aa9818da0bc9acb71875a1919462f065cbefec0c4`

## Procedure

Same as `fleet-cpg-baseline-zod`, substituting `pysrc2cpg` for `jssrc2cpg.sh`
and an 8-query set built from fastapi's own vocabulary (`Depends`, `Router`,
`__init__`, `*Response*`) in place of zod's. Full procedure is
`run_prototype.sh`.

## Results

Three repetitions, 8 queries × 2 variants, all `status: "ok"`, zero unstable
`result_count`s.

| Metric | cold | warm |
|---|---|---|
| `process_wall_time_ms` (median of 3) | 7,254 | 7,611 |
| `process_wall_time_ms` (all 3) | 7,184 / 7,254 / 7,323 | 7,570 / 7,611 / 8,028 |

**Acceptance rule not met — hypothesis rejected**, same direction as zod: warm
is not faster than cold.

Cold-build baseline:

| Metric | run-001 | run-002 | run-003 |
|---|---|---|---|
| wall time (s) | 1.57 | 1.55 | 1.70 |
| peak RSS (bytes) | 328,105,984 | 327,876,608 | 328,564,736 |

## Discordant cases

None beyond the one already established by zod: per-query timing runs after
the CPG is memory-resident, so it cannot see OS-page-cache effects; the ~7.2–8s
`process_wall_time_ms` figures are dominated by fixed JVM/Ammonite startup,
essentially unchanged from zod's ~7.5–8.8s despite fastapi's CPG being ~4×
smaller (21k vs. 84k LOC) and its peak RSS ~4× lower (328MB vs. ~1.2GB).
That process-wall-time floor barely moving while repo size and memory use
both drop sharply is itself the finding: the tax is dominated by JVM/script
bootstrap, not CPG size, at least in this size range.

## Limitations

Same as `fleet-cpg-baseline-zod` (3 repetitions, unpinned Joern release,
reduced 8-query set, single repo). Python-specific: fastapi's own decorator
and dependency-injection idioms (`Depends`) are exercised by name match only;
no attempt was made to verify Joern's Python frontend resolves them to the
same ground truth a `pyright`-based check would give — that resolution-quality
question is out of scope for this cost/latency benchmark.

## Decision

- [ ] accept mechanism for a bounded follow-up
- [x] reject hypothesis (consistent with `fleet-cpg-baseline-zod`; the
      JVM-startup-tax explanation now holds across two languages)
- [ ] revise and preregister a new experiment
- [x] propose architecture promotion with human approval (proposed 2026-09-05; acceptance pending)
