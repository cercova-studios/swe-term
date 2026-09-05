# Experiment: fleet-cpg-baseline-zod

Status: complete

Kind: benchmark

## Design reference

[Fleet CPG Engine — Phase 0 (Baseline)](https://claude.ai/code/artifact/0c277fff-2382-485d-8316-31af8f4179e7).
The Fleet CPG Engine is a proposed code-property-graph analysis engine for a
future multi-agent PR-review harness, and maps to swe-term's `Target`
analyzer/enrichment adapter extension point (`ARCHITECTURE.md` §9). Its
Phase 0 build-order step calls for measuring Joern's baseline cost on real
repositories before any of the engine's own O(diff) claims can be graded.
This experiment is that measurement for the first (smallest) of the four
repos named in that spec.

## Hypothesis

Full joern process wall time (JVM start, CPG open, golden-query execution,
teardown) is lower when Joern reuses an existing workspace project with a
warm OS page cache (variant `warm`) than when it opens the CPG for the first
time with no existing workspace project (variant `cold`), for zod's core
package (`packages/zod/src`) at a pinned revision.

## Null hypothesis

Warm-variant process wall time is not meaningfully lower than cold-variant
process wall time; any difference is within repetition noise.

## Independent variable

Workspace/page-cache state before the joern process starts. The CPG binary,
source revision, and query set are held fixed.

## Frozen inputs

- Source: `https://github.com/colinhacks/zod.git` @ `b801439b5fb160d651084ff7d7c27a61e24a7334`
- Source content digest: `sha256:5c20f08b0bc5ce42fd1458c1940218cdbf767f53686ffdfe31a7a4bb409e4005` (`git archive <rev> | sha256sum`)
- Scope: `packages/zod/src` only (the library, tests included; ~84k lines of
  `.ts`, measured with `find | wc -l` — `cloc` was not installed and the
  earlier ~25k estimate in the design doc was wrong by more than 3×)
- Joern version: whatever `joern-install.sh` resolved as latest at install
  time (2026-08-30); not pinned to a specific release tag in this run — a gap
  to close before treating any future rerun as comparable
- Evaluator rubric digest: `sha256:61bcb2cab58042894e754f0aa9818da0bc9acb71875a1919462f065cbefec0c4`

## Procedure

1. Clone the pinned revision fresh (or reuse if already materialized) into
   `experiments/runs/fleet-cpg-baseline-zod/<run-id>/materialized/repo`.
2. Cold build: `jssrc2cpg.sh packages/zod/src -o artifacts/zod.cpg.bin`, timed
   with `/usr/bin/time -l` for wall time and peak RSS.
3. Delete any existing joern workspace for this CPG.
4. Variant `cold`: run the golden query script (`queries.sc`, 8 queries across
   4 categories — symbol resolution, reference set, transitive call graph,
   type hierarchy) via `joern --script`, wall-clock the whole process, and
   record each query's in-process `System.nanoTime` latency.
5. Variant `warm`: immediately rerun the identical command — same workspace
   project, warm OS page cache.
6. Repeat steps 1–5 three times (`run-001..003`).

Full procedure is `experiments/specs/fleet-cpg-baseline-zod/run_prototype.sh`.

## Metrics

- Primary: `process_wall_time_ms` (cold vs. warm, per repetition)
- Secondary: `cold_build_wall_time_ms`, `cold_build_peak_rss_bytes`,
  `query_latency_p50_ms` by category
- Safety-critical counters that may not be averaged away: any query row with
  `status != "ok"`; any `(query_id, variant)` pair whose `result_count`
  changes across repetitions (a nondeterministic CPG would invalidate every
  other number in this run)

## Acceptance and stop conditions

Accept: median warm `process_wall_time_ms` ≤ 80% of median cold across all 3
repetitions. Reject: medians within 20% of each other, or warm is higher.
Cold-build wall time and peak RSS are reported as the Phase 0 baseline
deliverable regardless of which way this resolves.

## Risks and confinement

- Network: `inherit`, justified — `git clone` from github.com, and a one-time
  Joern release download via the project's official install script. No other
  endpoints contacted.
- Filesystem: writes confined to `experiments/runs/fleet-cpg-baseline-zod/`
  and a scratchpad Joern install outside the repo; no writes to the
  materialized clone beyond `git checkout`.
- Process: JVM-heavy (Joern); no containment beyond the `process` adapter —
  acceptable here since Joern and its inputs are not attacker-controlled in
  this run (contrast with the production extraction-worker sandboxing the
  design doc specifies for untrusted PR code).
- Secrets: none used or exposed.
- Requested vs. enforced: the `process` adapter enforces none of the above
  mechanically today; this run's isolation is procedural (a human-followed
  script), not a runner-enforced boundary. This is an accurate account of
  Phase 1 (deterministic local runner) not existing yet, not a claim that it
  does.

## Results

Three repetitions, 8 queries × 2 variants = 48 query rows + 6 process-level
rows per repetition, all `status: "ok"`, zero unstable `result_count`s.

| Metric | cold | warm |
|---|---|---|
| `process_wall_time_ms` (median of 3) | 7,575 | 7,997 |
| `process_wall_time_ms` (all 3) | 5,652 / 7,575 / 8,327 | 7,820 / 7,997 / 8,767 |

**Acceptance rule not met** — warm is not faster than cold; if anything it
trended slightly slower. **Hypothesis rejected.**

Cold-build baseline (the Phase 0 deliverable, independent of the rejected
hypothesis):

| Metric | run-001 | run-002 | run-003 |
|---|---|---|---|
| wall time (s) | 2.65 | 3.41 | 3.69 |
| peak RSS (bytes) | 1,407,172,608 | 1,503,969,280 | 1,136,230,400 |

Post-load, in-graph query latency (from the pilot run, consistent with the
official run's per-query rows): sub-millisecond to ~2–3 ms across all 8
golden queries, cold and warm alike — see Discordant cases.

## Discordant cases

The hypothesis was built around the wrong mechanism. Per-query
`System.nanoTime` timing runs entirely *after* the CPG is already
memory-resident inside the JVM — by that point the OS page-cache state that
existed before the process started can no longer affect anything being
measured. Both cold and warm per-query latencies land in the same
sub-millisecond-to-a-few-milliseconds band (see `experiments/runs/.../pilot-*`
for the exploratory run that surfaced this before preregistration).

The 7.5–8.8 second `process_wall_time_ms` figures are dominated by something
neither variant touches: Joern/Ammonite's fixed per-invocation JVM and script
REPL bootstrap. That startup tax is roughly constant regardless of workspace
or cache state, which is exactly why cold and warm come out statistically
indistinguishable — the independent variable this experiment manipulated
is dwarfed by a cost neither variant controls for.

## Limitations

- Sample size is 3 repetitions per variant — enough to see that warm isn't
  meaningfully faster, not enough to bound the true JVM-startup variance
  tightly.
- Joern's release version was not pinned; a rerun today may resolve a
  different build. Re-pin explicitly before treating any future run as
  comparable to this one.
- The golden query set here is a reduced 8-query subset (4 of the design
  doc's 6 categories: no interprocedural taint, no diff-scoped blast radius)
  chosen to get a first real number quickly, not the full ~80-query set the
  design doc specifies per repo.
- This run measured one repo (zod, small, TypeScript). The design doc's other
  three repos (fastapi, excalidraw, home-assistant/core) are not yet run;
  none of the cold-build numbers above should be assumed to hold at 10×–40×
  the LOC, especially the huge repo, which is precisely the point of running
  it separately rather than extrapolating.
- Joern does support a resident `--server` mode, which would eliminate the
  per-invocation JVM tax this run found dominant. That mode was not
  benchmarked here; it is the natural next experiment if a "would a resident
  Joern process close this gap" question becomes load-bearing for the design
  doc's argument.

## Decision

- [ ] accept mechanism for a bounded follow-up
- [x] reject hypothesis (cold vs. warm page-cache state, as operationalized
      here, does not meaningfully change process wall time)
- [ ] revise and preregister a new experiment (candidate: `joern --server`
      resident-process wall time vs. per-invocation cold-start, which is the
      comparison this run's discordant case actually motivates)
- [ ] propose architecture promotion with human approval

The cold-build wall-time and peak-RSS numbers stand as valid Phase 0 baseline
data independent of the rejected hypothesis, but promotion into the design
doc's baseline table has not been done here — that step requires the human
review this framework's evidence-promotion rules call for, and repeating this
procedure on the other three repos first.
