# Experiment: fleet-cpg-phase3-differential-zod

Status: complete

Kind: benchmark

## Design reference

[Fleet CPG Engine — Phase 3 (Prove steady-state operation)](https://claude.ai/code/artifact/0c277fff-2382-485d-8316-31af8f4179e7).
Phase 3's gate is "correctness under continuous update": replace full
re-index on push with differential base maintenance, and pass golden-query
differential tests — incremental result equals from-scratch result, on
every query, after every push, zero exceptions. Phases 1 and 2 both held
the base fixed and tested one overlay against it; this is the first
experiment that tests a *sequence* of real pushes.

## Hypothesis, null hypothesis, independent variable

**Hypothesis**: walking a real 7-commit chain on zod's own history (every
commit touching `packages/zod/src`), maintaining one base incrementally
step by step with no full re-scan after the initial build, both L1 facts
and an L2 blast-radius relation computed on the incrementally-maintained
base match a from-scratch full extraction at every single step, with
per-step update cost that does not grow across the chain.

**Null hypothesis**: any step's incrementally-maintained facts or blast
radius diverge from a from-scratch extraction at that commit, or per-step
update cost grows as the chain progresses.

**Independent variable**: chain length — `short-chain` (3 commits, 2
steps) vs. `full-chain` (7 commits, 6 steps), both real prefixes of the
same zod commit sequence, each repeated 3 times.

## Mechanism under test — corrected mid-pilot

**First attempt (discarded): re-flattening the entire base into a new full
snapshot after every step.** The first pilot run measured `write_wall_time_ms`
of ~8,500ms per step — write cost scaling with total repo size (1,384 defs,
43,557 calls), not with that step's diff. This directly contradicted the
claim this phase exists to test, and is exactly the anti-pattern the
design doc's LSM-style storage section exists to avoid ("differential
updates append small segments, background compaction merges them" —
compaction is periodic and amortized, not done on every push).

**Corrected mechanism: layered (LSM-style) merge.** Each step now writes
*only* its own small delta layer (defs/calls for the files that step
changed) to `store/layers/<sha>/`, appended to an ordered `layers.json`
manifest — the base at `store/base/<initial-sha>/` is never rewritten. The
current merged view is reconstructed at query time via a newest-wins union
chain: iterate delta layers newest→oldest, each excluding files already
claimed by a newer layer, with the base as the catch-all at the bottom.
This is genuinely O(diff) to write; the union-chain query is the read-path
cost of *not* compacting, tracked as its own metric (`merge_query_wall_time_ms`),
not conflated with write cost (`layer_write_wall_time_ms`).

Full procedure is `run_phase3.sh` (walks the chain, calling `init` once and
`step` per commit) plus `differential_base.py` (extraction, layered store,
merge-query construction, verification against ground truth).

## Results

**Hypothesis accepted. All 6 repetitions (2 variants × 3 reps), covering 24
step-verifications total, matched ground truth exactly** — L1 facts and L2
blast radius both correct at every single step, zero exceptions.

| step (position) | files changed | median incremental update | median merge-query | n layers at this step |
|---|---|---|---|---|
| 1 | 1 | 8.0ms | 17.0ms | 1 |
| 2 | 3 | 301–305ms (both variants) | 17.5ms | 2 |
| 3 | 1 | 7.8ms | 17.0ms | 3 |
| 4 | 3 | 555.0ms | 17.6ms | 4 |
| 5 | 1 | 7.8ms | 17.8ms | 5 |
| 6 | 3 | 182.8ms | 17.6ms | 6 |

Update cost per step tracks that step's own file count (1-file steps: ~8ms;
3-file steps: 175–555ms, variance dominated by the same DuckDB per-call
connection overhead documented in `fleet-cpg-phase1-commit-zod` and
`fleet-cpg-phase2-blast-radius-zod`) — **not chain position**. The cleanest
evidence: `short-chain`'s step 2 and `full-chain`'s step 2 are the identical
commit transition, and cost the same (301ms vs. 305ms) whether it's the
*last* step of a 2-step chain or an early step in a 6-step one. Merge-query
time stayed essentially flat (17.0–17.8ms) as the layer count grew from 1
to 6, with no visible upward trend at this depth.

## Limitations

- **6 layers is far too shallow to test the compaction question the
  design doc actually cares about.** A production base under continuous
  push accumulates an unbounded, ever-growing layer chain; this experiment
  cannot say whether merge-query cost stays flat at hundreds or thousands
  of uncompacted layers — only that no growth is visible yet at 6. This is
  a scale limitation to carry into a future experiment, not evidence that
  background compaction is unnecessary.
- Correctness verification re-derives ground truth via a full from-scratch
  extraction at every step — that's the validation method, deliberately
  not counted toward the primary `incremental_update_wall_time_ms` metric,
  same convention as Phase 1's `verify_wall_time_ms`.
- Same real-but-small commits as both Phase 1/2 spikes (1–3 files per
  step); a step with a much larger diff, or one that deletes files
  entirely (untested here — no commit in this chain deletes a touched
  file), is unexercised.
- A correctness failure at one step is designed to propagate forward
  uncorrected (no self-healing) — this held true by construction in this
  spike (every step matched, so it was never exercised), but the
  fail-forward behavior itself is asserted by the driver's design, not
  independently demonstrated by injecting a deliberate fault.
- No exploratory pilot data included in evidence: `pilot-001` (run twice —
  once with the discarded flatten-every-step mechanism, once with the
  corrected layered mechanism) informed the design correction above; its
  run directory was removed before the official 6 reps per this
  framework's pilot/official split.

## Decision

- [x] accept mechanism — differential base maintenance, implemented as a
      layered (LSM-style) store with query-time merge, holds correctness
      across a real 7-commit chain with zero exceptions, and keeps write
      cost independent of chain position. The compaction question (merge
      cost at real, unbounded layer depth) is explicitly out of scope for
      this spike and flagged as the next thing to measure, not assumed
      solved.
- [ ] reject hypothesis
- [ ] revise and preregister a new experiment
- [x] propose architecture promotion with human approval (proposed 2026-09-05; acceptance pending)

Promoting this into the design doc's Phase 3 gate as officially "passed"
still requires the human review this framework's evidence-promotion rules
call for, and a follow-on measurement of merge cost at real layer depth
before the LSM-without-compaction design can be trusted at production
scale. Phase 4 (a second language, testing the L0/L1 layering boundary) is
the next gate; it does not depend on the compaction question being closed
first.
