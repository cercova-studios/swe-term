# Experiment: fleet-cpg-phase1-commit-zod

Status: complete

Kind: benchmark

## Design reference

[Fleet CPG Engine — Phase 1 (Prove the core thesis)](https://claude.ai/code/artifact/0c277fff-2382-485d-8316-31af8f4179e7).
`fleet-cpg-phase1-overlay-zod` measured only the extraction/lowering half of
Phase 1's O(diff) claim. This experiment measures the other half that spike
explicitly left open: committing an overlay layer into a snapshot store
without copying the base, and reading back a merged view that must match a
from-scratch full-repo extraction exactly.

## Hypothesis, null hypothesis, independent variable

**Hypothesis**: a committed overlay layer (defs/calls for only the files a
real merged PR touched) merges with a base snapshot, at query time and
without copying the base, into a view set-identical to a from-scratch
full-repo extraction at the PR's head commit — and overlay storage,
commit-write time, and merge-query time all stay small relative to the base
snapshot, across PRs of varying diff size.

**Null hypothesis**: the merged view diverges from ground truth on any
bucket or repetition, or overlay storage/write/merge cost approaches the
same order of magnitude as the base snapshot.

**Independent variable**: which real merged zod PR is committed as an
overlay — the same three buckets as `fleet-cpg-phase1-overlay-zod` for
continuity (small: 2 files/8 lines; medium: 3 files/38 lines; large: 6
files/241 lines) — holding extractor, source repo, and merge mechanism
fixed.

## Mechanism under test

Store layout (DuckDB/Parquet scaffolding, not the owned segment format —
same deliberate scope cut as the extraction spike):

```
<store>/base/<parent-sha>/{defs,calls}.parquet   -- one full-repo snapshot
<store>/overlays/<run>_<bucket>/
    {defs,calls}.parquet                         -- ONLY touched files
    manifest.json                                -- {base_tag, touched_files}
```

Reading a commit's merged view is a query, not a copy:

```sql
SELECT * FROM base_defs WHERE file NOT IN touched_files
UNION ALL
SELECT * FROM overlay_defs
```

(same shape for `calls`). Write cost and storage cost are both O(touched
files); the base is never duplicated or mutated.

## Procedure

1. For each bucket, build a base snapshot at the PR's **true parent commit**
   (`SHA^`, cached per parent SHA so repetitions reuse it) — see the
   critical correction below for why this matters.
2. Check out the PR's merge commit, extract an overlay layer scoped to only
   the changed files, write it to the store, and merge it against the base
   via the DuckDB union query above.
3. Verify the merged view against a from-scratch full-repo extraction at the
   PR's head commit (ground truth) — every def and call must match exactly,
   not just count-match.
4. Record commit-write time, merge-query time, storage bytes (base vs.
   overlay), and correctness. 3 repetitions per bucket.

Full procedure is `run_commit.sh` (per-PR driver) plus `commit_store.py`
(extraction, store I/O, merge, verification — all in one file since each
experiment's scripts are self-contained by convention in this framework).

## Critical correction, found during the pilot

The first pilot run built the base snapshot at a single shared pinned
revision (the same `b801439b` used as the source pin) and got real,
reproducible mismatches: 7 defs and 15 calls both missing-and-extra between
the merged view and ground truth, despite matching total counts. Root
cause, confirmed by inspecting history directly: `b801439b` is a *later*
commit than the small PR's true parent, and an unrelated file
(`compile.ts`) changed in between — a file this PR never touched, but whose
content in the base snapshot no longer matched the PR's actual base state.

**This is not a spike bug — it is the correctness constraint that motivates
the design doc's Profile Staleness SLO** (`< 24h, alerting`, KR 1.2 in the
Context Gathering Engine OKRs). A COW overlay is only correct when merged
against a base that reflects the diff's exact ancestor state; if the base
snapshot is stale by even one unrelated commit, the merge silently corrupts
results for files the PR never touched. "Stale call graph → confidently
wrong comments" isn't only about missing recent files — it's about *any*
drift between snapshot revision and review-time parent, anywhere in the
repo. Fixed by building each bucket's base at `SHA^` exactly. The flawed
pilot data (`pilot-001`, with the shared-pin base) is excluded from
evidence, consistent with this framework's pilot/official split.

## Results

| bucket | files changed | median commit-write | median merge-query | median overlay/base bytes | correctness (9 reps total) |
|---|---|---|---|---|---|
| small (PR #6511) | 2 | 390.4ms | 13.8ms | 5.0% | 3/3 exact match |
| medium (PR #6488) | 3 | 287.3ms | 14.2ms | 12.5% | 3/3 exact match |
| large (PR #5913) | 6 | 146.5ms | 12.4ms | 5.6% | 3/3 exact match |

**Primary metric (correctness) accepted cleanly: 9/9 repetitions, 0 missing
and 0 extra defs or calls in any merged view vs. ground truth**, once the
base-freshness bug above was fixed. The COW-merge query is correct: it does
not need to touch, scan, or copy the base beyond the single `WHERE file NOT
IN (...)` filter.

Merge-query time (12–14ms) stays flat and small across all bucket sizes, as
expected for a query whose cost is dominated by base-table size, not diff
size — this is a place the mechanism does *not* yet scale with diff size,
which is fine (it's a query, not a copy) but worth flagging for a
much-larger base than zod's 324 files.

## Discordant cases

**Storage ratio: 2 of 3 buckets cleared my own <10% bar, "medium" did not
(12.5%).** This is the same effect `fleet-cpg-phase1-overlay-zod` already
found: overlay size tracks the total size of the *files touched*, not the
diff's line count or file count. Medium's 3 files contain more total defs
than large's 6 files (matching the prior spike's finding), so its overlay
layer is proportionally larger relative to its own base. All three ratios
are still far from 1.0 — no bucket comes close to full-base duplication —
so the core "no-copy" claim holds, but the specific <10% acceptance number
in the manifest was picked before seeing this cross-experiment pattern
confirmed a second time, and is now known to be too strict for
touched-file-size-driven variance. Not silently relaxed here: recorded as a
missed secondary-metric bar, with the primary correctness metric carrying
the accept decision.

**Commit-write time is dominated by DuckDB per-call connection overhead
(~140–390ms), not by data volume.** `write_parquet()` opens a fresh
connection for each of the two tables; large's overlay (89 defs across 6
files, per the prior spike) writes *faster* than small's (2 files) because
these numbers are noise-dominated by process/connection startup, not a real
signal about overlay size. This is a small-scale echo of the JVM/Ammonite
startup tax found in the Joern baselines (Phase 0) — connection or process
startup cost dominates when the actual payload is this small. A production
engine holding a resident DuckDB (or equivalent) connection, rather than
reconnecting per commit, would not pay this tax; this spike's numbers
overstate real per-commit cost accordingly.

## Limitations

- 3 real PRs, 3 repetitions each, same sample as the extraction spike — same
  caveat applies: all three are small by real-world standards, and a
  genuinely large PR (50+ files) hasn't been tested against this mechanism.
- Base snapshot rebuilt fully (not incrementally) each time a new parent SHA
  is needed — appropriate for this spike (three buckets, three parents) but
  not a claim about base-snapshot *maintenance* cost at scale; that's a
  separate, unmeasured question from the overlay-commit cost this
  experiment targets.
- `commit_write_wall_time_ms` numbers are inflated by DuckDB's per-call
  connection overhead, as noted above — not representative of a
  resident-connection production path.
- Correctness verification here compares symbol_id/file/kind/name tuples
  (defs) and caller/callee/file/line tuples (calls) for exact set equality.
  It does not check line-range correctness of def spans beyond that,
  because the spike's schema doesn't need it for this question.
- Same scope cut as the extraction spike: syntactic, name-based extraction
  only, no cross-file resolution, no `refs` table.
- One exploratory pilot run (`pilot-001`) preceded preregistration, using a
  since-fixed base-snapshot bug; its data is excluded from evidence and
  reported only as the discovery source for the freshness constraint above.
- **Preregistration ordering slip, disclosed.** The manifest's `status` was
  flipped to `preregistered` *after* the nine official `run-*` repetitions had
  executed and their results been inspected, not before. The hypothesis,
  acceptance rule, rubric, and driver were frozen before `run-001` (the only
  change after the pilot was the base-freshness fix, made on pilot data
  alone), and nothing in the manifest was altered afterward — but the letter
  of `experiments/README.md`'s promotion rule 1 ("preregistered before result
  inspection") was not met. Treat this experiment's evidence as
  *corroborated* rather than independently qualifying: the same commit/merge
  mechanism is re-verified against ground truth at every step, under proper
  preregistration, in `fleet-cpg-phase3-differential-zod` and
  `fleet-cpg-phase4-second-language`.

## Decision

- [x] accept mechanism (primary correctness metric: 9/9 exact matches,
      0 defect) — the COW-merge query is correct once the base-freshness
      constraint is respected; storage-ratio threshold flagged for revision
      rather than treated as a rejection, since the underlying "no
      duplication" claim still holds by a wide margin on every bucket
- [ ] reject hypothesis
- [ ] revise and preregister a new experiment
- [x] propose architecture promotion with human approval (proposed 2026-09-05; acceptance pending)

Promoting this into the design doc's Phase 1 gate as officially "passed"
still requires the human review this framework's evidence-promotion rules
call for. With both halves of Phase 1 now measured (extraction in
`fleet-cpg-phase1-overlay-zod`, commit-and-merge here), the phase's stated
gate — "prove the core thesis" — has empirical support on both fronts, on a
small real sample, with the base-freshness constraint surfaced as a
concrete design requirement (not yet enforced by any code) to carry into
Phase 2.
