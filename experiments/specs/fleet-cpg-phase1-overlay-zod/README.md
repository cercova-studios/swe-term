# Experiment: fleet-cpg-phase1-overlay-zod

Status: complete

Kind: benchmark

## Design reference

[Fleet CPG Engine — Phase 1 (Prove the core thesis)](https://claude.ai/code/artifact/0c277fff-2382-485d-8316-31af8f4179e7).
Phase 1's gate is O(diff), or the design is falsified: overlay build cost
must be proportional to diff size, not repo size. This experiment is the
fact-store-spike half of that phase — a minimal tree-sitter-to-Parquet
extractor, run full-repo once as a baseline and per-PR as the overlay, on
real merged zod PRs.

## Hypothesis, null hypothesis, independent variable

**Hypothesis**: overlay build wall time (re-extracting `defs`/`calls` facts
for only the files a real merged PR touches) is at least an order of
magnitude below full-repo build wall time, across PRs of varying diff size,
and does not grow with total repo size (324 files).

**Null hypothesis**: overlay time is within the same order of magnitude as
full-repo time, or tracks the repo's total size rather than the files a PR
actually touches.

**Independent variable**: which real merged zod PR is materialized — three
buckets by diff size (small: 2 files / 8 lines; medium: 3 files / 38 lines;
large: 6 files / 241 lines) — with extractor, source repo, and base revision
held fixed.

## Frozen inputs

- Source: `https://github.com/colinhacks/zod.git` @ `b801439b5fb160d651084ff7d7c27a61e24a7334`
  (same pin as `fleet-cpg-baseline-zod`)
- Source content digest: `sha256:5c20f08b0bc5ce42fd1458c1940218cdbf767f53686ffdfe31a7a4bb409e4005`
- Three real merged PRs (chosen via the GitHub REST API for closed/merged
  PRs, scoped to diffs touching `packages/zod/src`):
  - small — PR #6511, merge commit `5ff9566508e6c95873d2648a5bdcc3a371f1b757`
  - medium — PR #6488, merge commit `212b941791e7faae078e17645eb612824fd8f79a`
  - large — PR #5913, merge commit `555e5f46fed0e6ba25ecf931eae96f7b980a6650`
- Extractor: `extractor.py`, an isolated venv with `tree-sitter`,
  `tree-sitter-typescript`, `duckdb` (not committed; recreate per
  `run_prototype.sh`'s pattern — `python3 -m venv`, `pip install tree-sitter
  tree-sitter-typescript duckdb`)
- Evaluator rubric digest: `sha256:d53e423157b47702748e672dd5c5f6f6e7cd31fc025c7c4cfb36dbd081ff68ef`

## L1 schema (deliberately minimal — spike, not production coverage)

- `defs(symbol_id, file, kind, name, start_line, end_line)` — functions,
  classes, methods
- `calls(caller_symbol_id, callee_name, file, call_line)` — name-based
  callee resolution only (heuristic tier, per the design doc's tree-sitter
  fast path — no cross-file type resolution)

`refs` (every identifier use) is out of scope: it doesn't change whether
overlay cost tracks diff size, and building it would be scope creep on a
spike meant to answer one question.

## Procedure

1. Full-repo baseline: run `extractor.py --all` over all 324 `.ts` files in
   `packages/zod/src` at the pinned revision, 3 repetitions.
2. For each bucket: compute files changed within `packages/zod/src` between
   the PR's merge commit and its parent, check out the merge commit, and run
   `extractor.py` scoped to *only* those changed files (not the diff hunks —
   whole changed files). 3 repetitions per bucket.
3. Compare each bucket's median wall time against the full-repo median.

Full procedure is `run_overlay.sh` (per-PR driver) plus the full-repo
baseline command in `extractor.py`'s own `--all` mode.

## Results

| | files changed | lines changed | median overlay time | full-repo baseline |
|---|---|---|---|---|
| small (PR #6511) | 2 | 8 | 10.9ms | |
| medium (PR #6488) | 3 | 38 | 22.8ms | |
| large (PR #5913) | 6 | 241 | 9.6ms | |
| full-repo (324 files) | — | — | | 422.7ms (median of 3: 385.8 / 422.7 / 576.3) |

**Hypothesis accepted.** Every bucket clears the 10× rule by a wide margin —
overlay time is **18×–44× lower** than the full-repo baseline, not just one
order of magnitude. `defs`/`calls` counts were exactly stable across all 3
repetitions of every bucket (unlike Joern's `jssrc2cpg`, which showed a 1-off
`typeDecl` count wobble at ~173k LOC in `fleet-cpg-baseline-excalidraw`) —
this extractor's simpler, purely-syntactic pass is fully deterministic at
this scale.

## Discordant cases

**The "large" bucket (241 lines) was not the slowest — "medium" (38 lines)
was**, and by a clear margin (22.8ms vs. 9.6ms). This is not noise: `medium`'s
3 files contained 318 total `defs` after extraction; `large`'s 6 files
contained only 89. Overlay cost tracks **the total size of the files
actually touched** (because this spike re-extracts whole files, not diff
hunks), not the diff's line count. A PR that touches a few large files costs
more than a PR that touches more, smaller files — an important refinement to
"overlay cost is proportional to diff size," and a concrete design question
for the real engine: extract only the changed hunks (harder, needs stable
per-symbol boundaries surviving unrelated edits elsewhere in the file) or
whole changed files (simpler, this spike's choice, but couples cost to file
size rather than edit size). Neither choice is wrong; the difference matters
more as touched files grow large, which this spike's 3-PR sample doesn't
stress.

## Limitations

- 3 real PRs, 3 repetitions each — enough to falsify or confirm the
  order-of-magnitude claim, not enough to fit a real cost model.
- All three PRs are small by real-world standards (≤6 files, ≤241 lines) —
  consistent with zod's PR history generally skewing small (see the PR list
  pulled from the GitHub API), but this spike has not tested a genuinely
  large PR (e.g., 50+ files) where whole-file extraction cost could compound
  differently.
- The extractor is a syntactic, name-based heuristic pass — no cross-file
  resolution, no type information. It answers "does the mechanism have the
  right complexity class," not "is the resulting fact table correct or
  complete." That's a deliberate, documented scope cut for this spike, not
  an oversight.
- This measures extraction/lowering cost only — not the actual COW-overlay
  commit into a snapshot store (no segment format exists yet; this spike
  used DuckDB/Parquet as scaffolding per the design doc's own note). The
  "overlay builder on real PRs" half of Phase 1 (base snapshot + diff → COW
  overlay, measured end-to-end) is the next increment, not this one.
- One exploratory pilot run (`pilot-001`) preceded preregistration, used only
  to confirm the extractor and driver script worked end-to-end before
  committing to 3×3 official repetitions; its data is not counted as
  evidence.

## Decision

- [x] accept mechanism for a bounded follow-up (the fact-extraction /
      lowering step clears the O(diff) bar with room to spare; next: wire
      this into an actual base-snapshot + COW-overlay commit path, and widen
      the PR sample to include a genuinely large one)
- [ ] reject hypothesis
- [ ] revise and preregister a new experiment
- [ ] propose architecture promotion with human approval

Promoting this into the design doc's Phase 1 gate as officially "passed"
still requires the human review this framework's evidence-promotion rules
call for, and the still-missing overlay-commit half of the phase.
