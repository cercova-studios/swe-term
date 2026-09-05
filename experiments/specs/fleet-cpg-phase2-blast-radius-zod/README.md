# Experiment: fleet-cpg-phase2-blast-radius-zod

Status: complete

Kind: benchmark

## Design reference

[Fleet CPG Engine — Phase 2 (Prove the value path)](https://claude.ai/code/artifact/0c277fff-2382-485d-8316-31af8f4179e7).
Phase 2's gate is "first real end-to-end packet": PR event → overlay →
impact-set relation → materialized packet served to a stub agent. Neither
Phase 1 spike (`fleet-cpg-phase1-overlay-zod`,
`fleet-cpg-phase1-commit-zod`) chained its mechanism to the next stage, and
neither built an L2 derived relation or the file-tree context packet the
harness's progressive-disclosure doctrine calls for. This experiment is the
first one that does.

## Hypothesis, null hypothesis, independent variable

**Hypothesis**: for a real merged zod PR, the full pipeline (base snapshot
at the PR's true parent → overlay extraction and commit → query-time merge
→ reverse-call-graph blast radius → context-packet materialization →
stub-agent readback) completes end to end with no manual step, the
materialized packet round-trips as a plain file tree, and end-to-end wall
time stays under ~1s at zod's scale.

**Null hypothesis**: any stage requires manual intervention or fails to
compose, the packet fails to round-trip, or end-to-end time is large enough
to threaten the context-readiness SLO even at this small scale.

**Independent variable**: which real merged zod PR drives the pipeline —
the same three buckets as both Phase 1 spikes (small, medium, large).

## Mechanism under test (new pieces beyond Phase 1)

- **L2 blast radius**: reverse-BFS over the merged call graph, starting
  from every def in a touched file, over name-resolved edges (same
  heuristic tier as the L1 extractor — a callee name can match multiple
  defs; every match becomes an edge, which is honest ambiguity at this
  tier, not something silently resolved to one caller). Output: every
  symbol that transitively calls a touched symbol, ranked by BFS distance.
- **Context packet**: a materialized file tree —
  `packet/README.md` (index), `packet/touched/<file>.json` (defs per
  touched file), `packet/impact/impact-set.json` (ranked impact list),
  `packet/impact/by-file.json` (per-file rollup) — drillable with `cat`/
  `jq`, per the harness's progressive-disclosure doctrine, not a query API.
- **Stub agent**: reads the packet back (`README.md` + `impact-set.json`)
  and the driver checks the readback count against what was computed, as
  the round-trip check.

Full procedure is `run_phase2.sh` (per-PR driver, reusing
`fleet-cpg-phase1-commit-zod/commit_store.py`'s `base` subcommand for the
shared base snapshot) plus `blast_radius.py` (extraction, commit, merge,
blast radius, packet, all timed as one process).

## Results

| bucket | files changed | median end-to-end | impact set size | max BFS depth | round-trip |
|---|---|---|---|---|---|
| small (PR #6511) | 2 | 457.4ms | 0 (all 3 reps) | 0 | 3/3 |
| medium (PR #6488) | 3 | 460.4ms | 573 (all 3 reps) | 3 | 3/3 |
| large (PR #5913) | 6 | 245.5ms | 729 (all 3 reps) | 5 | 3/3 |

**Hypothesis accepted.** All 9 repetitions completed end to end with zero
manual intervention, every packet round-tripped exactly, and every bucket's
median end-to-end time is well under the 1s bar (245–460ms). Impact-set
size and max BFS depth were exactly stable across every repetition of every
bucket — the pipeline is deterministic end to end, same as both Phase 1
spikes.

Stage breakdown (medians, ms) — `commit_write` dominates every bucket
(195–412ms of the 245–460ms total), the same DuckDB per-call connection
overhead `fleet-cpg-phase1-commit-zod` already found, recurring here rather
than being newly discovered. Every other stage — extraction, merge,
blast-radius computation, packet materialization, stub-agent readback —
stays under 25ms even on the largest bucket.

## Blast-radius spot check (no independent ground truth exists for this
relation at zod's scale, so this is a hand-verified sample, not a
statistical claim — same caveat Phase 0's golden queries carried for
transitive call graphs on the huge repo)

**Small bucket (impact set = 0) is genuinely correct, not a bug.** Touched
files are `v4/core/index.ts` (a barrel re-export file — 0 defs match the
extractor's `DEF_KINDS`, correctly) and a test file whose local helper
functions (`callSite`, `expectMatch`, `valid`, …) are never called outside
that file. `callSite` initially looked like a miss — `grep` found the name
in two other test files — but inspection showed those are independently
*defined* local functions sharing the name, not calls to the touched
symbol. The 0 is correct.

**Medium bucket surfaced a real, concrete false positive from name-only
resolution.** `v3/helpers/util.ts`'s `joinValues` appears in the impact set
at distance 1. Its body is `array.map(...).join(separator)` — plain
`Array.prototype` calls. But the touched `v4/classic/schemas.ts` happens to
define its own function literally named `map` (line 1992, unrelated to
`Array.prototype.map`), and the extractor's callee resolution — property
name only, per the L1 schema's stated heuristic tier — matches
`joinValues`'s `.map(...)` call to that unrelated touched `map` function
purely by name collision. This is exactly the ambiguity the design doc's
"surface confidence, don't fake precision" caveat exists for, now with a
concrete instance: a name-only resolver **will** produce impact-set false
positives whenever a common method name (`map`, `join`, `format`, …)
collides with an unrelated top-level function of the same name. Not a bug
in the BFS or the packet — a real, expected cost of the heuristic tier that
compiler-integrated extraction (the design doc's "dual-path extraction"
upgrade) is meant to close.

## Limitations

- Same 3-PR sample as both Phase 1 spikes — small by real-world standards;
  a genuinely large PR's blast radius (hundreds of touched defs) is
  untested.
- Blast radius has no independent ground truth at this scale; correctness
  here is a hand-verified spot check on two buckets, not a statistical
  claim. This mirrors Phase 0's own admission that hand-verifying transitive
  call graphs on a large corpus isn't tractable.
- `commit_write_wall_time_ms` is still inflated by DuckDB's per-call
  connection overhead, as documented in `fleet-cpg-phase1-commit-zod` — a
  resident-connection production path would not pay this tax, so
  end-to-end numbers here are an upper bound, not a floor.
- Blast radius stops at reverse-call reachability; it doesn't yet rank or
  filter by anything beyond BFS distance (e.g. file ownership, test vs.
  production code, recency) — the packet's `by-file.json` rollup is the
  only aggregation offered.
- The "1s bar" in the acceptance rule was picked as a sanity threshold at
  zod's small scale, not derived from the harness's actual latency SLO
  (Context Readiness Rate's freshness join, KR 1.1) — it says nothing yet
  about behavior at home-assistant/core's 3.03M-LOC scale.
- No exploratory pilot data included in evidence: `pilot-001` (small
  bucket) and `pilot-002` (medium bucket) both ran clean end to end and
  informed the spot-check writeup above, but their run directories were
  removed before the official 3×3 reps per this framework's pilot/official
  split.

## Decision

- [x] accept mechanism — the value path composes end to end with no manual
      steps, packets round-trip, timing is well within the sanity bar, and
      the one correctness risk this phase was meant to surface (heuristic
      name-resolution false positives in the blast-radius relation) showed
      up concretely and is now documented rather than latent
- [ ] reject hypothesis
- [ ] revise and preregister a new experiment
- [ ] propose architecture promotion with human approval

Promoting this into the design doc's Phase 2 gate as officially "passed"
still requires the human review this framework's evidence-promotion rules
call for. Phase 3 (differential base maintenance) is the next gate; the
freshness constraint `fleet-cpg-phase1-commit-zod` surfaced (base must be
keyed to the true parent, not an arbitrary pin) is the concrete requirement
that phase has to satisfy under continuous push events, not just single
static builds.
