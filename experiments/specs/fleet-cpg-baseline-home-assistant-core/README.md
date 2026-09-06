# Experiment: fleet-cpg-baseline-home-assistant-core

Status: complete

Kind: benchmark

## Design reference

[Fleet CPG Engine — Phase 0 (Baseline)](https://claude.ai/code/artifact/0c277fff-2382-485d-8316-31af8f4179e7).
Fourth and last of the Phase 0 repos — the design doc's "one huge repo",
chosen because its shape (enormous base, small per-PR diffs) is structurally
what the fleet's O(diff) thesis actually has to survive, not a synthetic
stress test.

## Hypothesis, null hypothesis, independent variable

Identical framing to the other three Phase 0 runs (see
`fleet-cpg-baseline-zod`'s README for the full statement), retargeted at
home-assistant/core's `homeassistant/` package: does ~3M LOC — two orders of
magnitude past the largest prior repo (excalidraw, ~173k) — change the
already-rejected cold/warm outcome, and do cold-build wall time/RSS keep
scaling the way the three smaller repos suggested?

## Frozen inputs

- Source: `https://github.com/home-assistant/core.git` @ `6ad726ba56d517e56536cdd6fa2ba9f358bbc0ef`
  (shallow-cloned, `--depth 1`)
- Source content digest: `sha256:9ceecdf3a83da1c485a14fa13c2f9d2d552491ab4e1f47bd270037cbc9230e49`
- Scope: `homeassistant/` package only — **~3.03M lines of `.py`**, 3× past
  the design doc's "1M+" estimate; corrected here.
- JVM heap capped at 24GB (`_JAVA_OPTIONS=-Xmx24g`) on a 48GB-RAM host
- Evaluator rubric digest: `sha256:61bcb2cab58042894e754f0aa9818da0bc9acb71875a1919462f065cbefec0c4`

## Procedure

Same as `fleet-cpg-baseline-zod`, with a shallow clone and capped JVM heap
forced by scale. Query set is an 8-query set built from Home Assistant's own
vocabulary (`async_setup_entry`, `HomeAssistant`, `Entity`, `add_entities`).
Full procedure is `run_prototype.sh`.

## Pilot run

One exploratory cold build ran before preregistration
(`pilot-001`, not treated as evidence) purely to answer "does this fit in
memory and finish in reasonable time at all" before committing to three
official repetitions. It completed in 57.8s at ~12.3GB peak RSS, which set
the 24GB heap cap and 3600s timeout in the manifest.

## Results

Three repetitions, 8 queries × 2 variants, all `status: "ok"`, **zero
unstable `result_count`s** (unlike excalidraw at ~173k LOC — see Discordant
cases in that experiment's README; the nondeterminism found there did not
recur here).

| Metric | cold | warm |
|---|---|---|
| `process_wall_time_ms` (median of 3) | 213,006 | 216,186 |
| `process_wall_time_ms` (all 3) | 207,279 / 213,006 / 214,855 | 200,232 / 216,186 / 223,717 |

**Acceptance rule not met — hypothesis rejected**, same direction as all
three smaller repos: warm is not faster than cold, at any scale tested.

Cold-build baseline:

| Metric | run-001 | run-002 | run-003 |
|---|---|---|---|
| wall time (s) | 49.79 | 59.72 | 61.28 |
| peak RSS (bytes) | 15,558,819,840 | 14,098,595,840 | 13,183,139,840 |

### Scaling check against the other three repos

| Repo | LOC | LOC ratio | Cold-build time | Time ratio | Peak RSS | RSS ratio |
|---|---|---|---|---|---|---|
| fastapi | 21k | 1.0× | 1.6s | 1.0× | 328MB | 1.0× |
| zod | 84k | 4.0× | 3.4s | 2.1× | 1.35GB | 4.1× |
| excalidraw | 173k | 8.2× | 6.8s | 4.2× | 2.13GB | 6.5× |
| home-assistant/core | 3.03M | **144.3×** | 59.7s (median) | **37.3×** | 14.3GB (avg) | **43.5×** |

**Correction to the artifact's prior claim**: the three smaller repos alone
looked "roughly linear," but that was three points spanning less than one
order of magnitude — not enough range to tell linear from sublinear apart.
With this fourth point two orders of magnitude further out, the actual shape
is clearly **sublinear**: a 144× increase in LOC produced only a 37× increase
in wall time and a 44× increase in RSS. Cost per line of code goes *down* as
the repo gets bigger, at least across this range and this tool. A plausible
mechanism (not verified here): Home Assistant's `homeassistant/components/`
tree is ~1,000+ largely-similar integration packages sharing enormous
boilerplate and common base classes, which a real-world parser's internal
caching (interned strings, shared AST fragments, symbol-table reuse across
structurally similar files) would exploit — a repo that size but built from
1,000 *unrelated* codebases might not show the same curve. Confirming that
would need another, differently-shaped huge repo; out of scope here.

Caveat: this comparison also crosses two different Joern frontends
(`jssrc2cpg` for zod/excalidraw, `pysrc2cpg` for fastapi/home-assistant), so
some of the ratio isn't purely a LOC effect — language and frontend
implementation differences are folded in.

## Discordant cases

None beyond the already-established one: per-query timing runs after the CPG
is memory-resident, so cache state cannot appear there. Notably, the
excalidraw-scale `typehier-all-typedecls` nondeterminism (Section: Discordant
cases in `fleet-cpg-baseline-excalidraw`) **did not recur at this much larger
scale** — all three repetitions agreed exactly. That one data point isn't
enough to say the earlier finding was a fluke rather than a rare race that
simply didn't trigger here again; it remains an open, unbisected question.

## Limitations

- Same 3-repetition, unpinned-Joern-release, 8-of-~80-query limitations as
  the other three runs.
- The sublinear-scaling finding rests on 4 points across 2 languages and 2
  Joern frontends — a real pattern worth taking seriously, but not something
  to extrapolate confidently to a 5th repo of a different shape (e.g. a huge
  monorepo of many unrelated small codebases rather than many similar
  integration packages).
- 24GB heap cap and a 48GB host mean this run does not establish where Joern
  actually runs out of memory — only that ~3M LOC of Python fits comfortably
  under 16GB peak RSS in practice.

## Decision

- [ ] accept mechanism for a bounded follow-up
- [x] reject hypothesis (consistent with all three smaller repos; JVM-startup
      tax dominates at every scale tested, cache state never matters)
- [ ] revise and preregister a new experiment (candidates: bisect the
      cold/warm-irrelevant `--server` mode; test a huge repo shaped like many
      *unrelated* codebases to see if sublinear scaling is Home Assistant's
      integration-boilerplate structure or a general Joern property)
- [x] propose architecture promotion with human approval (proposed 2026-09-05; acceptance pending)

Phase 0 is now complete for all four repos named in the design doc. The
cold-build and process-wall-time numbers across all four, and the corrected
sublinear-scaling finding, are folded into the design doc's Phase 0 section;
promoting them into an actual acceptance gate for Phase 1 still requires the
human review this framework's evidence-promotion rules call for.
