# Experiment: fleet-cpg-phase4-second-language

Status: complete

Kind: benchmark

## Design reference

[Fleet CPG Engine — Phase 4 (Prove generality)](https://claude.ai/code/artifact/0c277fff-2382-485d-8316-31af8f4179e7).
Phase 4's gate is "the agnosticism test": add language #2, and it must
require zero changes below the lowering layer (L0 grammar pin + L1
lowering rules). Any change reaching the engine itself means the IR
boundary is wrong — fix it now, not after five languages have calcified
around the mistake. This is the final phase in the design doc's build
order.

## Hypothesis, null hypothesis, independent variable

**Hypothesis**: a shared, language-agnostic core (store I/O, COW/LSM
merge, L2 blast radius, packet materialization — `core.py`), finalized
against a TypeScript lowering module, requires zero source changes to also
support a Python lowering module on a real merged fastapi PR. `core.py`'s
sha256 hash is bitwise identical whether the pipeline runs TypeScript or
Python, and both produce facts that exactly match a from-scratch
extraction at their respective commits.

**Null hypothesis**: supporting Python requires any change to `core.py` (a
different hash across runs, or a hash that no longer matches the file on
disk), or either language's extracted facts diverge from ground truth.

**Independent variable**: which lowering module drives the shared core —
`typescript` (zod, reproducing `fleet-cpg-phase2-blast-radius-zod`'s small
bucket as a regression check) vs. `python` (a real merged fastapi PR,
genuinely new).

## Mechanism under test

`core.py` was extracted, unmodified in behavior, from the store/merge/
blast-radius/packet logic duplicated across `fleet-cpg-phase2-blast-radius-zod`
and `fleet-cpg-phase3-differential-zod` (each self-contained per this
framework's convention) — this is the first time that logic is centralized
as the actual shared spine it always conceptually was. `core.py` operates
purely on plain lists of dicts already in the L1 schema; it contains no
grammar, node-type, or language-specific code whatsoever.

Two lowering modules, each the *only* file that should need to exist per
language:
- `lowering_typescript.py` — tree-sitter-typescript, ported verbatim from
  Phase 2/3's inline extraction logic.
- `lowering_python.py` — tree-sitter-python, written fresh. Genuinely
  different from a copy-with-renamed-node-types: Python has no separate
  "method definition" node (a method is a `function_definition` lexically
  nested in a class — `kind` is derived from scope, not node type alone),
  decorators wrap definitions in a `decorated_definition` node with no
  `name` field of its own, and call syntax uses `call`/`attribute` instead
  of `call_expression`/`member_expression`.

The gate is checked mechanically, not by inspection: `phase4_run.py`
computes and records `core.py`'s sha256 on every single run; the evaluator
independently re-hashes the actual file on disk and requires all recorded
hashes plus the current file to be identical.

## Results

**Agnosticism gate passed.** `core.py`'s sha256
(`f5c4c402b8fb17a2a9140d992975cd7407fce2fa594599ee2d58c7f27e862d87`) is
identical across all 6 official runs (3 TypeScript, 3 Python) and matches
the file on disk at evaluation time — zero bytes changed to add Python
support.

| variant | language | merged defs | merged calls | impact set | defs/calls match truth | median end-to-end |
|---|---|---|---|---|---|---|
| typescript | TypeScript | 1388 (3/3) | 43888 (3/3) | 0 (3/3) | 3/3 | 11.3s* |
| python | Python | 502 (3/3) | 2649 (3/3) | 76 (3/3) | 3/3 | 1.1s |

*\*Dominated by a one-time full-repo base build (~10.8s) that only runs
once per fresh store; not comparable to Phase 2's per-overlay numbers,
which reused a pre-built base across repetitions.*

**Regression check passed**: the TypeScript variant's numbers
(`merged_defs=1388`, `merged_calls=43888`, `impact_set_size=0`) are exactly
`fleet-cpg-phase2-blast-radius-zod`'s official small-bucket result — proof
the `core.py` extraction didn't change any behavior, only where the code
lives.

**Python correctness passed on real, unseen code**: `lowering_python.py`
was written fresh for this experiment (no prior spike used Python) and its
output matched a from-scratch extraction exactly across all 3 repetitions,
including a hand-verified spot check — `@functools.wraps(cmgr)` at
`fastapi/routing.py:226` correctly appears as a call to `wraps`, proving
decorator-call capture works via the generic AST recursion with no
special-casing required, and the `method` vs. `function` vs. `class`
`kind` split (93 methods, 52 functions, 14 classes across 159 defs in the
touched file) matches manual inspection of the source.

## Discordant cases

None on the design claim itself — this is the cleanest result of all five
Fleet CPG experiments to date, with the mechanical hash check leaving no
room for a "mostly agnostic" verdict.

One environment gotcha worth recording for future spikes in this repo: an
early manual test of the driver logic (before `run_phase4.sh` existed)
failed because ad hoc multi-line commands in this environment run under
`zsh`, which does not word-split unquoted shell variables the way `bash`
does — a `$changed` variable holding multiple newline-separated filenames
silently became one merged, unparseable path. Every driver script in this
family (`run_commit.sh`, `run_phase2.sh`, `run_phase3.sh`, and now
`run_phase4.sh`) avoids this by being an actual `#!/usr/bin/env bash`
script invoked via `bash script.sh`, not inline shell — worth calling out
explicitly since it's easy to reintroduce by testing logic inline before
it's saved to a script file.

## Limitations

- Both PRs are small and real but modest (2 files/8 lines for zod, 1
  file/22 lines for fastapi) — same scale caveat as every prior Fleet CPG
  spike in this series.
- Only two languages tested. The design doc's own stated bar (Kythe,
  stack-graphs) required maintaining resolution rules across many more
  languages before their true maintenance cost became visible; two is
  enough to falsify a boundary violation, not enough to prove the pattern
  holds indefinitely as more languages are added.
- Both lowering modules use the same heuristic, name-only callee
  resolution tier — Python's genuinely different dynamic-typing profile
  (documented in the design doc as where this tier is weakest) wasn't
  specifically stress-tested here beyond the one real PR sampled.
- `core.py`'s hash check proves no *code* changed; it doesn't prove the L1
  schema itself would survive a language with a structurally different
  need (e.g. a language where "defs/calls" alone can't represent a core
  construct) — that would be a real schema-evolution question a third
  language might surface, deliberately out of scope here.
- One design decision the L1 schema already relies on: `kind` values
  (`function`/`method`/`class`) are treated as opaque strings by `core.py`
  and never branched on — this is what let Python's scope-derived `kind`
  assignment (vs. TypeScript's node-type-derived one) slot in without any
  core change. Worth naming explicitly since it's a boundary decision that
  could easily have been violated by accident.

## Decision

- [x] accept mechanism — the layered IR boundary holds under a real
      second-language addition: zero changes below the lowering layer,
      mechanically verified by hash, with both languages producing
      ground-truth-correct facts on real code.
- [ ] reject hypothesis
- [ ] revise and preregister a new experiment
- [x] propose architecture promotion with human approval (proposed 2026-09-05; acceptance pending)

Promoting this into the design doc's Phase 4 gate as officially "passed"
still requires the human review this framework's evidence-promotion rules
call for. With this experiment complete, all five phases in the "Build
order" section of the Fleet CPG Engine design doc now have empirical
evidence on record — the design doc's own stated open questions (overlay
extraction granularity from Phase 1, base-freshness enforcement from
Phase 1's commit spike, layer-compaction cost at real depth from Phase 3)
remain the concrete carry-forward items for whatever comes after the build
order, not settled by this phase's pass.
