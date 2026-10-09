# Temporal control-journal monitor

Status: complete — raw run record and manual evaluator verdict in
[`experiments/evidence/temporal-journal-monitor/`](../../evidence/temporal-journal-monitor/).

## Scope

This is Experiment 3 from
[`docs/plans/2026-08-27-harness-hypothesis-experiments.md`](../../../docs/plans/2026-08-27-harness-hypothesis-experiments.md).
It tests only a pure in-memory reducer across four rules: approval before
governed mutation, one active lease, declared effects, and a current receipt
before lifecycle promotion.

It does not reproduce Agent-C’s constrained generation or formal DSL, Progent’s
programmable privileges, AgentSpec’s policy language, a tool loop, durable
journal, database, model, or external environment.

## Hypothesis and null

A small monitor over closed, versioned events can reject each frozen unsafe
ordering trace with a stable rule ID while accepting legal traces and reaching
the same state after prefix replay.

Null: the fixed vocabulary misses a required transition, accepts an illegal
trace, rejects a legal trace, has replay-sensitive state, or requires unbounded
resident state.

## Procedure

Run one deterministic repetition of each variant against the frozen trace
corpus. The control is the agreement trace: a passed receipt recorded against
its unchanged target authorizes the lifecycle claim, which an outcome-only gate
and `ApplyControlEvent` both permit. The treatment runs `ApplyControlEvent`
over the whole corpus, including the contrast trace where the same receipt is
recorded after the target changed, to enforce ordering and current receipt
identity; its command lists every corpus test so the primary metric is
reproducible from the manifest alone.

## Metrics and stop conditions

- Primary: illegal treatment traces accepted (must be zero).
- Secondary: legal trace acceptance, stable rejection rule IDs, immediate replay
  idempotence, prefix-replay state equality, and maximum declaration size.
- Reject or revise if any legal trace fails, any illegal trace passes, or replay
  produces different state. Abort if corpus, reducer contract, or evaluator
  changes after the ready gate.

## Results

Control and treatment run under `experiments/runs/temporal-journal-monitor/run-001/`
(the v1 manifest commands with `-v` added, all exit 0). Treatment rejects a
receipt recorded against a stale target (`ControlReceiptStale`); the
remaining trace tests, run as a separate safety suite because the v1
treatment command covered only the contrast trace (8 illegal-trace subtests,
legal-trace, replay, prefix-replay, and bounds tests), pass every case with a
stable rule ID and zero unsafe lifecycle promotions. Full manual rubric verdict:
[`experiments/evidence/temporal-journal-monitor/README.md`](../../evidence/temporal-journal-monitor/README.md).

`run-001` ran against corpus `v1`. Review then revised the reducer to scope
the receipt target by obligation, retain valid failed receipts, invalidate
receipts on observed effects, and report lease and sequence violations under
their own rule IDs; corpus `v2` adds those rows, folds the whole corpus into
the treatment command, and turns the control into a reducer run. The v2 rows
need a recorded `run-002` before they count as results (see the evidence
README, "Corpus revision after run-001").

## Decision

- [x] accept a bounded durable-journal follow-up
- [ ] reject hypothesis
- [ ] revise the event vocabulary and preregister a new corpus
- [ ] propose architecture promotion with human approval

## Disclosure: a stated falsifier was reached after `status: complete`

On 2026-10-06, outside the experiment's own corpus, the subject was shown to
have **replay-sensitive state** — one of the conditions this manifest's
`null_hypothesis` names explicitly as a falsifier.

`copyControlEvent` stored `Effects` through `append([]string(nil), …)`, which
normalises an empty-but-non-nil slice to `nil`, while `controlEventsEqual`
compared the field with `reflect.DeepEqual`, which distinguishes the two. Any
accepted event carrying `Effects: []string{}` was therefore rejected on an
identical replay with `control.event.sequence`, contradicting the idempotence
promised in `control_monitor.go`'s `ControlEvent` doc comment.

The nine pinned examples could not observe this: every fixture in corpus `v2`
constructs `Effects` as `nil` or as a populated literal, so the empty-slice
shape is unreachable from the corpus. The lock held — nothing changed
silently — but the corpus was not varied along that axis.

Fixed at the comparator (`slices.Equal`, which treats nil and empty as equal)
and covered by `control_monitor_replay_test.go`, which enumerates the four
shapes the field can take and was confirmed to fail against the unfixed
reducer. The fix is source-only; the pinned test files and their digests are
untouched.

**Resolved 2026-10-08: re-graded on corpus `v3`.** The decision was taken as
"null hypothesis survives, corpus gap" rather than "falsified": the reducer's
rules were never wrong, the corpus simply could not express the shape that
broke replay. The record was reopened to `preregistered`, corpus `v3` was cut
adding `TestControlEventReplayIsIdempotentAcrossEffectsShapes` to the pinned
lock and the `treatment` command, the preregistration gate was re-passed, both
variants were run as `run-003`, and the status returned to `complete`.

A second gap surfaced during the regrade: there is no `run-002`. The v2 rows
were recorded in the manifest but never executed, so the previous
`status: complete` rested on `run-001` against corpus **v1**. The v3 run is
numbered `run-003` to keep that visible.

The new row was checked for discriminating power, not assumed to have it: it
fails against the pre-fix comparator and passes against the fix, with the
other eight corpus tests passing in both configurations. See
[`experiments/evidence/temporal-journal-monitor/`](../../evidence/temporal-journal-monitor/),
section "Regrade on corpus v3".
