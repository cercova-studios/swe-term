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
