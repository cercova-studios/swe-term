# Temporal control-journal monitor

Status: preregistered — deterministic implementation is complete, but no raw
run record has been produced by an execution runner.

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
corpus. The control permits lifecycle success based on a passing receipt outcome
without comparing it to the changed target. The treatment uses
`ApplyControlEvent` to enforce ordering and current receipt identity. Evaluate
all package-level trace tests separately as a safety suite.

## Metrics and stop conditions

- Primary: illegal treatment traces accepted (must be zero).
- Secondary: legal trace acceptance, stable rejection rule IDs, immediate replay
  idempotence, prefix-replay state equality, and maximum declaration size.
- Reject or revise if any legal trace fails, any illegal trace passes, or replay
  produces different state. Abort if corpus, reducer contract, or evaluator
  changes after the ready gate.

## Results

Do not record results here until a runner writes append-only raw output under
`experiments/runs/temporal-journal-monitor/`. Passing package tests verifies the
prototype only; it is not a completed experiment or an architecture promotion.

## Decision

- [ ] accept a bounded durable-journal follow-up
- [ ] reject hypothesis
- [ ] revise the event vocabulary and preregister a new corpus
- [ ] propose architecture promotion with human approval
