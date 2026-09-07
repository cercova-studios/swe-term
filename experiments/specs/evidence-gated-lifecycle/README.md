# Experiment: evidence-gated-lifecycle

Status: preregistered

Kind: mechanism-hypothesis

## Discovery provenance (mechanism-hypothesis only)

- Discovery packet: [`docs/research/papers/evidence-gated-lifecycle/`](../../../docs/research/papers/evidence-gated-lifecycle/)
- Paper-note disposition: `experiment-candidate`
- Local failure or architecture decision: lifecycle invariant 6 requires a
  current receipt bound to current source and verifier state before a lifecycle
  claim can be accepted.
- Competing mechanisms rejected or deferred: unattended-loop orchestration is
  rejected as a confound; persistent storage and the temporal monitor are
  deferred until the deterministic receipt semantics are stable.

Paper-backed experiments must link a discovery packet whose synthesis marks the
mechanism `experiment-candidate`. Discovery is selection evidence, not proof that
the mechanism works locally.

## Paper claim

Proof-or-Stop proposes that lifecycle transitions should require fresh,
tracked-source-state-bound, mechanically verifiable evidence. This experiment
tests only the identity-bound receipt gate. It does not reproduce the paper's
unattended loop, reviewer topology, local-key bundle format, model evaluation,
or self-application results.

## Hypothesis

A sealed passing receipt whose identity exactly matches the current obligation
identity prevents false lifecycle promotion when any relevant verification input
changes.

## Null hypothesis

The identity omits a relevant changed input, accepts a tampered or stale receipt,
or rejects an unchanged verification scope.

## Independent variable

Control accepts a passing receipt's result. Treatment additionally compares its
sealed identity to current scoped source, verifier, arguments, configuration,
runtime, and lockfile identities.

## Frozen inputs

- Source/fixture digest: filled into `manifest.json` from the frozen fixture
  directory before the ready gate.
- Paper revision: arXiv `2607.14890v1`.
- Prompt digest: not applicable; no model is used.
- Tool-schema digest: not applicable; no tool loop is used.
- Evaluator rubric digest: filled into `manifest.json` from the closed trace
  rubric before the ready gate.

## Procedure

Run one deterministic repetition of each variant against the frozen trace corpus.
The control records which stale or tampered receipts would be promoted under a
result-only rule. The treatment exercises the pure reducer across fresh,
missing, failed, changed-source, changed-verifier, changed-arguments,
changed-config, changed-runtime, changed-lockfile, tampered, unchanged-scope,
and replay traces.

## Metrics

- Primary: false lifecycle promotions.
- Secondary: fresh pass acceptance, stale rejection by rule ID, tamper rejection,
  and replay determinism.
- Safety-critical counters: treatment false promotions must be zero.

## Acceptance and stop conditions

Accept only if every legal trace is accepted, every stale, missing, failed, or
tampered trace is rejected with a stable rule ID, and replay reaches identical
state. Reject or revise if an omitted identity field lets a stale receipt pass or
if unchanged scope is rejected. Abort if the fixture corpus or evaluator changes
after ready-gate approval.

## Risks and confinement

Network is disabled. The process needs only the local Go toolchain and the frozen
fixture directory. There is no model, tool invocation, secret, workspace
mutation, database, or container. The process adapter requests one worker and a
five-minute wall-time limit; it does not independently enforce hard memory or CPU
limits in Phase 0.

## Results

Do not fill this section until `just experiment-ready evidence-gated-lifecycle`
passes. Raw trial output belongs under `experiments/runs/evidence-gated-lifecycle/`.

## Discordant cases

Inspect every control promotion that the treatment rejects. A treatment rejection
of unchanged scope is also discordant and requires identity-scope review.

## Limitations

The trace corpus cannot establish semantic program correctness, real-world test
scope adequacy, persistence durability, or model-loop outcomes. It only tests
closed receipt-identity invalidation semantics.

## Decision

- [ ] accept receipt gate for a bounded persistence follow-up
- [ ] reject hypothesis
- [ ] revise identity fields and preregister a new trace corpus
- [ ] propose architecture promotion with human approval
