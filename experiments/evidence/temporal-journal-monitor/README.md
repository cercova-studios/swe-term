# Evidence: temporal-journal-monitor

Kind: mechanism-hypothesis · Status: complete · Evaluator: manual (rubric below)

## Preregistered manifest

- Spec: [`experiments/specs/temporal-journal-monitor/manifest.json`](../../specs/temporal-journal-monitor/manifest.json)
- Paper: [Enforcing Temporal Constraints for LLM Agents (Agent-C)](https://huggingface.co/papers/2512.23738)
- Source: `directory` fixture `temporal-monitor-trace-corpus-v1`,
  content digest `sha256:81a4e4e9b441977d8cdd2424695a75659981e1ae7fc0526102c089d76ece44c5`
- Rubric digest: `sha256:6a1fb1c2f2e8c731534628b2dd88a4b4030b9b32dbe21d1f050228d0d8ef8ce5`
- Variants: `control` (`TestControlMonitorControlTrace`), `treatment` (`TestControlMonitorTreatmentTrace`) · 1 repetition each, per manifest

## Raw runs

`experiments/runs/temporal-journal-monitor/run-001/{control,treatment,safety_suite}.log` —
exact `go test ./internal/core -run <name> -count=1 -v` invocations from the
manifest, executed 2026-09-20. All exit 0.

## Manual rubric verdict

Rubric: "Inspect every trace against the closed vocabulary: approval must
precede a governed lease, only one lease is active, observed effects must be
declared, and lifecycle claims require a valid passing receipt matching the
current target; cancellation cannot become success."

- **Control** (`TestControlMonitorControlTrace`, `internal/core/control_monitor_test.go:33`):
  exercises `controlPermitsLifecycle` — a result-only baseline that accepts
  any passed receipt without comparing it to a changed target identity.
  Confirms the fixture models a real changed-target case (`staleReceipt`
  vs. `changedTarget`) the control would wrongly permit. Matches the
  manifest's control description exactly.
- **Treatment** (`TestControlMonitorTreatmentTrace`, `:49`): sets a receipt
  target, then records a receipt bound to the *original* identity after the
  target changed — `ApplyControlEvent` returns `ControlReceiptStale`, not
  acceptance. This is the causal contrast the hypothesis claims.
- **Safety suite** (`TestControlMonitorLegalTrace`,
  `TestControlMonitorRejectsIllegalTracesWithStableRules` [8 subtests],
  `TestControlMonitorReplayDuplicateAndSequenceRules`,
  `TestControlMonitorPrefixReplayMatchesUninterruptedTrace`,
  `TestControlMonitorBoundsEffectDeclaration`): covers every rule the
  rubric names — approval-before-lease (`ControlApprovalRequired`),
  single-lease (`ControlLeaseConflict`), declared-effects
  (`ControlEffectUndeclared`), receipt-gated lifecycle
  (`ControlLifecycleReceiptRequired`, `ControlReceiptStale`),
  cancellation-cannot-succeed (`ControlLifecycleCancelled`), an unsupported
  event kind failing closed (`ControlEventSchemaUnsupported`), immediate-
  duplicate idempotence, altered-duplicate and sequence-gap rejection
  (`ControlEventSequence`), prefix-replay state equality, and a bounded
  effect-declaration size (`maxDeclaredEffects = 32`,
  `ControlEffectDeclarationInvalid`).

Every acceptance-rule clause is met: legal traces accepted, every frozen
illegal trace rejected with a stable rule ID, prefix replay reaches
identical state, zero unsafe lifecycle promotions across all traces
exercised.

## Discordant cases

None found. Every rejection carries the rule ID the rubric or a table-driven
subtest name predicts; no illegal trace was accepted and no legal trace was
rejected.

## Limitations

- Tests a pure in-memory reducer only — no durable journal, no
  concurrent/interleaved event streams, no tool loop, no model. This is the
  scope the spec's README states explicitly ("It does not reproduce ...
  a tool loop, durable journal, database, model, or external environment").
- The trace corpus is hand-authored (8 illegal-trace cases + the legal/
  replay/bounds suites), not an adversarially generated or fuzzed corpus;
  it can prove the reducer handles the traces it was given, not that no
  trace exists outside this set that would slip through.
- `maxDeclaredEffects = 32` is an arbitrary bound picked in the
  implementation, not derived from any measured production workload —
  untested whether it's the right number for real tool effect sets.
- Replay/idempotence is tested for exact and altered immediate duplicates
  and one sequence gap; it is not tested under concurrent/out-of-order
  delivery, since the journal's own delivery-ordering guarantee is a
  separate, `Target` contract (ARCHITECTURE.md §6) this experiment assumes
  rather than tests.

## Decision

- [x] accept a bounded durable-journal follow-up — the reducer holds under
      every rule and replay case tested; the natural next increment is
      wiring it behind an actual `ControlJournal` (still `Target` per
      ARCHITECTURE.md §5) rather than the in-memory harness used here.
- [ ] reject hypothesis
- [ ] revise the event vocabulary and preregister a new corpus
- [ ] propose architecture promotion with human approval

This is a mechanism-hypothesis experiment; promoting `ApplyControlEvent`
into an architectural invariant or a `Target` → `Implemented` contract
change is a separate human decision this evidence bundle does not make.
