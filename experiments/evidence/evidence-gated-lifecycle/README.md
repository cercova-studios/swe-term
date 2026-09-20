# Evidence: evidence-gated-lifecycle

Kind: mechanism-hypothesis · Status: complete · Evaluator: manual (rubric below)

## Preregistered manifest

- Spec: [`experiments/specs/evidence-gated-lifecycle/manifest.json`](../../specs/evidence-gated-lifecycle/manifest.json)
- Paper: [Proof-or-Stop](https://huggingface.co/papers/2607.14890) (arXiv `2607.14890v1`)
- Source: `directory` fixture `receipt-trace-corpus-v2`,
  content digest `sha256:85b3ff76efa3a901f738af47ed588f9b910e82557ed1a9c29dbf1412d43e5e32`
  (includes `fixtures/executable-corpus.sha256`, which pins
  `internal/core/receipt_gate_test.go`). `run-001` below was executed against
  `receipt-trace-corpus-v1`
  (`sha256:074122b12828b693c56140ae29c58f6b59a444e16185505c97afd3ea6440d38d`);
  see "Corpus revision after run-001".
- Rubric digest: `sha256:1dee9cc640fc93aab0b6e1a49909660f772730bf507ff1ffa865e47f0f727f63`
- Variants: `control` (`TestReceiptGateControlTrace`), `treatment`
  (`TestReceiptGateTraces`) · 1 repetition each, per the manifest `run-001`
  ran under. The treatment filter has since been widened to also run
  `TestReceiptGateReplayIsByteEquivalent` and
  `TestReceiptGateDoesNotMutateInputState`, so the declared command produces
  the replay evidence the acceptance rule asks for; `run-001` ran those two
  separately (below).

## Raw runs

`experiments/runs/evidence-gated-lifecycle/run-001/{control,treatment,safety_suite}.log`
— executions of the manifest's `control` and `treatment` commands as declared
at the time (`^TestReceiptGateControlTrace$`, `^TestReceiptGateTraces$`), with
`-v` added to capture subtest names, plus an extra `safety_suite` run of
`TestReceiptGateReplayIsByteEquivalent` and
`TestReceiptGateDoesNotMutateInputState` that was not a manifest variant when
it ran. Executed 2026-09-20. All exit 0.

## Manual rubric verdict

Rubric: "Inspect every trace: a lifecycle claim is allowed only with an
untampered passing receipt whose identity exactly equals the current
obligation identity; stale, missing, and failed receipts are rejected."

- **Control** (`TestReceiptGateControlTrace`,
  `internal/core/receipt_gate_test.go:11`): `resultOnlyControlAllows`
  accepts any `ReceiptOutcome == Passed`, ignoring identity entirely. The
  test explicitly demonstrates the false-promotion baseline on two cases —
  a receipt whose `SourceDigest` changed after sealing, and one whose
  `ConfigurationDigest` was tampered — both of which the result-only rule
  would wrongly allow. Matches the manifest's control description.
- **Treatment** (`TestReceiptGateTraces`, `:28`, 11 subtests): exercises
  `ApplyReceiptGateEvent` across every case the manifest's secondary
  metrics name — fresh pass (accepted, claim recorded), missing receipt
  (`receipt.missing`), failed receipt (`receipt.failed`), and a stale
  rejection for **six of the seven identity dimensions independently**
  (`source`, `verifier`, `arguments`, `configuration`, `runtime`,
  `lockfiles` — all `receipt.stale`; `scope` had no dedicated staleness
  case in v1, see Limitations), a tampered receipt
  body (`receipt.invalid`, caught by the `BodyDigest` reseal check before
  identity is even compared), and — critically for the null hypothesis —
  an **unchanged-scope trace that still succeeds** (`unchanged scope
  preserves valid receipt` → `ClaimVerified`), proving the identity gate
  doesn't over-reject.
- **Safety suite** (`TestReceiptGateReplayIsByteEquivalent`,
  `TestReceiptGateDoesNotMutateInputState`): replaying a duplicate
  `setTarget` event before the same trace produces byte-identical JSON
  state; applying an event never mutates the caller's input state map.

Acceptance rule fully met: zero false lifecycle promotions in the treatment
across all 11 traces, every fresh-pass trace accepted, every stale/missing/
failed/tampered trace rejected with a stable rule ID
(`receipt.stale`/`receipt.missing`/`receipt.failed`/`receipt.invalid`), and
byte-equivalent replay confirmed.

## Discordant cases

None — the spec's own instruction was to flag any control promotion the
treatment rejects, or any treatment rejection of unchanged scope. The
control-baseline test exists specifically to demonstrate two cases it
would wrongly promote (stale source, tampered config); the treatment
correctly rejects both, and separately accepts the one unchanged-scope
case tested. No case fell outside this expected pattern.

## Corpus revision after run-001

Review of the v1 reducer found gaps that the v1 corpus did not exercise:

- a valid replacement receipt (for example a later failed run) replaced the
  stored receipt but left the lifecycle claim accepted on the strength of the
  old one;
- the record event's own obligation field was ignored, so an event naming one
  obligation could record a receipt sealed for another;
- `scope` was the one identity dimension without an independent staleness
  trace.

The reducer now withdraws the claim on every non-idempotent receipt
replacement and rejects a record event whose obligation differs from the
receipt's with `receipt.obligation_mismatch`. Corpus `v2` adds those rows plus
the scope-change row (14 treatment subtests) and asserts that every rejected
trace leaves no claim standing. The `run-001` verdict above stands for the v1
rows it inspected; the v2 rows have not been through a recorded run or manual
rubric verdict yet, so they are not admissible as results until a `run-002`
against v2 is recorded here.

## Limitations

- In `run-001`, six of the seven identity dimensions were tested for
  staleness independently (source, verifier, arguments, configuration,
  runtime, lockfiles — one field changed at a time). `scope` had no
  dedicated staleness case: the `unchanged scope` trace re-sends the
  identical baseline and expects acceptance only. Corpus v2 adds a scope
  row, pending `run-002`. Combinations of simultaneous changes are not
  separately exercised, though the identity comparison is a full struct
  equality so this is unlikely to hide a gap — untested, not assumed safe
  by inspection alone.
- `BodyDigest` reseal-and-compare (in `VerificationReceipt.Valid()`) is a
  corruption check, not authentication: it is an unkeyed SHA-256 that any
  caller can recompute, so a receipt edited and resealed with
  `SealVerificationReceipt` passes `Valid()`. The gate therefore assumes
  receipts are constructed only by a trusted verifier; a forged
  self-consistent receipt from an untrusted party is outside what this
  mechanism detects and would need a keyed MAC or signature. The test
  corpus covers one tampered-without-reseal field (`ConfigurationDigest`),
  not a fuzz across every receipt field.
- No persistence, database, or durable storage is tested — this experiment
  is explicitly scoped to the closed reducer only, per the spec's
  Limitations section, which the discovery packet's "competing mechanisms
  deferred" note also states.
- No model or tool loop is involved; this validates the gate mechanism in
  isolation, not its integration into an actual obligation/verifier
  pipeline (both still `Target` contracts per ARCHITECTURE.md §5).

## Decision

- [x] accept receipt gate for a bounded persistence follow-up — the sealed-
      identity comparison mechanism holds against every stale, missing,
      failed, and tampered case tested, and correctly does not over-reject
      an unchanged scope. The natural next increment is binding it to an
      actual `Obligation`/`VerificationReceipt` persistence layer (still
      `Target`) rather than the in-memory map used here.
- [ ] reject hypothesis
- [ ] revise identity fields and preregister a new trace corpus
- [ ] propose architecture promotion with human approval

This is a mechanism-hypothesis experiment; promoting the receipt-identity
gate into ARCHITECTURE.md invariant 6's enforced implementation is a
separate human decision this evidence bundle does not make.
