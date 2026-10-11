# Evidence: temporal-journal-monitor

Kind: mechanism-hypothesis · Status: complete · Evaluator: manual (rubric below)

## Preregistered manifest

- Spec: [`experiments/specs/temporal-journal-monitor/manifest.json`](../../specs/temporal-journal-monitor/manifest.json)
- Paper: [Enforcing Temporal Constraints for LLM Agents (Agent-C)](https://huggingface.co/papers/2512.23738)
- Source: `directory` fixture `temporal-monitor-trace-corpus-v2`,
  content digest `sha256:9cb353fea951c0320dcc5e91263e0ab59a5945838bb7cf394966c4942e67e0fc`
  (includes `fixtures/executable-corpus.sha256`, which pins
  `internal/core/control_monitor_test.go`). `run-001` below was executed
  against `temporal-monitor-trace-corpus-v1`
  (`sha256:81a4e4e9b441977d8cdd2424695a75659981e1ae7fc0526102c089d76ece44c5`);
  see "Corpus revision after run-001".
- Rubric digest: `sha256:f2fc70bd5c465deb4b190ba09693d8cb21a01eddfbf1f55c0e1bee2540e18e82`,
  SHA-256 over the exact `evaluator.rubric` string (UTF-8, no trailing
  newline), the same rendering the other manifests use. The manifest shipped
  with `run-001` carried `sha256:6a1fb1c2…8ef8ce5`, which does not hash any
  rendering of the rubric text; the rubric applied below is the quoted text,
  and the digest was corrected to match it.
- Variants, per the current manifest: `control` (`TestControlMonitorControlTrace`),
  `treatment` (the eight corpus tests listed in the manifest command) · 1
  repetition each. The v1 manifest `run-001` executed limited `treatment` to
  `TestControlMonitorTreatmentTrace` alone, so the rest of the corpus ran as
  the separate `safety_suite` below.

## Raw runs

`experiments/runs/temporal-journal-monitor/run-001/{control,treatment,safety_suite}.log` —
executions of the manifest's `control` and `treatment` commands (with `-v`
added to capture subtest names) plus an extra `safety_suite` run of the
remaining `TestControlMonitor*` tests, which are not manifest variants.
Executed 2026-09-20. All exit 0.

## Manual rubric verdict

Rubric: "Inspect every trace against the closed vocabulary: approval must
precede a governed lease, only one lease is active, observed effects must be
declared, and lifecycle claims require a valid passing receipt matching the
current target; cancellation cannot become success."

- **Control** (`TestControlMonitorControlTrace`, `internal/core/control_monitor_test.go:33`,
  v1 form): exercised a test-local `controlPermitsLifecycle`, a result-only
  baseline that accepts any passed receipt without comparing it to a changed
  target identity, and confirmed the fixture models a real changed-target
  case (`staleReceipt` vs. `changedTarget`). Both assertions were
  tautological given how the fixtures are built, so this variant contributed
  no reducer evidence in `run-001`; see "Corpus revision after run-001" for
  its v2 form.
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

## Corpus revision after run-001

Review of the v1 reducer found gaps that the v1 corpus did not exercise
and that the rubric's "matching the current target" clause did not pin down:

- the receipt target carried only an identity, so a valid passing receipt for
  a *different* obligation with the same digests could unlock a lifecycle
  claim;
- a valid `ReceiptFailed` receipt was rejected as `ControlReceiptInvalid`
  instead of being retained, so a legitimate failed verification was
  flattened into "no receipt";
- an accepted `effect_observed` after a recorded receipt left that receipt
  usable for a later lifecycle claim, although the governed mutation may have
  changed the verified source;
- re-declaring the identical target discarded a still-current receipt;
- an `effects_declared` event under a lease that was not active was reported
  as `control.effect.declaration_invalid`, and a zero-sequence event as
  `control.event.schema_unsupported`, so the same violated invariant carried
  different rule IDs depending on event kind;
- the `control` variant asserted only fixture properties and never called
  `ApplyControlEvent`, and the `treatment` command ran a single test although
  the primary metric counts every frozen illegal trace.

The reducer now scopes the target by obligation (`ControlSetReceiptTarget`
requires `Obligation`; a receipt for another obligation is rejected with
`control.receipt.obligation_mismatch`), retains valid failed receipts,
refusing the lifecycle claim with `control.lifecycle.receipt_failed`, clears
the current receipt on every accepted observed effect, and keeps it when the
same obligation and identity are re-declared, and reports a missing or
non-matching lease on `effects_declared` as `control.lease.required` and a
zero sequence as `control.event.sequence`. Corpus `v2` adds those rows and
the corresponding tests (13 illegal-trace subtests, a zero-sequence check in
the replay test, `TestControlMonitorRetainsFailedReceipt`, and
`TestControlMonitorRepeatedTargetKeepsReceipt`). The `control` variant now
runs the reducer on the agreement trace (passed receipt against its unchanged
target, claim accepted), and the `treatment` command lists every corpus test,
so the variant-to-metric mapping is reproducible from the manifest alone. The `run-001` verdict above
stands for the v1 rows it inspected; the v2 rows have not been through a
recorded run or manual rubric verdict yet, so they are not admissible as
results until a `run-002` against v2 is recorded here.

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

---

## Regrade on corpus v3 (2026-10-08)

**Status: re-graded `complete` on `temporal-monitor-trace-corpus-v3`
(`sha256:6bd4b6cffcdd2fa59e22597a6de89914f6003121943b7277bafdc2846596e016`),
which pins `internal/core/control_monitor_test.go` and
`internal/core/control_monitor_replay_test.go`.**

### Why the record was reopened

The v2 manifest's `null_hypothesis` names "has replay-sensitive state" as a
falsifier. The reducer had exactly that. `copyControlEvent` stored `Effects`
through `append([]string(nil), ...)`, normalising an empty-but-non-nil slice
to `nil`, while `controlEventsEqual` compared the field with
`reflect.DeepEqual`, which distinguishes the two. An accepted event carrying
`Effects: []string{}` was rejected on identical replay:

```
first apply: accepted=true  rule=""
replay:      accepted=false rule="control.event.sequence"
```

Every fixture in v1 and v2 built `Effects` as `nil` or as a populated
literal, so the empty-slice shape was unreachable from the corpus. The
sha256 lock held — nothing changed silently — but the corpus was never
varied on that axis.

### Two gaps found, not one

1. **The falsifier was reachable and unobserved** (above). Fixed at the
   comparator with `slices.Equal`, which treats nil and empty as equal, in
   commit `fed174d`.
2. **The v2 rows were never run.** `experiments/runs/temporal-journal-monitor/`
   contains `run-001` and now `run-003`; there is no `run-002`. The v2
   corpus revision was recorded in the manifest and the "Corpus revision
   after run-001" note above explicitly says the v2 rows "need a recorded
   `run-002` before they count as results" — that run never happened. So the
   v2 `status: complete` rested on `run-001`, executed against corpus **v1**.
   This run is numbered `run-003` rather than `run-002` to avoid implying it
   is the missing v2 run.

### Corpus v3

Adds one row the earlier corpora could not express:
`TestControlEventReplayIsIdempotentAcrossEffectsShapes` enumerates the four
shapes `Effects` can take (nil, empty, one, several) and asserts that an
accepted event replays idempotently and leaves state unchanged for each. It
is now listed in the manifest's `treatment` command and pinned by
`fixtures/executable-corpus.sha256`.

### Raw runs

`experiments/runs/temporal-journal-monitor/run-003/` —

| Log | Command | Result |
|---|---|---|
| `control.log` | manifest `control` variant, `-v` added | exit 0 |
| `treatment.log` | manifest `treatment` variant, `-v` added | exit 0, 43 RUN/PASS lines |
| `falsification-prefix-comparator.log` | the same `treatment` command against the **pre-fix** comparator, everything else held at the post-fix tree | **exit 1**, as intended |

The third log is the one that matters. A corpus row that cannot fail proves
nothing, so the v3 row was checked for discriminating power rather than
assumed to have it: it fails against the unfixed reducer with
`control.event.sequence` on the `empty` shape and passes against the fix.
The other eight corpus tests pass in both configurations, which localises
the new row's power to exactly the defect it was added for.

### Verdict

**Hypothesis accepted on v3 evidence; the v2 acceptance is superseded.** A
closed deterministic reducer does mechanically protect the target ordering
invariants, and now also replays idempotently across every shape its event
fields can take. The decision recorded in the spec README is unchanged in
substance — accept a bounded durable-journal follow-up, not an architecture
promotion — but it now rests on a corpus that was run and that is known to
discriminate.

What this episode is evidence *for*, beyond the reducer: an example corpus
encodes the shapes its author thought of. Nine hand-written traces, a
cryptographic lock, and a passing suite did not notice a two-valued axis
(nil versus empty) on a field the comparator read. That is the standing
argument for property-shaped rows over additional examples, recorded here
because it was paid for rather than argued.
