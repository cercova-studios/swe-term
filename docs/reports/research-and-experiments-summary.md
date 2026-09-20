# Research and experiments summary

A single running digest of every research finding and experiment result in
this repository, with citations, kept current instead of scattered across
per-topic and per-experiment documents. Update this file — don't replace it —
whenever research or experiment work is about to be committed; it is the
compiled counterpart to `docs/research/papers/<topic>/synthesis.md` and
`experiments/specs/<id>/README.md`, not a replacement for either. Those stay
the detailed, per-topic and per-experiment record; this is the one place that
answers "what do we actually know so far, across everything."

Last compiled: 2026-09-20.

## 1. Fleet CPG Engine — completed benchmark experiments

Design reference throughout: [Fleet CPG Engine artifact](https://claude.ai/code/artifact/0c277fff-2382-485d-8316-31af8f4179e7).
All nine experiments below are `kind: benchmark`, preregistered, complete, and
promoted as curated evidence under `experiments/evidence/fleet-cpg-*/`
(ARCHITECTURE.md §9/§11, accepted 2026-09-05).

### Phase 0 — Joern baseline, four real repos

| Repo | Source | LOC | Cold build | Peak RSS | Process wall time |
|---|---|---|---|---|---|
| [zod](https://github.com/colinhacks/zod) | `b801439b` | ~84k | 2.65–3.69s | 1.1–1.5GB | cold 7,575ms / warm 7,997ms |
| [fastapi](https://github.com/fastapi/fastapi) | `49033471` | ~21k | 1.55–1.70s | ~328MB | cold 7,254ms / warm 7,611ms |
| [excalidraw](https://github.com/excalidraw/excalidraw) | `e1bb9ff8` | ~173k | 6.65–6.94s | 2.0–2.3GB | cold 13,398ms / warm 13,420ms |
| [home-assistant/core](https://github.com/home-assistant/core) | `6ad726ba` | ~3.03M | 49.79–61.28s | 13.2–15.6GB | cold 213,006ms / warm 216,186ms |

Cold-vs-warm page-cache hypothesis **rejected on all four** — in-graph query
latency is sub-millisecond once the CPG is JVM-resident; the dominant,
unhypothesized cost is per-invocation JVM/Ammonite startup (7.2s at 21k LOC →
~224s at 3.03M LOC). Cost scaling with LOC is sublinear (144× more LOC over
fastapi produced only 37× wall time, 44× RSS), correcting an earlier
three-point "roughly linear" read. Excalidraw's `typehier-all-typedecls`
query returned an unstable count (5,915 vs. 5,916) across repetitions on a
pinned revision — a measured CPG-nondeterminism counterexample, not repeated
on the 3.03M-LOC repo.
Specs: `fleet-cpg-baseline-{zod,fastapi,excalidraw,home-assistant-core}`.

### Phase 1 — prove the core thesis (O(diff))

**Extraction** (`fleet-cpg-phase1-overlay-zod`): overlay extraction on 3 real
merged zod PRs ran 18–44× below the 422.7ms full-repo baseline (small
10.9ms / medium 22.8ms / large 9.6ms). Discordant: cost tracked touched-file
size, not diff line count, because whole files are re-extracted, not hunks.

**Commit/merge** (`fleet-cpg-phase1-commit-zod`, corroborated — its
`preregistered` status was set after the official runs; re-verified under
proper preregistration by phases 3 and 4): COW merge exact vs. ground truth,
9/9 reps, 0 mismatches; overlay storage 5.0–12.5% of base. Pilot caught a
real bug: a base snapshot one unrelated commit later than the diff's parent
silently corrupted facts for untouched files — the empirical case for the
freshness precondition now in ARCHITECTURE.md §9.

### Phase 2 — prove the value path (`fleet-cpg-phase2-blast-radius-zod`)

End-to-end PR → overlay → L2 blast radius → materialized packet → stub-agent
readback, 9/9 reps, zero manual steps, packets round-tripped exactly, median
end-to-end 245–460ms. Concrete false positive found and hand-verified: a
touched function named `map` pulled an unrelated caller into the impact set
via `Array.prototype.map()` name collision — the first fixture for a
compiler-integrated resolution upgrade.

### Phase 3 — prove steady-state operation (`fleet-cpg-phase3-differential-zod`)

Real 7-commit zod chain, 24 step-verifications (2 chain lengths × 3 reps),
zero exceptions — L1 facts and L2 blast radius both exact vs. from-scratch at
every step. First mechanism attempt re-flattened the whole base each push
(~8.5s/step, O(repo)) and was caught in pilot; corrected to a layered
(LSM-style) store with query-time newest-wins merge. Merge-query cost stayed
flat (17.0–17.8ms) at 1–6 layers — untested beyond that depth.

### Phase 4 — prove generality (`fleet-cpg-phase4-second-language`)

Shared `core.py`, finalized against TypeScript then given a fresh Python
lowering module for a real merged fastapi PR. Gate checked mechanically:
`core.py`'s sha256 identical across all 6 runs (3 TypeScript, 3 Python) and
matching the file on disk — zero bytes changed below the lowering layer.
Python output exact vs. ground truth on every rep; hand-verified
`@functools.wraps(cmgr)` correctly captured as a call via generic recursion.

### Where this goes next

Implementation plan: [`docs/plans/2026-09-05-fleet-cpg-engine-implementation.md`](../plans/2026-09-05-fleet-cpg-engine-implementation.md)
(Rust engine, sibling Cargo workspace, five gated slices). Open questions
carried forward, not resolved by any phase passing: overlay granularity
(file vs. hunk), freshness enforcement in code, compaction cost at real layer
depth, heuristic-resolution false-positive rate, L1 schema under a third
language.

## 2. Harness mechanism-hypothesis research

Three research packets under `docs/research/papers/` reached `synthesis:
complete`. Of their six associated specs: two (`evidence-gated-lifecycle`,
`temporal-journal-monitor`) are now **complete with results** (§2.1, §2.3
below); four (`structured-verifier-feedback`, `verified-external-task-state`,
`protected-spine-compaction`, `evaluator-validity-audit`) remain `draft`,
with no results yet. Don't read "complete synthesis" as "complete
experiment" for those four.

### 2.1 Evidence-gated lifecycle claims — complete

Decision informed: whether swe-term can fail closed when a lifecycle claim
(`verified`, `done`, `ready_to_merge`) relies on a receipt whose verification
inputs have since changed (target `VerificationReceipt`/`Obligation`,
invariant 6).

- [Proof-or-Stop](https://huggingface.co/papers/2607.14890) — sealed receipt
  identity compared against current obligation identity. `experiment-candidate`,
  reducible to deterministic source-bound receipt traces. Limitation carried
  forward: v1 preprint, model/corpus-limited evaluation, no independent
  confirmation found in this packet.
- Spec: `evidence-gated-lifecycle` (`complete`). **Result: hypothesis
  accepted.** `internal/core/receipt_gate.go`'s `ApplyReceiptGateEvent` is
  the preregistered treatment (`TestReceiptGateTraces`, 11/11 subtests
  pass) — its exact command match to the manifest was verified directly,
  not assumed from resemblance. Zero false lifecycle promotions across
  fresh/missing/failed/7-independently-tested-stale-dimensions/tampered/
  unchanged-scope traces; byte-equivalent replay confirmed. Evidence:
  [`experiments/evidence/evidence-gated-lifecycle/`](../../experiments/evidence/evidence-gated-lifecycle/).
  Decision: accept for a bounded persistence follow-up (binding the gate to
  an actual `Obligation`/`VerificationReceipt` store, still `Target`) — not
  an architecture promotion, which stays a separate human decision.

### 2.2 Model-dependent harness mechanisms

Four independent mechanisms, each isolated to a treatment that changes one
variable while information, task order, budget, and authority stay fixed;
all four specs held `draft` pending a chosen reproducible model/provider and
usage budget.

- [Structured Feedback Improves Repair in an LLM Agent Loop](https://huggingface.co/papers/2607.14167)
  — typed verifier feedback (failure location, observed value, admissible
  alternatives) vs. raw diagnostics, 50 paired cases. `experiment-candidate`.
  Spec: `structured-verifier-feedback` (`draft`).
- [LongHorizon-Harness](https://huggingface.co/papers/2608.01964) — task
  state updated only from independently attributable facts. `experiment-candidate`.
  Spec: `verified-external-task-state` (`draft`).
- [TRUSTMEM](https://huggingface.co/papers/2606.25161) — memory-transition
  verifier checking coverage, preservation, and faithfulness.
  `experiment-candidate`. Spec: `protected-spine-compaction` (`draft`).
- [Benchmarking the Benchmarks](https://huggingface.co/papers/2607.02577) —
  decomposed evaluator signals, judge disagreement audit. `experiment-candidate`.
  Spec: `evaluator-validity-audit` (`draft`).

### 2.3 Temporal control-journal monitor — complete

Decision informed: whether a small deterministic reducer can mechanically
protect swe-term's target ordering invariants, ahead of any production tool
loop (advances state-expansion evidence; does not claim a durable control
journal is implemented).

- [Enforcing Temporal Constraints for LLM Agents (Agent-C)](https://huggingface.co/papers/2512.23738)
  — closed event reducer over four fixed rules. `experiment-candidate`,
  table-driven legal/illegal/replay traces. Remaining unknown: whether four
  fixed rules are sufficient and replay-safe locally.
- [AgentSpec](https://huggingface.co/papers/2503.18666), [Progent](https://huggingface.co/papers/2504.11703)
  — programmable runtime policy enforcement, both `reference-only`: support
  runtime enforcement generally but weren't adopted as the local mechanism
  (a policy language would confound the four fixed repository invariants
  with policy-authoring quality).
- Spec: `temporal-journal-monitor` (`complete`). **Result: hypothesis
  accepted.** `internal/core/control_monitor.go`'s `ApplyControlEvent` is
  the preregistered treatment (`TestControlMonitorTreatmentTrace` +
  8-subtest safety suite, all pass) — verified by exact command match to
  the manifest, not inferred from resemblance. Rejects a receipt recorded
  against a stale target (`ControlReceiptStale`); every illegal trace
  rejected with a stable rule ID (approval-before-lease, single-lease,
  declared-effects, receipt-gated lifecycle, cancellation-cannot-succeed,
  unsupported-event, replay/sequence rules); prefix replay reaches
  identical state. Evidence:
  [`experiments/evidence/temporal-journal-monitor/`](../../experiments/evidence/temporal-journal-monitor/).
  Decision: accept for a bounded durable-journal follow-up (wiring the
  reducer behind an actual `ControlJournal`, still `Target`) — not an
  architecture promotion.
- Post-run revision (review of the PR that landed the reducer): the v1
  target matched receipts by identity only, so a passing receipt for a
  different obligation with the same digests could unlock a claim, and a
  valid failed receipt was rejected as invalid instead of retained. The
  reducer now scopes the target by obligation
  (`control.receipt.obligation_mismatch`) and keeps failed receipts,
  refusing the claim with `control.lifecycle.receipt_failed`. Corpus
  `temporal-monitor-trace-corpus-v2` adds those rows; they have no recorded
  run yet and are **not** part of the accepted result until `run-002`
  is recorded and evaluated.

Both specs' fixtures now carry `executable-corpus.sha256`, pinning the Go
test file that is the corpus's executable form inside the digested
directory (`TestFrozenExperimentCorpusMatchesLock` enforces it), so a corpus
edit can no longer leave the manifest `content_digest` unchanged. The
evidence-gated-lifecycle limitation on `BodyDigest` was also corrected: it is
an unkeyed integrity check that assumes a trusted verifier constructs
receipts, not a defense against a party who can reseal a forged receipt.

Resolution of an earlier open item in this report: `internal/core/control_monitor.go`
and `internal/core/receipt_gate.go` (committed 2026-09-05, before this
report first existed) *are* the preregistered treatments for these two
specs — confirmed 2026-09-20 by matching each manifest's exact `variant.command`
test-function names against the actual test files and running them, not by
resemblance or session-history inference (which was tried first and came
back inconclusive — see git log around 2026-09-20 for that dead end).

## Full citation list

| Paper | HF Papers link | Disposition | Topic packet |
|---|---|---|---|
| Proof-or-Stop | [2607.14890](https://huggingface.co/papers/2607.14890) | experiment-candidate | evidence-gated-lifecycle |
| Structured Feedback Improves Repair in an LLM Agent Loop | [2607.14167](https://huggingface.co/papers/2607.14167) | experiment-candidate | model-dependent-harness |
| LongHorizon-Harness | [2608.01964](https://huggingface.co/papers/2608.01964) | experiment-candidate | model-dependent-harness |
| TRUSTMEM | [2606.25161](https://huggingface.co/papers/2606.25161) | experiment-candidate | model-dependent-harness |
| Benchmarking the Benchmarks | [2607.02577](https://huggingface.co/papers/2607.02577) | experiment-candidate | model-dependent-harness |
| Enforcing Temporal Constraints for LLM Agents (Agent-C) | [2512.23738](https://huggingface.co/papers/2512.23738) | experiment-candidate | temporal-journal-monitor |
| AgentSpec | [2503.18666](https://huggingface.co/papers/2503.18666) | reference-only | temporal-journal-monitor |
| Progent | [2504.11703](https://huggingface.co/papers/2504.11703) | reference-only | temporal-journal-monitor |
| SWE-agent | [2405.15793](https://huggingface.co/papers/2405.15793) | reference-only | (worked example, `docs/research/papers/README.md`) |
| AutoSaddler | [2608.23041](https://huggingface.co/papers/2608.23041) | defer | (worked example, `docs/research/papers/README.md`) |
