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
unhypothesized cost is per-invocation JVM/Ammonite startup (7.3s at 21k LOC →
~213s at 3.03M LOC, cold medians above). Cost scaling with LOC is sublinear
(144× more LOC over fastapi produced only ~29× process wall time, ~37× cold
build, ~44× RSS), correcting an earlier
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
  fresh/missing/failed/6-independently-tested-stale-dimensions/tampered/
  unchanged-scope traces (`scope` had no dedicated staleness case in v1);
  byte-equivalent replay confirmed. Evidence:
  [`experiments/evidence/evidence-gated-lifecycle/`](../../experiments/evidence/evidence-gated-lifecycle/).
  Decision: accept for a bounded persistence follow-up (binding the gate to
  an actual `Obligation`/`VerificationReceipt` store, still `Target`) — not
  an architecture promotion, which stays a separate human decision.
- Post-run revision (review of the PR that landed the reducer): a valid
  replacement receipt left the lifecycle claim accepted on the strength of
  the receipt it replaced, and the record event's obligation was ignored.
  The reducer now withdraws the claim on every receipt replacement and
  rejects a mismatched record event (`receipt.obligation_mismatch`). Corpus
  `receipt-trace-corpus-v2` adds those rows and a scope-change row; they
  have no recorded run yet and are **not** part of the accepted result
  until `run-002` is recorded and evaluated. The manifest treatment filter
  now also runs the replay and no-mutation tests, which `run-001` ran as a
  separate safety suite.

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
  valid failed receipt was rejected as invalid instead of retained; an
  observed effect left a recorded receipt usable, and re-declaring the same
  target discarded a current one. The reducer now scopes the target by
  obligation (`control.receipt.obligation_mismatch`), keeps failed receipts,
  refusing the claim with `control.lifecycle.receipt_failed`, clears the
  receipt on every observed effect, and keeps it across an identical
  re-declared target. Corpus
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

## 3. Design research — not yet experiments

### 3.1 Test quality and architectural fitness steering (2026-09-20)

Full doc: [`docs/plans/2026-09-20-test-quality-and-architectural-fitness-steering.md`](../plans/2026-09-20-test-quality-and-architectural-fitness-steering.md).
**No results — this is a proposal.** Four experiments are proposed there and
none is preregistered yet.

Question: how can swe-term, as a harness, steer agents away from brittle
mock-heavy tests and from shipping locally-correct patches that compound
architectural debt?

Findings worth carrying regardless of whether the proposal proceeds:

- Agent over-mocking is empirically documented ([arXiv:2602.00409](https://arxiv.org/abs/2602.00409), MSR 2026).
- **Prompt interventions on agent test-writing do not significantly change
  outcomes** ([arXiv:2602.07900](https://arxiv.org/abs/2602.07900)) — so an
  `AGENTS.md` instruction is the weakest available mechanism. That paper also
  finds agents use tests mainly as *observational probes* (print statements
  over assertions), not as specifications.
- Mutation score beats coverage as a test-quality proxy but is itself
  contested: correlation with real-bug detection can vanish once suite size is
  controlled ([arXiv:2607.22880](https://arxiv.org/abs/2607.22880)). Anything
  we gate on belongs under `evaluator-validity-audit`.
- Antithesis' own docs rule out naive adoption of deterministic simulation
  testing for brownfield: the FoundationDB pluggable approach is *"generally
  impractical for systems already in production"*.
- The proposal adds **no new invariant** — it gives §5's `Obligation`
  "minimum V&V rung" the definition it currently lacks, and points the Fleet
  CPG overlay at debt signals.

**Thoughtworks harness-engineering frame (added 2026-09-20).** Böckeler's
[*Harness engineering for coding agent users*](https://martinfowler.com/articles/harness-engineering.html)
(2026-04-02) supplies the vocabulary: `Agent = Model + Harness`, with
**guides** (feedforward, steer before acting) and **sensors** (feedback,
observe after, enable self-correction), each either **computational**
(deterministic) or **inferential** (LLM). Her diagnostic — *"you get either an
agent that keeps repeating the same mistakes (feedback-only) or an agent that
encodes rules but never finds out whether they worked (feed-forward-only)"* —
applied to swe-term yields the most actionable finding so far: **this repo is
guide-heavy and sensor-poor.** It has a 14-invariant architecture contract, a
preregistration framework, and explicit testing doctrine, and almost nothing
that checks whether any of it held. Also imported: Ashby's Law reframes the
Fleet CPG engine as *the regulator's model of the system* (a regulator can
only regulate what it has a model of), the brownfield paradox — *"the harness
is most needed where it is hardest to build"* — and the separation of
per-change sensors from continuous drift sensors. Counter-evidence kept: she
calls the current behaviour-harness state of the art, including mutation
testing, *"not good enough yet"*, which is a direct challenge to this doc's
own M3.

**Acted on 2026-09-20:** the diagnostic above was closed by one degree —
`internal/architecture/fitness_test.go` now mechanically enforces five
dependency rules `ARCHITECTURE.md` previously only asserted in prose (core
vendor-free per invariant 13, core depends on nothing internal, TUI owns no
provider semantics, provider adapters don't cross-import, experiment tooling
is not a second state model). Stdlib only, no new dependency. It carries typed
feedback and a vacuity guard, and was mutation-tested against itself (inject a
vendor import into core → the rule fires as expected → revert). The vacuity
guard caught a real bug in the sensor's own first run. This is swe-term's
first computational sensor; the repo remains guide-heavy overall.

**Also acted on 2026-09-20:** `internal/core/vv_rung.go` implements the V&V
rung ladder — six rungs ordered by what class of defect the evidence can
catch, a closed hand-authored (kind, risk) → minimum-rung policy table where
absence is a failure rather than a default, and a gate rejecting under-rung
discharge plus downgrade-by-reclassification (16 trace cases). Invariant 7 is
made *structural*: there is no event that sets a minimum rung, so "a model may
never downgrade it" is unreachable rather than merely checked; the remaining
downgrade vector (relabel the work as lower-risk) is rejected explicitly, and
escalating the bar invalidates evidence that only cleared the old one. Status
matches `receipt_gate.go` — a pure reducer, no persistence or runtime loop, so
`ARCHITECTURE.md` §5 `Obligation` stays `Target`. **What it does not do:**
assign a rung to real evidence. The reducer is fed one. That remaining
question is what Experiment 8 was narrowed to.

**Build vs. adopt (§5 of the doc):** `hegel-go` (property-based testing, MIT,
by Hypothesis' author) and Bombadil (PBT for web **and terminal** UIs — swe-term
is a TUI) should be **adopted, not rebuilt**; they are commodity-but-deep test
engines and they implement the rungs of the proposed ladder rather than
competing with it. What swe-term should build is the **gate** — the
obligation → rung → receipt layer — because nobody else builds that. Caveats:
`hegel-go` is v0.9.5 beta and drives a native Rust `libhegel` via FFI, so it
is a test-path dependency with CI-hermeticity implications, and Go's native
`testing.F` fuzzing must be measured against it first (proposed experiment
`hegel-vs-native-fuzzing`).

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
| Hora \& Robbes, *Are Coding Agents Generating Over-Mocked Tests?* | [2602.00409](https://arxiv.org/abs/2602.00409) | experiment-candidate (unfiled) | design research §3.1 |
| *Rethinking the Value of Agent-Generated Tests* | [2602.07900](https://arxiv.org/abs/2602.07900) | reference-only (falsifies prompt-only fix) | design research §3.1 |
| Zhao et al., *Do Coverage and Mutation Scores Correlate with Effectiveness?* | [2607.22880](https://arxiv.org/abs/2607.22880) | evaluation challenge | design research §3.1 |
| *Mutation-Guided LLM-based Test Generation at Meta* | [2501.12862](https://arxiv.org/pdf/2501.12862) | reference-only | design research §3.1 |
| Böckeler, *Harness engineering for coding agent users* | [martinfowler.com](https://martinfowler.com/articles/harness-engineering.html) | frame-setting | design research §3.1 |
| Thoughtworks, *Exploring AI coding sensors* | [blog](https://www.thoughtworks.com/en-de/insights/blog/generative-ai/harness-engineering-agent-feedback-exploring-ai-coding-sensors) | reference-only | design research §3.1 |
| *Approved Fixtures* (Augmented Coding Patterns) | [pattern](https://lexler.github.io/augmented-coding-patterns/patterns/approved-fixtures/) | reference-only | design research §3.1 |
