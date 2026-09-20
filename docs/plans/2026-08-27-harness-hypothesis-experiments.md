# Harness hypothesis experiments

Status: active portfolio; experiments 2 and 3 have recorded raw runs and
curated evidence, the remaining four have not.

## Current experiment state

| ID | State | Meaning |
|---|---|---|
| 1 Structured verifier feedback | draft | Hypothesis and model-dependent contract are specified; fixture corpus and model identity remain open. |
| 2 Evidence-gated lifecycle | complete (v1 corpus); v2 rows await `run-002` | `run-001` recorded and manually evaluated against corpus v1; hypothesis accepted for a bounded persistence follow-up. Review then added replacement-receipt, obligation-mismatch, and scope-change rows (corpus v2) that are not yet admissible as results. Evidence: [`experiments/evidence/evidence-gated-lifecycle/`](../../experiments/evidence/evidence-gated-lifecycle/). |
| 3 Temporal journal monitor | complete (v1 corpus); v2 rows await `run-002` | `run-001` recorded and manually evaluated against corpus v1; hypothesis accepted for a bounded durable-journal follow-up. Review then added obligation-scoped target, failed-receipt, observed-effect invalidation, and identical-target rows (corpus v2) that are not yet admissible as results. Evidence: [`experiments/evidence/temporal-journal-monitor/`](../../experiments/evidence/temporal-journal-monitor/). |
| 4 Verified external task state | draft | Hypothesis and model-dependent contract are specified; task corpus and model identity remain open. |
| 5 Protected-spine compaction | draft | The primary mechanical design is specified; the synthetic corpus remains open. |
| 6 Evaluator validity audit | draft | Must wait for raw trajectories from earlier experiments and a named judge/annotation plan. |

Raw runs for experiments 2 and 3 were executed by hand (the exact manifest
`variant.command` invocations, logged under ignored `experiments/runs/`); the
general execution runner is still planned and does not write those records
itself.

## Goal

Reproduce the mechanisms suggested by recent agent-harness papers with small,
falsifiable experiments before promoting them into swe-term's core contracts.
The objective is not to reproduce headline benchmark scores. It is to determine
whether each mechanism improves a failure mode we can observe locally.

This plan advances the root architecture's in-flight control, state, and
verification refactors. It must not introduce a second agent loop, model-authored
policy language, or harness self-modification before deterministic gates exist.

## Experiment discipline

Each experiment must be preregistered before results are inspected:

- one sentence hypothesis and explicit null;
- frozen fixture set and source commit;
- fixed model, provider, parameters, prompt, and tool schema versions;
- paired variants using the same task order and budgets;
- raw append-only trajectories, not only aggregate scores;
- component metrics and failures, not a single correctness score;
- a short human-authored interpretation that names surprises and limitations.

For model-dependent experiments, run at least three repetitions per task and
report dispersion. Treat small samples as exploratory. Do not claim statistical
generality from them.

The implemented scaffold separates reviewed specifications, raw local runs, and
curated evidence:

```text
experiments/specs/<experiment-id>/
  README.md                    # preregistration and final interpretation
  manifest.json                # source/model/prompt/schema/verifier identities
  fixtures/                    # frozen inputs
experiments/runs/<experiment-id>/
  <run-id>/*.jsonl             # ignored, append-only raw trajectories
experiments/evidence/<experiment-id>/
  labels.csv                   # reviewed human and evaluator labels
  summary.json                 # mechanically derived component metrics
  report.md                    # curated cases, limitations, and decision
```

[`experiments/README.md`](../../experiments/README.md) defines the agent preflight
and promotion rules. Phase 0 implements creation and preregistration validation;
the bounded execution runner remains planned.

## Portfolio

| ID | Experiment | Mechanism | Model required? | Promotion decision |
|---|---|---|---|---|
| 1 | Structured verifier feedback | Agent-computer interface quality | Yes | Shape tool/verifier result contracts |
| 2 | Evidence-gated lifecycle | Fresh receipts bound to source state | No | Define receipt and obligation types |
| 3 | Temporal journal monitor | Deterministic ordering constraints | No | Define control events and monitor rules |
| 4 | Verified external task state | State updates from attributable facts | Yes | Define `TaskSnapshot` transitions |
| 5 | Protected-spine compaction | Verified state-preserving compression | Optional | Define compaction checkpoint contract |
| 6 | Evaluator validity audit | Judge disagreement and variance | Yes | Calibrate the evaluation harness |

Experiments 2 and 3 come before a production tool loop because they test the
safety state machine without depending on model quality.

## Experiment 1: Structured verifier feedback

### Source hypothesis

[SWE-agent](https://huggingface.co/papers/2405.15793) and
[Structured Feedback Improves Repair in an LLM Agent Loop](https://huggingface.co/papers/2607.14167)
suggest that interface and diagnostic information design materially affect
repair performance.

### Hypothesis

Given the same failed check, feedback containing the failure location, observed
state, required invariant, admissible alternatives, and evidence handle will
produce fewer invalid follow-up edits than raw stdout/stderr alone.

Null: after equalizing information and token budget, structured feedback does
not improve task completion, attempts, or invalid-edit rate.

### Smallest experiment

- Freeze 8–12 tiny Go repair fixtures spanning compile errors, assertion
  failures, formatting, and one cross-file contract mismatch.
- Run paired variants with the same model and maximum of three repair attempts:
  - **A:** raw command output;
  - **B:** a deterministic adapter transforms that same output into typed fields.
- Keep tool powers identical. Do not add hints to B that cannot be recovered
  from A; otherwise this tests extra information rather than structure.

### Measures

- Primary: completion within three attempts.
- Secondary: invalid edit count, verifier calls, input tokens, elapsed time, and
  whether the agent repaired the wrong location.
- Inspect every discordant pair manually.

### Promotion rule

Promote only the fields that explain repeated wins or prevent a concrete failure.
If B wins solely because it contains extra semantic hints, rerun with
information-equivalent A before changing the core contract.

## Experiment 2: Evidence-gated lifecycle

### Source hypothesis

[Proof-or-Stop](https://huggingface.co/papers/2607.14890) proposes that lifecycle
claims remain untrusted until fresh, source-bound evidence satisfies a gate.

### Hypothesis

A deterministic receipt gate can prevent false `verified` and `done`
transitions when source, verifier, configuration, runtime, or lockfiles change.

Null: the proposed receipt identity is insufficient to distinguish current from
stale evidence, or rejects legitimate unchanged evidence unpredictably.

### Smallest experiment

Implement only a pure state-transition function and table-driven traces. Do not
add SQLite, tools, or an agent loop yet. Cover at least:

- fresh passing receipt;
- missing and failed receipt;
- source changed after verification;
- verifier binary or arguments changed;
- config, sandbox/runtime, or lockfile changed;
- tampered receipt body;
- unrelated file change inside and outside the declared verification scope;
- replay of the same receipt.

### Measures

- False promotions are the safety-critical metric and must be zero in the frozen
  trace suite.
- False rejections must be explained by an explicit identity field rather than
  event ordering or map iteration.
- Replaying the same trace must produce byte-equivalent terminal state.

### Promotion rule

Promote the minimal receipt envelope and invalidation rules into Architecture
Sections 5 and 10 only after the trace suite passes. Persistence remains a
separate decision.

## Experiment 3: Temporal control-journal monitor

### Source hypothesis

[Enforcing Temporal Constraints for LLM Agents](https://huggingface.co/papers/2512.23738),
[Progent](https://huggingface.co/papers/2504.11703), and
[AgentSpec](https://huggingface.co/papers/2503.18666) suggest that deterministic
runtime enforcement can prevent unsafe tool sequences.

### Hypothesis

A small monitor over closed event types can enforce swe-term's core ordering
rules without an SMT solver or model-authored policy language.

### Smallest experiment

Define an in-memory event vocabulary and pure reducer for only four rules:

1. approval before governed mutation;
2. one active mutation lease;
3. observed effects stay within the declaration;
4. a receipt must be current before lifecycle promotion.

Exercise legal traces plus illegal reorderings, duplicate events, cancellation,
crash-and-replay, and unknown event/schema versions. Generate bounded event
permutations around each hand-authored trace to find missed transitions.

### Measures

- Every frozen illegal trace fails closed with a stable rule ID.
- Every legal trace is accepted.
- Prefix replay reaches the same monitor state as uninterrupted execution.
- Monitor work is linear in event count and keeps bounded resident state.

### Promotion rule

Land the event and monitor contracts before durable storage. Reject any design
that needs free-form model interpretation at enforcement time.

## Experiment 4: Verified external task state

### Source hypothesis

[LongHorizon-Harness](https://huggingface.co/papers/2608.01964) externalizes task
state and updates it only from independently verified environmental facts.

### Hypothesis

A typed task snapshot built from attributable observations reduces stale actions
and repeated discovery after context pressure compared with conversation-only
state.

Null: an external snapshot provides no benefit once token budget and accessible
evidence are held constant, or it anchors the agent to stale facts more often.

### Smallest experiment

- Freeze 8–10 multi-step repository tasks with one mid-task environmental change:
  another edit, a newly failing check, or an invalidated assumption.
- Pair two variants:
  - **A:** transcript plus ordinary summarization;
  - **B:** transcript plus a compact typed snapshot whose facts include source,
    version, freshness, and verification status.
- Force one context reset or compaction at the same step in both variants.
- Keep B read-only: no manager/executor/auditor process split is required.

### Measures

- Primary: stale-action rate after the environmental change.
- Secondary: redundant reads, time to rediscover the change, completion rate,
  token use, and incorrect snapshot promotions.
- Any incorrect promotion is inspected as a state-transition failure, not blamed
  on the model generically.

### Promotion rule

Promote only snapshot fields and transition authorities demonstrated by the
traces. Do not adopt a multi-agent topology unless a later experiment isolates a
benefit that typed state alone cannot provide.

## Experiment 5: Protected-spine compaction

### Source hypothesis

[TRUSTMEM](https://huggingface.co/papers/2606.25161) treats memory consolidation
as a state transition requiring coverage, preservation, and faithfulness checks.

### Hypothesis

Separating a mechanically protected spine from summarizable context prevents
loss of active constraints, approvals, errors, disproven hypotheses, dirty-file
state, obligations, and artifact handles at a comparable token budget.

Null: the protected representation does not improve preservation or causes
enough stale retention to offset its benefit.

### Smallest experiment

- Construct 20 synthetic session fixtures with known must-retain, superseded,
  conflicting, and untrusted facts.
- Compare:
  - **A:** ordinary free-form summary;
  - **B:** mechanical spine extraction, deduplication, then summary of the
    unprotected remainder.
- First score the transition mechanically. Use a model continuation only as a
  secondary behavioral probe.

### Measures

- Must-retain preservation and invented protected facts.
- Correct removal of explicitly superseded facts.
- Token size after compaction.
- Downstream constraint violations on a small continuation task.

### Promotion rule

The protected spine requires 100% preservation on the frozen mechanical suite.
If it accumulates stale facts, refine explicit invalidation rather than allowing
the summarizer to silently decide what no longer matters.

## Experiment 6: Evaluator validity audit

### Source hypothesis

[Benchmarking the Benchmarks](https://huggingface.co/papers/2607.02577),
[AgentProcessBench](https://huggingface.co/papers/2603.14465), and
[AgentRewardBench](https://huggingface.co/papers/2504.08942) warn that outcome,
process, and judge signals disagree in consequential ways.

### Hypothesis

A single aggregate or LLM-judge score will conceal disagreements among task
completion, side effects, evidence validity, and process quality in swe-term
trajectories.

This is a calibration experiment, not a pass/fail feature test.

### Smallest experiment

- Sample 30 trajectories across successful, failed, truncated, stale-receipt,
  and unsafe-effect cases from Experiments 1–5.
- Label each independently on:
  - task outcome;
  - tool correctness;
  - side effects;
  - evidence validity;
  - process errors and neutral exploration.
- Compare deterministic checks with an LLM judge repeated at least three times.
- Obtain two human passes. If only one reviewer is available, blind and reshuffle
  the second pass after a delay; report that limitation.

### Measures

- Pairwise disagreement by dimension, not only total agreement.
- Judge run-to-run variance.
- False `done` and unsafe-success rates.
- Cases where deterministic checks and human judgment each miss real behavior.

### Promotion rule

Keep dimensions whose disagreements would change a release decision. Never
collapse safety-critical failures into an average score, and preserve a human
audit sample for every later harness comparison.

## Deferred experiment: harness self-repair

[AutoSaddler](https://huggingface.co/papers/2608.23041) should be tested only after
Experiments 2, 3, and 6 produce stable gates and a trustworthy failure corpus.
The first version may propose a patch in an isolated branch, but it cannot merge,
alter its evaluator, or update the frozen corpus. A candidate survives only if it
fixes held-out failures without regressing the full evidence-gated suite.

## Execution sequence

1. Run Experiments 2 and 3 as pure deterministic prototypes.
2. Freeze the first repair fixtures and run Experiment 1.
3. Reuse the event and receipt vocabulary for Experiment 4.
4. Run Experiment 5 before implementing production compaction.
5. Audit the accumulated trajectories with Experiment 6.
6. Write one architecture decision per promoted mechanism; rejected hypotheses
   remain documented with their evidence.

The stopping point after each experiment is a written decision, not automatic
implementation. A negative result is useful if its fixtures, traces, and
limitations are reproducible.
