# Paper synthesis: model-dependent harness mechanisms

Status: complete — handoff drafts only

| Experiment | Paper mechanism | Isolated local treatment | Explicitly excluded |
|---|---|---|---|
| structured verifier feedback | validation information design | deterministic typed feedback adapter over identical raw output | extra hints, new tools, larger budget |
| verified external task state | state updated only from attributable facts | read-only typed snapshot after an environmental change | multi-agent topology |
| protected-spine compaction | verify coverage, preservation, and faithfulness of memory transitions | mechanical protected-fact extraction plus ordinary remainder summary | learned consolidation or RL |
| evaluator validity audit | decompose evaluator signals and audit disagreement | deterministic, human, and repeated judge labels by dimension | one aggregate score as authority |

The sources are mechanisms to isolate, not mandates to reproduce. The four
experiment specifications below are intentionally `draft` until the user chooses
a reproducible model/provider and its usage budget:

- [`structured-verifier-feedback`](../../../../experiments/specs/structured-verifier-feedback/)
- [`verified-external-task-state`](../../../../experiments/specs/verified-external-task-state/)
- [`protected-spine-compaction`](../../../../experiments/specs/protected-spine-compaction/)
- [`evaluator-validity-audit`](../../../../experiments/specs/evaluator-validity-audit/)

## Rejected and deferred paths

| Candidate or mechanism | Disposition | Reason | Reopen condition |
|---|---|---|---|
| manager/executor/auditor topology ([LongHorizon-Harness](notes/2608.01964.md)) | reject | changes agent topology and state at once, confounding the typed-snapshot question | a typed-snapshot experiment isolates a residual failure that state alone cannot fix |
| learned or preference-guided memory consolidation ([TRUSTMEM](notes/2606.25161.md)) | reject | requires training and confounds the mechanical verifier with model quality | the mechanical protected spine passes and stale retention remains the dominant failure |
| extra semantic hints or new tools in the feedback treatment ([Structured Feedback](notes/2607.14167.md)) | reject | would test extra information rather than information structure | never as a treatment; only as a separately preregistered variable |
| paper-specific corrected benchmark components ([Benchmarking the Benchmarks](notes/2607.02577.md)) | reject | benchmark-bound artifacts that do not describe swe-term traces | a local trajectory corpus exists and a benchmark comparison becomes the question |
| AgentRewardBench (`2504.08942`) | reference-only | comparison point for judge-versus-human agreement, not a transplantable mechanism | the evaluator audit needs an external agreement baseline |
| harness self-repair (AutoSaddler, per the portfolio plan) | defer | needs stable gates and a trustworthy failure corpus first | experiments 2, 3, and 6 produce those |
| LLM judge as sole authority | reject | the evaluator audit exists to measure exactly its disagreement and variance | never |

## Research-taste update

The useful question is: “what variable can change while information, task order,
budget, and authority stay fixed?” A paper that supplies an impressive harness
topology but not that contrast is a source of hypotheses, not a blueprint.

## Search closure

The search stopped because each of the four families has one candidate whose
mechanism reduces to a single isolated variable, and because the remaining
blocker is not a literature gap but a material experimental choice (a named
model, provider, parameters, prompt and tool-schema digests, frozen fixtures,
and an evaluator identity) that a human must make before preregistration.
Independent support for three of the four families was not established (see
the search plan's stop conditions).

Reopen the search for a family when: its experiment cannot be preregistered
without a model choice the candidate does not constrain; a discordant result
points at a mechanism the candidate does not name; or a replication or
comparison paper appears for one of the single-preprint families.
