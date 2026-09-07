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

## Research-taste update

The useful question is: “what variable can change while information, task order,
budget, and authority stay fixed?” A paper that supplies an impressive harness
topology but not that contrast is a source of hypotheses, not a blueprint.
