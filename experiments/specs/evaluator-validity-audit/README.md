# Experiment: evaluator-validity-audit

Status: draft. This calibration study must wait for raw trajectories from
earlier experiments and a named judge and annotation plan.

Kind: mechanism-hypothesis

## Discovery provenance (mechanism-hypothesis only)

- Discovery packet: [`docs/research/papers/model-dependent-harness/`](../../../docs/research/papers/model-dependent-harness/)
- Paper-note disposition: `experiment-candidate`
  ([`notes/2607.02577.md`](../../../docs/research/papers/model-dependent-harness/notes/2607.02577.md))
- Local failure or architecture decision: the target evaluation harness has
  no measured basis for trusting a single aggregate or model-judge score over
  swe-term trajectories; a release decision built on such a score could hide
  a safety-relevant disagreement.
- Competing mechanisms rejected or deferred: an LLM judge as sole authority is
  rejected; AgentRewardBench is reference-only as an external agreement
  comparison; benchmark-specific corrected components are not imported.

Paper-backed experiments must link a discovery packet whose synthesis marks the
mechanism `experiment-candidate`. Discovery is selection evidence, not proof that
the mechanism works locally.

## Paper claim

Benchmarking the Benchmarks reports that tool-calling evaluation can contain
evaluator-human disagreement and repeated-run variance large enough to alter
capability conclusions. This experiment tests only the decomposition of
evaluator signals over local trajectories; it does not reproduce the paper's
benchmark corrections or its headline agreement numbers.

## Hypothesis

A single aggregate or model-judge score conceals release-relevant
disagreements among task completion, side effects, evidence validity, and
process quality.

## Null hypothesis

A single score agrees sufficiently with component labels and repeated judging
to support the same release decisions.

## Independent variable

Collapsed evaluation score versus independently labeled component dimensions
with deterministic, human, and repeated model-judge signals.

## Frozen inputs

- Source/fixture digest: pending. The corpus is a sample of about 30
  trajectories across successful, failed, truncated, stale-receipt, and
  unsafe-effect cases from earlier experiments; it does not exist yet
  (`fixtures/`).
- Paper revision: arXiv `2607.02577`; version to be confirmed at
  preregistration.
- Prompt digest: pending; the judge prompt is frozen with the judge.
- Tool-schema digest: pending; not applicable unless the judge calls tools.
- Evaluator rubric digest: pending; filled from the annotation rubric before
  the ready gate.

## Procedure

Label every sampled trajectory independently on task outcome, tool
correctness, side effects, evidence validity, and process errors versus
neutral exploration. Produce three label sources per trajectory: deterministic
checks, two blinded human passes, and at least three repeated runs of one
named model judge under frozen settings. The control collapses these into a
single score or single judge result; the treatment keeps the dimensions and
sources separate. If only one reviewer is available, the second human pass is
a delayed blind reshuffle, and the limitation is recorded.

## Metrics

- Primary: pairwise disagreement by evaluation dimension.
- Secondary: judge run-to-run variance, false `done` rate, unsafe-success
  rate, and cases where deterministic checks and human judgment each miss
  real behavior.
- Safety-critical counters that may not be averaged away: false `done` and
  unsafe-success counts. A safety failure never becomes a compensating
  average.

## Acceptance and stop conditions

Retain every dimension whose disagreement would change a release decision and
preserve a human-audited sample for later harness comparisons. The null is
supported if the collapsed score reaches the same release decisions as the
component labels across the sample. Abort if the trajectory sample, judge
identity, or rubric changes after ready-gate approval, or if the second human
pass cannot be blinded.

## Risks and confinement

The `external` adapter with network inherited is requested because a named
model judge and a human-labeling workflow are involved; both are selected
before preregistration and recorded in `manifest.json`. Trajectories are
reviewed and redacted before they enter the corpus, so no secret, customer
code, or credential is exposed to the judge. No workspace is mutated and no
tool is executed; the experiment labels recorded traces only. The adapter
enforces the wall-time limit but not memory or CPU limits.

## Results

Do not fill this section until the manifest is preregistered and
`just experiment-ready evaluator-validity-audit` passes.

## Discordant cases

Inspect every trajectory where the collapsed score and any component label
disagree, and every trajectory where the repeated judge disagrees with itself.

## Limitations

This section requires human-authored interpretation before evidence promotion.

## Decision

- [ ] accept mechanism for a bounded follow-up
- [ ] reject hypothesis
- [ ] revise and preregister a new experiment
- [ ] propose architecture promotion with human approval
