# Experiment: __EXPERIMENT_ID__

Status: draft

## Discovery provenance

- Discovery packet:
- Paper-note disposition:
- Local failure or architecture decision:
- Competing mechanisms rejected or deferred:

Paper-backed experiments must link a discovery packet whose synthesis marks the
mechanism `experiment-candidate`. Discovery is selection evidence, not proof that
the mechanism works locally.

## Paper claim

Identify the exact claim, method section, and boundary conditions being tested.
Do not substitute the paper's headline result for a mechanism.

## Hypothesis

Write one falsifiable sentence.

## Null hypothesis

State what observation would show the mechanism provides no useful improvement.

## Independent variable

Name the one intentional difference between control and treatment.

## Frozen inputs

- Source/fixture digest:
- Paper or implementation revision:
- Prompt digest:
- Tool-schema digest:
- Evaluator rubric digest:

## Procedure

Describe the smallest run matrix capable of falsifying the hypothesis.

## Metrics

- Primary:
- Secondary:
- Safety-critical counters that may not be averaged away:

## Acceptance and stop conditions

State promotion, rejection, and abort conditions before observing results.

## Risks and confinement

Document network, filesystem, process, model, and secret exposure. Distinguish
requested limits from limits the selected environment adapter can enforce.

## Results

Do not fill this section until the manifest is preregistered and
`just experiment-ready __EXPERIMENT_ID__` passes.

## Discordant cases

Inspect paired cases where variants or evaluators disagree.

## Limitations

This section requires human-authored interpretation before evidence promotion.

## Decision

- [ ] accept mechanism for a bounded follow-up
- [ ] reject hypothesis
- [ ] revise and preregister a new experiment
- [ ] propose architecture promotion with human approval
