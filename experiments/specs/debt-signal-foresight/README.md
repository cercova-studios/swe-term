# Experiment: debt-signal-foresight

Status: preregistered

Kind: benchmark

## Design reference

This benchmark measures the §8 claim in
[`docs/plans/2026-09-20-test-quality-and-architectural-fitness-steering.md`](../../../docs/plans/2026-09-20-test-quality-and-architectural-fitness-steering.md):
the Fleet CPG overlay can expose blast radius and change coupling before an
agent chooses a plan. It informs the target analyzer/enrichment adapter and
`ContextPacket` contract in `ARCHITECTURE.md` §§5 and 9.

## Hypothesis

Injecting CPG blast radius and change coupling into the pre-decision packet
changes the agent's plan and reduces structural debt, measured by coupling and
public-surface growth, compared with the same task without that context.

The null is that the packet does not change plan choice or resulting debt beyond
random variance.

## Frozen inputs

- Fixture tree: `fixtures/`, revision `debt-signal-foresight-fixtures-v1`
- Source digest: recorded in `manifest.json` by `experimentctl digest`
- Same task corpus, model identity, budget, source revision, and evaluator for
  both variants
- Rubric digest: recorded in `manifest.json`

## Procedure

Run each frozen brownfield task in paired control and treatment conditions.
Control receives no CPG context; treatment receives the blast-radius and
change-coupling packet before plan selection. Capture the plan, patch, terminal
state, and structural measurements for every repetition. Evaluate both variants
with the same CPG revision and compare task-level deltas rather than aggregate
means alone.

## Metrics

- Primary: change in structural debt across paired patches, using coupling and
  public-surface growth.
- Secondary: plan-choice divergence, blast-radius size, change-coupling score,
  files and lines changed, and task completion.
- Never average away missing terminal records, stale or incomplete graph scope,
  failed tasks, or invalid paired inputs.

## Acceptance and stop conditions

Support the hypothesis only when treatment produces a measurable debt reduction
without material task-completion loss. Retain the null when plan and debt stay
within the preregistered variance band. Abort a pair for mismatched inputs,
stale context, incomplete graph provenance, or missing terminal output.

## Risks and confinement

The process adapter runs with network disabled and only `PATH` allowed. Runs use
a temporary execution directory and must not mutate the author's worktree.
No model secrets or ambient credentials are exposed. Unknown graph scope remains
unknown rather than being treated as zero impact.

## Results, limitations, and decision

These sections remain empty until a ready-gated run has complete terminal
records. A human must inspect discordant pairs, limitations, and any proposed
architecture promotion.
