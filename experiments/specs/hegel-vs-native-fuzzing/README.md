# Experiment: hegel-vs-native-fuzzing

Status: preregistered

Kind: benchmark

## Design reference

This benchmark measures the §8 claim in
[`docs/plans/2026-09-20-test-quality-and-architectural-fitness-steering.md`](../../../docs/plans/2026-09-20-test-quality-and-architectural-fitness-steering.md):
hegel-go must beat native `testing.F` on swe-term's defect classes by enough to
justify its cgo-backed beta dependency. The result informs the V&V rung ladder
and the core's cgo-free portability boundary.

## Hypothesis and null

hegel-go finds defect classes in swe-term's core that native Go fuzzing does not,
by enough margin to justify the dependency. The null is that native
`testing.F` detects equivalent classes with equal or lower runtime and
flakiness on the frozen corpus.

## Frozen inputs

- Fixture tree: `fixtures/`, revision `hegel-vs-native-fuzzing-fixtures-v1`
- Source digest: recorded in `manifest.json` by `experimentctl digest`
- Identical core defect corpus, seed schedule, task order, resource budget, and
  environment allowlist for both engines
- Rubric digest: recorded in `manifest.json`

## Procedure

Run native `testing.F` and hegel-go against the same frozen defect corpus for
the same number of repetitions and equal time budgets. Classify findings by
distinct defect class, replay every counterexample, and record whether it
minimizes and reproduces. Record setup and native-library failures as invalid
runs, not as missed defects.

## Metrics

- Primary: distinct defect classes detected per equal-budget repetition.
- Secondary: runtime, flaky failure rate, counterexample minimality,
  reproducibility, and setup/dependency cost.
- Never average away invalid runs, replay failures, or safety-critical
  corpus omissions.

## Acceptance and stop conditions

Adoption is justified only if hegel-go finds materially more distinct,
reproducible defect classes with acceptable runtime and flakiness. Reject the
adoption hypothesis when native fuzzing is equivalent or better. Abort on
mismatched corpus, seeds, limits, environment, or incomplete terminal records.

## Risks and confinement

The process adapter runs with network disabled and only `PATH` allowed. Runs use
a temporary directory and do not mutate the author's worktree. The native
hegel-go library is treated as an experiment dependency; loading, build, or
replay failures are explicit invalid-run states. No credentials are exposed.

## Results, limitations, and decision

These sections remain empty until a ready-gated run has complete terminal
records. A human must inspect discordant findings and limitations before any
dependency or architecture decision.
