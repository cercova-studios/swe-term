# Harness experiments

This directory is the stable entrypoint for small, reproducible experiments on
agent-harness mechanisms. It separates experimental intent from execution
infrastructure so an agent can reproduce a paper claim without inventing a
workspace layout, evidence format, or promotion process.

The research portfolio lives in
[`docs/plans/2026-08-27-harness-hypothesis-experiments.md`](../docs/plans/2026-08-27-harness-hypothesis-experiments.md).
The infrastructure design lives in
[`docs/plans/2026-08-29-experiment-infrastructure-design.md`](../docs/plans/2026-08-29-experiment-infrastructure-design.md).
Paper-backed hypotheses enter through the discovery and research-taste contract
in [`docs/research/papers/README.md`](../docs/research/papers/README.md).

## Current command surface

```sh
just experiment-new structured-verifier-feedback
just experiment-digest structured-verifier-feedback
just experiment-validate structured-verifier-feedback
just experiment-ready structured-verifier-feedback
just experiment-list
```

- `experiment-new` creates a tracked specification from the versioned template.
- `experiment-digest` computes the canonical SHA-256 identity of a directory
  source so fixtures can be frozen without an ad-hoc hashing recipe.
- `experiment-validate` checks schema and internal consistency while a draft is
  being written.
- `experiment-ready` applies the stricter preregistration gate. An experiment
  must pass it before any result-producing run.
- `experiment-list` reports known specifications and their status.

The execution runner is intentionally not part of this first scaffold. Until it
exists, an experiment-specific prototype may run only after the preregistration
gate passes and must follow the isolation and evidence rules below.

## Directory contract

```text
experiments/
  README.md
  schema/
    manifest.schema.json       # portable machine-readable contract
  templates/
    README.md                  # preregistration/report template
    manifest.json
    FIXTURES.md
  specs/
    <experiment-id>/
      README.md
      manifest.json
      fixtures/
        README.md
  runs/
    <experiment-id>/<run-id>/  # raw append-only output; ignored by Git
  evidence/
    <experiment-id>/           # reviewed, redacted, publishable evidence
```

Specifications are reviewed inputs. Raw runs are local working data. Evidence
is a deliberate promotion of selected results after secret scanning and human
review; it is never an automatic copy of a run directory.

## Two experiment kinds

Every manifest declares `kind`:

- **`mechanism-hypothesis`** — the original shape. A paper-backed, falsifiable
  causal claim about an agent-harness mechanism, tested as a control/treatment
  A/B pair. Gated by `papers` (at least one, with title/url/claim) at the ready
  gate.
- **`benchmark`** — a direct, descriptive measurement of an existing tool or
  artifact: no causal claim, nothing manipulated, no academic paper backing it.
  Still fully preregistered, isolated, and evidence-promoted like any
  experiment — but gated by `design_references` (title/url/claim, pointing at
  an internal design document and the specific claim under measurement)
  instead of `papers`. `hypothesis`, `null_hypothesis`, and
  `independent_variable` stay required; frame them around what is actually
  varied (e.g. cache state across repeated queries), not a fabricated causal
  story. Fabricating a paper citation to satisfy the `papers` gate is not an
  acceptable workaround — use `kind: benchmark` instead.

A `benchmark` experiment exists to put a number on something a target
architecture depends on (an external tool's baseline cost, a competitor
system's ceiling) before that architecture is built. It is how a target
extension point in `ARCHITECTURE.md` §9 earns evidence before code exists for
it.

## Agent preflight

Before implementing or running an experiment:

1. For a paper-backed hypothesis, read its discovery packet and confirm that the
   paper is marked `experiment-candidate`. Read the experiment specification and
   its cited paper section.
2. Read root `ARCHITECTURE.md`, especially Sections 5, 9, 10, and 12.
3. Run `just experiment-validate <id>` while editing.
4. Freeze fixtures, variants, budgets, model/tool identities, metrics, and the
   source content digest. Declare randomness as deterministic, seeded, or
   uncontrolled; uncontrolled runs require a written justification.
5. Set status to `preregistered` and run `just experiment-ready <id>`.
6. Stop if the ready gate fails. Do not weaken the validator to make a draft
   pass.

A discovery disposition does not replace preregistration. If the experiment
cannot preserve the isolated mechanism, boundary conditions, or declared
independent variable, return it to discovery as `defer` rather than testing a
different claim under the original citation.

## Reproducibility rules

- Change exactly one independent variable between paired variants.
- Use the same fixture order, budgets, environment policy, and repetition count
  across variants.
- Model-dependent experiments run at least three repetitions per task and report
  dispersion rather than only a mean.
- Seeded experiments provide one unique seed per repetition. Provider or system
  randomness that cannot be controlled is recorded as `uncontrolled`, never
  implied to be deterministic.
- Every run records the manifest digest, source digest, command arguments,
  allowed environment-variable names, tool/model identities, timestamps, exit
  state, resource observations, and artifact digests.
- Raw events are append-only JSONL. A failure to capture the terminal state makes
  the run invalid, not successful with missing telemetry.
- Unknown provenance, freshness, pricing, confinement, or evaluator state stays
  explicit.
- Negative and inconclusive results remain first-class evidence.

### Timing measurements

When an experiment needs command-level timing, the runner may invoke
[`hyperfine`](https://github.com/sharkdp/hyperfine) as an adapter. The manifest
must specify repetition and warmup policy explicitly; automatic benchmark
defaults are not part of the experiment contract. Store hyperfine's JSON output
under run artifacts and record the command, hyperfine version, source digest,
environment names, and resource observations in the normal run record.
Hyperfine provides timing statistics only; it does not replace terminal,
effect, artifact, or evidence-promotion records.

The directory digest binds sorted relative paths, executable bits, file sizes,
and streamed file contents. Symlinks and non-regular files fail closed so a
fixture cannot escape its declared tree. `experiment-ready` recomputes this
digest for `directory` sources.

## Isolation rules

The experiment contract is VCS-neutral. `source.kind` chooses how input is
materialized:

- `directory` — copy a frozen fixture subtree into a new temporary directory;
- `generated` — deterministically build fixtures from a versioned generator;
- `archive` — unpack an immutable, content-addressed archive;
- `jj` — materialize a named Jujutsu revision through an adapter;
- `git` — materialize a Git commit or worktree only when Git behavior is part of
  the hypothesis.

The default execution boundary is a new temporary directory with a scrubbed,
allowlisted environment. VCS workspaces are adapters, not the control plane.
Containers are an optional stronger boundary for untrusted dependencies or hard
resource enforcement; they are not required for deterministic reducer tests.

Never run a mutating trial in the author's working copy. Never expose ambient
credentials to a paper implementation. Network access is disabled by default and
must be justified in the manifest when changed.

## Evidence and promotion

An experiment may influence the harness only when:

1. its manifest was preregistered before result inspection;
2. all required repetitions produced complete terminal records;
3. component metrics and discordant cases were inspected;
4. the evidence directory contains a human-authored limitations section;
5. the proposed architecture change names the invariant or extension point it
   affects; and
6. a human explicitly accepts the promotion decision.

Agents may prepare evidence and propose a decision. They may not promote their
own result into an architectural invariant, alter the frozen corpus after seeing
results, or average away a safety-critical failure.

## Cleanup

Experiment-specific runners own only the temporary directories and run folders
they create. They must record those exact paths, handle interrupts, and remove
temporary execution state on success. Failed runs may retain their run directory
for diagnosis, but must say so explicitly. No cleanup command may target the
repository root, a home directory, or an unresolved environment variable.
