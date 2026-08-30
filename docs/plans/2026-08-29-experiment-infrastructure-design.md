# Constrained experiment infrastructure

Status: Phase 0 scaffold implemented; execution runner proposed

## Problem

Reimplementing an AI research paper usually fails for mundane reasons before it
fails scientifically: environment drift, undocumented prompts, unbounded
processes, mutable fixtures, missing raw traces, evaluator ambiguity, and an
accidental dependency on one VCS or cloud runtime.

swe-term needs a small experimentation substrate that makes those concerns
boring. The substrate should support pure deterministic state-machine tests,
model-mediated coding tasks, and isolated third-party paper implementations
without forcing every experiment into Git, containers, a database, or a
distributed scheduler.

## Design decision

The experiment is a versioned manifest plus immutable content, not a branch or
workspace. Version control is a source adapter. Execution isolation is a
separate adapter. Evidence is an append-only event stream plus content-addressed
artifacts.

The reference implementation is a local Go command using the standard library
where possible. It should run with bounded memory and concurrency, stream output
instead of buffering it, and use ordinary files before introducing a database.

Paper discovery is an upstream selection layer, governed by
[`docs/research/papers/README.md`](../research/papers/README.md). It records why a
mechanism is worth preregistering, including rejected candidates and evaluator
risks. It is not part of the execution runner and cannot bypass the experiment
ready gate.

## Minimal tool set

### 1. Specification compiler and validator

Responsibilities:

- parse a versioned manifest;
- reject unknown fields and unsupported schema versions;
- enforce preregistration completeness;
- resolve variants, repetitions, budgets, and evaluator contracts;
- make deterministic, seeded, and uncontrolled randomness explicit;
- emit a canonical manifest digest used by every run.

Phase 0 provides the manifest, JSON Schema, templates, and a dependency-free Go
validator.

### 2. Source materializer

Input is a `source.kind`, locator, immutable revision where applicable, and
content digest. Output is a fresh directory plus a verified digest.

Adapters, in implementation order:

1. `directory` — recursively copy a frozen fixture subtree;
2. `generated` — invoke a deterministic generator using an argv array;
3. `archive` — verify and unpack a content-addressed archive;
4. `jj` — materialize a Jujutsu revision without leaking workspace semantics;
5. `git` — use a commit/worktree when repository behavior is under test.

The runner consumes the resulting directory and provenance envelope. It does
not care which adapter produced it.

### 3. Environment adapter

The baseline `process` adapter provides:

- a fresh temporary working directory;
- an empty environment plus explicit allowlist;
- argv execution without a shell;
- wall-time deadline and process-group cancellation;
- bounded concurrent workers;
- streaming stdout/stderr with explicit output limits;
- distinct timeout, cancellation, signal, and exit-code states; and
- cleanup ownership recorded before execution begins.

For command-level timing, an optional `hyperfine` adapter may wrap the process
invocation after the manifest fixes repetitions and warmups. Its JSON result is
an input artifact, not the experiment record: the runner still owns source
identity, argv/environment provenance, terminal state, resource observations,
and evidence promotion. Enable the adapter only when timing is a declared
outcome.

Optional `container` and external sandbox adapters may add hard memory, CPU,
filesystem, or network confinement. The manifest must distinguish a requested
limit from an actually enforced limit.

### 4. Model adapter

The model boundary freezes:

- provider and model identity;
- decoding/reasoning parameters;
- system and task prompt digests;
- tool-schema digest;
- token budget and provider-reported usage; and
- raw provider terminal state.

The runner should use the same provider contract as swe-term when that contract
exists. Phase 0 does not add a second model client.

### 5. Event and artifact recorder

Each run writes append-only JSONL events and immutable artifacts. Required event
families are:

- run lifecycle;
- source materialization and digest verification;
- process/model request and terminal state;
- tool invocation and observed effects;
- evaluator output;
- resource observations; and
- cleanup result.

Large output is stored once by digest and referenced from events. Truncation is
an explicit event with a durable artifact handle.

### 6. Evaluator

Evaluators are composable commands or deterministic built-ins. They receive a
read-only evidence envelope and return typed component judgments. Outcome,
process quality, side effects, and evidence validity remain separate.

LLM judges are optional evaluators, never the sole safety gate. Repeated judge
runs record variance and exact rubric/model identity.

### 7. Repetition scheduler

The scheduler expands a manifest into a deterministic matrix of fixture,
variant, and repetition. It has a fixed maximum parallelism and stable run IDs.
It may resume missing cells but never overwrite a completed run.

### 8. Reporter

The reporter derives summaries from raw events, retains discordant pairs, and
produces machine-readable component metrics plus a human-authored report shell.
It does not decide whether a mechanism becomes architecture.

## Reference flow

```text
manifest + frozen fixtures
        │
        ▼
validate and preregister
        │
        ▼
expand deterministic run matrix
        │
        ▼
materialize fresh source directory
        │
        ▼
execute through bounded environment adapter
        │
        ├── stream JSONL events
        └── store content-addressed artifacts
        │
        ▼
run independent evaluators
        │
        ▼
derive summary + human limitations
        │
        ▼
accept, reject, or revise hypothesis
```

## Constrained-runtime posture

- One local process and files first; no resident service.
- Go standard library before third-party frameworks.
- JSON for contracts, JSONL for event streams, SHA-256 for portable content
  identity.
- Stream output and hashes; do not materialize entire trajectories in memory.
- Default parallelism is one. Higher concurrency is explicit and bounded.
- SQLite becomes justified only when cross-run queries over files are measurably
  painful. A server database is out of scope.
- Containers are a quality dial for stronger isolation, not the baseline runtime.
- No UI until command output and artifact contracts are stable.

## Failure and blast-radius analysis

| Failure | Required behavior |
|---|---|
| Manifest incomplete or unknown field | Refuse before execution |
| Source digest mismatch | Refuse and retain provenance error |
| Process exceeds deadline | Kill owned process group; record timeout |
| Output exceeds limit | Continue or terminate per manifest; retain artifact handle |
| Runner interrupted | Record owned paths before launch; recover or diagnose next run |
| Evaluator crashes | Run remains unevaluated, never converted to failure or success |
| Network requested but not enforceable | Mark confinement unknown or refuse by policy |
| Cleanup fails | Preserve exact path and emit visible cleanup failure |
| Agent proposes promotion | Require human acceptance and architecture update |

## Delivery stages

### Phase 0 — specification scaffold (this change)

- structured Hugging Face Papers discovery packet and research-taste rubric;
- command surface for new/list/validate/ready;
- canonical directory-source digest and ready-time recomputation;
- versioned templates and JSON Schema;
- agent guidelines and promotion rules;
- tracked specs, ignored raw runs, curated evidence boundary.

### Phase 1 — deterministic local runner

- `directory` and `generated` source adapters;
- process adapter with scrubbed environment, timeout, cancellation, and output
  limits;
- JSONL lifecycle events and artifact hashing;
- deterministic matrix expansion and resume-without-overwrite.

Done when Experiments 2 and 3 can run without a model, VCS, network, container,
or database.

### Phase 2 — model and repository adapters

- reuse the swe-term provider boundary;
- add optional `jj` and `git` source adapters;
- add Git-aware disposable repositories only for coding tasks that need them;
- record provider usage and complete terminal state.

### Phase 3 — evaluator and reporting calibration

- component evaluators and human labeling format;
- repeated LLM-judge adapter;
- disagreement and dispersion reports;
- curated evidence promotion workflow.

### Deferred

- container adapter and hard resource enforcement;
- SQLite run index;
- remote workers;
- visual dashboard;
- automatic harness patch proposals.

Each deferred item requires a reproduced local limitation. None is a prerequisite
for the first deterministic hypotheses.
