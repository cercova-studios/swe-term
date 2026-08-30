# swe-term Architecture

This is the authoritative architecture contract for the repository. It describes
both the implemented system and the accepted target design; every contract is
marked accordingly so plans are not mistaken for shipped behavior.

Long-form rationale lives in [`docs/core/GOLANG_TUI_PLAN.md`](docs/core/GOLANG_TUI_PLAN.md).
Comparative evidence lives in [`docs/core/FRAMEWORKS.md`](docs/core/FRAMEWORKS.md)
and [`docs/research/`](docs/research/).

## 1. Document Contract

- Read this file before architectural work, feature design, or non-trivial code
  changes.
- `Implemented` means the contract is present in current source and can be cited
  as runtime behavior.
- `Target` means the contract is an accepted design constraint but is not yet a
  runtime claim.
- The status of a contract changes only when source, tests, and this document
  change together.
- `docs/core/ARCHITECTURE.md` is a compatibility pointer, not a second source of
  truth.

## 2. Design Goals

- Minimal, stable Go core with explicit interface boundaries.
- Single-binary, local-first operation with optional sidecars and services.
- ACP-aligned protocol model at the frontend/core boundary.
- Deterministic safety and clear operational behavior.
- Cloud-agnostic capability ports without vendor lock-in.
- Cheap operation in constrained environments: stream instead of buffer, keep
  resident state bounded, and load capabilities lazily.

## 3. Core Principles

- Keep the core small; push specialized logic to extensions.
- Treat architecture as contracts: interfaces over framework magic.
- Prefer immutable snapshots and explicit state transitions.
- Separate agent infrastructure from product surface concerns.
- Optimize for debuggability, replay, and predictable rollback.
- Spend complexity only on correctness or a measured operational constraint.

## 4. System Shape and Boundaries

### Layers

1. **Frontends** — TUI, headless/pipe mode, and future RPC/server mode.
2. **Core (Go)** — provider event contracts today; target agent loop, state,
   approval, policy, journal, and verification control plane.
3. **Extensions** — providers, tools, analyzers, and sidecar adapters.
4. **External services (optional)** — retrieval, indexing, and sandboxes behind
   narrow adapters.

### Boundaries

- Frontend/core communication remains wire-protocol friendly.
- Core/extension interaction is interface-driven.
- Extension internals never leak into the core state model.
- Frontends render state; they do not own provider pricing, policy, verification,
  or persistence semantics.
- Heavy or language-specific engines run out of process behind bounded adapters.

### Current runtime slice

`main.runOnce` sends one `StreamRequest` to a `Provider`, collects the terminal
stream into a `Response`, and renders it. The interactive TUI consumes the same
provider event contract. There is not yet a multi-step tool-using agent loop.

## 5. Domain Types

| Contract | Status | Role | Source or intended home |
|---|---|---|---|
| `Message`, `StreamRequest`, `StreamEvent`, `Model` | Implemented | Provider-neutral request and streaming event vocabulary | [`internal/core/provider.go`](internal/core/provider.go) |
| `Provider` | Implemented | Streams completions and exposes model capabilities | [`internal/core/provider.go`](internal/core/provider.go) |
| `Usage`, `UsageTotals`, `Response` | Implemented | Provider-reported completion and session accounting | [`internal/core/usage.go`](internal/core/usage.go) |
| `Tool` | Target | Declares a versioned schema and executes a bounded invocation | [`internal/core/`](internal/core/) |
| `EffectDeclaration` | Target | Per-invocation paths, processes, network destinations, and secret names | [`internal/core/`](internal/core/) |
| `SessionStore` | Target | Persists sessions, journal checkpoints, task snapshots, and receipt indexes | [`internal/core/`](internal/core/) |
| `TaskSnapshot` | Target | Externalized, typed task state updated only from attributable evidence | [`internal/core/`](internal/core/) |
| `ControlEvent` / `ControlJournal` | Target | Durable ordered lifecycle record used by the safety monitor | [`internal/core/`](internal/core/) |
| `VerificationReceipt` | Target | Fresh, source-bound evidence that an obligation was discharged | [`internal/core/`](internal/core/) |
| `Obligation` | Target | Required check with kind, risk policy, status, and minimum V&V rung | [`internal/core/`](internal/core/) |
| `MutationLease` | Target | Enforces a single active mutation owner | [`internal/core/`](internal/core/) |

Target type names describe contracts, not frozen Go identifiers. A feature plan
may refine their representation without weakening Sections 6 or 10.

## 6. Control and Safety Model

- Risky or mutating actions pass through policy-gated approval.
- Read-only scheduling and safety authorization are separate decisions.
- A tool invocation declares bounded effects before execution. Observed effects
  outside that declaration fail closed and produce an auditable mismatch event.
- Approval, mutation lease, observed effect, snapshot, and verification events
  are written to a durable ordered control journal. Lossy telemetry is separate.
- A small deterministic monitor enforces ordering and closed, hand-authored rule
  IDs. The model may explain rules; it cannot create or weaken them.
- Tool truncation, timeout, cancellation, denial, and execution failure are
  distinct states. Omitted output has a durable artifact handle.
- Sandbox and escalation are adapters; missing required confinement is an error.
- Secret names may enter declarations and receipts. Secret values never do.

## 7. State and Context

- Prefer immutable snapshots and typed transitions over mutable global state.
- A token threshold decides when to compact, not what survives compaction.
- Compaction preserves a protected spine: active constraints and approvals,
  unresolved errors, disproven hypotheses, dirty-file state, obligations, and
  artifact handles.
- Every injected context item carries its source, snapshot/version, and
  freshness. Staleness is checked rather than assumed away.
- A verification receipt binds source state to verifier identity: binary digest,
  arguments, rule/config digest, sandbox/runtime, and relevant lockfiles.
- Graph reachability is three-valued: `found`,
  `not_found_in_complete_scope`, or `unknown`.
- Recoverability requires persisted sessions, resume semantics, and replayable
  control events.
- Multiple bounded read-only investigations may run concurrently under a shared
  budget, but they join before mutation.

## 8. Portability and Deployment

- Provider/model, storage, sandbox/compute, and retrieval/indexing are capability
  ports with swappable backends.
- The reference deployment is a local modular monolith and single Go binary.
- Sidecars are optional, independently restartable processes with versioned,
  bounded protocols.
- Moving to services requires a measured isolation, scaling, or ownership pain;
  it is not the default evolution path.
- State formats and journal schemas outlive implementations and require explicit
  versioning and migration paths.

## 9. Extension Points

- **Provider adapters — implemented.** Add model backends without changing core
  stream semantics.
- **Tool adapters — target.** Add bounded actions with the same manifest, effect,
  journal, and receipt contracts in every implementation language.
- **Analyzer/enrichment adapters — target.** Add pre-model context with explicit
  source, version, scope completeness, and freshness.
- **Session stores — target.** Swap persistence without changing state-machine
  semantics.
- **Policy and lifecycle hooks — target.** Intercept declared phases without
  rewriting the loop; hooks cannot bypass deterministic invariants.
- **Frontend protocol — target beyond the TUI.** TUI, headless, and ACP clients
  consume the same versioned event/state model.

Capabilities have two lanes: runtime/ad-hoc for iteration and compiled for
durable first-class support. Both emit the same versioned manifest and
policy-relevant events. Discovery is lazy and task-scoped; selected schema
versions are frozen into the session snapshot.

A capability moves into core only when independent extensions must agree on it
for safety, replay, or correctness. Popularity alone is not sufficient.

## 10. Invariants

These are non-negotiable and require mechanical enforcement or tests:

1. Exactly one terminal completion exists for a successful provider stream.
2. Approval precedes every mutating action that requires approval.
3. At most one mutation lease is active for a workspace.
4. Observed effects remain within the approved declaration.
5. Read-only investigations join and refresh contested state before mutation.
6. A lifecycle claim such as `verified`, `done`, or `ready_to_merge` requires a
   current receipt bound to the current source and verifier state.
7. Obligation kind and risk determine a minimum V&V rung; a model may escalate
   but never downgrade it.
8. Control events required for replay and safety do not drop.
9. Truncation, timeout, cancellation, denial, and failure are never flattened
   into successful text output.
10. Compaction cannot silently evict the protected spine.
11. Incomplete graph absence is `unknown`, not proof of non-existence.
12. Unknown pricing, provenance, freshness, or confinement remains explicit; it
    is never converted to a reassuring zero or success.
13. Extension internals and vendor-specific types do not enter core state.
14. Secret values are absent from prompts, journals, receipts, and artifacts.

## 11. Verification and Operations

- Table-driven trace tests cover fail-closed ordering and journal monitor rules.
- User-journey tests use mock providers and real local tool wiring; they assert
  observable behavior rather than implementation helpers.
- Receipt verification is independent of the model that requested the check.
- Evaluations separate tool invocation, task outcome, side effects, evidence
  quality, and reproducibility. Aggregate scores do not replace component data.
- Harness research follows the VCS-neutral specification, preregistration,
  isolation, and evidence-promotion contract in
  [`experiments/README.md`](experiments/README.md). Experiment infrastructure is
  tooling around the core, not a second agent loop or state model.
- Paper-backed hypotheses enter that contract through the mechanism-first
  discovery, source fallback, signal/noise triage, and selection workflow in
  [`docs/research/papers/README.md`](docs/research/papers/README.md). Discovery
  may propose an experiment; it cannot promote a claim into architecture.
- Operational failures must retain enough state to resume, explain the failed
  rule, and identify whether retry is safe.
- Rollback restores versioned session/harness records; it does not erase the
  audit trail.

## 12. In-Flight Refactors

These are accepted direction, not completed runtime claims:

1. **One-shot stream to agent loop.** Introduce tool dispatch and steering while
   retaining the current provider completion invariant.
2. **Tool safety contract.** Replace the planned `ReadOnly() bool` design with
   per-invocation effect declarations before landing the first mutating tool.
3. **State expansion.** Define session snapshot, obligation ledger, mutation
   lease, receipt index, and durable control journal before production
   persistence.
4. **Frontend protocol seam.** Keep TUI behavior on core events while defining a
   versioned protocol usable by headless and ACP clients.
5. **Architecture path migration.** Root `ARCHITECTURE.md` is now authoritative;
   the old `docs/core/ARCHITECTURE.md` path remains a temporary compatibility
   pointer.

Plans that touch these items must state whether they advance, defer, or conflict
with the refactor. Conflicts require an explicit architecture decision.

## 13. Source Hierarchy

When documents conflict:

1. `ARCHITECTURE.md` at the repository root
2. `docs/core/PLAN.md`
3. `docs/core/FRAMEWORKS.md`
4. Detailed evidence in `docs/research/`
5. Extended rationale in `docs/core/GOLANG_TUI_PLAN.md`

Scoped operational contracts govern their own workflows unless they conflict
with this file: [`CONTRIBUTING.md`](CONTRIBUTING.md) for change delivery and
[`experiments/README.md`](experiments/README.md) for research execution and
evidence promotion. Paper discovery and experiment selection are governed by
[`docs/research/papers/README.md`](docs/research/papers/README.md).

Current source and tests decide whether an architectural target is implemented.
If code contradicts an implemented claim here, stop and resolve the drift before
building on either version.

## 14. How to Keep This Document Honest

Update this file in the same change when work alters:

- a layer or dependency direction;
- a domain contract in Section 5;
- an extension point or promotion boundary;
- an invariant, safety rule, or minimum verification requirement;
- a state, journal, receipt, persistence, or protocol schema;
- deployment shape, sidecar boundary, or source hierarchy;
- the status of an in-flight refactor.

For every update:

1. Label contracts `Implemented` or `Target` from current source evidence.
2. Link the owning source, test, plan, or evidence document.
3. Search for superseded names and paths across code, tests, prompts, and docs.
4. Remove stale compatibility surfaces unless their owner and removal condition
   are documented.
5. Run reference checks and the smallest relevant verification suite.
