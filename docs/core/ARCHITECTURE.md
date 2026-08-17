# swe-term Architecture (Canonical)

This is the canonical architecture spec.

For deep rationale, alternatives, and long-form references, see `GOLANG_TUI_PLAN.md`.

---

## Design Goals

- Minimal, stable Go core with explicit interface boundaries.
- Single-binary local-first operation with optional sidecars/services.
- ACP-aligned protocol model at the frontend/core boundary.
- Deterministic safety and clear operational behavior.
- Cloud-agnostic capability ports (no hard vendor lock-in).

## Core Principles

- Keep the core small; push specialized logic to extensions.
- Treat architecture as contracts: interfaces over framework magic.
- Prefer immutable snapshots and explicit state transitions.
- Separate agent infrastructure from product surface concerns.
- Optimize for debuggability and predictable rollback paths.

---

## System Shape

## Layers

- **Frontends**
  - TUI, headless/pipe mode, RPC/server mode.
- **Core (Go)**
  - Agent loop, protocol/event model, state/session control, approval boundary.
- **Extensions**
  - Providers, tools, analyzers, and sidecar adapters.
- **External services (optional)**
  - Retrieval/search/indexing/sandbox services behind narrow adapters.

## Contract Boundary

- Frontend <-> core communication should remain wire-protocol friendly.
- Core <-> extension interaction is interface-driven.
- Extension internals must not leak into core state model.

---

## Canonical Interfaces

- `Provider`
  - Streams model output/events and exposes model capabilities.
- `Tool`
  - Declares schema/behavior and executes bounded actions.
- `SessionStore`
  - Persists and restores session/thread context.
- `Hook` / policy interfaces
  - Enables lifecycle interception without loop rewrites.
- Analyzer-like enrichment interfaces
  - Add pre-LLM context enrichment outside the loop core.

The core loop orchestrates these contracts; it should not absorb their implementation detail.

---

## Safety Model

- Policy-gated approval for risky and mutating actions.
- Clear separation between read-only and mutating tool paths.
- A single read-only boolean is a scheduling signal, not a complete safety
  declaration: tools should be able to declare bounded effects (paths
  touched, processes spawned, network destinations, secret access) so
  approval can reason about risk independently of concurrency scheduling.
  Observed effects exceeding the declaration fail closed and emit an
  auditable mismatch event.
- Timeouts/cancellation propagated consistently.
- Split a durable ordered **control journal** from lossy telemetry. Approval,
  mutation lease, observed effects, snapshot, and verification-receipt events
  must not drop; frontend/metrics subscribers may. A small deterministic
  monitor over that journal enforces core ordering (approval before mutation,
  one active mutation owner, effects within declaration, receipts still
  current). Rule IDs are hand-authored and closed — not a model-authored
  policy language.
- Explicit event emission for approvals, denials, and side effects.
- Tool output is never silently lossy: truncation, timeout, and cancellation
  are distinct, explicit states with a durable handle to the omitted
  remainder — never flattened into an ambiguous text blob.
- Sandbox/escalation strategy is adapter-driven, not hardcoded per vendor/runtime.

---

## State and Context

- Prefer explicit state snapshots and typed transitions over ad-hoc mutable global state.
- Maintain token/context budget with deterministic compaction triggers, but a
  token threshold decides *when* to compact, not *what survives*. Compaction
  is a typed checkpoint with a protected spine — active constraints and
  approvals, unresolved errors, disproven hypotheses, dirty-file state,
  artifact handles — that mechanical deduplication and summarization operate
  around, never evict silently.
- Every injected or analyzer-sourced context item (enrichment, tool result,
  verification receipt) carries source, snapshot/version, and freshness.
  Treat it as stale unless checked — the same contested-tree discipline
  already required between parallel agents applies to context provenance.
  A receipt is an **envelope**: file Merkle plus verifier identity (binary
  digest, args, rule/config digest, sandbox/runtime, lockfile hashes) — not
  file hashes alone. Record secret *names*, never secret values.
- Graph reachability is three-valued (`found` / `not_found_in_complete_scope`
  / `unknown`). Incomplete name-graph absence is not a safety proof; it opens
  an obligation.
- Obligation kind and risk policy set a **mechanical minimum V&V rung**. The
  LLM may add or escalate checks; it must not downgrade required discharge.
- Preserve recoverability: session persistence, resume semantics, and traceable events.
- One active mutation owner by default; multiple bounded, cancellable
  read-only investigations may run concurrently under a shared budget, with
  an explicit join before any mutation. Serial reasoning is a good default —
  serial waiting on independent reads is not.

---

## Extensibility Strategy

- New capability should usually be a plugin/adapter, not a core change.
- Support two lanes:
  - **Runtime/ad-hoc lane** for fast iteration.
  - **Compiled lane** for durable first-class capabilities.
- Both lanes emit the same capability manifest and policy-relevant events.
  The ad-hoc lane is exactly where trust-boundary drift appears first; it
  does not get a lighter approval/audit contract for being faster to write.
- Promotion path: repeated ad-hoc need -> formalized extension -> optional core contract if necessary.
- Promotion test for what belongs in core rather than an extension:
  something is core when independent extensions must agree on it for safety,
  replay, or correctness — not merely once it becomes popular.   Cancellation,
  effect accounting, artifact identity, context-retention rules, control-journal
  semantics, and verification-receipt identity meet this bar; providers and
  analyzers generally do not.
- Capability discovery is lazy and task-scoped: expose a compact manifest by
  default, load full tool/skill/policy contracts only on selection. Freeze
  selected schema versions into the session snapshot so resume stays
  reproducible. Do not inject every installed capability into every turn.

---

## Portability Strategy

- Capability ports for provider/model, storage, compute/sandbox, retrieval/indexing.
- Backends chosen per concern, swappable behind interfaces.
- Avoid assumptions tied to one API protocol or cloud deployment substrate.

---

## Source Hierarchy

When documents conflict:

1. `ARCHITECTURE.md` (this file)
2. `PLAN.md`
3. `FRAMEWORKS.md`
4. Detailed evidence in deep-dive and critique docs
5. Extended rationale in `GOLANG_TUI_PLAN.md`
