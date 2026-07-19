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
- Timeouts/cancellation propagated consistently.
- Explicit event emission for approvals, denials, and side effects.
- Sandbox/escalation strategy is adapter-driven, not hardcoded per vendor/runtime.

---

## State and Context

- Prefer explicit state snapshots and typed transitions over ad-hoc mutable global state.
- Maintain token/context budget with deterministic compaction triggers.
- Preserve recoverability: session persistence, resume semantics, and traceable events.
- Support steerable single active task semantics by default; add parallelism intentionally.

---

## Extensibility Strategy

- New capability should usually be a plugin/adapter, not a core change.
- Support two lanes:
  - **Runtime/ad-hoc lane** for fast iteration.
  - **Compiled lane** for durable first-class capabilities.
- Promotion path: repeated ad-hoc need -> formalized extension -> optional core contract if necessary.

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
