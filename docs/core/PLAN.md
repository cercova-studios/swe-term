# swe-term Roadmap (Compressed)

This file is the delivery plan only.

Architecture and design rationale live in `ARCHITECTURE.md`.
Framework synthesis lives in `FRAMEWORKS.md`.

---

## Scope

- Build a Go-first agent harness with explicit interfaces and ACP-aligned protocol types.
- Keep core minimal and stable; push specialized capabilities into extensions/adapters.
- Preserve portability through narrow capability ports for providers, compute, and storage.

## Non-Goals

- Rebuilding a Claude/Codex-scale product surface in v1.
- Embedding heavyweight runtimes into core for optional behavior.
- Locking architecture to a single model protocol or cloud vendor.

---

## Milestones

### Phase 1: Core Loop and Contracts

Deliver:

- Core interfaces for `Provider`, `Tool`, `SessionStore`, and hooks.
- Streaming agent loop with tool dispatch and steering/follow-up channels.
- ACP-aligned protocol types and JSON-RPC framing.
- Baseline file/shell tools.
- SQLite-backed session persistence.

Done when:

- Prompt -> tool call(s) -> final assistant response works end-to-end.
- Integration tests pass with mock provider + real tool wiring.

### Phase 2: Safety, Hooks, and Context Budget

Deliver:

- Pre/post tool and compaction hook lifecycle.
- Token budget manager and compaction triggers.
- Policy-gated approval path for risky/mutating actions.

Done when:

- Long sessions compact predictably while preserving required task context.
- Mutating operations are explicit, visible, and policy controlled.

### Phase 3: Extensibility Path

Deliver:

- Runtime extension lane (scripted or subprocess adapters).
- Compiled extension lane (first-class Go tools).
- Clear promotion path from ad-hoc capability to permanent tool.

Done when:

- New capabilities can be added without modifying the core loop.

### Phase 4: Surface Layer

Deliver:

- Stable headless/pipe mode.
- TUI integration via protocol-backed event stream.
- Session navigation and resume ergonomics.

Done when:

- Same core loop is operable from both TUI and headless modes.

### Phase 5: Hardening

Deliver:

- Observability events for debugging and performance.
- Reliability guardrails (timeouts, retries where appropriate).
- Single-binary packaging flow and reproducible local verification.

Done when:

- Core workflows are reproducible, testable, and operationally boring.

---

## Cross-Cutting Requirements

- **Single source of truth:** architecture decisions live in `ARCHITECTURE.md`.
- **Minimal core:** if a feature can be an extension, it should not be in core.
- **Deterministic safety:** risky actions require explicit policy pathways.
- **Portability:** backends stay swappable without core rewrites.

---

## Active Queue

See `BACKLOG.md` for tooling and integration candidates.

---

## References

- `ARCHITECTURE.md`
- `FRAMEWORKS.md`
- `../services/AST_SERVICE_ARCHITECTURE.md`
