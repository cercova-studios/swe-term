# swe-term Roadmap (Compressed)

This file is the delivery plan only.

Architecture truth lives in [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md).
Extended design rationale lives in `GOLANG_TUI_PLAN.md`.
Framework synthesis lives in `FRAMEWORKS.md`.

---

## Scope

- Build a Go-first agent harness with explicit interfaces and ACP-aligned protocol types.
- Keep core minimal and stable; push specialized capabilities into extensions/adapters.
- Preserve portability through narrow capability ports for providers, compute, and storage.
- Grow with model capability through agent-editable extensions and evidence-based
  removal of obsolete scaffolding; see the
  [agent-editable harness direction](../plans/2026-09-05-agent-editable-harness.md).
- Make what counts as "done" mechanically checkable, so verification strength is
  a gate the model can escalate but not downgrade, rather than an instruction it
  can ignore; see the
  [test-quality and architectural-fitness direction](../plans/2026-09-20-test-quality-and-architectural-fitness-steering.md).

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

Follow-on target, once obligations and receipts exist in core:

- Define the **V&V rung ladder** that `Obligation` already references but
  nothing specifies (ARCHITECTURE.md §5, invariant 7). Rungs are keyed to what
  class of defect the evidence can catch, not to a filename — so a mock-heavy
  interaction test cannot discharge a high-risk obligation.
- The deterministic enforcement this depends on is already validated:
  `ApplyControlEvent` and `ApplyReceiptGateEvent`, evidence under
  [`experiments/evidence/`](../../experiments/evidence/).

### Phase 3: Extensibility Path

Deliver:

- Runtime extension lane (scripted or subprocess adapters).
- Compiled extension lane (first-class Go tools).
- Clear promotion path from ad-hoc capability to permanent tool.

Done when:

- New capabilities can be added without modifying the core loop.

Follow-on target, after loop, safety, persistence, and extension contracts exist:

- Pilot agent-authored tool extensions in isolated copies, with a frozen
  baseline, independent evaluation, human-approved activation, and rollback.
- Demonstrate rejection of evaluator/authority changes, interruption-safe
  activation, and removal of an obsolete intervention.
- Keep core-loop self-rewrites and autonomous persistent promotion outside this
  first slice. The [design](../plans/2026-09-05-agent-editable-harness.md) defines
  the scope and evidence requirements; no improvement runtime is implemented.

Analyzer/enrichment adapter — the first concrete extension of this kind
(ARCHITECTURE.md §9, §12 item 6):

- Ship the **Fleet CPG engine** as an out-of-process Rust sidecar (`cpgd`)
  behind `internal/analyzer/cpg`, producing a `ContextPacket` with explicit
  source, snapshot version, freshness, and three-valued scope. Five gated
  slices in the [implementation plan](../plans/2026-09-05-fleet-cpg-engine-implementation.md);
  nine benchmark experiments already back the design.
- Its highest-value consumer is structural-cost context: blast radius, change
  coupling, and duplication delta injected *before* the model decides, which is
  the one lever against patches that are locally correct and compound debt.

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

- **Single source of truth:** architecture decisions live in the root
  [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
- **Minimal core:** if a feature can be an extension, it should not be in core.
- **Deterministic safety:** risky actions require explicit policy pathways.
- **Portability:** backends stay swappable without core rewrites.
- **Constraints need sensors, not just statements.** A boundary that is only
  written down is feedforward-only — the agent (or human) "encodes rules but
  never finds out whether they worked"
  ([Böckeler](https://martinfowler.com/articles/harness-engineering.html)).
  Where a contract in `ARCHITECTURE.md` is mechanically checkable, it should
  have a check. The dependency rules in §4/§8/§10/§11 now do:
  [`internal/architecture/fitness_test.go`](../../internal/architecture/fitness_test.go).
  Adding a boundary to the architecture contract should come with adding its
  sensor, or an explicit note on why it can't have one.

---

## Active Queue

See `BACKLOG.md` for tooling and integration candidates.

---

## References

- [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md)
- `FRAMEWORKS.md`
- `../services/AST_SERVICE_ARCHITECTURE.md`
- [Research and experiments summary](../reports/research-and-experiments-summary.md) —
  the compiled digest of every finding and result, with citations
- [Harness hypothesis experiments](../plans/2026-08-27-harness-hypothesis-experiments.md) —
  the experiment portfolio and execution sequence
- [Fleet CPG engine implementation](../plans/2026-09-05-fleet-cpg-engine-implementation.md)
- [Test quality and architectural fitness steering](../plans/2026-09-20-test-quality-and-architectural-fitness-steering.md)
