# swe-term Roadmap (Compressed)

This file is the delivery plan only.

Architecture truth lives in [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md).
Extended design rationale lives in `GOLANG_TUI_PLAN.md`.
Framework synthesis lives in `FRAMEWORKS.md`.

---

## Scope

- Build a general-purpose, minimal, Go-first agent harness with explicit
  interfaces and ACP-aligned protocol types.
- Ship one runtime behind three surfaces: TUI, headless, and Python and
  TypeScript (Node/Bun) SDKs, in the manner of the Claude Agent SDK and pi's
  `@earendil-works/pi-coding-agent`. SDKs are thin clients of the `st`
  binary over the protocol (ARCHITECTURE.md §9 "SDK clients").
- obsidian-agent is the first SDK consumer and replaces LangGraph with it;
  its convergence gates C1–C3 are the SDK's acceptance tests. Requests and
  responses: [`../cross-project/obsidian-agent.md`](../cross-project/obsidian-agent.md).
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
- Minimal Python SDK client: open a session, stream events, and register
  client-hosted tools that the loop calls back over the protocol (OA-001).

Done when:

- Prompt -> tool call(s) -> final assistant response works end-to-end.
- Integration tests pass with mock provider + real tool wiring.
- obsidian-agent gate C1 passes: read-only vault Q&A runs through the loop
  from the Python client with a mock provider.

### Phase 2: Safety, Hooks, and Context Budget

Deliver:

- Pre/post tool and compaction hook lifecycle.
- Token budget manager and compaction triggers.
- Policy-gated approval path for risky/mutating actions.
- SDK typed output: the session result is validated against a caller-supplied
  JSON Schema; providers declare support; a violation is a failure (OA-002).

Done when:

- Long sessions compact predictably while preserving required task context.
- Mutating operations are explicit, visible, and policy controlled.
- obsidian-agent gate C2 passes: summarizer and Zettelkasten steps use SDK
  typed output, and its crash/replay/cancel/budget checks pass on swe-term.

Follow-on target, once obligations and receipts exist in core:

- The **V&V rung ladder** that `Obligation` references but nothing specified
  (ARCHITECTURE.md §5, invariant 7) now exists as a pure reducer:
  [`internal/core/vv_rung.go`](../../internal/core/vv_rung.go). Rungs are keyed
  to what class of defect the evidence can catch, not to a filename, and the
  closed (kind, risk) → minimum-rung table makes invariant 7 structural —
  there is no event that lowers a minimum, so downgrade is unreachable rather
  than merely forbidden. Remaining work is wiring it to a real obligation
  ledger, and deciding how a rung gets *assigned* to actual evidence.
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
- TypeScript SDK client for Node and Bun with the same contract as the Python
  client; a versioned, published SDK API.

Done when:

- Same core loop is operable from the TUI, headless mode, and both SDKs.
- obsidian-agent gate C3 (parity with LangGraph) passes on the Python SDK.

### Phase 5: Hardening

Deliver:

- Observability events for debugging and performance.
- Reliability guardrails (timeouts, retries where appropriate).
- Single-binary packaging flow and reproducible local verification.

Done when:

- Core workflows are reproducible, testable, and operationally boring.

---

## Competitive Bar: opencode V2

opencode V2 (v2.0.32, `501b15af26`) is the closest production harness to
swe-term's shape. Matching it is the floor; the roadmap wins only where
opencode has structurally declined. Evidence and checks:
[`HARNESS_VERDICTS.md` → opencode](HARNESS_VERDICTS.md#where-swe-term-must-beat-opencode-v2).

Parity (port, do not reinvent), by phase:

- **Phase 1:** history-only `Drain`; durable inbox (steer vs next-turn);
  incomplete tool input fails and never runs; shell tool reports exit only
  after output is captured; edit tool matches the LF view and preserves line
  endings.
- **Phase 2:** last-match permission rules (empty = deny) with typed
  `once|always|reject|corrected` replies; `opaque` shell scan ⇒ ask;
  checkpoint compaction with a verbatim recent tail; `OutputLimit` as a
  distinct outcome with bounded continuation.
- **Phase 4:** typed protocol with bounded subscribers that fail loudly on
  overflow.
- **Phase 5:** bounded provider retry with a capped `retry-after` and a
  separate timeout budget.

Differentiators (each needs its check before it is claimed):

1. **Confinement** (Phase 2): OS sandbox and default-deny egress under the
   permission layer. opencode has prompts only.
2. **Effect-scheduled tools** (Phase 1–2): `EffectDeclaration` decides
   parallel reads vs serial writes. opencode forks every call.
3. **Verifiable done** (Phase 2): obligations, receipts, and the V&V rung
   gate block exit. opencode ends when the model stops.
4. **Structural context** (Phase 3): the analyzer `ContextPacket` arrives
   before the model decides.
5. **Out-of-process third-party extensions** (Phase 3).
6. **Footprint** (Phase 5): single static binary with RSS and cold-start
   budgets, and a loop that stays under opencode's 363 lines.

Non-goals versus opencode: voice, mobile/desktop apps, hosted console, and a
second provider SDK.

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
  [`internal/architecture/fitness_test.go`](../../internal/architecture/fitness_test.go)
  (per-change gate, blocks). Slow-accumulating health is measured separately
  by [`cmd/drift`](../../cmd/drift) (`just drift`), which runs outside the
  change lifecycle, ratchets against a checked-in baseline, and reports
  rather than blocks. Adding a boundary to the architecture contract should
  come with adding its sensor, or an explicit note on why it can't have one.

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
