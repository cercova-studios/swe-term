# Framework Comparison (Compressed)

This document is the terse, explicit comparison layer for architecture decisions.
Detailed evidence remains in each framework's deep-dive and critique docs.

## Executive Takeaways

- **pi-mono** is the primary architectural baseline to port.
- **Codex** has the strongest protocol seam and execution safety model.
- **Claude Code** has production-hardened permissions/orchestration lessons, but not a portable architecture.
- **Flue** is valuable for headless framework/deploy shape, not core agent loop internals.
- **Deep Agents** validates middleware composition and adapter patterns, but not the dependency stack.
- **Prime Agent** is the long-running/detachable product built *on* pi: steal protocol versioning, host-RPC, harness rollback — not IPython-as-the-loop.
- **DeepSeek Harness** is the strongest *correctness spec* (log-derived requests, seams, fail-closed sandbox) and the worst template to copy (226 packages, Cordis).
- **oh-my-pi** is the strongest *tool-quality* fork of pi: steal hashline, per-model edit contracts, summarized reads, Rust engines as **sidecars** — not N-API-in-process or the IDE kitchen sink.

## Port / Avoid Matrix

| Framework | Port | Avoid |
|---|---|---|
| **pi-mono** | Layered modularity, provider abstraction, loop/tool separation | God object session, dynamic import extensions, heuristic token counting |
| **Codex** | Protocol-first seam (SQ/EQ), layered execution safety, steerable one-task loop | Responses API lock-in, oversized runtime surface, shared mutable session core |
| **Claude Code** | Permission pipeline patterns, read-parallel/write-serial tool orchestration, startup prefetch | Monolithic query engine/state, feature-flag architecture as core structure |
| **Flue** | Headless-first framework framing, deployable harness lifecycle, host-side tool boundary | Treating borrowed core as original architecture, vendor-coupled durability assumptions |
| **Deep Agents** | Middleware composition model, pluggable backend protocol, adapter strategy (ACP/evals) | Deep framework dependency chain, LangSmith-shaped observability assumptions |
| **Prime Agent** | Client ≠ runtime, capability-gated local protocol, closed host-RPC, subagent admission+depth cap, harness snapshot/rollback, session catalog without loading runtimes | IPython as the only LLM tool, 11k-line `AgentSession`, three-process daemon as v1, dual TS/Python harness writers, process-split marketed as sandbox |
| **DeepSeek Harness** | `deriveMessages` + dispatch invariant, turn/step events, waterfall hook phases, capability seam triple, fail-closed argv sandbox, profile-as-patches, ACP as loop-agnostic adapter | Cordis kernel, 226-package granularity, per-file 100% coverage, in-process `eval` plugins, dual host/client runtimes, Typert/catalog gate factory |
| **oh-my-pi** | Hashline/snapshot-verified edits, per-model tool contracts, summarized read/grep, `ToolContext`, JSONL+SQLite split, prompts-as-files, mechanical compaction, harness metrics | In-process N-API as plugin ABI, eval→full tool registry, 9k-line session, kitchen-sink core, snapcompact as default, Bun lock-in |

## Cross-cutting synthesis (this round)

Three independent pi-family / harness systems converge on the same cuts swe-term already wants — and diverge on the packaging we must refuse.

| Concern | Best source | swe-term shape |
|---|---|---|
| Loop layering | pi-mono | Go `core/agent` over interfaces |
| Ghost-context prevention | DeepSeek Harness invariant | Reconstruct messages from `SessionStore` before every `Provider.Stream` |
| Detach / multi-frontend | Prime Agent protocol + Flue/Codex | `Frontend` is a client; version + capabilities on the wire |
| Tool quality | oh-my-pi | Sidecar engines + model-parameterized schemas |
| Long-running memory | Prime harness rollback | Versioned records, agent cannot rewrite base prompt |
| Sandbox | DeepSeek fail-closed + Codex policy | Adapter wraps argv; missing confinement is an error |
| Subagents | Prime admission handles (depth default 1) | Job IDs, hard cap, topology in SQLite |
| Code-exec | Prime closed host-RPC, **not** omp full re-entry | Optional sidecar, enumerated methods |
| Native engines | omp's *what* (grep/edit/tokenize), dsh landlock *how* (separate binary) | Spawn sidecars; never N-API into core |

**Do not grow another `AgentSession`.** Prime (11,288 ln) and omp (9,428 ln) are the post-pi cautionary pair. dsh's 496-line driver is the size proof that hooks + a log keep the loop small.

## swe-term Decision Mapping

| Decision Area | swe-term Direction | Primary Source |
|---|---|---|
| Core architecture | Interface-driven Go core with explicit layering | `../research/PI_MONO_*`, `ARCHITECTURE.md` |
| Protocol boundary | Frontend/core boundary should be wire-protocol friendly | `../research/CODEX_*`, `../research/PRIME_AGENT_*` |
| Safety boundary | Layered approvals + deterministic policy + sandbox | `../research/CLAUDE_CODE_CRITIQUE.md`, `../research/CODEX_CRITIQUE.md`, `../research/DEEPSEEK_HARNESS_*` |
| Context management | Token budget + explicit compaction + log-derived history | `../research/CLAUDE_*`, `../research/DEEPAGENTS_*`, `../research/DEEPSEEK_HARNESS_*` |
| Tool quality | Model-parameterized edit/read/grep; engines as sidecars | `../research/OH_MY_PI_*` |
| Long-running / detach | Protocol first; resident daemon only when needed | `../research/PRIME_AGENT_*`, `../research/FLUE_*` |
| Extensibility | Plugins/adapters over loop forks | `../research/PI_MONO_*`, `../research/DEEPAGENTS_*`, `../research/DEEPSEEK_HARNESS_*` |
| Deployment posture | Single-binary local-first core, optional sidecars/services | `ARCHITECTURE.md`, `../research/FLUE_*`, `../research/OH_MY_PI_CRITIQUE.md` |

## Agent Self-Report Synthesis (2026-08)

Three coding-agent CLIs were asked directly — as agents that operate inside a
harness every day — where a harness shaped like swe-term's snapshot would
cost them turns, context, or correctness, and to push back from lived
operational experience rather than validate the plan. Two are the exact
harnesses already studied here (`prime-agent`, `omp` = oh-my-pi); one is
outside that family (`codex`, OpenAI). Full transcripts, method, and the
findings not carried into `ARCHITECTURE.md`:
`docs/research/HARNESS_SELF_REPORT_2026-08.md`.

| Finding | prime-agent | omp | codex | Applied to |
|---|---|---|---|---|
| Compaction needs a protected spine, not just a token trigger | yes | yes | yes | `ARCHITECTURE.md` State and Context |
| Capability/tool/skill catalogs must load lazily, not inject eagerly | yes | yes | yes | `ARCHITECTURE.md` Extensibility Strategy |
| Tool output must never silently truncate; truncation/timeout/cancel are distinct explicit states | yes | yes | yes | `ARCHITECTURE.md` Safety Model |
| Single-active-task should permit bounded concurrent read-only investigation, not serial waiting | yes | yes | — | `ARCHITECTURE.md` State and Context |
| Injected/analyzer context needs source+freshness, not just text | yes | yes | partial | `ARCHITECTURE.md` State and Context |
| `Tool.ReadOnly() bool` conflates safety signal with scheduling signal; needs an effect declaration | — | — | yes (grounded in `GOLANG_TUI_PLAN.md`'s actual planned interface) | `BACKLOG.md` — interface rework, not yet applied |
| Both extensibility lanes need the same trust boundary (ad-hoc lane is where policy drift starts) | — | yes | — | `ARCHITECTURE.md` Extensibility Strategy |

Three-way unanimous convergence on independently-elicited answers is unusually
strong signal — treat disagreement with these four points as requiring an
explicit counter-argument, not a default override.

## Canonical Rule

When analysis docs disagree, default to:

1. `ARCHITECTURE.md` for architecture truth
2. `PLAN.md` for delivery sequencing
3. Deep-dive/critique docs for supporting evidence
