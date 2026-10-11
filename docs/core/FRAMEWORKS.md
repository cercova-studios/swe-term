# Framework Comparison (Compressed)

This document is the terse, explicit comparison layer for architecture decisions.
Detailed evidence remains in each framework's deep-dive and critique docs.

## Executive Takeaways

- **pi-mono** (now `earendil-works/pi` 1.1.0) is the primary architectural baseline to port; it now ships a real `Provider` interface and before/after tool-call hooks.
- **Codex** has the strongest protocol seam and execution safety model.
- **Claude Code** has production-hardened permissions/orchestration lessons, but not a portable architecture; auto mode makes a model classifier the default gate — deterministic policy must still enforce.
- **Flue** (2.x) is valuable for headless framework/deploy shape and its accepted-work durability contract, not core agent loop internals (still pi's).
- **Deep Agents** validates middleware composition and adapter patterns, but not the dependency stack.
- **Prime Agent** is the long-running/detachable product built *on* pi: steal protocol versioning, host-RPC, harness rollback — not Python-REPL-as-the-loop. Now a Rust rewrite (0.10.0).
- **DeepSeek Harness** is the strongest *correctness spec* (log-derived requests, seams, fail-closed sandbox) and the worst template to copy (340 packages, Cordis). It dropped its own runtime invariant; swe-term keeps it.
- **opencode** (V2, 2.0.x) is the closest production match to swe-term's shape — protocol-first, event-sourced session, durable inbox — but ships no OS sandbox and in-process npm plugins.
- **oh-my-pi** is the strongest *tool-quality* fork of pi: steal hashline, per-model edit contracts, summarized reads, Rust engines as **sidecars** — not N-API-in-process or the IDE kitchen sink.

## Port / Avoid Matrix

Revalidated against current upstream on 2026-10-10; each deep dive has a "Changes since" section.

| Framework | Port | Avoid |
|---|---|---|
| **pi-mono** | Core-four layering, `Provider` interface, injected `streamFn` + before/afterToolCall hooks, built-in MCP/codemode as references | God-object session (4.4k ln), jiti dynamic-import extensions, URL-sniff compat fallback, chars/4 tail estimate, experimental durable/chord stack |
| **Codex** | Protocol-first seam (SQ/EQ, now a single app-server façade), layered execution safety, out-of-process code runtime, steerable one-task loop | Responses API lock-in, growing runtime surface (8 SQLite DBs, dual Guardian LLMs), shared mutable session core |
| **Claude Code** | Permission rules + OS sandbox with default-deny egress, read-parallel/write-serial tool orchestration, startup prefetch | Monolithic query engine/state, feature-flag architecture as core structure, model classifier as sole safety gate |
| **Flue** | Headless-first framework framing, accepted-work durability contract (submission → one terminal outcome, tool-batch repair, `step.do`), host-side tool boundary | Treating borrowed pi core as original architecture, opt-in durability (in-memory default), API churn (2.0 reset-only redesign) |
| **Deep Agents** | Middleware composition (named, replaceable defaults), pluggable backend protocol, thin ACP adapter, agent-as-standard-artifact for Harbor | Deep framework dependency chain, LangSmith as a core dependency, unconfined local shell backend |
| **Prime Agent** | Client ≠ runtime, capability-gated protocol (survived the TS→Rust rewrite unbumped), closed host-RPC, spawn + bounded collect with depth cap, harness snapshot/rollback, session catalog without loading runtimes | Python REPL as the only LLM tool, god session class (15k ln; needed a rewrite to split), resident daemon as v1, dual Rust/Python harness writers, process split marketed as sandbox |
| **DeepSeek Harness** | `deriveMessages`-built frozen requests (keep our runtime check; dsh dropped theirs), versioned session format + migrations, turn/step events, waterfall hook phases, capability seam triple, fail-closed argv sandbox, profile-as-patches, ACP v1 as loop-agnostic adapter | Cordis kernel, 340-package granularity, per-file 100% coverage, in-process `eval` plugins, dual host/client runtimes, Typert/catalog gate factory |
| **opencode** | History-only `drain`, event-sourced session + projector, durable steer/next-turn inbox, last-match permission rules (empty = deny), `opaque` shell scan, checkpoint compaction with verbatim tail | In-process npm plugins, no OS sandbox, fork-every-tool-call fan-out, dual provider SDKs, Effect as architecture, product mass in-repo |
| **oh-my-pi** | Hashline/snapshot-verified edits (Rust engine), per-model tool contracts, summarized read/grep, `ToolContext`, JSONL+SQLite split, prompts-as-files, mechanical + notes-backed compaction, harness metrics | In-process N-API as plugin ABI (now edit/VCS too), eval as tool gateway (Code Mode, browser/computer), 13.6k-line session, kitchen-sink core, snapcompact as default, Bun lock-in |

## Cross-cutting synthesis (this round)

Three independent pi-family / harness systems converge on the same cuts swe-term already wants — and diverge on the packaging we must refuse.

| Concern | Best source | swe-term shape |
|---|---|---|
| Loop layering | pi-mono | Go `core/agent` over interfaces |
| Ghost-context prevention | DeepSeek Harness invariant | Reconstruct messages from `SessionStore` before every `Provider.Stream` |
| Detach / multi-frontend | Prime Agent protocol + Flue/Codex/opencode | `Frontend` is a client; version + capabilities on the wire |
| Tool quality | oh-my-pi | Sidecar engines + model-parameterized schemas |
| Long-running memory | Prime harness rollback | Versioned records, agent cannot rewrite base prompt |
| Sandbox | DeepSeek fail-closed + Codex policy | Adapter wraps argv; missing confinement is an error |
| Subagents | Prime spawn + bounded collect (depth default 2) | Job IDs, hard cap, topology in SQLite |
| Code-exec | Prime closed host-RPC, **not** omp full re-entry | Optional sidecar, enumerated methods |
| Native engines | omp's *what* (grep/edit/tokenize), dsh landlock *how* (separate binary) | Spawn sidecars; never N-API into core |

**Do not grow another `AgentSession`.** Prime (15,101 ln before its Rust rewrite split it) and omp (13,598 ln) are the post-pi cautionary pair; upstream pi itself grew to 4,398 ln. dsh's 705-line driver and opencode's 375-line loop are the size proof that hooks + a log keep the loop small.

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

1. Root [`ARCHITECTURE.md`](../../ARCHITECTURE.md) for architecture truth
2. `PLAN.md` for delivery sequencing
3. Deep-dive/critique docs for supporting evidence
