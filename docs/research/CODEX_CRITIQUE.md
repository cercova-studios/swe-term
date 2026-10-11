# Critical Evaluation of Codex as an Agentic Harness

> Evaluated against the Harness (swe-term) philosophy:
> zero-dependency core, interface-driven plugins, immutable state,
> goroutine-native concurrency, content-addressable caching, single binary.
>
> **Validated against:** upstream `cc7ba3360` (2026-10-10). Previous: `b89ce9a2b` (2026-06-06).
> Evidence for every change below is in `CODEX_DEEP_DIVE.md` § "Changes since 2026-06-06".

---

## Verdict Changes (2026-10-10 revalidation)

| Verdict | Change | Why (evidence) |
|---------|--------|----------------|
| Port the protocol seam | **Strengthened** | Codex dropped its third façade (`codex mcp-server`, `531f3836a`); the app-server is now the only external surface, and `Submission` became core-internal (`core/src/session/submission.rs`). |
| Do not embed V8 | **Refined** | Codex itself moved V8 out of the agent process into a `codex-code-mode-host` binary over gRPC/stdio (`aa46f2deb`, `2dc0791c7`). That validates our subprocess (Monty) direction; it does not make V8 lighter. |
| "Two of everything" | **Partly resolved upstream** | `execpolicy-legacy` (`656a2d090`), legacy remote compaction (`1ac689cc7`), `core-skills` (`45f8cafa4`), and the `Op::UserTurn` alias are deleted. Multi-agent v1+v2 and the `ConversationManager = ThreadManager` alias (`core/src/lib.rs`) remain. |
| One DB, not four | **Strengthened** | SQLite files grew from 4 to 8 (`state/src/sqlite.rs`, `ext/agent-message-board/src/local/pool.rs`). |
| No LLM in the approval hot path | **Strengthened** | Guardian now has two implementations (`ext/guardian-reviewer`, ~16K-line `ext/guardian-v2` async scorer) plus a user override op. |
| Port the 10 hook events | **Updated** | Now 12 (`SessionEnd`, `Interrupt`) plus an `McpTool` handler (`hooks/src/lib.rs`). |
| Responses-API monoculture, shared-mutable session, agent identity | **Unchanged** | `WireApi` still has only `Responses`; the new `amazon-bedrock-runtime` provider uses Bedrock's OpenAI-compatible endpoint. `Mutex<SessionState>` remains (the I/O split, `e4711f2a3`, did not change it). |

---

## Verdict Summary

Codex is the **most architecturally serious** of the agents surveyed (Claude Code, pi-mono,
Flue, Codex). Where Claude Code bolted modularity onto a monolith, Codex was designed
**protocol-first**: a canonical Submission/Event queue in `codex-core`, with every surface
(TUI, headless exec, VS Code, SDKs) reduced to a *client* of that protocol. That seam is
exactly the boundary the Harness philosophy demands — and Codex draws it more cleanly than any
TypeScript agent here.

But Codex pays for its completeness with **mass and lock-in**. It is a ~155-member Rust-2024
workspace (up from 116 in June) with four OS sandbox backends, two language runtimes (V8 — now in a
separate host process — and Starlark), dual persistence (JSONL + eight SQLite files), two automated
review LLM implementations, an Ed25519 agent-identity system, and a hard bet on OpenAI's Responses
API. The agent loop is split across `session/mod.rs` (~5.3K lines), `session/turn.rs` (~3.2K), and
`client.rs` (~3.0K) — each roughly 40-60% larger than in June. It is extractable in *theory*
(the crate graph is real) but enormous in *practice*.

**Port the protocol seam and the execution safety model. Do not port the surface area, the
Responses-API assumption, or the shared-mutable session state.**

---

## 1. Effectiveness

### What Codex does well

**The SQ/EQ protocol seam is the reference design.** This is the thing to steal. `codex-core`
exposes one async queue pair (`Submission`/`Op` in, `Event`/`EventMsg` out), and *everything*
else is a client. The TUI doesn't import core internals; it speaks JSON-RPC to an app-server
that itself speaks SQ/EQ to core. This is precisely the Harness `Frontend` interface taken to
its logical end: the frontend is defined by a wire protocol, not a Go interface call. It means
the TUI, the headless mode, the IDE, and the SDK are genuinely interchangeable.

> **Lesson for Harness:** Our `Frontend` interface should be backed by a serializable
> event/command protocol from day one (the PLAN already aligns protocol types with ACP — good).
> The win is not "an interface"; it's that *the same protocol drives in-process and out-of-process
> frontends identically*. Codex's `InProcessAppServerClient` vs `RemoteAppServerClient` is the
> pattern: one client API, two transports.

**The execution safety model is production-grade and layered.** ExecPolicy (Starlark prefix
rules) → hooks → guardian/user → network approval → patch safety → OS sandbox. Three real native
sandboxes (landlock+seccomp, seatbelt, Windows restricted token + WFP), with policy-driven
escalation and *amendable* approvals (approve once → append an allow rule). This is far beyond
Claude Code's pattern-matching `bashClassifier` and is the most complete sandboxing story in any
open agent.

> **Lesson for Harness:** `ext/tools/terminal/safety.go` should model the layered pipeline, and
> the OS-sandbox wrapper (prepend a helper binary that applies seccomp/landlock) is the right
> shape for a single-binary Go agent — the sandbox helper can be the same binary via an arg0
> dispatch.

**One task per session is the right concurrency invariant.** Codex's `Session` runs exactly one
task at a time; new input either *steers* the active turn or *replaces* it (with a 100ms graceful
cancellation window). This is simpler and safer than ad-hoc parallel turns, and the steering /
mailbox model is a clean way to handle interrupts and inter-agent messages.

> **Lesson for Harness:** Adopt "one active turn, steerable" as the core invariant. The
> `internal/agent/queue.go` (steering + follow-up channels) in the PLAN is exactly this.

### What Codex gets wrong (for *our* purposes)

**The agent loop is not small, and it is growing.** `session/turn.rs` is ~3,209 lines (was ~2,191
in June); `session/mod.rs` is ~5,287 (was ~3,317). This is much better factored than Claude Code's 46K-line `QueryEngine.ts`, but it is
not the ~500-line pure function the Harness targets. The turn loop interleaves compaction
(pre/mid), skill/plugin injection, hook dispatch, steering drain, token-budget gating, and
streaming tool execution in one place.

> **Lesson:** Even Rust discipline doesn't keep an agent loop small if it owns every concern.
> Harness should push compaction, hooks, and injection *behind interfaces* the loop calls, not
> inline them.

**Responses-API monoculture.** `WireApi` has effectively one live variant: `Responses`. Chat
Completions is rejected. Bedrock/Ollama/LM Studio are supported only because they speak (or are
made to speak) the Responses shape — still true at `cc7ba3360`, including the new
`amazon-bedrock-runtime` provider (`model-provider/src/amazon_bedrock/runtime.rs`) — and
compaction is now *only* local or server-side remote v2. This is a deep coupling to one vendor's protocol.

> **Do not port:** A provider model that assumes the Responses API. The Harness `Provider`
> interface must be the *narrow waist* — `Stream(ctx, model, msgs, tools) -> <-chan Event` — and
> Responses-specific features (server-side conversation state, remote compaction) must live behind
> it, not leak into core.

---

## 2. Efficiency

### Compiled single binary: aligned

Codex compiles to native code with fat LTO and symbol stripping, and uses the **arg0 multitool**
pattern so one binary is the CLI, the Linux-sandbox helper, and the patch applier. (Code mode
is the exception: it now ships a separate `codex-code-mode-host` binary.) This is exactly the Harness "single binary" goal, and the arg0 trick is worth copying for
the sandbox helper specifically (Go supports it via `os.Args[0]` dispatch in `main`).

> **Port:** arg0 dispatch for the sandbox helper. It avoids shipping a second executable while
> keeping the sandboxed child in a separate process.

### Two embedded runtimes is two too many for us

Codex embeds **V8** (code mode — the model writes JS that orchestrates nested tool calls) and
**Starlark** (execpolicy). V8 (`v8 = "=150.4.0"`) is a heavy dependency with a long build and large
binary footprint. Code mode is a genuinely interesting idea — let the model batch tool calls in a
program instead of one-call-per-roundtrip — but shipping a full JS engine to get it is a large
bet. Since June, Codex has itself moved V8 out of the agent process: `code-mode-runtime` runs in a
`codex-code-mode-host` child that core drives over gRPC on stdio and kills/reaps on reconnect
(`core/src/code_mode_host.rs`). That is upstream evidence for process isolation of model-written
code — the same shape as our Monty subprocess.

> **Contrast with Harness:** The PLAN already chooses **Starlark** (Go-native, via
> `go.starlark.net`) for agent-authored tool defs and **Monty** (subprocess) for runtime code.
> That is the right instinct: get code-mode-like batching without linking a browser engine into
> your agent. Codex validates the *concept* of code mode; it does not validate *V8 in your binary*.

### Dual persistence is justified but heavy

JSONL rollout (canonical, append-only) + eight SQLite files (`state_5`, `logs_2`, `goals_1`,
`memories_1`, `memories_v2_1`, `queue_1`, `thread_history_1`, `agent_message_board_1`) via `sqlx`
with embedded migrations. This is more robust than pi-mono's JSONL-with-tree-walking (it has real
indexes and queryable metadata) but it's also six migrator sets (`state/src/migrations.rs`) plus the
message board's own, and a reconciliation layer (`LiveThread`) to keep JSONL and SQLite in sync.
The count doubled in four months.

> **Port the concept, simplify the instance:** The Harness PLAN already picks `modernc.org/sqlite`
> (pure Go, no CGo) for sessions. Good. But keep it to *one* DB with a `tree`/`branch` schema, not
> four. Codex's split (goals and memories in their own DBs) is product accretion, not architecture
> — it even has a migration that *drops* goals from the main DB to move them out (`0034_drop_thread_goals`).

### Managed network proxy: study it

Codex routes egress through a managed proxy (`network-proxy`, `responses-api-proxy`) with
allowlist approval. This is a strong pattern for sandboxed network access and is more principled
than "network on/off."

---

## 3. Simplicity

### Shared-mutable session state behind a Mutex

Codex's `Session.state: Mutex<SessionState>` is **shared mutable state**, not immutable snapshots.
`SessionState` carries the `ContextManager` history, rate limits, additional-context store, granted
permissions, auto-compact window, prewarm handle, connector selection, and more. Prompts are built
from a `clone_history()` snapshot taken under the lock.

This is *internally disciplined* — clippy denies `await_holding_lock`, and the snapshot-on-clone
pattern is sound — but it's the opposite of the Harness model.

> **Do not port:** A single mutable session struct behind a lock. Harness uses immutable
> `AgentState` snapshots (~10 fields) passed through the loop, so there's no lock to hold and no
> "everything reads the session" coupling. Codex's approach works *in Rust with lint enforcement*;
> in Go, a shared mutable struct behind a `sync.Mutex` is a data-race footgun and a coupling magnet.
> Snapshots are cheaper to reason about than disciplined locking.

### Two of everything (v1/v2, execpolicy/execpolicy-legacy)

In June the codebase carried duplicated subsystems mid-migration: multi-agent v1 *and* v2,
`execpolicy` *and* `execpolicy-legacy`, legacy `Op::UserTurn`/`ConfigureSession` aliases, and
deprecated type aliases. Since then Codex paid much of that debt down by deletion:
`execpolicy-legacy` (`656a2d090`), legacy remote compaction (`1ac689cc7`), the `core-skills` crate
(`45f8cafa4`), the `UserTurn` alias, `thread/rollback` (`3052bbcf8`), and `codex mcp-server`
(`531f3836a`) are gone. What remains: multi-agent v1 *and* v2 handlers (selected per model), the
`ConversationManager = ThreadManager` alias (`core/src/lib.rs`), and a new pair — Guardian
`ext/guardian-reviewer` *and* `ext/guardian-v2`.

> **Lesson:** This is what "evolve in place" looks like at scale — and Codex's own cleanup
> confirms that compile-time removal is the end state. Harness should go there directly: delete
> the old module instead of running both. Go's lack of a stable public API for an internal tool
> is an *advantage* here — there's no one to keep v1 working for.

### `TurnContext` and `Config` are large

`core/src/config/mod.rs` is ~5,071 lines and the runtime `Config` struct is huge (model, provider,
permissions, hooks, plugins, memories, features, guardian policy, sqlite home, …). Every turn
carries a `TurnContext`. This is closer to Claude Code's `ToolUseContext`-carries-the-universe
problem than the Harness ideal of tools that see only `context.Context` + `json.RawMessage`.

> **Do not port:** A `Config` that is also the runtime god-context. Harness keeps `Config` as a
> load-time struct (defaults → file → env → flags) and does *not* thread it into every tool.

---

## 4. Key Gaps (relative to Harness goals)

### No content-addressable cache

Like Claude Code, Codex has narrow, ad-hoc caches (model catalog cache, connector directory cache)
but no general `ContentKey(sha256) → value` store. File reads, tool outputs, and AST work are not
deduplicated by content.

> **Gap for Harness to fill:** `core/cache` with content addressing remains a differentiator.

### No analyzer concept

Codex's code intelligence is the model calling `exec_command`/`apply_patch`/`tool_search`. There
is no pre-LLM enrichment step — no repo map, no dependency graph injected into context before the
model sees it. (The `file-search` crate is a fuzzy filename matcher, not a semantic analyzer.)

> **Gap for Harness to fill:** The `Analyzer` interface (enrich context *before* the LLM) has no
> equivalent in Codex.

### Provider abstraction is shallow because the protocol is fixed

There *is* a `ModelProvider` trait, but because everything is Responses-API, the abstraction never
has to bridge genuinely different wire protocols (Anthropic Messages vs OpenAI Chat vs Responses).
The Harness `Provider` interface is doing harder work — and that's the point.

> **Gap/contrast:** Codex's provider layer is not a model for a *multi-protocol* agent. Ours must be.

---

## 5. Assumptions to NOT Port

### ❌ "The OpenAI Responses API is the universal model protocol"

Codex bets the whole core on it, including server-side compaction. A provider-agnostic harness
cannot make this bet. Responses-specific capabilities belong *behind* the `Provider` interface.

### ❌ "Ship V8 to get programmatic tool orchestration"

Code mode is a good idea executed with a heavy hammer. Get the batching benefit with a lightweight
embedded interpreter (Starlark) or a sandboxed subprocess (Monty), not a browser JS engine. Codex
now runs V8 in a separate host process rather than in the agent binary — which fixes isolation but
not weight.

### ❌ "Shared mutable session state behind a lock"

Works in lint-enforced Rust; a liability in Go. Prefer immutable snapshots threaded through the loop.

### ❌ "Four databases, two policy engines, v1+v2 of everything"

Accretive product growth. The Harness answer is one DB, one policy mechanism, and compile-time
deletion of superseded code.

### ❌ "A second LLM (Guardian) in the approval path by default"

Guardian auto-reviews `on-request` approvals with a model call (fail-closed, timeout, circuit
breaker). It's clever, but it puts a latency- and cost-bearing model in the hot path of *running a
command* — and since June it has grown a second, asynchronous background scorer
(`ext/guardian-v2`, ~16K lines) beside the synchronous reviewer. For Harness, approval should be deterministic (rules + sandbox); an LLM reviewer, if any,
is an *optional extension subscriber*, not core.

### ❌ "Agent identity needs Ed25519 keys and a backend JWKS"

`agent-identity` (Ed25519 + JWT + backend registration) is infrastructure for OpenAI's hosted
multi-agent/cloud story. A local-first harness does not need cryptographic agent identity to spawn
a subagent.

### ❌ "Surfaces should be 450K lines"

The TUI crate is enormous (~450K lines, up from ~290K; `chat_composer.rs` ~12.7K lines). The protocol seam is excellent; the
*frontend* is a monolith. Harness should keep the TUI thin (Bubble Tea / ratatui) precisely
*because* the protocol seam lets it stay thin.

---

## 6. What to Actually Port

| From Codex | To Harness | Why |
|-----------|------------|-----|
| SQ/EQ protocol seam | `internal/protocol` + `Frontend` | One wire protocol drives in-proc and remote frontends identically |
| In-process vs remote client, one API | `frontend/client.go` | `InProcessAppServerClient`/`RemoteAppServerClient` pattern |
| One-task-per-session + steering | `core/agent/loop.go` + `queue.go` | Correct concurrency invariant |
| Layered exec approval pipeline | `ext/tools/terminal/safety.go` | execpolicy → hooks → user → network → sandbox |
| arg0 multitool dispatch | `cmd/harness/main.go` | Single binary doubles as sandbox helper |
| OS sandbox via helper binary | `ext/sandbox/` | seccomp/landlock wrapper in a child process |
| Managed network proxy + allowlist | `ext/sandbox/netproxy.go` | Principled network egress control |
| 12 Claude-compatible hook events | `internal/hook` | Interop with existing hook ecosystems |
| Code-mode *concept* (batch tool calls in a program) | Starlark/Monty self-extension | Get the benefit without V8 |
| Out-of-process code runtime (host child, kill+reap on reconnect) | Monty subprocess | Codex's own move (`core/src/code_mode_host.rs`) confirms isolating model-written code |
| Amendable approvals | `core/approval` | Approve once → persist an allow rule |
| Lint-denied `unwrap`/`expect` | `golangci-lint` config | Codebase-wide error discipline |

---

## 7. Architectural Contrasts

```
Codex                                  Harness
─────────────────────────────────     ─────────────────────────────────
~155 crates, Rust 2024                 Target: <50 files core, Go
session/turn.rs: ~3.2K ln              core/agent/loop.go: ~500 ln
session/mod.rs: ~5.3K ln               core/agent/state.go: small
config/mod.rs: ~5.1K ln                core/config/config.go: one struct
Mutex<SessionState> (shared mutable)   Immutable AgentState snapshots
ratatui TUI: ~450K ln                  Bubble Tea / ratatui (thin)
Responses API only                     Provider interface (any protocol)
V8 (host process) + Starlark           Starlark (tools) + Monty (subproc)
JSONL + 8 SQLite files                 One SQLite DB (modernc, no CGo)
2 Guardian LLMs in approval path       Deterministic rules + sandbox
Ed25519 agent identity (backend)       Local subagent = new loop + state
multi-agent v1+v2, Guardian v1+v2      Compile-time deletion of old code
SQ/EQ protocol seam ✅ (steal this)    Protocol-backed Frontend interface
4 native sandbox backends ✅ (steal)   Sandbox helper via arg0 dispatch
```

---

## 8. Final Assessment

Codex is what happens when a serious systems team rebuilds an agent in Rust and gets the *seam*
right: a small canonical protocol with interchangeable surfaces. That seam is the single best idea
in the open-agent landscape and the Harness should adopt it wholesale — the PLAN's ACP alignment
is the start, but the deeper lesson is **"one protocol, two transports (in-proc + remote), every
frontend is a client."**

Everything *around* the seam, though, is a cautionary tale about scope: four sandbox backends, two
language runtimes, eight SQLite files, two versions of multiple subsystems, two LLM reviewers in
the approval path, a cryptographic identity system, and a hard bet on one vendor's API. Four months
of upstream change (2026-06 → 2026-10) narrowed the seam to one façade but grew nearly everything
behind it. None of that is wrong
for OpenAI's product; all of it is wrong for a zero-dependency, provider-agnostic, single-binary
learning harness.

The Harness philosophy is the corrective:

1. **Keep the protocol seam, shrink everything behind it.** Core is the loop, the bus, the state
   machine, the protocol. Compaction, hooks, sandboxing, providers, persistence are extensions.
2. **Immutable snapshots, not a locked mutable session.** Go rewards this; Rust merely tolerates
   the alternative.
3. **One provider interface that bridges real protocol differences** — the opposite of a
   Responses-API monoculture.
4. **One DB, one policy mechanism, no embedded browser engine.**
5. **Delete superseded code at compile time** instead of running v1 and v2 forever.

Steal the seam and the safety model. Leave the mass.
