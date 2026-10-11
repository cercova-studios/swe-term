# pi-mono Deep Dive — Architecture & Feature Analysis

> Analysis of [`earendil-works/pi`](https://github.com/earendil-works/pi) (formerly
> `badlogic/pi-mono`) — the agent monorepo started by Mario Zechner ("badlogic") and the
> **reference implementation swe-term is porting from**. Runtime: Node.js (≥22.19.0).
> Language: TypeScript. Distribution: npm workspaces (root package name is still `pi-monorepo`).
>
> **Validated against a local source checkout** at commit `ea448f454` (2026-10-10), all 14
> packages at version **`1.1.0`**, npm scope **`@earendil-works/*`**. The previous revision of
> this document was validated against a `0.65.2` checkout under the old `@mariozechner/*` scope
> (written 2026-06-06; upstream at that date was `ff3e9df5f`, already `@earendil-works/*` at
> `0.78.1`). Line counts below are exact (from the local source, test files excluded) unless
> noted.

---

## Table of Contents

1. [Changes since 0.65.2 (2026-06-06)](#changes-since-0652-2026-06-06)
2. [System Overview](#system-overview)
3. [Monorepo Package Layout](#monorepo-package-layout)
4. [pi-ai: The Provider Abstraction](#pi-ai-the-provider-abstraction)
5. [pi-agent-core: The Agent Framework](#pi-agent-core-the-agent-framework)
6. [The Agent Loop](#the-agent-loop)
7. [AgentSession: The Orchestrator](#agentsession-the-orchestrator)
8. [Tool System](#tool-system)
9. [Extension System](#extension-system)
10. [Session Storage (JSONL Tree)](#session-storage-jsonl-tree)
11. [Context Compaction](#context-compaction)
12. [pi-tui: The Terminal UI](#pi-tui-the-terminal-ui)
13. [Modes (Interactive / Print / JSON / RPC)](#modes-interactive--print--json--rpc)
14. [The Durable / Remote Stack (new, experimental)](#the-durable--remote-stack-new-experimental)
15. [Key Design Patterns](#key-design-patterns)
16. [Why swe-term Ports It](#why-swe-term-ports-it)
17. [Summary Statistics](#summary-statistics)

---

## Changes since 0.65.2 (2026-06-06)

About 2,360 commits landed between `ff3e9df5f` and `ea448f454`. The most significant for swe-term:

1. **Rebrand and org move (already done before 2026-06-06).** The npm scope moved
   `@mariozechner/*` → `@earendil-works/*` in `3e5ad67e0` (2026-05-07), and package
   repository metadata moved to `github.com/earendil-works/pi` in `93600d898` (2026-05-29).
   `pi-mom` and `pi` (pods) were removed in `0ed0d4343` (2026-04-30); `pi-web-ui` in `b141e1fa2`
   (2026-05-20). The earlier doc's 0.65.2 checkout predated all of this.
2. **v1.0.0** shipped in `a13d35a74` (2026-10-01); all packages now version in lockstep at `1.1.0`.
3. **Provider is now a first-class interface.** `Provider<TApi>` (`packages/ai/src/models.ts:150`)
   owns auth, model listing, `refreshModels()`, and `stream`/`streamSimple`; providers are held in a
   `Models` collection (`createModels()`, `models.ts:992`). Wire protocols moved to
   `packages/ai/src/api/*`, one module per provider to `packages/ai/src/providers/*`.
4. **Agent core decoupled from the provider catalog.** `pi-agent-core` no longer defaults to
   `pi-ai`'s `streamSimple`; hosts install a default via `setDefaultStreamFn()`
   (`packages/agent/src/stream-fn.ts`). Tool interception is now `beforeToolCall`/`afterToolCall`
   hooks on the loop config (`packages/agent/src/types.ts:326,341`).
5. **Built-in MCP and codemode** (`8562bcf66`, 2026-09-29): standalone MCP client
   `@earendil-works/pi-mcp` (`packages/mcp`) and QuickJS-sandboxed `@earendil-works/pi-codemode`
   (`packages/codemode`), wired in as bundled extensions (`packages/coding-agent/src/extensions/mcp`,
   `.../extensions/codemode`).
6. **Durable harness** `@earendil-works/pi-durable` (`080160162`, 2026-09-18), with JSONL **and
   SQLite** storage backends (`packages/durable/src/storage/{jsonl,sqlite}`; Durable Object SQLite
   adapter in `da866ada1`), an experimental coding agent on top of it (`5609b0d6c`), and SSH remote
   execution via `@earendil-works/pi-env` with a Rust daemon (`ba03e03f2`, 2026-10-05).
7. **Chord + remote-session protocol.** `@earendil-works/chord` (`28b49a6b3`, 2026-08-29) is an
   application-composition runtime (facets, services, replicated state); `pi-protocol`,
   `pi-server`, and `pi-client` carry remote sessions over framed CBOR.
8. **Telemetry package** `@earendil-works/pi-telemetry` extracted in `6b461b75b` (2026-08-05);
   `pi-ai` now depends on it.
9. **`AgentSession` kept growing:** 3,128 lines at `ff3e9df5f` → **4,398** lines now. The
   god-object trend is worse, not better.
10. **History re-rooted.** The first-parent history now starts at orphan commit `583f153d5`
    (2026-08-01); the old lineage (including `ff3e9df5f`) is still reachable through a merge.
    `git log --diff-filter=A` dates for packages present at that root are therefore not creation
    dates. A `storage` → `session-backends` package came and went in that window (`a80008b96`
    renamed it; `7fd478a2e` removed it with the experimental harness in `pi-agent-core`).

---

## System Overview

pi (repo `earendil-works/pi`, historically `pi-mono`) is a lean, hackable coding agent built as a
TypeScript monorepo. Its design philosophy is the opposite of Claude Code's: **layered,
independently publishable npm packages** where the LLM abstraction, the agent framework, the
coding agent, and the TUI are separable. It is the spiritual ancestor of both swe-term (which ports
its architecture to Go/Rust) and Flue (which builds a headless framework on top of `pi-agent-core`
+ `pi-ai`).

Since 1.0 the repo has also grown a second, **experimental** stack — durable sessions, a composition
runtime (Chord), and a remote-session protocol — alongside the original
`pi-ai` → `pi-agent-core` → `pi-coding-agent` core. See
[The Durable / Remote Stack](#the-durable--remote-stack-new-experimental).

**Tech Stack:**

| Layer | Technology |
|-------|-----------|
| Runtime | Node.js ≥22.19.0 (Bun entry points also exist: `packages/coding-agent/src/bun`) |
| Language | TypeScript |
| Package mgmt | npm workspaces (monorepo) |
| Schema | TypeBox (`typebox` 1.3.27, no longer `@sinclair/typebox`); validation via TypeBox `Compile()` (`packages/ai/src/utils/validation.ts`) |
| Extension loading | `jiti` 2.7.0 (run TS without a build step) |
| TUI | custom (`@earendil-works/pi-tui`, differential renderer) |
| Session format | JSONL (append-only, tree via `id`/`parentId`), session version 3 |
| Cancellation | `AbortController` / `AbortSignal` |

**Scale:** the headline number is still the `AgentSession` class, now **4,398 lines**
(`packages/coding-agent/src/core/agent-session.ts`). The whole repo is no longer "small":
`pi-coding-agent` is ~86K lines of non-test TS and `pi-ai` ~27K. The value for swe-term is still in
the *layering* of the core four packages.

---

## Monorepo Package Layout

Fourteen packages live under `packages/*/`, all `@earendil-works/*` at `1.1.0`:

| Package | Role | Internal deps |
|---------|------|---------------|
| **`pi-ai`** (`packages/ai`) | Unified LLM API: `Provider` interface, `Models` collection, ~45 provider modules over ~20 wire-protocol modules (`src/api/`), model catalog, auth/OAuth, streaming, usage, images, classifiers. | `pi-telemetry` |
| **`pi-agent-core`** (`packages/agent`) | Domain-free agent: `Agent` class (`agent.ts`, 613 ln), loop (`agent-loop.ts`, 949 ln), steering/follow-up queues, tool hooks, `proxy.ts`. | `pi-ai` |
| **`pi-coding-agent`** (`packages/coding-agent`) | The `pi` CLI: `AgentSession`, built-in tools, extensions, sessions, modes, plus an `experimental/` durable client/server agent. | `pi-ai`, `pi-agent-core`, `pi-tui`, `pi-mcp`, `pi-codemode`, `chord` |
| **`pi-tui`** (`packages/tui`) | Terminal UI library with differential rendering (~19,500 ln, 46 files). | — |
| `pi-mcp` (`packages/mcp`) | Standalone MCP client (stdio + Streamable HTTP); no official MCP SDK dependency. | — |
| `pi-codemode` (`packages/codemode`) | Runs model-written JS in a QuickJS/WASM VM whose only capability is calling injected tools. | — |
| `pi-durable` (`packages/durable`) | **Experimental** durable agent harness: commits turns/tool calls to storage before display; resumable. | `pi-ai`, `chord` |
| `pi-env` (`packages/env`) | Remote `ExecutionEnv` for pi-durable over SSH; Rust daemon on the remote host. | `pi-durable`, `chord` |
| `chord` (`packages/chord`) | Application-composition runtime (facets, services, replicated state, RPC). Explicitly "not a Pi package". | — |
| `pi-protocol` / `pi-server` / `pi-client` | Experimental CBOR-framed protocol (v8), server, and client for remote durable sessions. | `chord` (+ `pi-protocol`) |
| `pi-telemetry` (`packages/telemetry`) | Vendor-neutral telemetry contracts. | — |
| `pi-evals` (`packages/evals`) | Eval harness (no package description). | — |

The core dependency direction is still clean: `pi-coding-agent` → `pi-agent-core` → `pi-ai`, with
`pi-tui` as a presentation dependency. `pi-mom`, `pi-web-ui`, and `pods` no longer exist (see
[Changes](#changes-since-0652-2026-06-06)). The new experimental stack hangs off `chord` and
`pi-ai`, not off `pi-agent-core`. This layering is still the single most important thing pi gets
right, and the reason it is portable at all.

---

## pi-ai: The Provider Abstraction

`pi-ai` is the bottom layer: one API over many LLM providers. It is now organized in two tiers:

- **Wire protocols** in `packages/ai/src/api/` (e.g. `anthropic-messages.ts`,
  `openai-completions.ts`, `openai-responses.ts`, `google-generative-ai.ts`,
  `bedrock-converse-stream.ts`, `mistral-conversations.ts`), each with a `.lazy.ts` loader.
- **Providers** in `packages/ai/src/providers/` — one module per provider (≈45: `anthropic`,
  `openai`, `google`, `google-vertex`, `amazon-bedrock`, `azure`, `mistral`, `groq`, `cerebras`,
  `deepseek`, `openrouter`, `xai`, `zai`, `github-copilot`, `openai-codex`, `cloudflare-*`, …),
  most with a generated `*.models.ts` catalog.

### The `Provider` interface

`Provider<TApi>` (`packages/ai/src/models.ts:150`) is the concrete runtime unit: `id`, `name`,
optional `baseUrl`/`headers`, required `auth: ProviderAuth`, `getModels()` / `getAllModels?()`,
optional `refreshModels()` for dynamic catalogs, optional `filterModels()` credential policy, and
`stream()` / `streamSimple()`. Providers are aggregated by a `Models` collection
(`createModels()`, `models.ts:992`), and extensions can add or wrap providers with
`pi.registerProvider()` (`packages/coding-agent/src/core/extensions/types.ts:1793`).

### Capability compat

Provider quirks are still declared through a `compat` field on the `Model`, now as **TypeBox
schemas per wire protocol** in `packages/ai/src/providers/compat-schema.ts`
(`OpenAICompletionsCompatSchema`, `OpenAIResponsesCompatSchema`, `AnthropicMessagesCompatSchema`,
`BedrockCompatSchema`, `MistralConversationsCompatSchema`). `OpenAICompletionsCompat`
(`compat-schema.ts:621`) has grown from 14 to **27** top-level flags, including `supportsStore`,
`supportsDeveloperRole`, `supportsReasoningEffort`, `maxTokensField`, `thinkingFormat`,
`openRouterRouting`, `vercelGatewayRouting`, `zaiToolStream`, `supportsStrictMode`,
`cacheControlFormat`, `sessionAffinityFormat`, and `supportsLongCacheRetention`.

When `compat` is **not** set, `pi-ai` *still falls back to provider-slug/URL-substring detection*:
`detectCompat()` in `packages/ai/src/api/openai-completions.ts:1593` matches the provider id **or**
`baseUrl` against strings such as `api.z.ai`, `open.bigmodel.cn`, `api.together.ai`,
`api.moonshot.`, `openrouter.ai`, `api.cloudflare.com`, `integrate.api.nvidia.com`, `cerebras.ai`,
`deepseek.com`, `api.x.ai`, `chutes.ai`, and `opencode.ai`. The list has grown. So there are still
two mechanisms: declared capability (the intended path, now behind a real `Provider` interface) and
URL sniffing (the fallback, inside the OpenAI-compatible wire protocol).

### Usage accounting

`pi-ai` tracks real, provider-reported token usage. `Usage` (`packages/ai/src/types.ts:446`)
carries `input`, `output`, `cacheRead`, `cacheWrite`, and now `cacheWrite1h?` (Anthropic 1-hour
retention) and `reasoning?` (a subset of `output`), plus totals and cost. This is distinct from
*pre-flight* estimation (see [Compaction](#context-compaction)), whose helpers now live in
`packages/ai/src/utils/estimate.ts`.

### Streaming

`Models.streamSimple` is the streaming entry point that satisfies `pi-agent-core`'s `StreamFn`
(`packages/agent/src/types.ts:20`). The agent receives it by injection, not by import (next section).

---

## pi-agent-core: The Agent Framework

`pi-agent-core` (`packages/agent`) contains the **provider-agnostic agent**: an `Agent` class for
state management and tool calling, and the loop functions `agentLoop()` / `agentLoopContinue()` /
`runAgentLoop()` / `runAgentLoopContinue()`, plus a standalone `runToolCall()`
(`agent-loop.ts:38,71,102,128,811`). It deliberately knows nothing about the *coding* domain — no
file tools, no bash — so it can be reused (Flue builds on exactly this layer).

It is now also decoupled from the provider catalog: the default stream function is installed by the
host through `setDefaultStreamFn()`, and `getDefaultStreamFn()` throws if nothing was configured
(`packages/agent/src/stream-fn.ts`). An experimental harness that briefly lived in this package was
removed in `7fd478a2e` (2026-10-01).

`ThinkingLevel` (reasoning-effort control) lives here and is re-exported by downstream consumers.

---

## The Agent Loop

The loop lives in `packages/agent/src/agent-loop.ts` (949 lines) and is driven by the `Agent` class
in `agent.ts` (613 lines). The structure is unchanged: an **outer follow-up loop wrapping an inner
tool+steering loop**:

```
runAgentLoop(agent, input):
  outer loop:                              # follow-up messages
    inner loop while toolCalls or pending: # tool execution + steering
      inject steering messages
      stream assistant response (streamFn, injected; default via setDefaultStreamFn)
      execute tools (beforeToolCall → execute → afterToolCall; sequential or parallel)
      append tool results to history
      pending = getSteeringMessages()
    followUp = getFollowUpMessages()
    if followUp: continue outer loop
    else: break
```

Tool execution policy is explicit: `toolExecution: "sequential" | "parallel"`, and any tool can
force sequential execution (`agent-loop.ts:519`). Parallel results are collected with `Promise.all`
in source order (`agent-loop.ts:646`). Interception is via `beforeToolCall` (can block) and
`afterToolCall` (can override the result) on the loop config (`types.ts:326,341`), each receiving
an `AbortSignal`. The `Agent` reads `runtimeOptions.streamFn ?? getDefaultStreamFn()`
(`agent.ts:236`).

---

## AgentSession: The Orchestrator

**Path:** `packages/coding-agent/src/core/agent-session.ts` (**4,398 lines**, up from 3,060 in
0.65.2 and 3,128 at `ff3e9df5f`)

`AgentSession` is still the central object of the coding agent — and the one swe-term explicitly
singles out as a **god object to decompose** (PLAN gap #1). It wraps the `pi-agent-core` `Agent`
and coordinates between the frontends and the LLM. Its responsibilities (all in one class):

- Agent lifecycle and state access
- Event subscription and **automatic session persistence**
- **Model and thinking-level management** (including `_scopedModels`)
- **Manual and automatic compaction** (separate `AbortController`s)
- **Bash execution** (concurrent runs tracked in a set)
- **Session switching and branching** (tree navigation, branch summaries)
- **Extension integration** — it installs itself as the `Agent`'s `beforeToolCall` /
  `afterToolCall` (`agent-session.ts:653–654`) and drives `ExtensionRunner`
- Retry handling

Two small satellites now exist — `agent-session-services.ts` (233 ln, resource loading/diagnostics)
and `agent-session-runtime.ts` (447 ln, session start/replace/shutdown wiring) — but they are
helpers around the class, not a decomposition of it.

Cancellation: four single-slot controllers — `_compactionAbortController`,
`_autoCompactionAbortController`, `_branchSummaryAbortController`, `_retryAbortController`
(`agent-session.ts:403–411`) — plus `_bashAbortControllers: Set<AbortController>`
(`agent-session.ts:420`) so several bash runs can be cancelled together. This is the `AbortSignal`
pattern (PLAN gap #7) applied pervasively.

It still uses `any` (`_scopedModels: Array<{ model: Model<any>; … }>`, `agent-session.ts:383`;
~20 `any` occurrences in the file), despite strict-TS aspirations (PLAN gap #12).

---

## Tool System

Tools (`AgentTool`, `packages/agent/src/types.ts:466`) are TypeScript objects with `name`,
`label`, `description`, and `parameters` defined with **TypeBox** (`Type.Object({...})`).
Arguments are validated with TypeBox's compiler (`packages/ai/src/utils/validation.ts:277`).

Tool *execution* runs in the agent loop; `AgentSession` hooks it through `beforeToolCall` /
`afterToolCall` and forwards to `ExtensionRunner`, which lets extensions intercept, block, or
rewrite a call.

Built-in coding tools (`packages/coding-agent/src/core/tools/`): `read`, `write`, `edit`, `bash`,
`powershell`, `grep`, `find`, `ls`, with a per-file mutation queue (`file-mutation-queue.ts`) and
output truncation. Bundled extensions add `codemode`, `mcp`, and `tool-search` tools
(`packages/coding-agent/src/extensions/`).

---

## Extension System

**Mechanism:** extensions are TypeScript modules that `export default` a factory receiving an
`ExtensionAPI`. The API surface (`core/extensions/types.ts`, 2,273 ln) now includes
`registerTool()` (`:1643`), `registerProvider()` (`:1793`), and `registerMcpServer()` (`:726`), along
with event hooks and project-trust events.

**Loading:** still **`jiti`** (`core/extensions/loader.ts:576–581`, `jiti.import(path, { default:
true })`), with a static-module variant for embedded/bundled builds (`jiti-static-loader.ts`). A
tool registered by an extension is available to the LLM without a reload.

**Project trust:** project-local resources that can run code now require trust
(`core/project-trust.ts`, `core/trust-manager.ts`) — a new gate in front of the dynamic loader.

This is still PLAN gap #2: the extension system depends on dynamic module evaluation at runtime,
which is exactly the dynamism swe-term avoids (Starlark tool defs + Go interface hooks).

---

## Session Storage (JSONL Tree)

The coding agent's sessions are still **append-only JSONL**, one file per session named
`{timestamp}_{sessionId}.jsonl` (`core/session-manager.ts:1080`), with entries linked by
`id`/`parentId` into a **conversation tree**. `CURRENT_SESSION_VERSION = 3`
(`session-manager.ts:41`). `SessionManager` (2,013 ln) owns persistence and path resolution.

There is **no on-disk index**. An in-memory `byId: Map` is built on demand
(`buildEntryIndex()`, `session-manager.ts:381`) and path resolution walks `parentId` through it.
Loading is still an O(n) file read + index build (PLAN gap #4). Files live under
`{agentDir}/sessions/--<cwd with separators replaced by '-'>--/` (`session-manager.ts:589–596`);
the agent dir is overridable via **`PI_CODING_AGENT_DIR`** and the session dir via
**`PI_CODING_AGENT_SESSION_DIR`** (both derived from `APP_NAME`, `config.ts:584–586`).

Notably, the new **`pi-durable`** package ships both JSONL and **SQLite** storage
(`packages/durable/src/storage/sqlite/`, with migrations and a Cloudflare Durable Object adapter) —
upstream independently reached for indexed storage, but only in the experimental durable stack, not
in the shipping coding agent.

---

## Context Compaction

When context exceeds a configurable threshold, pi **summarizes older content** while preserving
recent turns:

1. Find a **cut point** walking backward (`keepRecentTokens` default 20000,
   `core/settings-defaults.ts:11`).
2. Extract the messages to summarize.
3. Generate a summary via the LLM.
4. Append a `CompactionEntry` to the session log.

Manual `/compact` exists; `AgentSession` owns both paths. Code lives in
`core/compaction/` (`compaction.ts`, `branch-summarization.ts`).

> **Note on token counting (corrected):** the trigger is *not* pure `chars/4`.
> `estimateContextTokens()` (`compaction.ts:196`) takes the **last provider-reported usage** as the
> anchor and applies the `chars/4` heuristic (`estimateTokens()`, `compaction.ts:298`, `Math.ceil(chars
> / 4)`) **only to messages after that usage**. The same helpers exist in
> `packages/ai/src/utils/estimate.ts` (`CHARS_PER_TOKEN`). This anchoring was already present at
> `ff3e9df5f`. PLAN gap #5 still holds for the trailing segment and for the first turn (no usage
> yet), but the error is bounded to un-reported messages.

---

## pi-tui: The Terminal UI

`@earendil-works/pi-tui` is a **custom terminal UI library** built around **differential
rendering**. `TUI` is now an interface implemented by `abstract class TuiBase extends Container`
(`packages/tui/src/tui.ts:506`), with `TuiMainScreen` and `TuiAltScreen` variants. Exported
components include `Box`, `Text`, `TruncatedText`, `Markdown`, `Editor`, `Input`, `Loader`,
`CancellableLoader`, `SelectList`, `SettingsList`, `Spacer`, `HStack`, `MouseRegion`, LaTeX
rendering, fuzzy matching, and `ProcessTerminal`.

The coding agent's `InteractiveMode` (`modes/interactive/interactive-mode.ts`, **7,109 lines**, up
from ~4,690) handles rendering and interaction, delegating business logic to `AgentSession`.

The package is now **~19,500 lines across 46 files** (e.g. `components/editor.ts` 2,473 ln,
`keys.ts` 1,455 ln, `tui.ts` 1,517 ln) — nearly double the 0.65.2 size — which swe-term replaces
with `ratatui` (Rust) or Bubble Tea (Go).

---

## Modes (Interactive / Print / JSON / RPC)

The coding agent separates **four modes** from the session (`packages/coding-agent/src/modes/`,
routed in `main.ts:113–116, 950–986`; `AppMode = "interactive" | "print" | "json" | "rpc"`):

- **Interactive** (`modes/interactive/interactive-mode.ts`, 7,109 ln) — the full TUI.
- **Print** (`modes/print-mode.ts`, 169 ln) — non-interactive stdout.
- **JSON** (`modes/json-event.ts`) — print-style run emitting JSON events.
- **RPC** (`modes/rpc/rpc-mode.ts`, 819 ln) — JSON stdin/stdout protocol (`runRpcMode()` +
  `RpcClient`, framing in `rpc/jsonl.ts`), with a dedicated `rpc-entry.ts`.

All drive the same `AgentSession`, so business logic is shared and the UI is swappable.

> **MCP is now built in.** The earlier "no built-in MCP" claim is obsolete. `pi-mcp` is a
> standalone MCP client, and the bundled `mcp` extension connects servers from `mcp.json` and from
> `pi.registerMcpServer()` (`core/mcp-servers.ts`). `codemode` lets the model call tools (including
> MCP tools) from a QuickJS script so nested calls stay out of context.

---

## The Durable / Remote Stack (new, experimental)

All of the following are marked experimental in their READMEs and are **not** on the main
`pi` CLI path:

- **`pi-durable`** — conversations, turns, tool calls, and app state are committed to storage
  before display; reopening storage resumes a mid-turn crash. Built on `pi-ai` + `chord`, **not**
  on `pi-agent-core`. Storage: memory, JSONL, SQLite (`packages/durable/src/storage/`).
- **`pi-env`** — remote `ExecutionEnv` over SSH; a Rust daemon on the remote host speaks a framed
  stdin/stdout protocol (`packages/env/docs/protocol.md`).
- **`chord`** — composition runtime: facets, services, replicated state ("Delta" codecs),
  pluggable remote-service boundary. Standalone, no pi dependencies.
- **`pi-protocol` / `pi-server` / `pi-client`** — protocol v8 over CBOR byte-stream framing;
  routes clients to durable sessions with per-presentation attachments.
- **Experimental coding agent** on pi-durable (`packages/coding-agent/src/experimental/`,
  `5609b0d6c`) with a coordinator, session workers, and a client TUI.

This is a second architecture forming inside the repo. It is too young (≤6 weeks, API "changes
without notice") to be a porting reference yet.

---

## Key Design Patterns

### 1. Layered, independently-publishable packages

`pi-ai` (providers) → `pi-agent-core` (agent) → `pi-coding-agent` (coding) + `pi-tui` (UI). Each is
usable on its own; Flue consumes the lower two directly. New leaf packages (`pi-mcp`,
`pi-codemode`, `chord`) have **zero** internal dependencies, which extends the discipline.

### 2. Provider-agnostic, provider-catalog-agnostic agent core

`pi-agent-core` knows nothing about coding, and since `setDefaultStreamFn()` it no longer chooses a
provider runtime either.

### 3. Provider interface + declared capability (with a sniffing fallback)

`Provider` is an interface with per-provider modules, and `Model.compat` is a TypeBox-typed
declaration per wire protocol. URL detection in `detectCompat()` remains the weak spot.

### 4. Zero-build extensions via `jiti`

Drop a `.ts` file, it loads and its tools are live. Now gated by project trust. Excellent iteration
speed; deep Node coupling.

### 5. JSONL tree for branchable history

Append-only log with `id`/`parentId` — free branching, human-readable — at the cost of O(n) load
and tree-walking. (The durable stack adds SQLite.)

### 6. `AbortController` everywhere for cancellation

Every long-running operation gets its own signal; tool hooks receive one. Maps directly to Go's
`context.Context`.

### 7. Orchestrator-as-god-object

`AgentSession` centralizes everything, and has grown ~44% since 0.65.2. Great for
discoverability, poor for testability — and the explicit thing swe-term decomposes.

---

## Why swe-term Ports It

`../core/PLAN.md` states the thesis: *"Porting the architecture of pi-mono — without its gaps."*
pi is the reference because its **core layering is right** (provider / agent / coding / UI). The
core-four scope is still sane even though the repo as a whole has grown. swe-term keeps the
layering and the loop, and fixes the twelve gaps — chiefly: decompose the `AgentSession` god
object, replace dynamic-`import()` extensions with Starlark + Go interfaces, replace URL-substring
provider detection with a clean `Provider` interface (upstream now has the interface but kept the
sniffing), replace JSONL tree-walking with SQLite, and replace `chars/4` estimation with a real
tokenizer.

See `../core/PLAN.md` → "pi-mono Gap Analysis" for the 12-row table; `PI_MONO_CRITIQUE.md` expands
each gap and records which verdicts changed in this refresh.

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Repo | `github.com/earendil-works/pi` (formerly `badlogic/pi-mono`); root package `pi-monorepo` |
| Language / runtime | TypeScript / Node.js (≥22.19.0) |
| Version (local checkout) | `1.1.0` (all packages), scope `@earendil-works/*`, commit `ea448f454` |
| Packages | 14: core `pi-ai`, `pi-agent-core`, `pi-coding-agent`, `pi-tui`; plus `pi-mcp`, `pi-codemode`, `pi-telemetry`, `pi-evals`, and experimental `pi-durable`, `pi-env`, `chord`, `pi-protocol`, `pi-server`, `pi-client` |
| Providers (pi-ai) | ≈45 provider modules over ≈20 wire-protocol modules; `Provider` interface at `models.ts:150` |
| `OpenAICompletionsCompat` | 27 flags, TypeBox schema (`providers/compat-schema.ts`) |
| `AgentSession` size | **4,398 lines** (single class) |
| Agent loop | `agent-loop.ts` (949 ln) + `agent.ts` (613 ln); `streamFn` injected or `setDefaultStreamFn()` |
| Tool hooks | `beforeToolCall` / `afterToolCall`; `toolExecution` sequential/parallel |
| Tool schema | TypeBox (`typebox` 1.3.27) + TypeBox `Compile()` validation |
| Extension loading | `jiti` 2.7.0, `pi.registerTool()` / `registerProvider()` / `registerMcpServer()`; project-trust gate |
| Session format | JSONL `{timestamp}_{id}.jsonl`, v3, tree via `id`/`parentId`; in-memory index only |
| Session dir env | `PI_CODING_AGENT_DIR`, `PI_CODING_AGENT_SESSION_DIR` → `…/sessions/--<cwd>--/` |
| Modes | interactive / print / json / **rpc** (all over one `AgentSession`) |
| Compaction | LLM summarization; backward cut point (keepRecentTokens 20000); `CompactionEntry`; `/compact` |
| Token estimation (pre-flight) | last provider usage + `chars/4` on trailing messages (`estimateContextTokens`) |
| Token usage (accounting) | provider-reported `Usage` (`input`/`output`/`cacheRead`/`cacheWrite`/`cacheWrite1h`/`reasoning`/cost) |
| Cancellation | 4 `AbortController`s + a bash controller `Set` in `AgentSession` (+ the `Agent`'s run controller) |
| TUI | custom differential renderer (`pi-tui`, ~19,500 ln / 46 files) |
| MCP | built in: `pi-mcp` client + bundled `mcp` extension; `codemode` sandbox |
| Downstream consumers | swe-term (port), Flue (builds on `pi-agent-core` + `pi-ai`) |

---

*Validated 2026-10-10 against the local source checkout at `/Users/rohit/sandbox/swe-term/pi`
(commit `ea448f454`, packages `1.1.0`, scope `@earendil-works/*`) plus the swe-term gap analysis in
`../core/PLAN.md`. Line counts are from that checkout, test files excluded. Previous validation:
pi-mono `0.65.2` (`@mariozechner/*`), 2026-06-06.*
