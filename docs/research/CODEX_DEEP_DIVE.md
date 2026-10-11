# Codex Deep Dive — Architecture & Feature Analysis

> Analysis of the OpenAI Codex source snapshot (`codex/codex-rs`, a ~155-member Cargo
> workspace + Python and TypeScript SDKs under `codex/sdk/`).
> Runtime: native (compiled). Language: Rust (edition 2024, `unwrap`/`expect` denied
> by lint). UI: ratatui + crossterm (terminal). License: Apache-2.0.
>
> **Validated against:** upstream `cc7ba3360` (2026-10-10; workspace `version = "0.0.0"`
> on `main`, so the commit is the version). Previous validation: `b89ce9a2b` (2026-06-06).

---

## Changes since 2026-06-06 (b89ce9a2b)

~4,900 upstream commits; workspace grew from 116 to 155 members. The architecturally
significant changes:

1. **MCP-server façade removed.** `codex mcp-server` and the `mcp-server` crate are gone
   (`531f3836a`). Codex is now an MCP *client* only; the app-server is the sole external façade.
2. **Code mode moved out of process.** V8 (now `v8 = "=150.4.0"`, `v8_enable_sandbox`) lives in
   `code-mode-runtime`, hosted by a separate `codex-code-mode-host` binary that core drives over
   gRPC on stdio pipes (`core/src/code_mode_host.rs`; `aa46f2deb`, `2dc0791c7`).
3. **Turn input unified.** `Op::UserInput` is replaced by `Op::TurnInput { request, mode, reply }`
   (`cbb7e82a8`); `Op::ThreadRollback` and `thread/rollback` were removed in favour of
   `thread/revert` (`3052bbcf8`). The `Submission` type is now core-internal
   (`core/src/session/submission.rs`).
4. **Guardian split and rebuilt.** `ext/guardian` is gone; synchronous review lives in
   `ext/guardian-reviewer` and a new asynchronous background scorer in `ext/guardian-v2`
   (~16K lines; `fe614a630`). New `Op::ApproveGuardianDeniedAction` lets a user override a denial.
5. **Multi-agent coordination grew a message board; agent jobs removed.** New
   `ext/agent-message-board` (local SQLite / in-memory / remote backends) wired into persistent
   multi-agent runtimes (`6149914a0`); CSV-backed agent jobs deleted (`687f05cb9`); roles extracted
   to `agent-roles` (`fb9311db5`). v1 and v2 multi-agent tool sets still coexist.
6. **Durable per-thread input queue.** `ext/queue` + `queue_1.sqlite` + `thread/queue/*` RPCs +
   `codex queue` subcommand (`b87981a51`, `bc8b25ea0`).
7. **Fourth sandbox backend.** `SandboxType::WindowsMxc` via the new `mxc-sandbox` crate
   (`60888d086`, `8b78600dc`), plus a `windows-sandbox-service` crate.
8. **Legacy removals.** `execpolicy-legacy` (`656a2d090`), `core-skills` → `ext/skills`
   (`45f8cafa4`), patched-zsh `shell-escalation` backend (`cd85a26ca`), legacy remote compaction
   (`1ac689cc7`), the one-shot `shell_command` tool (standardized on unified exec, `8a40095ea`), and
   the `untrusted` approval policy (`942af8447`).
9. **Hooks: 10 → 12 events** (`SessionEnd`, `Interrupt`) and an `McpTool` handler type
   (`hooks/src/lib.rs`, `config/src/hook_config.rs`).
10. **Persistence widened.** SQLite files went from 4 to 8 (`state/src/sqlite.rs`,
    `ext/agent-message-board/src/local/pool.rs`); managed Git worktrees added (`worktree/`, `e7ea5f4a8`).

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Workspace Layout](#workspace-layout)
3. [The SQ/EQ Protocol](#the-sqeq-protocol)
4. [Layered Architecture: Core → App-Server → Surfaces](#layered-architecture-core--app-server--surfaces)
5. [Agent Loop: Threads, Sessions, Turns, Tasks](#agent-loop-threads-sessions-turns-tasks)
6. [Streaming & The Model Client](#streaming--the-model-client)
7. [State Management](#state-management)
8. [Tool System](#tool-system)
9. [Command Execution](#command-execution)
10. [Sandboxing (3 Platforms, 4 Backends)](#sandboxing-3-platforms-4-backends)
11. [ExecPolicy (Starlark) & Approvals](#execpolicy-starlark--approvals)
12. [Code Mode (V8)](#code-mode-v8)
13. [Context Compaction](#context-compaction)
14. [Config System](#config-system)
15. [Session Persistence: Rollouts + SQLite](#session-persistence-rollouts--sqlite)
16. [Skills & AGENTS.md](#skills--agentsmd)
17. [Hooks](#hooks)
18. [Plugins & Connectors](#plugins--connectors)
19. [Multi-Agent System](#multi-agent-system)
20. [Guardian (Automated Approval Reviewer)](#guardian-automated-approval-reviewer)
21. [Memories & Goals](#memories--goals)
22. [Provider & Auth Layer](#provider--auth-layer)
23. [MCP Integration (Client only)](#mcp-integration-client-only)
24. [Terminal UI (ratatui)](#terminal-ui-ratatui)
25. [CLI & arg0 Multitool](#cli--arg0-multitool)
26. [Observability](#observability)
27. [Key Design Patterns](#key-design-patterns)
28. [Summary Statistics](#summary-statistics)

---

## System Overview

Codex is OpenAI's agentic coding CLI, rewritten from the original TypeScript version
into a large Rust workspace. Where Claude Code is a single Bun/TypeScript bundle, Codex
is a **multi-crate compiled system** with a hard architectural seam between the agent
core and every surface that drives it.

**Tech Stack:**

| Layer | Technology |
|-------|-----------|
| Language | Rust, edition 2024 |
| Async runtime | Tokio (1.x) |
| Terminal UI | ratatui 0.29 + crossterm 0.28 (forked) |
| CLI parsing | clap 4 |
| Serialization | serde, serde_json, `ts-rs` (TypeScript bindings), `schemars` (JSON Schema) |
| Persistence | JSONL rollouts + SQLite via `sqlx` 0.9 (bundled) |
| Policy engine | `starlark` 0.13 (execpolicy) |
| Code-mode runtime | `v8` 150.4 (JS isolate in a separate `codex-code-mode-host` process) |
| Patch grammar | Lark grammar (`apply_patch.lark`) |
| MCP | `rmcp` 1.7 (client + server) |
| Sandbox | bubblewrap+seccomp (Linux), seatbelt (macOS), restricted token/WFP or MXC (Windows) |
| Telemetry | OpenTelemetry 0.31 (OTLP), Statsig |
| Auth | OAuth 2.0 (PKCE), API key, Ed25519 agent identity |
| HTTP | reqwest 0.12, WebSocket via tungstenite (forked) |

**Scale:** ~155 workspace members. The `core` crate alone has ~586 `.rs` files. The TUI crate
has ~1,105 `.rs` files and ~450K lines (`chat_composer.rs` is ~12.7K lines). `protocol/src/protocol.rs`
is ~6,542 lines; `core/src/config/mod.rs` is ~5,071; `core/src/session/mod.rs` is ~5,287;
`core/src/session/turn.rs` (the turn loop) is ~3,209.

The defining characteristic: **Codex is a protocol-first system.** Everything is built
around an async submission/event queue (SQ/EQ). The agent core is a library; the TUI,
the headless `exec` mode, the VS Code extension, and the SDKs are all *clients* of that
protocol (via the app-server).

---

## Workspace Layout

The ~155 members fall into rough tiers (selected, not exhaustive):

| Tier | Crates |
|------|--------|
| **Protocol** | `protocol`, `app-server-protocol`, `exec-server-protocol`, `code-mode-protocol`, `codex-backend-openapi-models` |
| **Core** | `core`, `core-api`, `core-plugins`, `core-plugin-common`, `history`, `agent-roles` |
| **Surfaces** | `tui`, `exec`, `cli`, `app-server`, `app-server-daemon` |
| **Transports/clients** | `app-server-transport`, `app-server-client`, `codex-client`, `codex-mcp`, `rmcp-client`, `http-client`, `websocket-client`, `tcp-tunnel` |
| **Execution** | `exec-server` (~49K lines: execution/filesystem environments), `execpolicy`, `apply-patch`, `shell-command`, `code-mode`, `code-mode-host`, `code-mode-runtime`, `worktree` |
| **Sandbox** | `sandboxing`, `linux-sandbox`, `windows-sandbox-rs`, `windows-sandbox-service`, `mxc-sandbox`, `bwrap`, `network-proxy`, `responses-api-proxy` |
| **Providers** | `model-provider`, `model-provider-info`, `models-manager`, `ollama`, `lmstudio`, `chatgpt`, `aws-auth`, `login`, `workload-identity` |
| **State** | `state`, `thread-store`, `rollout`, `rollout-trace`, `message-history`, `agent-graph-store`, `agent-identity`, `attachment-store` |
| **Extensions** | `ext/extension-api`, `ext/agent`, `ext/agent-message-board`, `ext/connectors`, `ext/git-attribution`, `ext/goal`, `ext/guardian-reviewer`, `ext/guardian-v2`, `ext/history-notes`, `ext/items`, `ext/mcp`, `ext/memories`, `ext/queue`, `ext/skills`, `ext/image-generation`, `ext/web-search` |
| **Cloud** | `cloud-tasks`, `cloud-tasks-client`, `cloud-client`, `cloud-requirements`, `backend-client` |
| **Voice** | `voice-host`, `utils/audio` |
| **Utils** | ~30 crates under `utils/` (cache, pty, string, fuzzy-match, image, process, git-discovery, …) |

This is deliberate: each crate has explicit dependencies declared in the workspace
`Cargo.toml`, so the dependency graph is enforced by the compiler. There is no
"god crate" that everything imports — although `core` comes close.

---

## The SQ/EQ Protocol

**Path:** `protocol/src/protocol.rs` (~6,542 lines), `docs/protocol_v1.md`

The heart of Codex is a **Submission Queue / Event Queue** pattern:

```
//! Uses a SQ (Submission Queue) / EQ (Event Queue) pattern to asynchronously
//! communicate between user and agent.
```

- **SQ (Submission → agent):** `Submission { id, op, turn_extension_init, trace,
  parent_turn_id, root_turn_id, residency_guard }` — now a `pub(crate)` type in
  `core/src/session/submission.rs`, not a wire type.
- **EQ (agent → consumer):** `Event { id, msg }` (`protocol.rs`), where `id` correlates with the
  submission that started the current task.

### `Op` — what a client can ask for

The `Op` enum (tagged JSON, `"type": "snake_case"`) is the *entire* command surface of
the agent. Selected variants:

| Variant | Role |
|---------|------|
| `TurnInput { request, mode, reply }` | Start, steer, or queue a turn; replies with a `TurnInputSubmission` over a oneshot (replaced `UserInput`, `cbb7e82a8`) |
| `Interrupt` / `InterruptIfNoPendingInput` | Abort the current task |
| `RecoverTurn` / `SuspendTurnAndShutdown` | Turn recovery and suspend-on-shutdown |
| `Compact` | Trigger context compaction |
| `Review { review_request }` | Run a code review turn |
| `ThreadSettings` / `TurnSettings` | Change settings without starting a turn |
| `InterAgentCommunication` | Subagent mailbox message |
| `ExecApproval` / `PatchApproval` | Respond to approval prompts |
| `ApproveGuardianDeniedAction` | User override of a Guardian denial |
| `ResolveElicitation` / `UserInputAnswer` / `DynamicToolResponse` | Respond to MCP elicitation, structured input, client-side tools |
| `RequestPermissionsResponse` | Grant filesystem/network permissions |
| `RunUserShellCommand { command }` | `!cmd` one-off shell |
| `RealtimeConversation*` | Voice/realtime (WebRTC) |
| `RefreshMcpServers` / `ReloadUserConfig` / `SetThreadMemoryMode` | Hot reload / memory mode |
| `Shutdown` | Stop the agent |

`ThreadRollback` was removed (`3052bbcf8`); history undo is now the app-server's `thread/revert`.

### `EventMsg` — what the agent emits

`EventMsg` has **~84** variants grouped by concern: lifecycle (`TurnStarted`,
`TurnComplete`, `TurnAborted`, `SessionConfigured`, `ContextCompacted`), streaming
(`AgentMessageContentDelta`, `AgentReasoning`, `TokenCount`), tools/exec
(`ExecCommandBegin/OutputDelta/End`, `PatchApplyBegin/Updated/End`, `McpToolCallBegin/End`),
approvals (`ExecApprovalRequest`, `RequestPermissions`, `GuardianAssessment`, `GuardianWarning`),
multi-agent (`CollabAgentSpawn*`, `CollabAgentInteraction*`, `CollabWaiting*`, `CollabClose*`,
`CollabResume*`), hooks (`HookStarted`, `HookCompleted`), and review (`EnteredReviewMode`, `TurnDiff`).

### Serialization & bindings

Every wire type derives `Serialize`/`Deserialize`. Most also derive `schemars::JsonSchema`
and `ts-rs::TS`. TypeScript definitions are generated (`codex app-server generate-ts`),
which is how the Python/TypeScript SDK and the VS Code extension stay in sync with the
Rust core.

### Runtime channels

```rust
// core/src/session/mod.rs — session I/O split from session state (e4711f2a3)
pub(crate) struct SessionIo {
    pub(crate) tx_sub: Sender<Submission>,
    pub(crate) rx_event: Receiver<Event>,
    pub(crate) agent_status: watch::Receiver<AgentStatus>,
    pub(crate) session_loop_termination: SessionLoopTermination,
}

let (tx_sub, rx_sub) = async_channel::bounded(SUBMISSION_CHANNEL_CAPACITY); // 512
let (tx_event, rx_event) = async_channel::unbounded();
```

A background `submission_loop` (`session/handlers.rs`) consumes submissions until
`Op::Shutdown`. Consumers call `next_event().await`.

---

## Layered Architecture: Core → App-Server → Surfaces

This is the single most important architectural fact about Codex, and the one that most
distinguishes it from Claude Code:

```
┌──────────────────────────────────────────────────────────────┐
│  Surfaces                                                    │
│  TUI (ratatui)      codex exec       VS Code / SDKs          │
│       │                  │                 │                 │
│       └── app-server-client (in-process OR remote JSON-RPC) ─┘
└────────────────────────────┬─────────────────────────────────┘
                             │
                      codex-app-server
                      turn/start → Op::TurnInput
                      EventMsg → JSON-RPC notification
                             │
                      CodexThread.submit()  (ThreadManager)
                             ▼
                  ┌──────────────────────┐
                  │  codex-core Session  │
                  │  SQ async_channel(512)│
                  │  EQ async_channel(∞) │
                  │  submission_loop()   │
                  └──────────┬───────────┘
                             ▼
     External MCP servers (rmcp-client) · code-mode-host (gRPC) · exec-server environments
```

Key insight from the source: **the TUI and `codex exec` do not submit to core directly.**
They embed `codex-app-server` via `codex-app-server-client` (in-process by default, optionally
remote) and speak JSON-RPC. Only `codex-core` itself and `app-server` touch the raw SQ/EQ protocol.

So Codex now has **two façades over one core**:
1. The canonical `Submission`/`Event` SQ/EQ (in-core; `Submission` is `pub(crate)`).
2. A JSON-RPC app-server for rich clients (TUI, exec, VS Code, SDKs).

The third façade, `codex mcp-server`, was deprecated and then removed (`531f3836a`).

### The app-server

**Path:** `app-server/`, `app-server-protocol/`, `app-server-transport/`,
`app-server-client/`, `app-server-daemon/`

It is "JSON-RPC 2.0 minus the version field" by deliberate choice:

```
//! We do not do true JSON-RPC 2.0, as we neither send nor expect the
//! "jsonrpc": "2.0" field.
```

Transports: stdio (JSONL, default), unix socket (WebSocket over UDS), websocket, and
in-process (bounded Tokio channels — no process boundary). Method names are v2-style and
path-shaped:

- **Threads:** `thread/start`, `thread/resume`, `thread/fork`, `thread/archive`,
  `thread/delete`, `thread/list`, `thread/search`, `thread/read`, `thread/compact/start`,
  `thread/revert`, `thread/inject_items`, `thread/goal/set`, `thread/settings/update`, …
- **Queue/attachments (new):** `thread/queue/{add,list,update,reorder,delete,start}`,
  `thread/attachment/{add,list,remove}`, `thread/backgroundTerminals/*`
- **Turns:** `turn/start`, `turn/steer`, `turn/interrupt`, `turn/settings/update`
- **Review:** `review/start`
- **MCP/config/account:** `mcpServer/tool/call`, `config/read`, `account/login/start`, …

`turn/start` maps to `Op::TurnInput`; `EventMsg` is mapped to notifications such
as `item/agentMessage/delta` and `item/commandExecution/outputDelta`
(`app-server-protocol/src/protocol/event_mapping.rs`; method names in
`app-server-protocol/src/protocol/common.rs`).

---

## Agent Loop: Threads, Sessions, Turns, Tasks

**Path:** `core/src/thread_manager.rs`, `core/src/codex_thread.rs`,
`core/src/session/`, `core/src/tasks/`

The vocabulary matters here:

| Concept | Struct | Role |
|---------|--------|------|
| **Thread manager** | `ThreadManager` (~2,749 ln) | Owns thread store, model refresh; `start_thread`, `resume_thread_from_rollout`, `fork_thread` |
| **Thread façade** | `CodexThread` (~1,258 ln) | Wrapper: `submit(op)`, `steer_input`, `shutdown_and_wait` |
| **Session I/O** | `SessionIo` (in `session/mod.rs`, ~5,287 ln) | The SQ/EQ queue pair, split from session state (`e4711f2a3`) |
| **Session** | `Session` (`session/session.rs`, ~1,990 ln) | "At most **1** running task at a time"; holds `state`, `active_turn`, `input_queue` |

### From prompt to turn

1. A client submits `Op::TurnInput`.
2. `submission_loop` (`handlers.rs`) dispatches it to `user_input_or_turn`.
3. `Session::new_turn_with_sub_id` builds an `Arc<TurnContext>`.
4. If a **regular** turn is already running, the input is **steered** into the active
   turn's `pending_input`. Otherwise:
   ```rust
   sess.spawn_task(Arc::clone(&current_context), task_input, RegularTask::new()).await;
   ```
5. `spawn_task` (`tasks/mod.rs`) **always aborts any existing task first**
   (`TurnAbortReason::Replaced`) — a session runs one task at a time.

### Tasks

There is a `SessionTask` trait with a small set of implementations:

```rust
pub(crate) trait SessionTask: Send + Sync + 'static {
    fn kind(&self) -> TaskKind;
    fn run(self: Arc<Self>, session, ctx, input, cancellation_token)
        -> impl Future<Output = Option<String>> + Send;
    fn abort(&self, session, ctx) -> …;
}
```

| Task | Kind | Behavior |
|------|------|----------|
| `RegularTask` | `Regular` | Runs the `run_turn` loop; steerable; continues if pending input remains |
| `ReviewTask` | `Review` | Spawns a delegated one-shot review sub-thread; not steerable |
| `CompactTask` | `Compact` | Local/remote context compaction; not steerable |
| `UserShellCommandTask` | — | Runs `!cmd` shell outside the chat path |

### The turn loop (`run_turn`, `session/turn.rs`)

This is the agent loop proper. Documented inline: each **sampling request** yields tool
calls and/or assistant messages; tool output is fed into the *next* sampling request;
an assistant-only response can complete the turn.

```
run_turn:
  run_pre_sampling_compact()         ← pre-turn compaction
  inject skills/plugins, run SessionStart hooks
  loop:
    drain pending_input into history (steering/mailbox)
    build prompt from clone_history().for_prompt()
    run_sampling_request()           ← one model stream + tools
      → ModelClientSession.stream()
      → match ResponseEvent::* (deltas, OutputItemDone, Completed)
      → push tool calls into FuturesOrdered in_flight
      → drain_in_flight() records tool outputs into history
    if token_limit && needs_follow_up: run_auto_compact(); continue
    if !needs_follow_up: run Stop hooks; break
    else: continue (another sampling in same turn)
```

### Concurrency model — one task, steered

`Session` carries a comment: *"A session has at most 1 running task at a time, and can be
interrupted by user input."* The active task lives in:

```rust
pub(crate) struct ActiveTurn {
    pub(crate) task: Option<RunningTask>,
    pub(crate) turn_state: Arc<Mutex<TurnState>>,
}
```

`RunningTask` holds a `CancellationToken`, an `AbortOnDropHandle`, and a `Notify`.
Abort gives a graceful window (`GRACEFULL_INTERRUPTION_TIMEOUT_MS = 100`) before a hard
`handle.abort()`.

**Steering** pushes new user text onto `pending_input` and re-opens mailbox delivery;
it is rejected for Review/Compact turns. **Interrupt** aborts and, if there is pending
multi-agent mail, can auto-start a fresh `RegularTask`.

---

## Streaming & The Model Client

**Path:** `core/src/client.rs` (~3,045 ln), `core/src/client_common.rs`,
`codex-api/src/common.rs`

Two layers:

- **`ModelClient`** — session-scoped (auth, provider, thread id, websocket fallback).
- **`ModelClientSession`** — created **per turn**; lazily opens a Responses WebSocket and
  carries a sticky `x-codex-turn-state`.

```rust
pub struct ResponseStream {
    pub(crate) rx_event: mpsc::Receiver<Result<ResponseEvent>>,
    pub(crate) consumer_dropped: CancellationToken,
}
// impl Stream<Item = Result<ResponseEvent>>
```

`ResponseEvent` variants include `Created`, `OutputItemAdded/Done`, `OutputTextDelta`,
`ToolCallInputDelta`, `ReasoningSummaryDelta`, `RateLimits`, and `Completed { response_id,
token_usage, end_turn }`.

### Responses API only

`WireApi` (`model-provider-info/src/lib.rs`) has exactly **one** variant: `WireApi::Responses`.
There is no Chat Completions branch in the live client — legacy `chat` and `ollama-chat` are
explicitly rejected by the provider layer. Still true at `cc7ba3360`: even the new
`amazon-bedrock-runtime` provider targets Bedrock's OpenAI-compatible
`https://bedrock-runtime.{region}.amazonaws.com/openai/v1` (`model-provider/src/amazon_bedrock/runtime.rs`). Codex assumes the OpenAI **Responses API**
(stateful, server-managed conversation items, reasoning) as its native protocol.

Two transports:
- **WebSocket** (`stream_responses_websocket`) with beta header
  `responses_websockets=2026-02-06`.
- **HTTP SSE** fallback (`stream_responses_api`) via `eventsource_stream`.

Compaction has its own unary endpoint (`/responses/compact`).

---

## State Management

**Path:** `core/src/state/`, `core/src/context_manager/`

Codex uses **shared mutable state guarded by async locks** — not immutable snapshots.

```rust
// Session
state: Mutex<SessionState>,                 // tokio::sync::Mutex
active_turn: Mutex<Option<ActiveTurn>>,
```

`SessionState` (`state/session.rs`, ~461 ln) holds the live session: the
`ContextManager` history, rate limits, additional-context store, granted permissions,
auto-compact window, startup prewarm handle, active connector selection, and more.

`TurnState` (`state/turn.rs`) is per-turn and lives behind `Arc<Mutex<TurnState>>`. It
tracks pending approvals, pending elicitations, the `pending_input` queue, mailbox
delivery phase, tool-call counts, and token usage at turn start.

`ContextManager` (`context_manager/history.rs`, ~1,400 ln) is the transcript:
`items: Vec<ResponseItem>`, token info, and a `history_version`. Prompts are built from
`Session::clone_history()` (a snapshot taken under the lock, then released).

So the model is: **one mutable session protected by a Mutex, with cheap cloned snapshots
taken when building a prompt.** This is the opposite of an immutable-snapshot architecture
but is internally consistent and lock-disciplined (clippy denies `await_holding_lock`).

---

## Tool System

**Path:** `core/src/tools/` (registry, router, orchestrator, parallel, sandboxing,
handlers/, runtimes/), `tools/` crate

### The abstraction

There is **no central enum of tools.** Tools are trait objects keyed by name:

```rust
// tools/src/tool_executor.rs
#[async_trait]
pub trait ToolExecutor<Invocation>: Send + Sync {
    fn tool_name(&self) -> ToolName;
    fn spec(&self) -> ToolSpec;
    fn exposure(&self) -> ToolExposure { ToolExposure::Direct }
    fn supports_parallel_tool_calls(&self) -> bool { false }
    async fn handle(&self, invocation) -> Result<Box<dyn ToolOutput>, FunctionCallError>;
}
```

Core wraps this with `CoreToolRuntime` (adds search metadata, hook payloads, diff
consumers), stored in:

```rust
pub struct ToolRegistry { tools: HashMap<ToolName, Arc<dyn CoreToolRuntime>> }
```

`ToolExposure` is one of `Direct`, `Deferred` (discoverable via search), `DirectModelOnly`,
`Hidden`.

### Schema & assembly

Each handler returns a `ToolSpec`:

```rust
pub enum ToolSpec {
    Function(ResponsesApiTool),
    Namespace(ResponsesApiNamespace),
    ToolSearch { execution, description, parameters },
    ImageGeneration { output_format },
    WebSearch { … },
    Freeform(FreeformTool),       // e.g. apply_patch
}
```

`spec_plan.rs` (~1,722 ln) assembles the per-turn tool set by **feature flags**: it adds tool
sources, appends a tool-search executor when there are deferred tools, prepends code-mode
executors when in code mode, then builds the model-visible specs + registry.

### Built-in tool handlers

| Tool | Purpose |
|------|---------|
| `exec_command` / `write_stdin` | Unified exec (PTY or pipes) and stdin writes — the only model-facing shell since `8a40095ea` retired `shell_command` |
| `apply_patch` | Freeform patch application (Lark grammar) |
| `new_context_window` / `get_context_remaining` | Context-window management |
| `current_time` / `sleep` / `wait_for_environment` | Utility and environment waits |
| `view_image` | Attach an image for multimodal context |
| `update_plan` | Update the agent's plan state |
| `request_permissions` | Ask user for session/turn FS/network grants |
| `request_user_input` | Prompt user for structured input |
| `tool_search` | **BM25** search over deferred tool metadata |
| `list_mcp_resources` / `read_mcp_resource` / `…_templates` | MCP resources |
| `create_goal` / `get_goal` / `update_goal` | Goal CRUD (now an extension tool in `ext/goal`) |
| `spawn_agent` / `followup_task` / `send_message` / `interrupt_agent` / `wait_agent` / `list_agents` | Multi-agent v2 (`handlers/multi_agents_v2/`) |
| `spawn_agent` / `send_input` / `resume_agent` / `wait_agent` / `close_agent` | Multi-agent v1 (`handlers/multi_agents/`) |
| `exec` / `wait` | Code mode (V8, out-of-process) |
| `web_search` / `image_generation` | Provider-hosted (not in registry) |
| `request_plugin_install` / `list_available_plugins_to_install` | Plugin discovery |
| *(dynamic / MCP / extension)* | `DynamicToolHandler`, `McpHandler`, `ExtensionToolAdapter` |

### Concurrency: parallel vs sequential

`parallel.rs` gates execution with a single `Arc<RwLock<()>>`:

```rust
let _guard = if supports_parallel {
    Either::Left(lock.read().await)    // shared read → run in parallel
} else {
    Either::Right(lock.write().await)  // exclusive → serialize
};
```

Tools that opt in (`supports_parallel_tool_calls() == true` — e.g. `view_image`, MCP when
the server allows) run concurrently as `tokio::spawn` tasks with
`AbortOnDropHandle`. Everything else serializes. This is the read-parallel/write-sequential
pattern, expressed via a reader-writer lock.

### Orchestration

`orchestrator.rs` runs the approval → sandbox → execute → retry pipeline for the exec-style
tools (shell, apply_patch, unified exec): assess approval requirement → pick the initial
sandbox attempt → request network approval → run → on `SandboxErr::Denied`, optionally ask
the user to retry without sandbox.

---

## Command Execution

**Path:** `core/src/exec.rs` (~1,277 ln), `core/src/unified_exec/`,
`core/src/tools/runtimes/`, `apply-patch/`, `exec-server/`

Three distinct execution paths:

1. **One-shot subprocess (`exec.rs`).** Pipe-based sandboxed spawn with byte-capped output.
   The model-facing `shell_command` tool that used it was retired; shell execution is
   standardized on unified exec (`8a40095ea`).

2. **Unified exec (`unified_exec/`).** Backs `exec_command`/`write_stdin`. Spawns a **PTY**
   when `tty: true`, keeps a process store so the model can spawn and later write stdin to a
   live process, and buffers output in a `HeadTailBuffer`. Same orchestrator/approval/sandbox
   pipeline.

3. **apply_patch.** A Lark-grammar patch format:
   ```
   *** Begin Patch
   *** Update File: src/foo.rs
   @@
   - old line
   + new line
   *** End Patch
   ```
   Two entry points: the dedicated `ApplyPatchHandler` (a `Freeform` tool), and an
   **intercept** — `intercept_apply_patch` detects `["apply_patch", "<patch>"]` inside a
   shell/unified-exec invocation and reroutes it to the sandboxed patch runtime. Writes go
   through a `FileSystemSandboxContext` with per-file approval keys.

**Execution environments.** `exec-server` (~49K lines) now "owns the execution/filesystem
environments available to the Codex runtime" (`exec-server/src/environment.rs`): local or
remote environments with capability discovery, reconnection, and a filesystem sandbox, selected
per turn (`core/src/environment_selection.rs`, `wait_for_environment` tool). Managed Git
worktrees (`worktree/`, TUI tools in `e7ea5f4a8`) give new/forked threads their own checkout.

---

## Sandboxing (3 Platforms, 4 Backends)

**Path:** `sandboxing/`, `linux-sandbox/`, `windows-sandbox-rs/`, `mxc-sandbox/`,
`windows-sandbox-service/`

```rust
// protocol/src/sandbox.rs
pub enum SandboxType { None, MacosSeatbelt, LinuxSeccomp, WindowsRestrictedToken, WindowsMxc }
```

| Platform | Mechanism |
|----------|-----------|
| **Linux** | A separate `codex-linux-sandbox` helper binary is prepended to argv; it runs the command under **bubblewrap** + **seccomp** (legacy Landlock path behind `--use-legacy-landlock`). `linux-sandbox/src/linux_run_main.rs` is ~1,605 lines. |
| **macOS** | `/usr/bin/sandbox-exec` with a compiled `.sbpl` policy (`seatbelt_base_policy.sbpl`, `seatbelt_network_policy.sbpl`). `seatbelt.rs` ~1,125 lines. |
| **Windows** | Restricted-token / elevated-sandbox backends, ACL-based deny rules, WFP network filtering, ConPTY. `windows-sandbox-rs/src/lib.rs` ~1,085 lines. New: a native **MXC** backend (`mxc-sandbox`, "never enters MXC's ACL fallbacks"; `60888d086`), managed MXC requirements (`f90a578f7`), and a provisioning `windows-sandbox-service`. |

A verified sandbox launcher is now also exposed to local MCP servers (`78a19f5ab`).

The Linux wrapper passes the policy explicitly:

```rust
let mut linux_cmd = vec![
    "--sandbox-policy-cwd".into(), sandbox_policy_cwd,
    "--command-cwd".into(), command_cwd,
    "--permission-profile".into(), permission_profile_json,
];
// optional: --use-legacy-landlock, --allow-network-for-proxy
linux_cmd.push("--".into());
linux_cmd.extend(command);
```

### Policy modes

Legacy `SandboxPolicy`: `DangerFullAccess`, `ReadOnly { network_access }`,
`ExternalSandbox { network_access }`, `WorkspaceWrite { writable_roots, network_access,
exclude_tmpdir_env_var, exclude_slash_tmp }`.

Runtime model `PermissionProfile`: `Managed { file_system, network }`, `Disabled`,
`External { network }`. Built-in profile IDs map to read-only / workspace-write /
danger-full-access semantics.

Network egress is mediated by a **managed network proxy** (`network-proxy`,
`responses-api-proxy`) with allowlist-based approval.

---

## ExecPolicy (Starlark) & Approvals

**Path:** `execpolicy/`, `core/src/exec_policy.rs` (~1,175 ln),
`core/src/safety.rs`, `core/src/tools/network_approval.rs`

ExecPolicy is a **Starlark-based prefix-rule language** for deciding whether a command runs:

```starlark
prefix_rule(
    pattern = ["git", "status"],
    decision = "allow",            # allow | prompt | forbidden
    justification = "...",
)
host_executable(name = "git", paths = ["/usr/bin/git", ...])
```

```rust
pub enum Decision { Allow, Prompt, Forbidden }
```

`exec_policy.rs` lowers shell wrappers (`bash -lc`, PowerShell `-Command`) to token prefixes,
matches against the policy, and maps the result to an `ExecApprovalRequirement`:

```rust
pub(crate) enum ExecApprovalRequirement {
    Skip { bypass_sandbox, proposed_execpolicy_amendment },
    NeedsApproval { reason, proposed_execpolicy_amendment },
    Forbidden { reason },
}
```

When a user approves a prompt, the policy can be **amended** — an allow prefix rule is
appended to the policy file, so the same command is trusted next time.

### The full approval stack (in precedence order)

1. **ExecPolicy** prefix rules.
2. **PermissionRequest hooks** (can allow/deny before guardian/user).
3. **Guardian** automated review *or* user approval.
4. **Network approval** for managed-proxy allowlist misses (immediate or deferred,
   host/session caching).
5. **Patch safety** (`assess_patch_safety` → `AutoApprove | AskUser | Reject`).

```rust
pub enum SafetyCheck {
    AutoApprove { sandbox_type, user_explicitly_approved },
    AskUser,
    Reject { reason },
}
```

The older execv-oriented `execpolicy-legacy` matcher was deleted (`656a2d090`); Starlark
`execpolicy` is now the only policy engine. The `untrusted` approval policy was also retired
(`942af8447`).

---

## Code Mode (V8)

**Path:** `code-mode/`, `code-mode-protocol/`, `code-mode-host/`, `code-mode-runtime/`,
`core/src/code_mode_host.rs`, `core/src/tools/code_mode/`

Code mode is one of the more unusual features. When enabled (`ToolMode::CodeMode` /
`CodeModeOnly`), Codex exposes two tools — `exec` and `wait` — and runs the model's
JavaScript in a **V8 isolate** (`v8 = "=150.4.0"` with `v8_enable_sandbox`).

**Changed since June:** V8 is no longer linked into core. `code-mode-runtime` owns the isolate;
the `codex-code-mode-host` binary hosts it; core's `ProcessOwnedGrpcCodeModeSessionProvider`
spawns the host and talks gRPC over stdio pipes with HTTP/2 keep-alives, killing and reaping
the child on reconnect (`core/src/code_mode_host.rs`; `aa46f2deb`, `2dc0791c7`). The earlier
WebSocket host transport was removed (`725b3a44f`). Nested tools are
projected into the JS global as `global.tools.<normalized_name>(...)`. A
`CodeModeDispatchBroker` routes those nested calls back through the normal tool runtime with
`ToolCallSource::CodeMode`.

In `CodeModeOnly`, the top-level model only sees `exec`/`wait`; all other tools are reachable
*only* from JavaScript. This lets the model orchestrate many tool calls in a single
program rather than one tool call per model round-trip. (Note: this is V8/JavaScript, not
Starlark — Starlark appears only in execpolicy; "Lark" appears only in the apply-patch grammar.)

---

## Context Compaction

**Path:** `core/src/compact.rs`, `compact_remote_v2.rs` (~1,332 ln), `compact_remote_history.rs`,
`core/src/tasks/compact.rs`

Three trigger points:

| Phase | Trigger |
|-------|---------|
| **Pre-turn** | `run_pre_sampling_compact` at the start of `run_turn`; also a model-downshift inline compact |
| **Mid-turn** | `run_auto_compact` inside the loop when `token_limit_reached && needs_follow_up` |
| **Manual** | `Op::Compact` → `CompactTask` |

Two implementations, chosen by provider capability:

- **Remote v2** — `compact_remote_v2` (server-side compaction; the legacy v1 remote
  implementation was removed in `1ac689cc7`).
- **Local** — streams a summary via a normal `ModelClientSession` using `SUMMARIZATION_PROMPT`.

`InitialContextInjection` controls re-injection: `DoNotInject` (manual/pre-turn — the next
regular turn reinjects full context) vs `BeforeLastUserMessage` (mid-turn — the summary must
remain the last visible item). `PreCompact`/`PostCompact` hooks fire around the operation.

---

## Config System

**Path:** `config/` crate (loader in `config/src/loader/mod.rs`), `core/src/config/mod.rs` (~5,071 ln)

Config is layered TOML (`config.toml`) with a strict precedence ladder. **Requirements**
(cloud-managed, macOS admin, system `requirements.toml`) come first and **cannot be
overridden**. Then config layers merge, later overriding earlier:

```
admin (macOS managed)
  → /etc/codex/config.toml (system)
  → $CODEX_HOME/config.toml (user)
  → $CODEX_HOME/<name>.config.toml (selected profile)
  → $PWD/config.toml (cwd; disabled if untrusted)
  → .codex/config.toml (tree parent; disabled if untrusted)
  → <repo>/.codex/config.toml (repo; disabled if untrusted)
  → runtime (--config flags, UI overrides)
```

Project-local layers are denylisted from setting sensitive keys (`openai_base_url`,
`model_provider`, `otel`, …) via `PROJECT_LOCAL_CONFIG_DENYLIST`. Profiles (`ConfigProfile`)
bundle model, provider, approval, sandbox, reasoning, and features. Permission profiles live
under `[permissions.<name>]` with `extends` inheritance; built-ins are `:read-only`,
`:workspace` (default for trusted projects), and `:danger-full-access`.

---

## Session Persistence: Rollouts + SQLite

**Path:** `rollout/`, `thread-store/`, `state/`, `core/src/state_db_bridge.rs`

Codex persists sessions **two ways at once**:

1. **Rollout (canonical history)** — a JSONL file of `RolloutItem` events:
   ```
   $CODEX_HOME/sessions/rollout-{date}-{conversation_id}.jsonl
   ```
   Written by `RolloutRecorder` (`rollout/src/recorder.rs`, ~2,266 ln). The first line is
   `SessionMeta` (includes `forked_from_id` for lineage). Archived sessions move to
   `archived_sessions/`. A persistence policy decides what is written and truncates exec
   output in extended mode.

2. **SQLite state DBs (queryable metadata)** — via `sqlx` 0.9 with embedded migrations:
   ```rust
   // state/src/migrations.rs
   STATE_MIGRATOR, LOGS_MIGRATOR, GOALS_MIGRATOR, MEMORIES_MIGRATOR,
   QUEUE_MIGRATOR, THREAD_HISTORY_MIGRATOR  // each sqlx_macros::migrate!("./…")
   ```
   DB files (`state/src/sqlite.rs`): `state_5.sqlite` (thread metadata, spawn edges;
   migrations now at `0061_thread_read_state`), `logs_2.sqlite`, `goals_1.sqlite`,
   `memories_1.sqlite`, `memories_v2_1.sqlite`, `queue_1.sqlite`, `thread_history_1.sqlite`,
   plus `agent_message_board_1.sqlite` owned by `ext/agent-message-board`. Located in
   `sqlite_home` (`$CODEX_SQLITE_HOME` or `$CODEX_HOME`).

`thread-store`'s `LiveThread` ties them together: apply persistence policy → append JSONL →
patch SQLite metadata. Fork/spawn topology lives in the `0021_thread_spawn_edges` table and is
read via `agent-graph-store`.

---

## Skills & AGENTS.md

**Path:** `ext/skills/`, `core/src/skills.rs`, `core/src/agents_md.rs`, `core/src/agents_md_manager.rs`

### Skills

Skills are `SKILL.md` files with YAML frontmatter. The `core-skills` crate and legacy core
loader were removed; loading, catalog prompts, aliases, and host prompt injection now live in
the `ext/skills` extension (`33e365b19`, `beac16ccc`, `45f8cafa4`), and the remote skills
client was dropped (`3b366654f`). When a user mentions a skill via the `$skill` sigil, the full
`SKILL.md` body is read at call time and injected. (The June root list — `$HOME/.agents/skills`,
`<repo>/.codex/skills`, plugin roots, … — was not re-verified line by line in this pass.)

### AGENTS.md

Separate from skills. Codex finds the project root (default marker `.git`), collects
`AGENTS.override.md` / `AGENTS.md` from root→cwd, plus a global one in `$CODEX_HOME`, joins
them with a separator, and folds them into the user instructions (capped at 32 KiB:
`AGENTS_MD_MAX_BYTES` in `core/src/config/mod.rs`). User instructions are now loaded through
an injected provider (`236b50125`).

```rust
pub const DEFAULT_AGENTS_MD_FILENAME: &str = "AGENTS.md";
pub const LOCAL_AGENTS_MD_FILENAME: &str = "AGENTS.override.md";
```

---

## Hooks

**Path:** `hooks/`, `core/src/hook_runtime.rs` (~1,353 ln), `config/src/hook_config.rs`

Codex implements a **Claude-Code-compatible hook engine** with 12 lifecycle events
(`hooks/src/lib.rs`; `SessionEnd` and `Interrupt` added since June):

```rust
pub const HOOK_EVENT_NAMES: [&str; 12] = [
    "PreToolUse", "PermissionRequest", "PostToolUse", "PreCompact", "PostCompact",
    "SessionStart", "SessionEnd", "UserPromptSubmit", "SubagentStart", "SubagentStop",
    "Stop", "Interrupt",
];
```

Handler types (`Command` now has `async`, `commandWindows`, and an `additionalContext` spill
threshold; `McpTool` is new — executed via `core/src/hook_mcp_executor.rs`):

```rust
pub enum HookHandlerConfig { Command { command, … }, McpTool { … }, Prompt {}, Agent {} }
```

Hooks come from user config (`[hooks]`), managed requirements (admin/cloud), and plugin
bundles (`hooks/hooks.json`). Trust is enforced via `trusted_hash`. Outcomes can block a
tool, inject additional context, or stop the session.

---

## Plugins & Connectors

**Path:** `plugin/`, `core-plugins/`, `connectors/`, `core/src/plugins/`

**Plugins** are bundles described by a `PluginManifest` that can contribute skills
(`skills/`), hooks (`hooks/hooks.json`), MCP servers (`.mcp.json`), and app connectors
(`.app.json`). Capabilities are summarized per plugin; marketplaces are `openai-curated` and
`openai-bundled`. Install flows are first-class tools (`request_plugin_install`).

**Connectors** are different: they are **ChatGPT Apps directory entries** fetched from backend
APIs (`/connectors/directory/list`), cached on disk, normalized to `AppInfo` with install URLs
like `https://chatgpt.com/apps/{slug}/{id}`. They surface linked ChatGPT apps in the composer.

---

## Multi-Agent System

**Path:** `core/src/tools/handlers/multi_agents_v2/`, `core/src/tools/handlers/multi_agents/`,
`agent-roles/`, `ext/agent-message-board/`, `agent-identity/`, `agent-graph-store/`

The v2 tool set: `spawn_agent`, `followup_task`, `send_message`, `interrupt_agent`,
`list_agents`, `wait_agent` (`assign_task` and `close_agent` are no longer v2 tools). The v1 set
(`spawn_agent`, `send_input`, `resume_agent`, `wait_agent`, `close_agent`) still exists and is
selected per turn/model — e.g. v1 for some Bedrock models (`fc8c6b738`, `f3cd29942`). Spawn parses
model/reasoning/role overrides, emits `CollabAgentSpawnBegin`, and delegates with **depth limits**
(`DEFAULT_AGENT_MAX_DEPTH = 1`, `DEFAULT_MULTI_AGENT_V2_MAX_CONCURRENT_THREADS_PER_SESSION = 4`,
both in `core/src/config/mod.rs`). Partial-history forks were removed (`6221a217e`); CSV-backed
agent jobs (`spawn_agents_on_csv`) were deleted (`687f05cb9`). Roles are bounded config overrides
loaded by the `agent-roles` crate (`1a6e07a4f`, `fb9311db5`).

**Agent message board (new).** `ext/agent-message-board` provides "shared agent discussions with
interchangeable local, in-memory and remote backends"; board and caller identity come from the
host. The local backend is SQLite (`agent_message_board_1.sqlite`) with paginated queries,
channel permissions, and structured denials (`293de2717`, `8f2c15c39`, `36ae1561b`). It is wired
into persistent multi-agent runtimes (`6149914a0`); a remote backend uses
`agent-message-board-client`, whose session-scoped credentials "tools never see".

Two supporting crates:
- **`agent-identity`** — generates Ed25519 keys, registers agent tasks with the ChatGPT
  backend, and builds `AgentAssertion` auth headers (JWT, JWKS from
  `{chatgpt_base_url}/agent-identities/jwks`). It is a real `CodexAuth::AgentIdentity` mode.
- **`agent-graph-store`** — persists parent/child spawn edges in SQLite
  (`upsert_thread_spawn_edge`). `SubagentStart`/`SubagentStop` hooks fire on transitions.

Inter-agent communication flows over the session **mailbox** (`InterAgentCommunication`),
delivered into a turn's `pending_input`.

---

## Guardian (Automated Approval Reviewer)

**Path:** `core/src/guardian/`, `core/src/guardian_review.rs`, `ext/guardian-reviewer/`,
`ext/guardian-v2/` (`ext/guardian` was removed)

Guardian is **not** a sandbox — it is an **LLM that decides whether an `on-request` approval
should be auto-granted instead of shown to the user.**

```rust
pub(crate) fn routes_approval_to_guardian(turn: &TurnContext) -> bool {
    matches!(turn.approval_policy.value(),
             AskForApproval::OnRequest | AskForApproval::Granular(_))
        && turn.config.approvals_reviewer == ApprovalsReviewer::AutoReview
}
```

It builds a compact transcript, runs a review under the reviewer name `"guardian"`
(`GUARDIAN_REVIEWER_NAME`; model `codex-auto-review` in tests), and expects strict JSON
(`GuardianAssessment`; assessment text is now required, `be2e129ec`). It **fails closed** on
timeout (`REVIEW_TIMEOUT` from `codex_guardian_reviewer`), malformed output, or error, and has a
circuit breaker after `MAX_CONSECUTIVE_GUARDIAN_DENIALS_PER_TURN = 3`
(`ext/guardian-reviewer/src/circuit_breaker.rs`).

**Changed since June** (~430 Guardian commits):
- Synchronous review bookkeeping and policy moved to `ext/guardian-reviewer`, "independently of
  the host session runtime"; subagent-spawner plumbing was removed (`f2d9bccbd`).
- **Guardian V2** (`ext/guardian-v2`, ~16K lines) adds an `async_scorer` (background
  classification, budgets, its own Responses WebSocket connection pool) beside a
  `sync_reviewer` (`fe614a630`, `6ac8b1b50`).
- Users can override a denial (`Op::ApproveGuardianDeniedAction`, `thread/approveGuardianDeniedAction`).
- Opt-in trust for orchestrator connector identities (`75e0b4088`) and per-turn "Cyber access
  programs" forwarding (`10da56925`, `core/src/cyber_access_program.rs`).

---

## Memories & Goals

**Path:** `ext/memories/`, `memories/read`, `memories/write`, `ext/goal/`, `ext/history-notes/`

**Memories** (feature-gated): a read path exposing `memories.{list,read,search,add_ad_hoc_note}`
tools, and a write path that runs a consolidation pipeline at startup (summarizing rollouts
into `raw_memories.md`, `rollout_summaries/`, indexed in `memories_1.sqlite`). A second
`memories_v2_1.sqlite` now exists (`state/src/sqlite.rs`); per-thread memory mode is settable
via `Op::SetThreadMemoryMode` / `thread/memoryMode/set`. (Tool names not re-verified in this pass.)

**Goals** (`ext/goal`, ~5.9K lines; `core/src/goals.rs` no longer exists): `create_goal`/
`get_goal`/`update_goal` as extension tools (`ext/goal/src/tool.rs`, routed through
`core/src/tools/handlers/extension_tools.rs`) with an explicit **token budget**, budget-limit
steering prompts, and OTEL metrics, persisted in `goals_1.sqlite`.

---

## Provider & Auth Layer

**Path:** `model-provider-info/`, `model-provider/`, `models-manager/`, `ollama/`,
`lmstudio/`, `login/`

Built-in providers:

```rust
// model-provider-info/src/lib.rs
"openai", "amazon-bedrock", "amazon-bedrock-runtime", "ollama", "lmstudio"
```

All use `WireApi::Responses`; legacy chat APIs are rejected. Bedrock gained a runtime endpoint,
GovCloud support (`d4a475add`), and multi-agent v2 on some models (`8ffd91e42`). User providers can be added via
`[model_providers.<id>]`. `ModelsManager` caches the model catalog at
`$CODEX_HOME/models_cache.json` and merges remote catalogs with local `models.json` presets.

**Auth (`login/`)** — `CodexAuth` variants:
- `ApiKey` (`OPENAI_API_KEY` / `CODEX_API_KEY`)
- `Chatgpt` — OAuth tokens in `$CODEX_HOME/auth.json` or the OS keyring (PKCE, localhost
  callback server on port 1455, issuer `https://auth.openai.com`)
- `ChatgptAuthTokens` — token-only
- `AgentIdentity` — the agent-task JWT flow
- New since June (`login/src/auth/manager.rs`): `Headers`, `PersonalAccessToken`,
  `BedrockApiKey`, `BedrockAccessKeys`. Separate crates add `workload-identity` (token exchange,
  `12933b695`) and `user-verification` (device credentials and signing).

Local OSS: **Ollama** (requires ≥ 0.13.4 for Responses API, default `gpt-oss:20b`) and **LM
Studio** (default `openai/gpt-oss-20b`), selected by `oss_provider`/`--local-provider`.

---

## MCP Integration (Client only)

**Path:** `codex-mcp/`, `rmcp-client/`, `ext/mcp/`

Codex is now **only an MCP client**. The `codex mcp-server` command and `mcp-server/` crate
(which exposed Codex as an MCP tool over stdio, with approvals via elicitation) were deprecated
and removed (`531f3836a`); external hosts integrate through the app-server instead.

**As a client** (`codex-mcp` + `rmcp-client` + the new `ext/mcp` extension): connects to
external MCP servers. Transports (`rmcp-client/src/rmcp_client.rs`):

```rust
enum PendingTransport {
    InProcess { … }, Stdio { … }, StreamableHttp { … },
    StreamableHttpWithOAuth { … }, StreamableHttpWithAccessTokenOnly { … },
}
```

New since June: application network policy is enforced on MCP commands (`79773e882`), local MCP
servers can use a verified sandbox launcher (`78a19f5ab`), and MCP user verification
(`b38942ce7`) is supported.

The connection manager emits `McpStartupUpdate/Complete` and turns each remote MCP tool into a
local `McpHandler` in the tool registry.

---

## Terminal UI (ratatui)

**Path:** `tui/` (~1,105 `.rs` files, ~450K lines)

The TUI is a large, conventional ratatui + crossterm application with a custom terminal
backend for history insertion and hyperlinks. Modules: `app/` (run loop + event dispatch),
`chatwidget/` (the main chat surface and tool lifecycle), `bottom_pane/` (composer, approvals,
popups, slash commands — `chat_composer.rs` alone is ~12.7K lines), `history_cell/` (transcript
cell types), `render/`, `markdown_render.rs`, plus `status/`, `onboarding/`, `resume_picker/`.

Crucially, the TUI does **not** consume raw `EventMsg`. It embeds the app-server via
`AppServerClient` (in-process or remote) and reacts to typed `ServerNotification`s, which it
translates into internal `AppEvent`s and feeds to `ChatWidget::handle_server_notification`.

---

## CLI & arg0 Multitool

**Path:** `cli/` (~30K lines), `arg0/`

The CLI is a **multitool** that dispatches on `argv[0]` via `codex_arg0::arg0_dispatch_or_else`:
the same binary can re-exec itself as `apply_patch`, `codex-linux-sandbox`,
`codex-execve-wrapper`, etc., prepending temp PATH aliases so child processes can find the
helpers.

Subcommands (`enum Subcommand` in `cli/src/main.rs`) include: `exec` (headless), `review`,
`login`/`logout`, `mcp` (manage MCP servers), `plugin`, `agents`, `queue`, `app-server`,
`exec-server`, `remote-control`, `tcp-tunnel`, `app` (desktop), `apply`,
`resume`/`fork`/`archive`/`unarchive`/`delete`, `migrate-rollouts`, `cloud`, `doctor`,
`sandbox`, `debug`, `execpolicy`, `features`, `completion`, `update`. `mcp-server` is gone.
No subcommand → the interactive TUI.

---

## Observability

**Path:** `otel/`, `analytics/`

- **OpenTelemetry** (`otel/`, OTLP exporters, default Statsig endpoint
  `https://ab.chatgpt.com/otlp/v1/metrics`; disabled by default in debug builds). Metrics for
  hooks, sessions, and DB operations (`codex.db.*`, `codex.sqlite.init.*`).
- **Analytics** (`analytics/`) — an async event queue (256 events) reduced and shipped via
  app-server RPC. Facts: hook runs, skill invocations, guardian reviews, compaction, subagent
  starts, turn config, token usage. Gated by config.

---

## Key Design Patterns

### 1. Protocol-first, façade-layered

One canonical SQ/EQ protocol in `core`; a JSON-RPC app-server façade for every client (the MCP
façade was removed in `531f3836a`). Surfaces never reach into core internals — they speak a wire protocol,
even in-process. This is the inverse of Claude Code's "everything imports AppState."

### 2. One running task per session, with steering

Concurrency is intentionally constrained: a session runs exactly one task; new input either
steers the active turn or replaces it. Cancellation is a first-class `CancellationToken` with a
graceful window.

### 3. Tools as trait objects, assembled by feature flags

No central tool enum; tools are `Arc<dyn CoreToolRuntime>` in a `HashMap`, with the per-turn
set planned by `spec_plan.rs`. Read-parallel/write-sequential via a single RwLock.

### 4. Defense in depth for execution

ExecPolicy (Starlark) → hooks → guardian/user → network approval → patch safety → OS sandbox.
Four native sandbox backends (bubblewrap+seccomp, seatbelt, restricted token, MXC) with
policy-driven escalation and retry. Risky runtimes are increasingly pushed out of process
(V8 in `code-mode-host`, execution in `exec-server` environments).

### 5. Dual persistence

JSONL rollout for canonical, append-only history; SQLite (sqlx) for queryable metadata, spawn
graphs, goals, memories, queues, and message boards. The two are reconciled by `LiveThread`.

### 6. Responses-API-native

Codex assumes the OpenAI Responses API (stateful items, reasoning, server-side compaction). It
even has remote compaction endpoints. This is a bet on a single provider's protocol, traded for
Bedrock/Ollama/LM Studio via the same Responses shape.

### 7. Compiled multitool

`arg0` dispatch lets one binary be the CLI, the sandbox helper, and the patch applier. Code mode
is a deliberate exception: it ships as the separate `codex-code-mode-host` binary. Release builds use fat LTO + symbol stripping for size.

### 8. Lint-enforced safety

The workspace denies `unwrap_used`, `expect_used`, `await_holding_lock`, and dozens of clippy
"manual_*" lints — a codebase-wide discipline that Rust makes enforceable at compile time.

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Workspace members | ~155 (was 116 at `b89ce9a2b`) |
| `core` crate source files | ~586 |
| TUI source files / lines | ~1,105 / ~450K |
| Largest files | `protocol.rs` ~6.5K, `session/mod.rs` ~5.3K, `config/mod.rs` ~5.1K, `turn.rs` ~3.2K, `client.rs` ~3.0K |
| Protocol | SQ/EQ (`Op` / `EventMsg` ~84 variants), JSON-RPC-lite app-server |
| Surfaces | TUI, exec, app-server (VS Code/SDKs) |
| Built-in tools | ~25+ (plus MCP, dynamic, extension) |
| Tool exposure modes | Direct, Deferred, DirectModelOnly, Hidden |
| Sandboxes | 4 backends (Linux bwrap+seccomp, macOS seatbelt, Windows restricted token, Windows MXC) |
| Sandbox policy modes | read-only, workspace-write, danger-full-access (+ external) |
| Hook events | 12 (Claude-Code-compatible) |
| Compaction strategies | 2 (local, remote v2) × 3 triggers (pre/mid/manual) |
| Persistence | JSONL rollout + 8 SQLite files (sqlx) |
| Providers | OpenAI, Bedrock (+ runtime), Ollama, LM Studio (Responses API only) |
| Auth modes | 8 `CodexAuth` variants (API key, ChatGPT OAuth, tokens, headers, agent identity, PAT, 2× Bedrock) |
| MCP | client only (stdio/HTTP/in-process); server removed |
| Embedded runtimes | V8 (code mode, out-of-process host), Starlark (execpolicy) |
| Edition / license | Rust 2024 / Apache-2.0 |

---

*Generated from static analysis of the Codex source snapshot under `codex/codex-rs`; revalidated
against `cc7ba3360` on 2026-10-10.*
