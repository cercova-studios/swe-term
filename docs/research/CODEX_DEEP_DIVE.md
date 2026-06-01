# Codex Deep Dive — Architecture & Feature Analysis

> Analysis of the OpenAI Codex source snapshot (`codex/codex-rs`, a 120-crate Cargo
> workspace + a Python SDK under `codex/sdk/python`).
> Runtime: native (compiled). Language: Rust (edition 2024, `unwrap`/`expect` denied
> by lint). UI: ratatui + crossterm (terminal). License: Apache-2.0.

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
10. [Sandboxing (3 Platforms)](#sandboxing-3-platforms)
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
23. [MCP Integration (Client + Server)](#mcp-integration-client--server)
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
| Code-mode runtime | `v8` 147 (embedded JS isolate) |
| Patch grammar | Lark grammar (`apply_patch.lark`) |
| MCP | `rmcp` 1.7 (client + server) |
| Sandbox | landlock/seccomp (Linux), seatbelt (macOS), restricted token/WFP (Windows) |
| Telemetry | OpenTelemetry 0.31 (OTLP), Statsig |
| Auth | OAuth 2.0 (PKCE), API key, Ed25519 agent identity |
| HTTP | reqwest 0.12, WebSocket via tungstenite (forked) |

**Scale:** ~120 workspace crates. The `core` crate alone has ~344 `.rs` files. The TUI crate
has ~326 `.rs` files and ~290K lines (one composer file is ~10K lines). `protocol/src/protocol.rs`
is ~4,900 lines; `core/src/config/mod.rs` is ~3,846; `core/src/session/mod.rs` is ~3,317;
`core/src/session/turn.rs` (the turn loop) is ~2,191.

The defining characteristic: **Codex is a protocol-first system.** Everything is built
around an async submission/event queue (SQ/EQ). The agent core is a library; the TUI,
the headless `exec` mode, the VS Code extension, the SDK, and the MCP server are all
*clients* of that protocol.

---

## Workspace Layout

The 120 crates fall into rough tiers (selected, not exhaustive):

| Tier | Crates |
|------|--------|
| **Protocol** | `protocol`, `app-server-protocol`, `codex-backend-openapi-models` |
| **Core** | `core`, `core-api`, `core-plugins`, `core-skills` |
| **Surfaces** | `tui`, `exec`, `cli`, `app-server`, `app-server-daemon`, `mcp-server` |
| **Transports/clients** | `app-server-transport`, `app-server-client`, `codex-client`, `codex-mcp`, `rmcp-client` |
| **Execution** | `exec-server`, `execpolicy`, `execpolicy-legacy`, `apply-patch`, `shell-command`, `shell-escalation`, `code-mode` |
| **Sandbox** | `sandboxing`, `linux-sandbox`, `windows-sandbox-rs`, `bwrap`, `network-proxy`, `responses-api-proxy` |
| **Providers** | `model-provider`, `model-provider-info`, `models-manager`, `ollama`, `lmstudio`, `chatgpt`, `aws-auth`, `login` |
| **State** | `state`, `thread-store`, `rollout`, `rollout-trace`, `message-history`, `agent-graph-store`, `agent-identity` |
| **Extensions** | `ext/extension-api`, `ext/goal`, `ext/guardian`, `ext/memories`, `ext/image-generation`, `ext/web-search`, `memories/read`, `memories/write` |
| **Cloud** | `cloud-tasks`, `cloud-tasks-client`, `cloud-requirements`, `backend-client` |
| **Utils** | ~25 crates under `utils/` (cache, pty, string, fuzzy-match, image, …) |

This is deliberate: each crate has explicit dependencies declared in the workspace
`Cargo.toml`, so the dependency graph is enforced by the compiler. There is no
"god crate" that everything imports — although `core` comes close.

---

## The SQ/EQ Protocol

**Path:** `protocol/src/protocol.rs` (~4,900 lines), `docs/protocol_v1.md`

The heart of Codex is a **Submission Queue / Event Queue** pattern:

```
//! Uses a SQ (Submission Queue) / EQ (Event Queue) pattern to asynchronously
//! communicate between user and agent.
```

- **SQ (Submission → agent):** `Submission { id, op, client_user_message_id, trace }`
- **EQ (agent → consumer):** `Event { id, msg }`, where `id` correlates with the
  submission that started the current task.

### `Op` — what a client can ask for

The `Op` enum (tagged JSON, `"type": "snake_case"`) is the *entire* command surface of
the agent. Selected variants:

| Variant | Role |
|---------|------|
| `UserInput { items, environments, final_output_json_schema, … }` | Start/continue a turn |
| `Interrupt` | Abort the current task |
| `Compact` | Trigger context compaction |
| `Review { review_request }` | Run a code review turn |
| `ThreadSettings` | Change settings without starting a turn |
| `InterAgentCommunication` | Subagent mailbox message |
| `ExecApproval` / `PatchApproval` | Respond to approval prompts |
| `ResolveElicitation` | Respond to an MCP elicitation |
| `RequestPermissionsResponse` | Grant filesystem/network permissions |
| `ThreadRollback { num_turns }` | Undo N user turns |
| `RunUserShellCommand { command }` | `!cmd` one-off shell |
| `RealtimeConversation*` | Voice/realtime (WebRTC) |
| `RefreshMcpServers` / `ReloadUserConfig` | Hot reload |
| `Shutdown` | Stop the agent |

### `EventMsg` — what the agent emits

`EventMsg` has **dozens** of variants grouped by concern: lifecycle (`TurnStarted`,
`TurnComplete`, `TurnAborted`, `SessionConfigured`, `ContextCompacted`), streaming
(`AgentMessageContentDelta`, `AgentReasoning`, `TokenCount`), tools/exec
(`ExecCommandBegin/OutputDelta/End`, `PatchApplyBegin/End`, `McpToolCallBegin/End`),
approvals (`ExecApprovalRequest`, `ApplyPatchApprovalRequest`, `RequestPermissions`,
`GuardianAssessment`), multi-agent (`CollabAgentSpawnBegin/End`, `CollabWaitingBegin/End`),
hooks (`HookStarted`, `HookCompleted`), and review (`EnteredReviewMode`, `TurnDiff`).

### Serialization & bindings

Every wire type derives `Serialize`/`Deserialize`. Most also derive `schemars::JsonSchema`
and `ts-rs::TS`. TypeScript definitions are generated (`codex app-server generate-ts`),
which is how the Python/TypeScript SDK and the VS Code extension stay in sync with the
Rust core.

### Runtime channels

```rust
// core/src/session/mod.rs
pub struct Codex {
    pub(crate) tx_sub: Sender<Submission>,
    pub(crate) rx_event: Receiver<Event>,
    pub(crate) agent_status: watch::Receiver<AgentStatus>,
    pub(crate) session: Arc<Session>,
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
┌──────────────────────────────────────────────────────────────────────┐
│  Surfaces                                                              │
│  TUI (ratatui)      codex exec       VS Code / SDK       mcp-server     │
│       │                  │                 │                 │          │
│       └── app-server-client (in-process OR remote JSON-RPC) ─┘          │
└────────────────────────────┬───────────────────────────────┼──────────┘
                             │                               │
                      codex-app-server                       │ (direct SQ/EQ)
                      MessageProcessor                        │
                      turn/start → Op::UserInput              │
                      EventMsg → JSON-RPC notification        │
                             │                               │
                      CodexThread.submit()              ThreadManager
                             ▼                               │
                  ┌──────────────────────┐                   │
                  │  codex-core Session  │◄──────────────────┘
                  │  SQ async_channel(512)│
                  │  EQ async_channel(∞) │
                  │  submission_loop()   │
                  └──────────┬───────────┘
                             ▼
                  External MCP servers (rmcp-client)
```

Key insight from the source: **the TUI and `codex exec` do not call `Codex::submit`
directly.** They embed `codex-app-server` via `codex-app-server-client` (in-process by
default, optionally remote) and speak JSON-RPC. Only `codex-core` itself, `mcp-server`,
and `app-server` touch the raw SQ/EQ protocol.

So Codex has **three façades over one core**:
1. The canonical `Submission`/`Event` SQ/EQ (in-core).
2. A JSON-RPC app-server for rich clients (TUI, exec, VS Code, SDK).
3. An MCP server façade for MCP hosts.

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
  `thread/list`, `thread/read`, `thread/compact/start`, `thread/rollback`,
  `thread/inject_items`, `thread/goal/set`, `thread/settings/update`, …
- **Turns:** `turn/start`, `turn/steer`, `turn/interrupt`
- **Review:** `review/start`
- **MCP/config/account:** `mcpServer/tool/call`, `config/read`, `account/login/start`, …

`turn/start` maps directly to `Op::UserInput`; `EventMsg` is mapped to notifications such
as `item/agentMessage/delta` and `item/commandExecution/outputDelta`
(`app-server-protocol/src/protocol/event_mapping.rs`).

---

## Agent Loop: Threads, Sessions, Turns, Tasks

**Path:** `core/src/thread_manager.rs`, `core/src/codex_thread.rs`,
`core/src/session/`, `core/src/tasks/`

The vocabulary matters here:

| Concept | Struct | Role |
|---------|--------|------|
| **Thread manager** | `ThreadManager` (~1,572 ln) | Owns thread store, model refresh; `start_thread`, `resume_thread_from_rollout`, `fork_thread` |
| **Thread façade** | `CodexThread` (~572 ln) | Thin wrapper: `submit(op)`, `steer_input`, `shutdown_and_wait` |
| **Agent API** | `Codex` (in `session/mod.rs`, ~3,317 ln) | The SQ/EQ queue pair; owns `Arc<Session>` |
| **Session** | `Session` (`session/session.rs`, ~1,251 ln) | "At most **1** running task at a time"; holds `state`, `active_turn`, `input_queue` |

### From prompt to turn

1. A client submits `Op::UserInput`.
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

**Path:** `core/src/client.rs` (~2,253 ln), `core/src/client_common.rs`,
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

`WireApi` has effectively **one** variant in `client.stream`: `WireApi::Responses`. There is
no Chat Completions branch in the live client — legacy `chat` and `ollama-chat` are
explicitly rejected by the provider layer. Codex assumes the OpenAI **Responses API**
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

`SessionState` (`state/session.rs`, ~270 ln) holds the live session: the
`ContextManager` history, rate limits, additional-context store, granted permissions,
auto-compact window, startup prewarm handle, active connector selection, and more.

`TurnState` (`state/turn.rs`) is per-turn and lives behind `Arc<Mutex<TurnState>>`. It
tracks pending approvals, pending elicitations, the `pending_input` queue, mailbox
delivery phase, tool-call counts, and token usage at turn start.

`ContextManager` (`context_manager/history.rs`, ~772 ln) is the transcript:
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

`spec_plan.rs` (~979 ln) assembles the per-turn tool set by **feature flags**: it adds tool
sources, appends a tool-search executor when there are deferred tools, prepends code-mode
executors when in code mode, then builds the model-visible specs + registry.

### Built-in tool handlers

| Tool | Purpose |
|------|---------|
| `exec_command` / `write_stdin` | Interactive PTY exec (unified exec) and stdin writes |
| `shell_command` | One-shot shell command |
| `apply_patch` | Freeform patch application (Lark grammar) |
| `view_image` | Attach an image for multimodal context |
| `update_plan` | Update the agent's plan state |
| `request_permissions` | Ask user for session/turn FS/network grants |
| `request_user_input` | Prompt user for structured input |
| `tool_search` | **BM25** search over deferred tool metadata |
| `list_mcp_resources` / `read_mcp_resource` / `…_templates` | MCP resources |
| `create_goal` / `get_goal` / `update_goal` | Goal CRUD (feature-gated) |
| `spawn_agent` / `assign_task` / `send_message` / `wait_agent` / `close_agent` / `list_agents` | Multi-agent v2 |
| `spawn_agents_on_csv` / `report_agent_job_result` | Batch "agent jobs" |
| `exec` / `wait` | Code mode (V8) |
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

Tools that opt in (`supports_parallel_tool_calls() == true` — e.g. `shell_command`,
`view_image`, MCP when the server allows) run concurrently as `tokio::spawn` tasks with
`AbortOnDropHandle`. Everything else serializes. This is the read-parallel/write-sequential
pattern, expressed via a reader-writer lock.

### Orchestration

`orchestrator.rs` runs the approval → sandbox → execute → retry pipeline for the exec-style
tools (shell, apply_patch, unified exec): assess approval requirement → pick the initial
sandbox attempt → request network approval → run → on `SandboxErr::Denied`, optionally ask
the user to retry without sandbox.

---

## Command Execution

**Path:** `core/src/exec.rs` (~1,576 ln), `core/src/unified_exec/`,
`core/src/tools/runtimes/`, `apply-patch/`

Three distinct execution paths:

1. **One-shot subprocess (`exec.rs`).** `process_exec_tool_call` transforms the command
   through the sandbox, then `spawn_child_async` with pipes (not a PTY). Output is byte-capped.
   PTY support here is only used for process-group kills on timeout/cancel.

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

---

## Sandboxing (3 Platforms)

**Path:** `sandboxing/`, `linux-sandbox/`, `windows-sandbox-rs/`

```rust
pub enum SandboxType { None, MacosSeatbelt, LinuxSeccomp, WindowsRestrictedToken }
```

| Platform | Mechanism |
|----------|-----------|
| **Linux** | A separate `codex-linux-sandbox` helper binary is prepended to argv; it runs the command under **bubblewrap** + **seccomp** (legacy Landlock path behind `--use-legacy-landlock`). `linux-sandbox/src/linux_run_main.rs` is ~1,470 lines. |
| **macOS** | `/usr/bin/sandbox-exec` with a compiled `.sbpl` policy (`seatbelt_base_policy.sbpl`, `seatbelt_network_policy.sbpl`). `seatbelt.rs` ~746 lines. |
| **Windows** | Restricted-token / elevated-sandbox backends, ACL-based deny rules, WFP network filtering, ConPTY. `windows-sandbox-rs/src/lib.rs` ~811 lines. |

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

**Path:** `execpolicy/`, `execpolicy-legacy/`, `core/src/exec_policy.rs` (~1,048 ln),
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

The legacy `execpolicy-legacy` crate is an older, execv-oriented matcher (per-program arg
typing for `ls`/`cp`/`sed`/…) retained for compatibility.

---

## Code Mode (V8)

**Path:** `code-mode/`, `core/src/tools/code_mode/`

Code mode is one of the more unusual features. When enabled (`ToolMode::CodeMode` /
`CodeModeOnly`), Codex exposes two tools — `exec` and `wait` — and runs the model's
JavaScript **inside an embedded V8 isolate** (dependency `v8` 147). Nested tools are
projected into the JS global as `global.tools.<normalized_name>(...)`. A
`CodeModeDispatchBroker` routes those nested calls back through the normal tool runtime with
`ToolCallSource::CodeMode`.

In `CodeModeOnly`, the top-level model only sees `exec`/`wait`; all other tools are reachable
*only* from JavaScript. This lets the model orchestrate many tool calls in a single
program rather than one tool call per model round-trip. (Note: this is V8/JavaScript, not
Starlark — Starlark appears only in execpolicy; "Lark" appears only in the apply-patch grammar.)

---

## Context Compaction

**Path:** `core/src/compact.rs`, `compact_remote.rs`, `compact_remote_v2.rs`,
`core/src/tasks/compact.rs`

Three trigger points:

| Phase | Trigger |
|-------|---------|
| **Pre-turn** | `run_pre_sampling_compact` at the start of `run_turn`; also a model-downshift inline compact |
| **Mid-turn** | `run_auto_compact` inside the loop when `token_limit_reached && needs_follow_up` |
| **Manual** | `Op::Compact` → `CompactTask` |

Three implementations, chosen by provider capability:

- **Remote v2** (feature-gated) — `compact_remote_v2`.
- **Remote** — uses `/responses/compact` server-side compaction.
- **Local** — streams a summary via a normal `ModelClientSession` using `SUMMARIZATION_PROMPT`.

`InitialContextInjection` controls re-injection: `DoNotInject` (manual/pre-turn — the next
regular turn reinjects full context) vs `BeforeLastUserMessage` (mid-turn — the summary must
remain the last visible item). `PreCompact`/`PostCompact` hooks fire around the operation.

---

## Config System

**Path:** `config/` crate, `core/src/config/mod.rs` (~3,846 ln)

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
   Written by `RolloutRecorder` (`rollout/src/recorder.rs`, ~1,785 ln). The first line is
   `SessionMeta` (includes `forked_from_id` for lineage). Archived sessions move to
   `archived_sessions/`. A persistence policy decides what is written and truncates exec
   output in extended mode.

2. **SQLite state DBs (queryable metadata)** — via `sqlx` 0.9 with embedded migrations:
   ```rust
   pub(crate) static STATE_MIGRATOR:    Migrator = sqlx::migrate!("./migrations");
   pub(crate) static GOALS_MIGRATOR:    Migrator = sqlx::migrate!("./goals_migrations");
   pub(crate) static MEMORIES_MIGRATOR: Migrator = sqlx::migrate!("./memory_migrations");
   ```
   DB files: `state_5.sqlite` (thread metadata, spawn edges, agent jobs),
   `goals_1.sqlite`, `memories_1.sqlite`, `logs_2.sqlite`. Located in `sqlite_home`
   (`$CODEX_SQLITE_HOME` or `$CODEX_HOME`).

`thread-store`'s `LiveThread` ties them together: apply persistence policy → append JSONL →
patch SQLite metadata. Fork/spawn topology lives in the `0021_thread_spawn_edges` table and is
read via `agent-graph-store`.

---

## Skills & AGENTS.md

**Path:** `core-skills/`, `core/src/skills/`, `core/src/agents_md.rs`

### Skills

Skills are `SKILL.md` files with YAML frontmatter, discovered from many roots:
`$CODEX_HOME/skills` (deprecated), `$HOME/.agents/skills`, bundled `.system` skills,
`/etc/codex/skills`, `<repo>/.codex/skills`, `<repo>/.agents/skills` (walking root→cwd), plus
plugin skill roots. When a user mentions a skill via the `$skill` sigil, the full `SKILL.md`
body is read at call time and injected.

### AGENTS.md

Separate from skills. Codex finds the project root (default marker `.git`), collects
`AGENTS.override.md` / `AGENTS.md` from root→cwd, plus a global one in `$CODEX_HOME`, joins
them with a separator, and folds them into `Config.user_instructions` (capped at 32 KiB).

```rust
pub const DEFAULT_AGENTS_MD_FILENAME: &str = "AGENTS.md";
pub const LOCAL_AGENTS_MD_FILENAME: &str = "AGENTS.override.md";
```

---

## Hooks

**Path:** `hooks/`, `core/src/hook_runtime.rs` (~912 ln), `config/src/hook_config.rs`

Codex implements a **Claude-Code-compatible hook engine** with 10 lifecycle events:

```rust
pub const HOOK_EVENT_NAMES: [&str; 10] = [
    "PreToolUse", "PermissionRequest", "PostToolUse", "PreCompact", "PostCompact",
    "SessionStart", "UserPromptSubmit", "SubagentStart", "SubagentStop", "Stop",
];
```

Handler types:

```rust
pub enum HookHandlerConfig { Command { command, … }, Prompt {}, Agent {} }
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

**Path:** `core/src/tools/handlers/multi_agents_v2/`, `agent-identity/`, `agent-graph-store/`

The v2 tool set: `spawn_agent`, `close_agent`, `assign_task`, `send_message`, `list_agents`,
`wait_agent` (a v1 set still exists under `multi_agents/`). Spawn parses model/reasoning/role
overrides, picks a fork mode (full history vs fresh), emits `CollabAgentSpawnBegin`, and
delegates with **depth limits** (`DEFAULT_AGENT_MAX_DEPTH = 1`,
`DEFAULT_MULTI_AGENT_V2_MAX_CONCURRENT_THREADS_PER_SESSION = 4`).

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

**Path:** `core/src/guardian/`, `ext/guardian/`

Guardian is **not** a sandbox — it is an **LLM that decides whether an `on-request` approval
should be auto-granted instead of shown to the user.**

```rust
pub(crate) fn routes_approval_to_guardian(turn: &TurnContext) -> bool {
    matches!(turn.approval_policy.value(),
             AskForApproval::OnRequest | AskForApproval::Granular(_))
        && turn.config.approvals_reviewer == ApprovalsReviewer::AutoReview
}
```

It builds a compact transcript, spawns a dedicated guardian review subagent (named
`"guardian"`, default model `codex-auto-review`), and expects strict JSON
(`GuardianAssessment { risk_level, user_authorization, outcome, rationale }`). It **fails
closed** on timeout (90s), malformed output, or error, and has a circuit breaker after
`MAX_CONSECUTIVE_GUARDIAN_DENIALS_PER_TURN = 3`. Policy text is injectable via
`guardian_policy_config`.

---

## Memories & Goals

**Path:** `ext/memories/`, `memories/read`, `memories/write`, `ext/goal/`, `core/src/goals.rs`

**Memories** (feature-gated): a read path exposing `memories.{list,read,search,add_ad_hoc_note}`
tools, and a write path that runs a consolidation pipeline at startup (summarizing rollouts
into `raw_memories.md`, `rollout_summaries/`, indexed in `memories_1.sqlite`).

**Goals** (`ext/goal` + `core/src/goals.rs`, ~1,854 ln): `create_goal`/`get_goal`/`update_goal`
with an explicit **token budget**, budget-limit steering prompts, and OTEL metrics. Goals were
migrated out of the main state DB into a dedicated `goals_1.sqlite`.

---

## Provider & Auth Layer

**Path:** `model-provider-info/`, `model-provider/`, `models-manager/`, `ollama/`,
`lmstudio/`, `login/`

Built-in providers:

```rust
[ ("openai", …), ("amazon-bedrock", …), ("ollama", …), ("lmstudio", …) ]
```

All use `WireApi::Responses`; legacy chat APIs are rejected. User providers can be added via
`[model_providers.<id>]`. `ModelsManager` caches the model catalog at
`$CODEX_HOME/models_cache.json` and merges remote catalogs with local `models.json` presets.

**Auth (`login/`)** — `CodexAuth` variants:
- `ApiKey` (`OPENAI_API_KEY` / `CODEX_API_KEY`)
- `Chatgpt` — OAuth tokens in `$CODEX_HOME/auth.json` or the OS keyring (PKCE, localhost
  callback server on port 1455, issuer `https://auth.openai.com`)
- `ChatgptAuthTokens` — token-only
- `AgentIdentity` — the agent-task JWT flow

Local OSS: **Ollama** (requires ≥ 0.13.4 for Responses API, default `gpt-oss:20b`) and **LM
Studio** (default `openai/gpt-oss-20b`), selected by `oss_provider`/`--local-provider`.

---

## MCP Integration (Client + Server)

**Path:** `mcp-server/`, `codex-mcp/`, `rmcp-client/`

Codex is **both** an MCP server and an MCP client.

**As a server** (`mcp-server/`): `codex mcp-server` exposes Codex itself as an MCP tool over
**stdio** (newline-delimited JSON-RPC via `rmcp`). It runs a `ThreadManager` with
`SessionSource::Mcp` and talks to core directly via SQ/EQ, reusing the MCP request id as the
submission id. Approvals use MCP **elicitation**.

**As a client** (`codex-mcp` + `rmcp-client`): connects to external MCP servers. Transports:

```rust
enum PendingTransport {
    InProcess { transport: DuplexStream },
    Stdio { transport: StdioServerTransport },
    StreamableHttp { transport: StreamableHttpClientTransport<…> },
    StreamableHttpWithOAuth { transport: …, oauth_persistor: OAuthPersistor },
}
```

The connection manager emits `McpStartupUpdate/Complete` and turns each remote MCP tool into a
local `McpHandler` in the tool registry.

---

## Terminal UI (ratatui)

**Path:** `tui/` (~326 `.rs` files, ~290K lines)

The TUI is a large, conventional ratatui + crossterm application with a custom terminal
backend for history insertion and hyperlinks. Modules: `app/` (run loop + event dispatch),
`chatwidget/` (the main chat surface and tool lifecycle), `bottom_pane/` (composer, approvals,
popups, slash commands — `chat_composer.rs` alone is ~10K lines), `history_cell/` (transcript
cell types), `render/`, `markdown_render.rs`, plus `status/`, `onboarding/`, `resume_picker/`.

Crucially, the TUI does **not** consume raw `EventMsg`. It embeds the app-server via
`AppServerClient` (in-process or remote) and reacts to typed `ServerNotification`s, which it
translates into internal `AppEvent`s and feeds to `ChatWidget::handle_server_notification`.

---

## CLI & arg0 Multitool

**Path:** `cli/` (~15K lines), `arg0/`

The CLI is a **multitool** that dispatches on `argv[0]` via `codex_arg0::arg0_dispatch_or_else`:
the same binary can re-exec itself as `apply_patch`, `codex-linux-sandbox`,
`codex-execve-wrapper`, etc., prepending temp PATH aliases so child processes can find the
helpers.

Subcommands include: `exec` (headless), `review`, `login`/`logout`, `mcp` (manage MCP servers),
`mcp-server` (run as MCP server), `app-server`, `remote-control`, `app` (desktop), `apply`,
`resume`/`fork`/`archive`, `cloud`, `doctor`, `sandbox`, `debug`, `execpolicy`, `features`,
`completion`, `update`. No subcommand → the interactive TUI.

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

One canonical SQ/EQ protocol in `core`; a JSON-RPC app-server façade for rich clients; an MCP
façade for MCP hosts. Surfaces never reach into core internals — they speak a wire protocol,
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
Three native sandboxes (landlock/seccomp, seatbelt, restricted token) with policy-driven
escalation and retry.

### 5. Dual persistence

JSONL rollout for canonical, append-only history; SQLite (sqlx) for queryable metadata, spawn
graphs, goals, and memories. The two are reconciled by `LiveThread`.

### 6. Responses-API-native

Codex assumes the OpenAI Responses API (stateful items, reasoning, server-side compaction). It
even has remote compaction endpoints. This is a bet on a single provider's protocol, traded for
Bedrock/Ollama/LM Studio via the same Responses shape.

### 7. Compiled multitool

`arg0` dispatch lets one binary be the CLI, the sandbox helper, the patch applier, and the MCP
server. Release builds use fat LTO + symbol stripping for size.

### 8. Lint-enforced safety

The workspace denies `unwrap_used`, `expect_used`, `await_holding_lock`, and dozens of clippy
"manual_*" lints — a codebase-wide discipline that Rust makes enforceable at compile time.

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Workspace crates | ~120 |
| `core` crate source files | ~344 |
| TUI source files / lines | ~326 / ~290K |
| Largest files | `protocol.rs` ~4.9K, `config/mod.rs` ~3.8K, `session/mod.rs` ~3.3K, `turn.rs` ~2.2K, `client.rs` ~2.3K |
| Protocol | SQ/EQ (`Op` / `EventMsg`), JSON-RPC-lite app-server |
| Surfaces | TUI, exec, app-server (VS Code/SDK), mcp-server |
| Built-in tools | ~25+ (plus MCP, dynamic, extension) |
| Tool exposure modes | Direct, Deferred, DirectModelOnly, Hidden |
| Sandboxes | 3 (Linux landlock/seccomp, macOS seatbelt, Windows restricted token) |
| Sandbox policy modes | read-only, workspace-write, danger-full-access (+ external) |
| Hook events | 10 (Claude-Code-compatible) |
| Compaction strategies | 3 (local, remote, remote v2) × 3 triggers (pre/mid/manual) |
| Persistence | JSONL rollout + 4 SQLite DBs (sqlx) |
| Providers | OpenAI, Bedrock, Ollama, LM Studio (Responses API only) |
| Auth modes | API key, ChatGPT OAuth, token-only, agent identity (Ed25519/JWT) |
| MCP | client (stdio/HTTP/in-process) + server (stdio) |
| Embedded runtimes | V8 (code mode), Starlark (execpolicy) |
| Edition / license | Rust 2024 / Apache-2.0 |

---

*Generated from static analysis of the Codex source snapshot under `codex/codex-rs` (2026).*
