# Harness: A Go Agent Framework

> *Inspired by [pi-mono](https://github.com/badlogic/pi-mono). Distilled. Faster. Simpler.*

---

## Philosophy

The cheapest, fastest, and most reliable component is the one that does not exist.

**Harness** is a minimal, composable agent framework in Go. It takes the layered modularity of pi-mono — `pi-ai → pi-agent-core → pi-coding-agent → pi-tui` — and reframes it as a single, zero-dependency core that everything else plugs into.

Where pi-mono is a TypeScript monorepo of npm packages with EventEmitters, dynamic extension loading, and runtime type checking, Harness is:

- **A Go multi-module workspace** — each module compiles independently, imports are explicit
- **Interface-driven** — plugin contracts are Go interfaces, not registration APIs
- **Goroutine-native** — no async/await, no callback chains, just goroutines and channels
- **Single binary** — no runtime, no node_modules, no interpreter
- **Zero-dependency core** — the harness itself uses only Go stdlib

The CLI tools (X search, web→markdown, PDF parsing, codebase research) and AST services (tree-sitter, zoekt, ast-grep, semgrep) described in the project docs are not part of the core. They are **extensions** — first-class citizens that demonstrate the harness's power through composition.

### What pi-mono Gets Right

| Principle | pi-mono | Harness |
|-----------|---------|---------|
| Layered modularity | npm packages with clear dependencies | Go modules with `go.work` |
| Unified LLM abstraction | `streamFn()` + `ModelRegistry` | `Provider` interface + `Registry` |
| Extension system | `pi.registerTool()`, `pi.on()` | Go `Tool`/`Hook` interfaces + thin subprocess adapters for native services |
| External services | Wrapped inside TS extensions via `exec`/`spawn` | Dedicated `cli/*` adapters calling `extensions/*` binaries (e.g. `swe_distiller`) |
| Event streaming | EventEmitter with typed events | Typed channels with `context.Context` |
| Message queuing | `steeringQueue` + `followUpQueue` | Channel-based priority queue |
| Tool execution | parallel/sequential + before/after hooks | Same model, goroutine-native |
| State management | `AgentState` with mutators | Immutable snapshots + state machine |

### What Claude Code Proves Works (Credit: Anthropic)

Analysis of Claude Code's source (~512K LoC, ~1900 files) reveals production-hardened patterns that Harness should adopt and improve upon:

| Pattern | Claude Code Implementation | Harness Enhancement |
|---------|---------------------------|---------------------|
| **Permission system** | Multi-layer: bash classifier → rule matching → mode dispatch → UI prompt. Battle-tested across thousands of edge cases. | Promote to core `Approver` interface. Decouple from UI. Make rules first-class data, not code. |
| **Tool concurrency partitioning** | `toolOrchestration.ts` classifies tools as read-only (parallel) or write (serial). Context modifiers queue and apply in order. | Native via goroutines + `errgroup`. Tool interface gains `ReadOnly() bool`. |
| **Startup prefetch** | MDM settings, keychain reads, API preconnect fire as side-effects before heavy module evaluation. Cuts ~200ms. | Parallel prefetch in `main.go` before Bubble Tea mounts. Config, API preconnect, cache warming. |
| **Multi-strategy compaction** | 4 strategies: auto-compact, micro-compact, API-side compact, session-memory-compact. Extracts memories before truncating. | `Compactor` interface in core with pluggable strategies. Memory extraction is a core event hook. |
| **Structured diff rendering** | Unified diffs with syntax highlighting in terminal. | Port the UX pattern to Bubble Tea via lipgloss. |
| **Session persistence** | Session save/restore, cross-project resume, session discovery. | Core `Session` type with atomic persistence. Extensions add resume UI. |

### What claw-code Proves Works (Credit: Rust port in this repo)

[claw-code](../claw-code/) is a Rust reimplementation of a Claude Code–class agent (~40 tools, mock parity harness, typed policy). It is smaller and more explicit than upstream TypeScript — useful as a **second reference** beside pi-mono and Anthropic. Adopt the patterns below; do not port the full lane/team/PR orchestration surface unless Harness explicitly targets multi-lane SWE workflows.

| Pattern | claw-code | Harness adoption |
|---------|-----------|------------------|
| **Mock parity harness** | `mock-anthropic-service` + scripted scenarios in `mock_parity_harness.rs` + `mock_parity_scenarios.json` | Phase 1: `test/mockprovider` + `test/integration/scenarios.json`; CI runs clean-env E2E before TUI |
| **Honest capability matrix** | `PARITY.md` labels stub vs real per tool | `docs/HARNESS_PARITY.md` (or section in plan): what's implemented, what's delegated to `extensions/*` |
| **Lean agent binary** | `claw-analog`: FS tools only, no bash/MCP/plugins, NDJSON for automation | `cmd/harness-headless`: read/grep/write (mode-gated), stdout JSON — CI and agent-to-agent |
| **RAG / heavy retrieval sidecar** | `claw-rag-service` HTTP + SQLite; agent calls `retrieve_context` | Same as `swe_distiller`: sidecar or `extensions/*` binary, never embed indexing in core |
| **Permission modes** | `PermissionMode`: read-only, workspace-write, danger-full-access, prompt | Extend `Approver` with `PermissionProfile` on session; tools declare `RequiredMode()` |
| **Typed approval tokens** | `approval_tokens.rs`: scope, expiry, consumed/revoked, delegation hops | `ApprovalGrant` struct for one-shot exceptions (e.g. allow `distill_url` to internal docs host) |
| **Structured events > TUI scrape** | `lane_events.rs`: typed names, status, fingerprints, dedupe | Event bus payloads include `schema`, `seq`, fingerprint; TUI is never source of truth |
| **Versioned machine reports** | `report_schema.rs`: claims, confidence, projections, content hash | Optional `HarnessReport` for audit/compact artifacts; ACP-aligned JSON |
| **Deferred tool discovery** | `ToolSearch` when tool pool is large | `ToolRegistry.Search(query)` + dynamic schema injection for one turn |
| **Executable policy engine** | `policy_engine.rs`: conditions + prioritized actions | Keep agent loop simple; add `Policy` interface for automation hooks (retry, escalate) later |
| **Workspace trust gate** | `trust_resolver.rs`: allowlist, auto-trust, typed `TrustEvent` | Run once at session start before loop; emit `TrustResolved` on bus |
| **Bash safety decomposition** | Nine validation submodules (path, sed, sandbox decision, …) | `tools/terminal/safety/*` as separate files, each testable |
| **Task packet** | `task_packet.rs`: objective, scope, acceptance criteria, permission profile | Optional `Session.Task` for spawned child agents / SWE-bench runs |
| **In-crate vs subprocess tools** | `pdf_extract.rs` in `tools/` (small, pure Rust) | In Go only when small; else `extensions/*` + `cli/*` adapter (see swe_distiller) |

**Explicitly defer from claw-code** (complexity without Harness v1 goals): 40-tool surface parity, team/cron registries, lane board JSON, plugin marketplace install, 100+ slash commands, green-contract merge automation.

### What Flue Proves Works (Credit: Astro harness on pi-mono)

[flue](../flue/) (gitignored local copy; upstream [withastro/flue](https://github.com/withastro/flue)) is **“the agent harness framework”** built on `@earendil-works/pi-agent-core` / `pi-ai`. It is the closest product-shaped reference to what Go Harness aims to be: headless, deployable, sandbox-aware — not another LLM SDK and not a terminal IDE agent.

```text
pi-mono     →  agent loop, tools, extensions (library)
Flue        →  pi-mono + HTTP deploy + sandbox tiers + session API (product, TS)
claw-code   →  Claude Code parity + policy/events (Rust, terminal agent)
Harness     →  Go core + Bubble Tea + extensions/* (planned)
```

| Pattern | Flue | Harness adoption |
|---------|------|------------------|
| **Headless-first** | No baked-in TUI; `flue dev` / `flue run` / HTTP webhooks | `Frontend` is optional; `cmd/harness` (TUI) vs `cmd/harness-headless` (JSON) vs `cmd/harness-server` (HTTP) |
| **Instance / harness / session** | `POST /agents/<name>/<id>`; `init()` → harness; `harness.session()` | `InstanceID` (durable scope) → named `Harness` config → `Session` threads; don't conflate workspace with chat |
| **Sandbox tiers** | Default virtual (`just-bash`); `local()` for CI; connectors for Daytona/containers | `Sandbox` interface: `Virtual`, `Local`, `Subprocess`, `Remote`; default cheap tier |
| **Markdown-first logic** | `AGENTS.md`, `.agents/skills/`, `roles/*.md`; thin TS handlers | `prompts.yaml`, skills, `AGENTS.md` discovery; Go stays wiring-only |
| **Roles as overlays** | `call > session > harness`; not persisted in user history | `Role` overlays on `Prompt()` / `Task()`; system prompt only for that turn |
| **Child work: `task()`** | Shared sandbox, separate history, `cwd` + role, `MAX_TASK_DEPTH` | `SpawnChild` / `session.Task()` with depth limit + shared workspace FS |
| **Structured results** | `finish` / `give_up` tools + Valibot schema on `prompt()` | `PromptOption.ResultSchema`; inject validator tools; retry if model answers in prose |
| **Compaction** | Model-aware `deriveCompactionDefaults`; threshold + overflow→compact→retry | `Compactor` + `TokenBudget`; port overflow retry from `compaction.ts` |
| **Built-in truncation** | Read 2000 lines / 50KB in `agent.ts` | `cli/runner` + tool descriptions document limits |
| **MCP in trusted code** | `connectMcpServer()` in handler; tools passed to `init()`; no stdio auto-spawn | MCP wiring in `main.go` or agent bootstrap only; secrets in config/env |
| **Provider gateway** | `configureProvider()` in `app.ts` per request | Layered `Config` + per-provider `baseURL` / headers; no global registry mutation |
| **Env allowlist (local)** | `local({ env: { GH_TOKEN } })` — not full `os.Environ` | `Sandbox.Local` inherits PATH/HOME/locale only; explicit extra vars |
| **Sidecar knowledge** | Hydrate R2 → CF workspace once; not live bucket mount | Same hydrate-once pattern for RAG/index blobs into workspace |
| **Run observability** | `runId`, run registry, OpenAPI error envelopes | `RunID` on events; optional `GET /runs/:id/events` in `harness-server` |
| **Connectors UX** | `flue add daytona \| claude` → markdown recipe → `.flue/connectors/*.ts` | `docs/connectors/*.md` recipes for `extensions/*` + `cli/*` adapters (docs-only v1) |

**Explicitly defer from Flue** (different language/runtime goals): depending on pi-agent-core at runtime, Cloudflare Durable Objects / wrangler build pipeline, Astro-style multi-target bundler, connector marketplace hosted at flueframework.com — adopt **patterns**, not the TS stack.

### What Deep Agents Proves Works (Credit: LangChain harness on LangGraph)

[Deep Agents](https://github.com/langchain-ai/deepagents) is LangChain's "harness as composition": it writes **no agent loop**, assembling ~11 `AgentMiddleware` over `create_agent`/LangGraph. It's the opposite of Harness at the *substrate* level (three-framework Python stack, no single binary, asyncio not goroutines, LangSmith-coupled), but the **best organizational reference** in the survey — and its ACP and Harbor adapters are the patterns to copy for protocol + evals. Adopt the *organization*; do not inherit the stack.

| Pattern | Deep Agents | Harness adoption |
|---------|-------------|------------------|
| **Harness = composition, not a forked loop** | ~11 middleware (todos, FS, subagents, summarize, skills, HITL) handed to a generic loop | Validates our model: capabilities are ordered `Hook`/`Analyzer`/`Tool` interfaces; add a capability ≠ fork the loop |
| **One backend protocol, many impls** | `BackendProtocol` (state/disk/store/composite/remote); agent sees only tools | `Sandbox`/`Filesystem` interface; add **path-routed composite** (`/memories/` → store, rest → sandbox) |
| **Filesystem as context-overflow substrate** | Summaries → `/conversation_history/{thread}.md`; tool results >20K tok → `/large_tool_results/<id>`, leave a handle | Offload to the **content-addressable cache**, pass a `sha256` handle not bytes — cheaper *and* more debuggable than in-memory truncation |
| **Any compiled agent is a valid sub-agent** | `CompiledSubAgent` — delegation and composition share one seam | `SubAgent` interface accepts any `Harness` instance; `SpawnChild` already isolates context |
| **Two-tier eval scoring** | `.success()` hard-fails correctness; `.expect()` logs efficiency, never fails CI | Steal verbatim: correctness gates merges, trajectory efficiency is observed not gated |
| **ACP / Harbor as thin external adapters** | `AgentServerACP(acp.Agent)`; `DeepAgentsWrapper(BaseAgent)` + `HarborSandbox` — protocol/eval frameworks stay *out* of core | `frontend/acp` and `eval/harbor` adapters implement the *other* system's interface; core stays clean |
| **Model switch keeps session state** | `set_config_option` rebuilds the graph, reuses checkpointer + `thread_id` | Rebuild provider/config, keep `SessionStore` + thread — swap model mid-conversation |
| **ACP completeness gap (anti-lesson)** | Brings its *own* backend; skips client `fs/*`+`terminal/*`, full tool-status, reasoning stream, `session/load` | Do the half they skipped (below) — under ACP the **editor is the environment** |
| **"Trust the LLM" security (anti-lesson)** | Default `LocalShellBackend` runs on host with no isolation | Confirms our stance: layered `Approver` in core, on by default, *before* any sandbox |

**Explicitly defer from Deep Agents:** the three-framework dependency stack (LangGraph→create_agent→deepagents), asyncio/GIL concurrency, LangSmith-required observability and evals, Python-env deployment.

### Where Harness Improves

1. **No GC pressure** — Go's GC is simpler than V8's; goroutines are lighter than Promises
2. **Compile-time contracts** — interfaces checked at compile time, not runtime `instanceof`
3. **Natural concurrency** — `go func()` + channels vs `async/await` + `Promise.all`
4. **Startup in milliseconds** — no JIT warmup, no module resolution
5. **Content-addressable caching** — built into the core, not bolted on
6. **Single binary deployment** — `GOOS=linux go build` and ship

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTENDS                                 │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │  TUI     │  │  Web UI  │  │  RPC     │  │  Print/Pipe  │    │
│  │ (Bubble  │  │ (future) │  │  Server  │  │  (stdout)    │    │
│  │  Tea)    │  │          │  │          │  │              │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘    │
│       │              │              │               │            │
│       └──────────────┴──────┬───────┴───────────────┘            │
│                             │                                    │
│                      Frontend interface                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                     HARNESS CORE                                 │
│                     (zero external deps)                         │
│                                                                  │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌─────────────┐  │
│  │ Agent     │  │ Event     │  │ State     │  │ Transport   │  │
│  │ Loop      │  │ Bus       │  │ Machine   │  │ Layer       │  │
│  │           │  │           │  │           │  │             │  │
│  │ - turns   │  │ - typed   │  │ - session │  │ - stdio     │  │
│  │ - tools   │  │ - fan-out │  │ - history │  │ - jsonrpc   │  │
│  │ - steer   │  │ - context │  │ - persist │  │ - framing   │  │
│  │ - follow  │  │ - backpr. │  │ - snaps   │  │             │  │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └──────┬──────┘  │
│        │              │              │               │          │
│        └──────────────┴──────┬───────┴───────────────┘          │
│                              │                                   │
│                     Protocol Types (ACP)                         │
│                     - messages, content blocks                   │
│                     - tool calls, capabilities                   │
│                     - discriminated unions via iota               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                     Provider / Tool / Analyzer
                          interfaces
                               │
┌──────────────────────────────┴──────────────────────────────────┐
│                       EXTENSIONS                                 │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LLM Providers                                          │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │    │
│  │  │ OpenAI   │ │Anthropic │ │ Ollama   │ │  vLLM     │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └───────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Tools                                                   │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │    │
│  │  │ File I/O │ │ Terminal │ │ Browser  │ │ Approval  │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └───────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  CLI Plugins                                             │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │    │
│  │  │ X Search │ │ Web→MD   │ │ PDF Parse│ │ Code Resch│  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └───────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Analyzers (AST / Code Intelligence)                     │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │    │
│  │  │TreeSitter│ │ Zoekt    │ │ ast-grep │ │ Semgrep   │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └───────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Native Services (repo `extensions/`, non-Go binaries)    │    │
│  │  ┌──────────────┐  ┌──────────┐  ┌──────────┐           │    │
│  │  │ swe_distiller│  │ (future) │  │ swe-token│           │    │
│  │  │   (Rust)     │  │          │  │  (Rust)  │           │    │
│  │  └──────┬───────┘  └──────────┘  └──────────┘           │    │
│  │         │ subprocess / JSON stdout                       │    │
│  │         ▼                                                │    │
│  │  Go adapters in `harness/cli/*` implement `Tool`         │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### The Inversion

pi-mono builds up: `pi-ai` is a library, `pi-agent-core` wraps it, `pi-coding-agent` wraps that, `pi-tui` renders it. **Flue** productizes that stack for headless HTTP/CI deploy without replacing the loop.

Harness inverts: the **core is the orchestrator** (in Go, not pi-mono). Providers, tools, frontends, sandboxes, and analyzers all plug into it. Nothing wraps the core — the core calls out to extensions via interfaces. Flue is the behavioral spec for **product** concerns (instance scope, sandbox tiers, structured `prompt` results) that pi-mono leaves to the integrator.

This means:
- Swap `OpenAI` for `Ollama` by changing one line
- Replace the TUI with a web frontend without touching agent logic
- Add a new tool by implementing a 3-method interface
- Integrate an AST analyzer as a specialized tool
- Wrap a Rust/Python service (like `swe_distiller`) with a thin Go `Tool` adapter — same role pi-mono extensions play for `rg` or custom CLIs, without dynamic `import()`
- Expose hosted agents via `POST /agents/{name}/{instanceID}` with stable instance scope (Flue) while keeping the interactive path on Bubble Tea
- Back each infrastructure concern (compute, retrieval, blobs, KV) with the *best* provider behind a narrow port — Modal for sandboxes, turbopuffer for retrieval, S3/GCS/R2 for blobs — none of them load-bearing, each swappable for a self-host adapter
- Let **agents extend the framework** without writing Go — author a Markdown skill, a sandboxed Monty program, or a Starlark tool def at runtime; promote to a compiled `Tool` only when proven (see *Metaengineering: A Framework Agents Can Extend*)

---

## Core Interfaces

These are the contracts. The entire system is defined by these five interfaces plus the event types.

```go
package harness

// Provider streams LLM completions.
// Equivalent to pi-mono's streamFn() — but as an interface, not a function.
type Provider interface {
    // Stream sends messages to an LLM and returns a channel of events.
    // The channel closes when the response is complete.
    // Errors are delivered as events, never as Go errors (pi-mono contract).
    Stream(ctx context.Context, req *StreamRequest) (<-chan StreamEvent, error)

    // Models returns available models from this provider.
    Models(ctx context.Context) ([]Model, error)
}

// Tool executes actions in the environment.
// Equivalent to pi-mono's custom tool registration.
//
// ReadOnly signals whether a tool only observes (read files, search, glob)
// or mutates (write files, run commands). The agent loop uses this to
// partition concurrent execution — read-only tools run in parallel,
// mutating tools serialize. Pattern proven by Claude Code's
// toolOrchestration.ts, adapted for goroutines.
type Tool interface {
    Name() string
    Description() string
    Schema() json.RawMessage        // JSON Schema for arguments
    ReadOnly() bool                 // true = safe for concurrent execution
    Execute(ctx context.Context, args json.RawMessage) (*ToolResult, error)
}

// Approver decides whether a tool invocation should proceed.
//
// This is the safety boundary — the most important interface in the system.
// Claude Code's permission system (bashClassifier, dangerousPatterns,
// pathValidation, rule-based allow/deny) is its most battle-tested
// component. Harness promotes it to a core interface so it cannot be
// bypassed by extensions.
//
// Implementations:
//   - InteractiveApprover: prompt user via Frontend (default)
//   - AutoApprover: classify command safety automatically
//   - AlwaysApprover: bypass for trusted environments (CI, sandboxes)
//   - RuleApprover: glob-based allow/deny rules
//   - ChainApprover: compose multiple approvers (rules first, then interactive)
type Approver interface {
    // Approve returns an ApprovalResult for a pending tool invocation.
    // The Approver sees the tool name and arguments but NOT the full
    // application state — this is the key difference from Claude Code's
    // ToolUseContext which carries ~60 fields of product state.
    Approve(ctx context.Context, req *ApprovalRequest) (*ApprovalResult, error)
}

// ApprovalRequest is what the Approver sees.
type ApprovalRequest struct {
    ToolName    string
    Args        json.RawMessage
    SessionID   string
    WorkDir     string              // workspace root
    ReadOnly    bool                // from Tool.ReadOnly()
}

// ApprovalResult is what the Approver returns.
type ApprovalResult struct {
    Decision    Decision            // Allow, Deny, Ask
    Reason      string              // human-readable explanation
    Rule        string              // which rule matched (for audit)
}

type Decision int
const (
    Allow Decision = iota
    Deny
    Ask                             // escalate to Frontend for user input
)

// Frontend renders agent state and captures user input.
// pi-mono couples TUI into the agent; Harness decouples it.
type Frontend interface {
    // Run starts the frontend event loop. Blocks until ctx is cancelled.
    Run(ctx context.Context, bus *EventBus, state *StateReader) error
}

// Analyzer provides code intelligence over a set of files.
// This is the interface for AST services, zoekt, ast-grep, semgrep.
type Analyzer interface {
    Name() string
    Analyze(ctx context.Context, files []FileRef) (*Analysis, error)
    Query(ctx context.Context, q *AnalysisQuery) (*AnalysisResult, error)
}

// Sandbox is where tool side-effects run. Flue defaults to a cheap virtual tier;
// opt into Local or Remote only when the workload needs real Linux or host CI tools.
type Sandbox interface {
    ReadFile(ctx context.Context, path string) ([]byte, error)
    WriteFile(ctx context.Context, path string, data []byte) error
    Exec(ctx context.Context, argv []string, opts ExecOpts) (ExecResult, error)
}

// SessionStore persists conversation state per (instanceID, harnessName, sessionName).
// Flue: reuse URL <id> for instance; harness.session("thread") for parallel threads.
type SessionStore interface {
    Load(ctx context.Context, key SessionKey) (*SessionData, error)
    Save(ctx context.Context, key SessionKey, data *SessionData) error
}
```

**Structured prompt results (Flue `finish` / `give_up`):** When `PromptOpts` includes a JSON Schema, the loop registers ephemeral `finish` and `give_up` tools and retries if the model returns plain text instead of calling `finish`. This is how webhook/CI agents return typed data without fragile “respond with JSON” instructions.

**Why interfaces, not registration functions:**

pi-mono requires `pi.registerTool({ name, schema, execute })` — runtime registration, dynamic dispatch, no compile-time checks. In Go, you implement the `Tool` interface and pass it to the harness. The compiler verifies the contract. No ceremony.

**Why Approver is in core, not in extensions:**

Claude Code's permission system lives in `utils/permissions/` and `hooks/toolPermission/` — deep in the utility layer, not in the architectural surface. This means extensions CAN bypass it by calling tool execution directly. In Harness, the agent loop calls `Approver.Approve()` before every `Tool.Execute()`. It is not optional. It cannot be circumvented. The safety boundary is structural, not conventional.

---

## Module Layout

```
harness/
├── go.work                          # Go workspace (links all modules)
│
├── core/                            # THE HARNESS — zero external deps
│   ├── go.mod                       # module github.com/user/harness/core
│   ├── harness.go                   # Harness struct, wire everything
│   ├── interfaces.go                # Provider, Tool, Approver, Frontend, Analyzer
│   ├── agent/
│   │   ├── loop.go                  # Agent loop (turn → stream → approve → tools → repeat)
│   │   ├── state.go                 # AgentState, immutable snapshots
│   │   ├── queue.go                 # Steering + follow-up message queues
│   │   ├── instance.go              # InstanceID scope (Flue URL <id> — sandbox + harness grouping)
│   │   ├── session.go               # Session lifecycle, persistence, named threads
│   │   ├── budget.go                # Token budget tracking, compaction triggers
│   │   ├── compact.go               # Compaction + overflow retry (Flue compaction.ts)
│   │   ├── spawn.go                 # Child tasks (Flue session.task — depth-limited)
│   │   └── result.go                # Structured prompt results (finish / give_up tools)
│   ├── sandbox/
│   │   ├── sandbox.go               # Sandbox interface (virtual / local / remote)
│   │   ├── virtual.go               # Restricted in-process exec (Flue just-bash analogue)
│   │   └── local.go                 # Host FS + shell, env allowlist (Flue local())
│   ├── platform/
│   │   ├── platform.go              # Narrow ports: BlobStore, KVStore, Secrets, Queue, Lock
│   │   ├── runtime.go               # Runtime capability descriptor (FS? subprocess? ephemeral GB?)
│   │   └── retrieve.go              # Retriever port (vector + BM25 + metadata filter)
│   ├── archive/
│   │   ├── catalog.go               # ArchiveCatalog — list entries w/o full extraction
│   │   ├── reader.go                # ArchiveReader — JIT stream a single path from zip/tar.gz
│   │   └── search.go                # ArchiveSearch — grep over streamed entries (archaeology)
│   ├── bus/
│   │   ├── bus.go                   # Typed event bus (channels, fan-out)
│   │   ├── events.go                # All event types (agent, turn, message, tool)
│   │   └── subscriber.go            # Subscription with filters
│   ├── transport/
│   │   ├── transport.go             # Transport interface
│   │   ├── stdio.go                 # stdin/stdout + newline-delimited JSON
│   │   └── framing.go               # Length-prefixed and newline framing
│   ├── protocol/
│   │   ├── jsonrpc.go               # JSON-RPC 2.0 request/response/notification
│   │   ├── acp.go                   # ACP types (ContentBlock, ToolCall, Session)
│   │   ├── codec.go                 # Marshal/unmarshal with discriminated unions
│   │   └── capabilities.go          # Capability negotiation
│   ├── cache/
│   │   ├── cache.go                 # Content-addressable cache interface
│   │   ├── memory.go                # In-process LRU (L1)
│   │   └── disk.go                  # On-disk with hash keys (L2)
│   └── process/
│       ├── spawn.go                 # Spawn agent subprocess
│       ├── lifecycle.go             # Init → session → shutdown
│       └── signal.go                # SIGTERM/SIGINT handling
│
├── provider/                        # LLM provider extensions
│   ├── go.mod                       # module github.com/user/harness/provider
│   ├── openai/
│   │   └── openai.go                # implements harness.Provider
│   ├── anthropic/
│   │   └── anthropic.go             # implements harness.Provider
│   ├── ollama/
│   │   └── ollama.go                # implements harness.Provider
│   └── registry.go                  # ModelRegistry (like pi-mono's ModelRegistry)
│
├── selfext/                         # Self-extension tiers — the agent-facing extension surface
│   ├── go.mod                       # module github.com/user/harness/selfext
│   ├── skills/
│   │   └── loader.go                # Rung 0: Markdown sub-programs, read at call time (Flue-style)
│   ├── starlark/
│   │   ├── loader.go                # Rung 2: load .star tool defs, dynamic register
│   │   ├── runtime.go               # Sandboxed Starlark execution environment
│   │   └── bridge.go                # Expose curated Go fns to Starlark (file I/O, HTTP)
│   └── monty/
│       ├── executor.go              # Rung 1: spawn monty-cli, run throwaway Python
│       ├── bridge.go                # External functions the Python code may call
│       └── sandbox.go               # Resource limits, timeout, isolation
│
├── tools/                           # Built-in tool extensions (Rung 3: compiled Go)
│   ├── go.mod                       # module github.com/user/harness/tools
│   ├── filesystem/
│   │   └── fs.go                    # Read, write, list, search — implements Tool
│   ├── terminal/
│   │   ├── exec.go                  # Command execution — implements Tool
│   │   └── safety.go                # Command classification, dangerous patterns,
│   │                                # path validation (adapted from Claude Code's
│   │                                # bashSecurity.ts, sedValidation.ts, pathValidation.ts)
│   ├── browser/
│   │   └── browse.go                # Web fetch, screenshot — implements Tool
│   ├── skill/
│   │   └── skill.go                 # Tool that invokes a selfext/skills Markdown sub-program
│   └── approval/                    # Approver implementations
│       ├── rules.go                 # Glob-based allow/deny rules — implements Approver
│       ├── auto.go                  # Auto-classify command safety — implements Approver
│       ├── interactive.go           # Prompt user via Frontend — implements Approver
│       └── chain.go                 # Compose approvers (rules → auto → interactive)
│
├── tui/                             # Terminal UI frontend
│   ├── go.mod                       # module github.com/user/harness/tui
│   ├── app.go                       # Bubble Tea model — implements Frontend
│   ├── update.go                    # Elm-architecture update loop
│   ├── view.go                      # Rendering
│   ├── components/
│   │   ├── chat.go                  # Message display with markdown
│   │   ├── input.go                 # User input with history
│   │   ├── status.go                # Status bar, model info
│   │   ├── toolview.go              # Tool execution display
│   │   └── filetree.go              # File browser
│   ├── markdown/
│   │   └── render.go                # Glamour-based markdown rendering
│   └── streaming/
│       └── stream.go                # Real-time token streaming display
│
├── acp/                             # Agent Client Protocol frontend (Zed, editors)
│   ├── go.mod                       # module github.com/user/harness/acp
│   ├── server.go                    # Implements Frontend; acp.Agent over JSON-RPC/stdio
│   ├── translate.go                 # Stream events ↔ session/update; full tool-status lifecycle
│   ├── permission.go                # HITL interrupt → session/request_permission
│   ├── bridge.go                    # Client fs/* + terminal/* → Sandbox (editor IS the environment)
│   └── session.go                   # session/new|load|fork|resume via core SessionStore
│
├── eval/                            # Eval adapters (frameworks stay OUT of core)
│   ├── go.mod                       # module github.com/user/harness/eval
│   ├── scorer.go                    # Two-tier: Success() gates, Expect() logs (Deep Agents)
│   ├── parity/                      # Deterministic mock-parity scenarios (claw-code) — gates CI
│   └── harbor/
│       ├── agent.go                 # Wraps Harness as Harbor BaseAgent-equivalent
│       ├── sandbox.go               # HarborSandbox: exec for read/grep; upload/download for write
│       └── trajectory.go            # ATIF trajectory.json; Harbor verifier produces reward
│
├── cli/                             # Agent-facing adapters (implement Tool)
│   ├── go.mod                       # module github.com/user/harness/cli
│   ├── runner.go                    # Subprocess runner + truncation (pi truncated-tool pattern)
│   ├── xsearch/
│   │   └── xsearch.go              # X API search — implements Tool
│   ├── distiller/
│   │   └── distiller.go            # Wraps extensions/swe_distiller — implements Tool
│   ├── webmd/
│   │   └── webmd.go                 # Web→Markdown (may delegate to distiller or pure Go)
│   ├── pdfparse/
│   │   └── pdfparse.go             # PDF→text (like liteparse) — implements Tool
│   └── codereview/
│       ├── review.go                # Codebase research — implements Tool + Analyzer
│       ├── architecture.go          # Architecture-as-code extraction
│       ├── intent.go                # Intent formalization
│       └── verify.go                # Formal verification helpers
│
├── analyzers/                       # AST / code intelligence extensions
│   ├── go.mod                       # module github.com/user/harness/analyzers
│   ├── treesitter/
│   │   ├── parser.go                # Tree-sitter parsing — implements Analyzer
│   │   ├── tags.go                  # Symbol extraction (defs/refs)
│   │   └── queries/                 # Language-specific .scm query files
│   ├── zoekt/
│   │   └── zoekt.go                 # Zoekt code search — implements Analyzer
│   ├── astgrep/
│   │   └── astgrep.go              # ast-grep structural search — implements Analyzer
│   ├── semgrep/
│   │   └── semgrep.go              # Semgrep SAST — implements Analyzer
│   └── graph/
│       ├── dependency.go            # Dependency graph builder
│       ├── ranking.go               # PageRank, betweenness centrality
│       └── repomap.go               # Repository map generation (aider-style)
│
├── platform/                        # Cloud backend adapters (implement core/platform ports)
│   ├── go.mod                       # module github.com/user/harness/platform
│   ├── local/
│   │   └── local.go                 # Filesystem blobs, BoltDB KV — dev + self-host
│   ├── gcp/
│   │   └── gcp.go                   # GCS BlobStore, Firestore KV, Secret Manager
│   ├── aws/
│   │   └── aws.go                   # S3 BlobStore, DynamoDB KV, Secrets Manager, SQS
│   ├── cf/
│   │   └── cf.go                    # R2 BlobStore, KV/DO — NO local FS / subprocess tier
│   ├── modal/
│   │   └── modal.go                 # Remote Sandbox via modal.Sandbox (exec/fs/snapshot)
│   └── turbopuffer/
│       └── turbopuffer.go           # Retriever: namespace-per-repo, vector + BM25 + filters
│
├── cmd/                             # Binaries
│   ├── harness/                     # Interactive agent CLI (Bubble Tea)
│   │   └── main.go                  # Wire core + providers + tools + TUI → run
│   ├── harness-headless/            # CI / pipes — JSON in/out (Flue flue run, claw-analog)
│   │   └── main.go                  # No TUI; caps + --output-format json
│   └── harness-server/              # Hosted agents (Flue flue dev / build HTTP)
│       └── main.go                  # POST /agents/<name>/<instanceId>; run registry
│
├── test/
│   ├── integration/                 # Cross-module integration tests
│   ├── fixtures/                    # Test data
│   └── mock/
│       ├── provider.go              # Mock LLM provider for testing
│       └── tool.go                  # Mock tool for testing
│
└── docs/
    ├── architecture.md              # This document
    ├── getting-started.md
    ├── writing-extensions.md        # How to write providers, tools, analyzers
    ├── connectors/                  # Flue-style install recipes for extensions/* + cli/*
    └── examples/

# Repo root (sibling to harness/) — native implementations, not Go modules
extensions/
├── swe_distiller/                   # Rust: URL → markdown/json (see ARCHITECTURE.md)
│   ├── Cargo.toml
│   └── src/                         # Library + `swe_distiller` binary
└── (future native services)
```

---

## Core Design Deep Dive

### 1. Agent Loop

The agent loop is the heart. It mirrors pi-mono's `runAgentLoop` but leverages goroutines and an explicit typed state machine.

**What Claude Code gets wrong here:** `QueryEngine.ts` is 46K lines — a single file containing the entire agent loop, LLM streaming, tool execution, retry logic, token counting, thinking mode, and analytics. `query/transitions.ts` defines its "state machine" as `type Terminal = any; type Continue = any`. There are no typed states. Harness corrects this.

#### Typed States

```go
// LoopState is an explicit enum — not booleans, not `any`.
type LoopState int

const (
    StateIdle         LoopState = iota  // waiting for prompt
    StateEnriching                      // analyzers injecting context
    StateBudgetCheck                    // check token budget, compact if needed
    StateStreaming                      // streaming LLM response
    StateApproving                      // awaiting Approver decision on tool call
    StateExecuting                      // tools running (parallel read, serial write)
    StateCollecting                     // collecting tool results
    StateSteered                        // steering message received, restart turn
    StateCompleted                      // turn done, check for follow-ups
    StateCompacting                     // context window full, compacting
    StateError                          // recoverable error, retry with backoff
    StateDone                           // agent loop finished
)
```

#### Flow

```
                    ┌─────────────┐
                    │   prompt()  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  ENRICHING  │──── Analyzers inject repomap, symbols
                    └──────┬──────┘     (Claude Code has no equivalent)
                           │
                    ┌──────▼──────┐
                    │ BUDGET CHECK│──── Compact if over token budget
                    └──────┬──────┘     (from Claude Code's autoCompact)
                           │
                    ┌──────▼──────┐
              ┌────►│  STREAMING  │◄──── steering messages (interrupt)
              │     │  (Provider) │
              │     └──────┬──────┘
              │            │
              │     ┌──────▼──────┐
              │     │  APPROVING  │──── Approver.Approve() per tool call
              │     │             │     (Claude Code: hooks/toolPermission/)
              │     └──────┬──────┘
              │            │
              │     ┌──────▼──────┐
              │     │  EXECUTING  │──── read-only: parallel (errgroup)
              │     │  (tools)    │     mutating: sequential
              │     └──────┬──────┘     (from Claude Code's toolOrchestration.ts)
              │            │
              │     ┌──────▼──────┐
              │     │ COLLECTING  │──── gather results, update state
              │     └──────┬──────┘
              │            │
              │      more tools?
              │        │      │
              │       yes     no
              │        │      │
              └────────┘  ┌───▼───────┐
                          │ COMPLETED │
                          │ follow-up │
                          │ queued?   │
                          └───┬───┬───┘
                           yes   no
                            │     │
                    ┌───────┘     └──► DONE
                    │
             ┌──────▼──────┐
             │ DEQUEUE     │
             │ follow-up   │
             └──────┬──────┘
                    │
                    └────► (back to ENRICHING)
```

#### Signature

The loop is a pure function. No file should exceed ~500 lines.

```go
func RunLoop(
    ctx       context.Context,
    provider  Provider,
    tools     []Tool,
    approver  Approver,
    analyzers []Analyzer,
    state     *AgentState,
    bus       *EventBus,
    budget    *TokenBudget,
) error
```

**Key differences from pi-mono and Claude Code:**
- **Typed state machine** with explicit `LoopState` enum (Claude Code uses implicit booleans)
- **Enrichment phase** before streaming — analyzers inject context automatically
- **Budget check** before every LLM call — compact proactively, not reactively
- **Approval phase** is structural — `Approver.Approve()` is called by the loop, not by tools
- **Tool partitioning** via `Tool.ReadOnly()` — read-only parallel, writes serial (from Claude Code)
- **Steering via `context.CancelFunc`** — not queue polling (from pi-mono)

### 2. Event Bus

Typed events over channels. Every event flows through the bus. Frontends, loggers, metrics collectors all subscribe.

```go
// Event types mirror pi-mono's but are Go-native
type EventKind int

const (
    AgentStart EventKind = iota
    AgentEnd
    TurnStart
    TurnEnd
    MessageStart
    MessageUpdate        // streaming delta
    MessageEnd
    ToolExecStart
    ToolExecUpdate       // progress
    ToolExecEnd
    ApprovalRequest      // tool awaiting approval
    ApprovalResult       // approval decision (allow/deny/ask)
    BudgetWarning        // approaching token limit
    CompactStart         // compaction beginning
    CompactEnd           // compaction complete
    ChildAgentStart      // sub-agent spawned
    ChildAgentEnd        // sub-agent completed
    StateChange          // immutable state snapshot
    Error
)

type Event struct {
    Kind      EventKind
    Timestamp time.Time
    SessionID string
    Data      any       // type-assert based on Kind
}
```

Subscribers receive events via a buffered channel. Slow subscribers get dropped events (configurable: drop-oldest or block). This is the backpressure story that pi-mono's EventEmitter lacks.

### 3. State Machine

Agent state is an immutable snapshot. Every mutation produces a new snapshot and emits a `StateChange` event. Frontends render from snapshots — never from mutable state.

```go
type AgentState struct {
    SessionID    string
    Mode         SessionMode          // ask, code, architect
    Model        Model
    SystemPrompt string
    Messages     []Message            // conversation history
    IsStreaming   bool
    StreamDelta  string               // current streaming chunk
    PendingTools map[string]ToolCall  // in-flight tool executions
    Error        error
}

// Snapshot returns an immutable copy
func (s *AgentState) Snapshot() AgentState { ... }
```

**Why immutable snapshots:**
pi-mono mutates `_state` directly and re-renders. This creates race conditions when tools execute concurrently. Snapshots eliminate this — the frontend always renders a consistent state.

### 4. Content-Addressable Cache

Every expensive computation is cached by content hash. Built into core, used by extensions.

```go
type Cache interface {
    Get(key [32]byte) ([]byte, bool)
    Put(key [32]byte, value []byte) error
    Has(key [32]byte) bool
}

// ContentKey hashes the input to produce a cache key
func ContentKey(data []byte) [32]byte {
    return sha256.Sum256(data)
}
```

Two implementations in core:
- **MemoryCache**: LRU with configurable max size (L1, microseconds)
- **DiskCache**: `~/.harness/cache/{hex(key)[:2]}/{hex(key)}` (L2, milliseconds)

Extensions (like the AST analyzer) use this transparently:
```go
key := harness.ContentKey([]byte(fileContent))
if cached, ok := cache.Get(key); ok {
    return unmarshal(cached)
}
result := expensiveParse(fileContent)
cache.Put(key, marshal(result))
```

### 5. Transport Layer

Abstracts the wire protocol. Default is stdio + newline-delimited JSON (same as pi-mono).

```go
type Transport interface {
    Send(ctx context.Context, msg []byte) error
    Receive(ctx context.Context) ([]byte, error)
    Close() error
}
```

Implementations:
- `StdioTransport`: newline-delimited JSON over stdin/stdout (core)
- `TCPTransport`: for remote agents (extension)
- `GRPCTransport`: for high-performance service mesh (extension)

### 6. Token Budget & Compaction

**Credit:** Claude Code's `services/compact/` implements four compaction strategies (auto, micro, API-side, session-memory) and extracts memories before truncating context. This is one of the most practically important subsystems in any agent — without it, long sessions crash into context limits. Harness adopts the concept and improves the architecture.

```go
// TokenBudget tracks context window usage and triggers compaction.
type TokenBudget struct {
    MaxTokens      int     // model's context window
    WarnThreshold  float64 // e.g., 0.80 — emit BudgetWarning event
    CompactAt      float64 // e.g., 0.90 — trigger compaction
    CurrentTokens  int     // estimated tokens in current context
}

func (b *TokenBudget) ShouldCompact() bool {
    return float64(b.CurrentTokens)/float64(b.MaxTokens) >= b.CompactAt
}

// Compactor reduces context size. Extensions implement this.
// The agent loop calls Compact() when budget threshold is exceeded.
type Compactor interface {
    // Compact receives the current messages and returns a reduced set.
    // The bus receives CompactStart/CompactEnd events for observability.
    Compact(ctx context.Context, messages []Message) ([]Message, error)
}
```

Compactor implementations (all in extensions, not core):
- **SummaryCompactor**: LLM-generated summary of old messages (like Claude Code's autoCompact)
- **SlidingWindowCompactor**: drop oldest messages beyond a window
- **MemoryExtractCompactor**: extract key learnings before compacting (from Claude Code's sessionMemoryCompact — this is the most valuable strategy)
- **OffloadCompactor**: spill summarized history and oversized tool results to the **content-addressable cache / filesystem**, leaving a `sha256` handle in the transcript (Deep Agents' `/conversation_history/` + `/large_tool_results/` pattern). A single huge grep/read result is stored once and referenced, not re-sent — cheaper and more debuggable than in-memory truncation. This is where Harness's content-addressable cache pays off that other harnesses lack.
- **ChainCompactor**: extract memories THEN offload large artifacts THEN summarize THEN truncate

### 7. Multi-Agent Spawning

**Credit:** Claude Code's `AgentTool` + `coordinator/coordinatorMode.ts` prove that multi-agent orchestration works for real-world coding tasks. Their implementation (tmux backends, layout managers, iTerm2 setup) is over-engineered for a harness — but the core idea is sound.

Harness keeps it simple: an agent can spawn a child agent. The child has its own state, its own conversation, and a subset of tools. The parent receives events from the child via the event bus.

```go
// SpawnChild creates a new agent loop that runs concurrently.
// The child gets its own AgentState and communicates via the shared EventBus.
// The parent's ChildAgentStart/ChildAgentEnd events track lifecycle.
func (h *Harness) SpawnChild(ctx context.Context, opts ChildOpts) (<-chan Event, error) {
    childState := &AgentState{
        SessionID:    newSessionID(),
        ParentID:     opts.ParentSessionID,
        Model:        opts.Model,
        SystemPrompt: opts.SystemPrompt,
        Messages:     []Message{userMessage(opts.Task)},
    }
    childBus := h.bus.Scoped(childState.SessionID) // events prefixed with child ID

    go func() {
        h.bus.Emit(Event{Kind: ChildAgentStart, ...})
        err := RunLoop(ctx, h.provider, opts.Tools, h.approver, nil, childState, childBus, opts.Budget)
        h.bus.Emit(Event{Kind: ChildAgentEnd, ...})
    }()

    return childBus.Subscribe(), nil
}
```

**What NOT to do (from Claude Code):**
- No tmux/iTerm2 backends — the frontend decides how to render multiple agents
- No layout managers — that's a TUI concern, not a core concern
- No permission sync protocol — children inherit the parent's Approver
- No 30-file swarm infrastructure — spawning a child is ~50 lines

---

## Extension Design Patterns

### Writing a Provider

```go
package openai

type OpenAI struct {
    apiKey  string
    client  *http.Client
    baseURL string
}

func New(apiKey string) *OpenAI {
    return &OpenAI{
        apiKey:  apiKey,
        client:  &http.Client{Timeout: 5 * time.Minute},
        baseURL: "https://api.openai.com/v1",
    }
}

func (o *OpenAI) Stream(ctx context.Context, req *harness.StreamRequest) (<-chan harness.StreamEvent, error) {
    ch := make(chan harness.StreamEvent, 64)
    go func() {
        defer close(ch)
        // ... SSE stream from OpenAI, emit events to ch
    }()
    return ch, nil
}

func (o *OpenAI) Models(ctx context.Context) ([]harness.Model, error) {
    // ... GET /v1/models
}
```

### Writing a Tool

```go
package filesystem

type ReadFile struct{}

func (r *ReadFile) Name() string        { return "read_file" }
func (r *ReadFile) Description() string { return "Read contents of a file" }
func (r *ReadFile) Schema() json.RawMessage {
    return json.RawMessage(`{
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read"}
        },
        "required": ["path"]
    }`)
}

func (r *ReadFile) Execute(ctx context.Context, args json.RawMessage) (*harness.ToolResult, error) {
    var params struct{ Path string `json:"path"` }
    if err := json.Unmarshal(args, &params); err != nil {
        return nil, err
    }
    content, err := os.ReadFile(params.Path)
    if err != nil {
        return &harness.ToolResult{IsError: true, Content: err.Error()}, nil
    }
    return &harness.ToolResult{Content: string(content)}, nil
}
```

### Writing an Analyzer

```go
package treesitter

type TreeSitter struct {
    cache harness.Cache
}

func (ts *TreeSitter) Name() string { return "treesitter" }

func (ts *TreeSitter) Analyze(ctx context.Context, files []harness.FileRef) (*harness.Analysis, error) {
    var tags []Tag
    for _, f := range files {
        key := harness.ContentKey(f.Content)
        if cached, ok := ts.cache.Get(key); ok {
            tags = append(tags, unmarshalTags(cached)...)
            continue
        }
        parsed := parse(f)
        ts.cache.Put(key, marshalTags(parsed))
        tags = append(tags, parsed...)
    }
    return buildAnalysis(tags), nil
}

func (ts *TreeSitter) Query(ctx context.Context, q *harness.AnalysisQuery) (*harness.AnalysisResult, error) {
    // Symbol lookup, dependency queries, repomap generation
}
```

### Wrapping Native Services (`swe_distiller`)

pi-mono extensions are TypeScript modules loaded at runtime (`~/.pi/agent/extensions/`, `-e path.ts`). They extend the agent via `pi.registerTool()` and `pi.on()`. When the capability lives in another language or a standalone CLI, the extension **wraps a subprocess** — see [truncated-tool.ts](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/examples/extensions/truncated-tool.ts), which registers an `rg` tool that shells out to ripgrep, truncates output, and optionally spills full results to a temp file.

Harness does not use dynamic extension loading. The equivalent is a **two-layer split**:

| Layer | Location | Responsibility |
|-------|----------|----------------|
| **Native service** | `extensions/<name>/` | Fast, focused binary or library (Rust, Python, etc.). Stable CLI contract. No knowledge of the agent loop. |
| **Go adapter** | `harness/cli/<name>/` | Implements `harness.Tool`. Spawns the binary, maps JSON schema ↔ CLI flags, truncates/caches, returns `ToolResult`. |

`swe_distiller` is the reference implementation of this pattern.

#### pi-mono vs Harness (same intent, different wiring)

```
pi-mono (runtime extension)                Harness (compile-time composition)

  extension.ts                               cmd/harness/main.go
       │                                          │
       ├─ pi.registerTool({ name: "…" })          ├─ core.WithTools(distiller.New(cfg))
       └─ execute() → spawn swe_distiller         └─ DistillerTool.Execute() → process.Run(...)
              │                                          │
              ▼                                          ▼
         swe_distiller binary                      extensions/swe_distiller binary
```

| Concern | pi-mono | Harness |
|---------|---------|---------|
| Discovery | Auto-scan `~/.pi/agent/extensions/` or `-e` | Explicit import in `main.go` (or config-driven list) |
| Tool registration | `pi.registerTool()` at runtime | Struct satisfies `Tool`; passed to `core.WithTools()` |
| Hooks | `pi.on("tool_call", …)` | `Hook` interface implementations (compile-time) |
| Subprocess | Node child-process APIs in extension | `core/process.Runner` with `context.Context` cancel |
| Truncation | `truncateHead` from coding-agent SDK | `cli/runner.go` shared limits (50KB / 2000 lines) |
| Custom TUI for tool | `renderCall` / `renderResult` on tool def | `Frontend` renders from `ToolExec*` events + optional `ToolResult.Details` |
| Session fork state | `details` on tool result | Same: opaque `Details` JSON on `ToolResult` |

#### Stable CLI contract (owned by the native service)

The adapter depends on a **versioned subprocess contract**, not on Rust types. Today `swe_distiller` exposes:

```bash
swe_distiller <url> --stdout                    # markdown on stdout (agent-friendly)
swe_distiller <url> --mode json --stdout        # structured metadata + content
swe_distiller <url> --llm -o out.md             # optional LLM override pipeline
```

Harness adapters should prefer `--stdout` and `--mode json` so nothing writes into the workspace unless the LLM explicitly asks for a file path. Environment (proxy, provider keys) stays in the **native** service (`SWE_DISTILLER_*`, `JINA_API_KEY`, etc.) — the Go layer only forwards allowlisted env vars from config, never logs secrets.

#### Go adapter sketch

Shared subprocess plumbing lives in `cli/runner.go` (timeouts, stderr capture, cancellation, truncation). Each service adapter is ~80–120 lines.

```go
package distiller

import (
    "context"
    "encoding/json"
    "fmt"

    "github.com/user/harness/cli/runner"
    "github.com/user/harness/core"
    "github.com/user/harness/core/process"
)

// DistillerTool wraps extensions/swe_distiller for the agent loop.
// Equivalent to a pi-mono extension that registerTool({ name: "distill_url", execute: ... }).
type DistillerTool struct {
    Bin    string              // default: lookup PATH or cfg.Extensions.Distiller
    Runner process.Runner      // core/process — ctx cancel, SIGTERM tree
    Cache  core.Cache          // ContentKey(url + flags) → markdown
    Limits runner.OutputLimits // align with pi DEFAULT_MAX_BYTES / DEFAULT_MAX_LINES
}

func (d *DistillerTool) Name() string        { return "distill_url" }
func (d *DistillerTool) ReadOnly() bool      { return true } // fetch-only; no workspace mutation

func (d *DistillerTool) Schema() json.RawMessage { /* url, lang?, llm?, mode? */ }

func (d *DistillerTool) Execute(ctx context.Context, args json.RawMessage) (*core.ToolResult, error) {
    var params struct {
        URL  string `json:"url"`
        Lang string `json:"lang,omitempty"`
        LLM  bool   `json:"llm,omitempty"`
        Mode string `json:"mode,omitempty"` // "markdown" | "json"
    }
    if err := json.Unmarshal(args, &params); err != nil {
        return nil, err
    }

    cacheKey := core.ContentKey([]byte(params.URL + params.Mode + params.Lang + fmt.Sprint(params.LLM)))
    if cached, ok := d.Cache.Get(cacheKey); ok {
        return &core.ToolResult{Content: string(cached)}, nil
    }

    argv := []string{params.URL, "--stdout"}
    if params.LLM {
        argv = append(argv, "--llm")
    }
    if params.Lang != "" {
        argv = append(argv, "--lang", params.Lang)
    }
    if params.Mode == "json" {
        argv = append(argv, "--mode", "json")
    }

    out, err := d.Runner.Run(ctx, process.Command{
        Path: d.Bin,
        Args: argv,
        // Env: only SWE_DISTILLER_* + proxy vars from config — never dump os.Environ()
    })
    if err != nil {
        return &core.ToolResult{IsError: true, Content: err.Error()}, nil
    }

    text, details, spill := runner.TruncateHead(out.Stdout, d.Limits)
    if spill != "" {
        details["full_output_path"] = spill // pi-mono pattern: LLM can read_file if needed
    }

    d.Cache.Put(cacheKey, []byte(text))
    return &core.ToolResult{
        Content: text,
        Details: details, // title, word_count when mode=json — for TUI + session fork
    }, nil
}
```

`process.Runner` uses `exec.CommandContext` with **argv slices only** (no shell interpolation) — the Go equivalent of pi-mono's safer `execFile` style, not string-concatenated shell commands.

#### Configuration and wiring

```go
// config.Extensions — resolved at startup (no runtime plugin registry)
type Extensions struct {
    Distiller string `toml:"distiller"` // path to binary; default: "swe_distiller" on PATH
}

// cmd/harness/main.go
distillerTool := distiller.New(distiller.Options{
    Bin:    cfg.Extensions.Distiller, // or ${REPO_ROOT}/extensions/swe_distiller/target/release/swe_distiller
    Runner: process.DefaultRunner(),
    Cache:  diskCache,
})

h := core.New(
    core.WithTools(
        filesystem.NewReadFile(),
        distillerTool,           // replaces a monolithic "webmd" tool when distiller is available
        // webmd.NewFallback(),  // optional pure-Go fallback if binary missing
    ),
    // ...
)
```

**Verification:**

```bash
# Native service alone
cd extensions/swe_distiller && cargo run -- "https://example.com" --stdout | head

# Through harness (once wired)
echo '{"tool":"distill_url","args":{"url":"https://example.com"}}' | go run ./cmd/harness/
```

#### When to use which integration style

| Style | When | Example |
|-------|------|---------|
| **In-process Go `Tool`** | Logic is small, no special runtime | `read_file`, glob rules |
| **Subprocess adapter** (`cli/*` → `extensions/*`) | Heavy native stack (HTML, ML, browsers) | `swe_distiller`, future `swe-tokenizer` |
| **Long-running sidecar** | Amortize startup (index servers) | Zoekt, tree-sitter daemon (later) |

Rule: keep the **native** crate/script focused on one job (extract, tokenize, parse). Keep the **Go adapter** focused on agent concerns (schema, cache, truncation, approval, events). Do not re-implement distillation in Go while `extensions/swe_distiller` exists — that duplicates the pipeline described in `extensions/swe_distiller/ARCHITECTURE.md`.

#### Hooks without `pi.on()`

pi-mono extensions intercept lifecycle with `pi.on("tool_call", …)`. Harness equivalents are explicit `Hook` implementations registered beside tools:

```go
// Optional: block distill_url to non-HTTPS or allowlisted domains
type DistillerPolicyHook struct{}
func (h *DistillerPolicyHook) OnToolCall(ctx context.Context, e *core.ToolCallEvent) (*core.ToolCallMutation, error) {
    if e.ToolName != "distill_url" { return nil, nil }
    // mutate or block before Execute — same phase as pi tool_call handler
}
```

---

## Metaengineering: A Framework Agents Can Extend

The extension patterns above describe how a *human* writes a Go extension. But the primary extenders of Harness are **AI agents** (and the OSS community working through agents). The framework's structure must therefore align with how agents *succeed* and *fail* — so that a strong, stable Go core can be extended by the community without anyone (human or model) holding the whole framework in their head, and without the maintainers patching every time a frontier lab ships a new agentic tool.

> **Meta-principle:** *A correct extension is the path of least resistance; an incorrect one fails loudly, locally, and early — before the agent has burned context or done damage.*

### Agent strengths/weaknesses → design implications

| Agent is **good** at | Agent is **bad** at | Framework therefore… |
|---|---|---|
| Imitating a clear example | Inventing structure from scratch | Ships a golden-path template per extension point |
| Local, bounded reasoning | Holding a 500K-LoC mental model | Keeps extensions single-file, single-directory, zero-core-edit |
| Following explicit schemas/types | Tracking implicit global invariants | Encodes invariants in types + a validator, not docs/convention |
| Iterating against feedback | Knowing when it's wrong | Offers a fast, deterministic, offline loop (mock provider + parity test) |
| Generating code in known languages | Novel framework-specific glue | Lets agents author in **Python/Starlark/Markdown they already know** |
| Reading error text and acting | Spooky action at a distance | Teaching errors; unidirectional deps enforced by the compiler |

### The Capability Ladder

The core insight (carried from `docs/PLAN.md`'s self-extension model and confirmed by Codex's "code mode" and Flue's skills): **don't make agents write compiled Go to extend the system.** Offer a ladder of extension surfaces, each trading friction for permanence/trust. Agents live mostly on the bottom rungs, where there is **no wiring and no rebuild** — and where mistakes are sandboxed and cheap.

| Rung | Surface | Agent authors | Persistence | Isolation | Wiring cost |
|------|---------|---------------|-------------|-----------|-------------|
| **0** | **Skill** (Markdown sub-program, Flue-style) | natural language | file, read at call time | n/a (prompt-level) | none |
| **1** | **Monty** (Python subset, `monty-cli` subprocess) | throwaway code | discarded after run | **sandboxed subprocess** | none |
| **2** | **Starlark** tool definition (`go.starlark.net`) | a tool def | session/runtime | **sandboxed interpreter** | dynamic register |
| **3** | **Go `Tool`/`Hook`** (compiled) | Go (agent or human) | permanent | in-process (compiled) | **explicit (`main.go`)** |

```
Hot path:   need capability NOW    → write Python  → Monty executes → throwaway
Warm path:  reusable tool          → write Starlark→ dynamic register, no rebuild
Cold path:  recurring / proven     → write Go tool → go build (~3s) → restart → permanent
Rule of three: 1st ad-hoc (Monty) → 2nd notice → 3rd promote to Go.
```

This is also why Go was chosen: it is *simple enough for an LLM to self-modify*, so the cold-path promotion (and the Phase 7 recursive self-build — agent writes Go, `go build`, exec the new binary) is realistic rather than aspirational.

### Three properties that make agent authorship safe

1. **Sandboxed-by-default authoring.** Monty runs in a resource-limited subprocess; Starlark runs in a restricted interpreter with only the bridges we expose. An agent-written extension on rungs 0–2 *cannot* escape its tier, so agents can experiment freely — the blast radius is contained by construction. This is the single biggest enabler of agent self-extension.
2. **Host-side execution, schema-only to the agent** (Flue). A `Tool`'s *implementation* reads secrets from env/config host-side; the agent only ever sees the tool's JSON Schema. Agents author and invoke behavior without ever touching credentials — secrets never enter args, prompts, or filesystem context.
3. **Rule-of-three promotion as a workflow.** Agents iterate cheaply at the bottom; only proven, recurring capabilities graduate to typed, reviewed Go. This plays to agent strengths (fast, disposable iteration) and human strengths (reviewing the small permanent surface). Codex's code-mode corollary: let agents **batch many tool calls in one Monty/Starlark program** instead of one-call-per-round-trip.

### Why explicit wiring (not codegen) for the cold path — for now

The `Approver`-in-core boundary, immutable state, and compiler-enforced unidirectional module deps already make Go extensions safe to *write*. The only friction left is *connecting* a compiled tool to the binary. Three options exist — explicit `main.go` wiring, `go:generate` codegen, and `init()` self-registration — but the **rule of three makes cold-path promotion deliberately rare and review-worthy**, so we choose the simplest:

- **v1: explicit wiring in `main.go`** (the one obvious, fully compile-checked, 3am-debuggable place). For a rare, intentional, reviewed commitment, "visible in one file" is a feature, not a tax.
- **Defer codegen** (`go:generate` scanning a manifest) until the number of compiled extensions makes `main.go` churn a *measured* pain. Static Go has no autoloading, so codegen — not `init()`+blank-import (a runtime registry with silent-failure modes) — would be the eventual answer. But building that subsystem now is the "leave the mass / measure before heroics" anti-pattern every reference critique warns against.

A `harness new tool|skill|starlark <name>` scaffold and a `harness doctor` validator (deterministic pass/fail beyond compilation: naming, schema validity, banned imports, contract-test presence) give agents crisp feedback across *all* rungs — that, not codegen, is where the leverage is.

### The framework is itself a corpus

Jacob Young's *Use boring languages with LLMs* argues that agents produce reliable output in **low-variance, strong-convention** ecosystems (one Rails) and unreliable output in fragmented ones (a dozen JS frameworks). The non-obvious corollary: **Harness's own API surface is a mini-corpus the agent pattern-matches against.** If there are three ways to write a `Tool`, a god-context, or feature-flags-as-architecture, we recreate the JS-fragmentation problem *inside our own framework* — every extra degree of freedom is variance the model must gamble on.

So the goal is *"there is essentially one Rails"* applied to ourselves: **there is essentially one way to write a Harness `Tool`.** This is *why* the Anti-Patterns section is load-bearing for agent extensibility, not just for human maintainers — singularity of the extension surface is a feature the model consumes. It also corroborates the wiring choice: explicit `core.WithTools(...)` is plain, in-corpus Go any agent recognizes; a bespoke `// +harness:tool` codegen DSL is novel and out-of-corpus.

### One-right-way tooling is the agent's feedback substrate

The same essay is emphatic that one-right-way *runtime tooling* is the best half of the combo — it enforces conventions *without prompting the agent into compliance*. Harness makes the standard Go trinity a **first-class, documented part of the extension loop**, not an afterthought:

- **`gopls`** — real-time semantic feedback while the agent edits (types, references, undefined symbols).
- **`go vet`** — catches the bounded footgun set (shadowed `err`, unused `Errorf`, lock copies).
- **`golangci-lint`** — statically enforces house style and banned primitives via committed config, so review comments become compile-time failures.
- **`harness doctor`** + the **mock-parity harness** (claw-code) — framework-specific contracts and offline, network-free behavioral checks.

This is the deterministic, teaching feedback agents iterate against. Pair it with Go's GC (agents fight Rust's borrow checker; Rust stays confined to the TUI renderer + tokenizer) and the result is the essay's exact prescription: a boring, consistent substrate where the *median* agent output is already correct.

---

## Wiring: The Main Binary

The power of the harness pattern: everything is composed at the top level.

```go
package main

import (
    "github.com/user/harness/core"
    "github.com/user/harness/provider/openai"
    "github.com/user/harness/provider/anthropic"
    "github.com/user/harness/tools/filesystem"
    "github.com/user/harness/tools/terminal"
    "github.com/user/harness/cli/distiller"
    "github.com/user/harness/cli/pdfparse"
    "github.com/user/harness/analyzers/treesitter"
    "github.com/user/harness/tui"
)

func main() {
    // Parallel prefetch — fire before heavy initialization.
    // Pattern from Claude Code's main.tsx (startMdmRawRead, startKeychainPrefetch).
    go core.PreconnectAPI(os.Getenv("ANTHROPIC_BASE_URL"))
    go core.WarmDiskCache("~/.harness/cache")

    h := core.New(
        core.WithProviders(
            openai.New(os.Getenv("OPENAI_API_KEY")),
            anthropic.New(os.Getenv("ANTHROPIC_API_KEY")),
        ),
        core.WithTools(
            filesystem.NewReadFile(),
            filesystem.NewWriteFile(),
            terminal.NewExec(),
            distiller.New(cfg.Extensions),
            pdfparse.New(),
        ),
        core.WithApprover(
            approval.NewChain(
                approval.NewRules("~/.harness/rules.toml"),  // glob allow/deny
                approval.NewAuto(),                           // auto-classify
                approval.NewInteractive(),                    // prompt user
            ),
        ),
        core.WithAnalyzers(
            treesitter.New(),
        ),
        core.WithFrontend(tui.New()),
        core.WithCache(core.NewDiskCache("~/.harness/cache")),
    )

    if err := h.Run(context.Background()); err != nil {
        fmt.Fprintf(os.Stderr, "harness: %v\n", err)
        os.Exit(1)
    }
}
```

Want a headless server instead? Same core, different frontend:

```go
h := core.New(
    core.WithProviders(openai.New(key)),
    core.WithTools(filesystem.NewReadFile()),
    core.WithFrontend(rpc.NewGRPCServer(":50051")),
)
```

---

## Cloud Portability & Composable Backends

Harness runs three ways: a local single binary (default), a hosted server, and an unbundled cloud deployment where each infrastructure concern is backed by a best-of-breed provider. The interactive path never needs any of this — these ports only exist so the *deployed* harness isn't married to one cloud.

### Two workloads, two homes

The single biggest deployment mistake is treating "run it in the cloud" as one thing. Harness has two workloads with opposite shapes:

| Workload | Shape | Wants |
|----------|-------|-------|
| **Orchestrator** (`harness-server`, agent loop, sessions, run registry) | long-lived, light CPU, stateful | live near your DB/secrets; steady-state economics |
| **Sandbox / archaeology** (clone, extract, `rg`/`bsdtar`/`swe_distiller`, agent-authored code) | ephemeral, per-repo, untrusted, bursty | FS + arbitrary subprocess; per-job isolation; scale-to-zero |

Cloudflare Workers can host the orchestrator but **cannot** be the sandbox (no FS, no subprocess). That asymmetry is the whole argument for keeping the sandbox tier behind a swappable port.

### Capability ports (decouple from any one cloud)

Everything cloud-specific hides behind narrow, behavior-first interfaces in `core/platform`. The core depends on these, never on a vendor SDK:

```go
package platform

// BlobStore — object storage (S3 / GCS / R2 / local FS).
type BlobStore interface {
    Get(ctx context.Context, key string) (io.ReadCloser, error)
    Put(ctx context.Context, key string, r io.Reader) error
    List(ctx context.Context, prefix string) ([]string, error)
}

// KVStore — small mutable state (DynamoDB / Firestore / CF KV / BoltDB).
type KVStore interface {
    Get(ctx context.Context, key string) ([]byte, bool, error)
    Put(ctx context.Context, key string, val []byte) error
    Delete(ctx context.Context, key string) error
}

// Secrets, Queue, Lock follow the same shape — minimal verbs, no vendor types.

// Retriever — vector + BM25 + metadata filter (turbopuffer / pgvector / OpenSearch).
// ns maps to one repo/instance (turbopuffer namespace-per-prefix).
type Retriever interface {
    Upsert(ctx context.Context, ns string, docs []Doc) error   // batch — respect WAL cadence
    Query(ctx context.Context, ns string, q Query) ([]Hit, error)
    Warm(ctx context.Context, ns string) error                 // pre-flight cache hint
}
```

A `Runtime` capability descriptor lets the core adapt to environments that lack a filesystem or subprocess execution (Cloudflare) instead of crashing:

```go
type Runtime struct {
    LocalFS       bool   // can we write to a real disk?
    Subprocess    bool   // can we exec rg / bsdtar / swe_distiller?
    EphemeralGB   int    // scratch budget for hydration
    Region        string
}
```

When `Subprocess` is false, archaeology tools degrade to the `Remote` sandbox tier (Modal/Fargate) instead of running in-process.

### Ports × tiers: pick a backend per concern

The payoff of narrow ports: each one is backed by the *best* provider, none load-bearing, every one with a self-host fallback.

| Port | Self-host | Cloud-native | Bespoke best-of-breed |
|------|-----------|--------------|----------------------|
| `Sandbox` (compute) | local subprocess | Fargate / Cloud Run Jobs | **Modal**, Daytona, E2B |
| `Retriever` (index) | pgvector, Qdrant | OpenSearch, Vertex | **turbopuffer** |
| `BlobStore` | MinIO / local FS | S3 / GCS / R2 | — (commodity) |
| `KVStore` | BoltDB | DynamoDB / Firestore / CF KV | — (commodity) |
| `Provider` (LLM) | Ollama | Bedrock / Vertex | OpenAI / Anthropic |

### Sandbox compute tier (Modal vs cloud-native)

`modal.Sandbox` is purpose-built for "execute agent code": gVisor isolation, `sb.Exec([...])` runs **any** binary (so `swe_distiller`, `rg`, `bsdtar` work unchanged), volumes, FS snapshots, web tunnels, idle-timeout + scale-to-zero, and a **Go SDK** (`mc.Sandboxes.Create`, `sb.Exec`) that maps onto our `Sandbox` interface 1:1. It's the same role Flue gives Daytona — a *remote sandbox connector*, not foundational.

```go
// platform/modal — Remote tier of core/sandbox.Sandbox
func (m *ModalSandbox) Exec(ctx context.Context, argv []string, o ExecOpts) (ExecResult, error) {
    p, err := m.sb.Exec(ctx, argv, &modal.SandboxExecParams{Timeout: o.Timeout})
    // stream stdout/stderr, map exit code → ExecResult
}
```

- **Reach for Modal** when code is untrusted/agent-generated, load is bursty, ops budget is thin, or GPU is occasionally needed.
- **Reach for Fargate / Cloud Run Jobs** when data-residency, VPC-private DBs, or high *sustained* throughput (reserved capacity) dominate.

### Retrieval tier (turbopuffer)

turbopuffer is the natural backend for the RAG sidecar (the `claw-code` lesson) and `AST_SERVICE_ARCHITECTURE.md`'s embeddings layer: Rust query nodes over object storage, **namespace-per-prefix** (≈ per-repo index), cheap scale-to-zero storage, hybrid **BM25 + ANN + metadata filter** in one query (ideal for code: exact symbols *and* semantics), cold ~500ms / warm p50 ~14ms with a warm-up hint, strong consistency by default.

- **Index in batch** — one WAL entry/sec/namespace, batched. Bulk-index a repo; never per-keystroke.
- **Data gravity** — embeddings live in turbopuffer's layout; migration = re-embed + re-upsert. BYOC exists for residency.

### Archive archaeology (search without extraction)

For code archaeology over large repos/archives, prefer **one-time hydration** to local ephemeral storage over live object-store mounts (`gcsfuse` is convenient but each `stat`/`open` is a network round-trip). When full extraction is wasteful, the `core/archive` ports JIT-stream individual paths:

- **`ArchiveCatalog`** lists entries from the central directory (ZIP) or index without inflating the body — cheap `git ls-tree`-style traversal.
- **`ArchiveReader`** streams exactly one path on demand (range-read the blob, inflate only that member).
- **`ArchiveSearch`** wraps the catalog + reader to grep across entries lazily — the archaeology equivalent of `unzip -p file.zip path | rg pattern` without ever expanding the archive to disk.

Shallow clone vs archive is an agent-need decision: shallow `git clone --depth=1` when history/blame matters; a `.tar.gz`/`.zip` + catalog when the agent only reads current state (smaller, no `.git`, streamable from `BlobStore`).

### Decision rule & the real risk

Buy a bespoke managed primitive when **all** hold: (1) the capability is genuinely hard (secure sandboxes, object-store vector search), (2) it sits behind a narrow port, (3) its data gravity is acceptable or BYOC exists, (4) the alternative is operating stateful infra at 3am. Otherwise use cloud-native or self-host.

The risk being managed is **not** "bespoke vs big cloud" — it's **vendor sprawl**: N providers = N auth surfaces, DPAs, and failure domains. Bound it by buying bespoke only for hard capabilities and always shipping a self-host/cloud-native adapter behind the same port, so any provider is replaceable in an afternoon.

```text
Composable backend map → Harness ports

  Modal sandbox        →  platform/modal        (core/sandbox Remote tier)
  turbopuffer          →  platform/turbopuffer  (core/platform Retriever)
  S3 / GCS / R2        →  platform/{aws,gcp,cf}  (core/platform BlobStore)
  Cloudflare (no FS)   →  Runtime{Subprocess:false} → degrade to Remote sandbox
  zip/tar.gz on blobs  →  core/archive (catalog → reader → search, JIT stream)
```

---

## Implementation Phases

### Phase 1: The Core (Weeks 1–3)

Build the harness with zero external dependencies.

| Module | What | Learn |
|--------|------|-------|
| `core/protocol` | JSON-RPC 2.0 types, ACP types, codec | Go structs, JSON tags, discriminated unions with iota |
| `core/transport` | stdio transport, framing | `io.Reader`/`io.Writer`, `bufio.Scanner`, interfaces |
| `core/bus` | Typed event bus with channels | Goroutines, channels, fan-out, `context.Context` |
| `core/agent` | Agent loop, typed state machine, budget, compaction interface | State machines, `select {}`, `errgroup` |
| `core/cache` | Content-addressable LRU + disk cache | SHA-256, file I/O, `sync.Map`, atomic operations |
| `core/process` | Subprocess spawning, lifecycle | `os/exec`, pipes, signal handling |

**Deliverable:** A working harness that can spawn an agent process, send a prompt, stream a response, and display it on stdout. No TUI yet — stdout is the frontend.

**claw-code alignment:** Add `test/mockprovider` and 3–5 scenarios from claw's milestone-1 set (`streaming_text`, `read_file_roundtrip`, `write_file_denied`) so behavior is regression-locked before Bubble Tea.

**Flue alignment:** Introduce `InstanceID` + `SessionStore` key shape early; default sandbox to virtual/restricted; sketch `Prompt` with optional `ResultSchema` (finish/give_up) before HTTP server lands.

**Verification:**
```bash
echo '{"prompt": "Hello"}' | go run ./cmd/harness/
# Streams response tokens to stdout
go test ./test/integration/ -run Parity
```

### Phase 2: Providers (Weeks 3–4)

Implement the `Provider` interface for real LLMs.

| Module | What | Learn |
|--------|------|-------|
| `provider/openai` | OpenAI chat completions + streaming | HTTP clients, SSE parsing, API design |
| `provider/anthropic` | Anthropic Messages API | Provider abstraction validation |
| `provider/ollama` | Local Ollama models | Localhost integration, model management |
| `provider/registry` | Model registry + resolution | Registry pattern, model metadata |

**Deliverable:** Send prompts to real LLMs, receive streaming responses through the harness event bus.

### Phase 3: Tools + Approval (Weeks 4–5)

Implement built-in tools + the Approver chain.

| Module | What | Learn |
|--------|------|-------|
| `tools/filesystem` | Read, write, list, search | File I/O, sandboxing, path validation |
| `tools/terminal` | Command execution + safety classification | `os/exec`, PTY, output capture |
| `tools/terminal/safety` | Dangerous pattern detection, path validation | Adapted from Claude Code's `bashSecurity.ts`, `dangerousPatterns.ts` |
| `tools/approval/rules` | Glob-based allow/deny rules | Pattern matching, TOML config parsing |
| `tools/approval/auto` | Auto-classify command safety | Command parsing, heuristics |
| `tools/approval/interactive` | Prompt user via Frontend event | Event bus, Frontend protocol |
| `tools/approval/chain` | Compose: rules → auto → interactive | Chain of responsibility pattern |

**Deliverable:** Agent can call tools, each invocation passes through the Approver chain, results feed back to the LLM. Denied tools return structured errors to the LLM.

### Phase 4: TUI Frontend (Weeks 5–7)

Build the Bubble Tea frontend. This is purely a rendering concern — no agent logic lives here.

| Module | What | Learn |
|--------|------|-------|
| `tui/app.go` | Bubble Tea model, implements Frontend | Elm architecture, event-driven UI |
| `tui/components/` | Chat, input, status, tool views | Component composition, lipgloss styling |
| `tui/markdown/` | Glamour markdown rendering | Terminal rendering, ANSI codes |
| `tui/streaming/` | Real-time token display | Buffering, render throttling |
| `acp/server.go` | ACP `Frontend` (Zed/editors) over JSON-RPC/stdio | Protocol-as-frontend seam; thin adapter implementing `acp.Agent` |
| `acp/bridge.go` | Route file/shell ops through client `fs/*`+`terminal/*` | "Editor is the environment" — the half Deep Agents skipped |

**Deliverable:** Full interactive TUI that renders agent state from immutable snapshots. The same `Frontend` interface also drives an ACP server: file/terminal ops bridge to the editor's capabilities, tool status reports the full `pending→in_progress→completed/failed` lifecycle, and `session/load`/fork/resume ride on the core `SessionStore`.

### Phase 5: CLI Plugins + Analyzers (Weeks 7–9)

The extensions that make this an **agent harness**, not just an agent.

| Module | What | Learn |
|--------|------|-------|
| `cli/runner` | Shared subprocess + truncation for native CLIs | `process.Runner`, argv-only spawn, output limits |
| `cli/distiller` | Go `Tool` adapter for `extensions/swe_distiller` | Polyglot extension pattern (pi `registerTool` → compile-time `Tool`) |
| `core/archive` | Catalog/reader/search over zip+tar.gz w/o extraction | JIT streaming, central-directory parsing, lazy grep |
| `cli/xsearch` | X API search as a tool | OAuth, API clients, pagination |
| `cli/webmd` | Optional pure-Go fallback if distiller binary absent | When not to subprocess |
| `cli/pdfparse` | PDF→text extraction | Binary formats, streaming parsers |
| `cli/codereview` | Codebase research tool | Architecture analysis, static analysis |
| `analyzers/treesitter` | Tree-sitter AST parsing | Tree-sitter bindings, tag extraction |
| `analyzers/graph` | Dependency graphs, PageRank | Graph algorithms, NetworkX-style analysis |
| `analyzers/zoekt` | Zoekt code search integration | Index-based search, trigram indexing |
| `analyzers/astgrep` | ast-grep structural search | AST patterns, structural matching |

**Deliverable:** A fully extensible agent that can search X, parse PDFs, extract web content, analyze codebases via AST, and generate repository maps — all as pluggable tools and analyzers.

### Phase 6: Production Hardening (Weeks 9–10)

| Concern | What | Learn |
|---------|------|-------|
| Observability | Structured logging (`slog`), OpenTelemetry traces | `log/slog`, trace propagation |
| Persistence | Session save/restore, SQLite history | Atomic writes, migrations; instance-scoped keys (Flue) |
| Platform ports | `BlobStore`/`KVStore`/`Secrets` + `Runtime` capability descriptor | Ports-and-adapters, capability negotiation, cloud-neutral core |
| Compute tier | `platform/modal` Remote sandbox; `platform/{aws,gcp,cf}` blobs/KV | Vendor SDK isolation, scale-to-zero, gVisor exec |
| Retrieval tier | `platform/turbopuffer` RAG sidecar (namespace-per-repo) | Hybrid BM25+ANN, batch upsert, warm-up hints |
| HTTP agents | `harness-server`: `/agents/{name}/{instanceID}`, run registry | REST design, idempotent instance routing |
| Structured API results | `finish`/`give_up` on webhook agents | JSON Schema validation, CI-friendly responses |
| Context offload | `OffloadCompactor`: spill large results/history to content-addressable cache, pass `sha256` handle | Deep Agents' FS-as-overflow on our cache substrate |
| Eval scoring | `eval/scorer`: two-tier `Success()` (gates) / `Expect()` (logs) | Deep Agents two-tier scoring; correctness gates, efficiency observed |
| Harbor evals | `eval/harbor`: BaseAgent wrapper + `HarborSandbox`; Terminal-Bench 2.0 | Agent-on-orchestrator/IO-in-container; ATIF trajectory; Harbor verifies |
| Configuration | TOML/YAML config, env vars, flags | `flag`, config layering; provider gateway (Flue `configureProvider`) |
| Testing | Integration tests, mock provider/tools | Table-driven tests, `httptest` |
| Connectors docs | `docs/connectors/*.md` install recipes | Extension onboarding without dynamic plugin load |
| Release | GoReleaser, cross-compilation | Build systems, CI/CD |

---

### Phase 7: Self-Extension & Recursive Self-Build (Weeks 10–12)

The capability ladder that makes Harness extensible *by agents* (see *Metaengineering*). Build bottom-up so each rung is usable before the next exists.

| Module | What | Learn |
|--------|------|-------|
| `selfext/skills` | Rung 0: Markdown sub-programs, read at call time (Flue-style) | Frontmatter parsing, prompt assembly, hot-reload from disk |
| `tools/skill` | `Tool` that invokes a skill by name | Bridging prompt-level extension into the tool protocol |
| `selfext/monty` | Rung 1: spawn `monty-cli`, run throwaway sandboxed Python | Subprocess sandboxing, resource limits, external-fn bridges |
| `selfext/starlark` | Rung 2: load `.star` tool defs, dynamic register (`go.starlark.net`) | Embedding interpreters, hermetic execution, curated bridges |
| `cmd/harness new` | Scaffold a skill / starlark def / Go tool from a template | Code generation from templates, golden-path DX |
| `cmd/harness doctor` | Validator: naming, schema validity, banned imports, contract-test presence | Deterministic, teaching feedback beyond `go build` |
| (recursive self-build) | Agent writes a Go tool → `go build ./cmd/...` → exec new binary | The cold-path promotion loop; Go-as-self-modifiable substrate |

**Deliverable:** An agent can (a) drop a Markdown skill, (b) run a one-shot Monty program, (c) register a Starlark tool at runtime with no rebuild, and (d) promote a proven capability to a compiled Go `Tool` — each rung sandboxed and validated by `harness doctor` + the Go toolchain.

**Boring-language alignment (Jacob Young, *Use boring languages with LLMs*):** the agent's feedback substrate on every rung is the **one-right-way Go toolchain** — `gopls` (semantic), `go vet`, and `golangci-lint` — plus `harness doctor` and the mock-parity harness. These enforce conventions *without prompting the agent into compliance*. Watch the **Monty/Starlark variance caveat**: we keep Python/Starlark *syntax* (high model recall) but strip the ecosystem (no package managers, curated bridges); the further the subset drifts from the stdlib idioms the model expects (`requests`, `asyncio`, comprehensions), the more inconsistency we reintroduce. Keep bridges aligned to the most-reinforced idioms and lean on teaching errors.

---

## Key Design Decisions

### 1. Why Go, Not TypeScript (like pi-mono)

| Dimension | TypeScript (pi-mono) | Go (Harness) |
|-----------|---------------------|--------------|
| Startup | ~200ms (Node.js) | ~5ms |
| Memory | ~50MB baseline | ~5MB baseline |
| Concurrency | Event loop + Promises | Goroutines (M:N scheduling) |
| Deployment | node + node_modules | Single static binary |
| Type safety | Structural, runtime gaps | Interface-checked at compile time |
| Dependencies | npm (deep trees) | Go modules (flat, vendorable) |

### 2. Why Interfaces, Not a Plugin Registry

pi-mono uses `pi.registerTool()` — a runtime function that adds to a mutable array. This requires:
- A `pi` context object to exist
- Runtime type checking of the registered object
- Dynamic dispatch through the registry

Go interfaces are implicit. Any struct that has the right methods IS a Tool. No registration, no context object, no runtime checks. The compiler verifies it.

### 3. Why Immutable State, Not Mutable Mutators

pi-mono's `AgentState` has methods like `setModel()`, `setTools()`, `replaceMessages()` that mutate in-place. When tools execute concurrently and the TUI renders simultaneously, you get data races (Node.js avoids this via single-threaded event loop, but it constrains performance).

Go has real concurrency. Immutable snapshots are the safe, performant answer. The agent loop produces snapshots. The frontend consumes them. No locks on the hot path.

### 4. Why Content-Addressable Caching in Core

pi-mono has no built-in caching. The AST service doc in this project describes content-addressable caching as a separate cloud service. Harness puts it in core because:
- Every extension benefits (LLM responses, parsed ASTs, tool outputs)
- The interface is tiny (Get/Put/Has with a [32]byte key)
- It enables offline operation and instant re-analysis of unchanged files
- Merkle-tree change detection for incremental updates comes naturally

### 5. Why Analyzers Are a Separate Interface

Tools execute actions. Analyzers provide intelligence. The distinction matters:
- Tools are called by the LLM via tool-use protocol
- Analyzers are called by the harness to enrich context before sending to the LLM
- A tool says "read this file." An analyzer says "here are the 20 most important symbols in this codebase for this query."

The harness can use analyzers automatically (e.g., before every prompt, run treesitter on changed files and inject a repomap into context) — without the LLM explicitly requesting it.

### 6. Why Approver Is a Core Interface (Lesson from Claude Code)

Claude Code's permission system is its most battle-tested component — multi-layer classification (bash classifier, dangerous patterns, path validation, sed validation) with glob-based allow/deny rules. It works. But it lives in `utils/permissions/`, a utility directory that any extension can bypass.

Harness promotes safety from "convention" to "structure":
- The agent loop calls `Approver.Approve()` before every `Tool.Execute()`
- Extensions cannot bypass this — the interface is enforced by the loop, not by convention
- The Approver receives minimal context (`ApprovalRequest`: tool name, args, workspace, read-only flag) — NOT the entire application state
- Multiple Approvers compose via `ChainApprover`: rules first, then auto-classify, then interactive prompt

This is the single most important architectural difference from Claude Code. Safety is not a feature. It is the architecture.

### 7. Why Native Services Use Subprocess Adapters (Lesson from pi-mono + swe_distiller)

pi-mono can load a TypeScript extension that wraps `rg`, a Python script, or any CLI via `registerTool` + subprocess. Harness replaces runtime discovery with **explicit wiring**, but keeps the same boundary: the agent never embeds Rust/HTML/ML stacks inside the Go core.

- **`extensions/swe_distiller`** owns extraction quality (fetch, DOM, markdown, optional LLM). It ships as a standalone binary with a stable `--stdout` contract.
- **`cli/distiller`** owns agent semantics: JSON schema, content-addressable cache, truncation, `ReadOnly()`, Approver visibility, event bus.
- **No CGO/FFI** unless profiling proves subprocess overhead is dominant — subprocess + cache is simpler and matches how pi-mono wraps ripgrep.

New native capabilities follow the same template: implement in `extensions/<name>/`, expose a narrow CLI, add `harness/cli/<name>/` implementing `Tool`.

### 8. Lessons from claw-code (Testing, Sidecars, and Typed Policy)

claw-code's main gift to Harness is **operational honesty** and **machine-readable contracts**, not feature breadth.

**Adopt now (Phases 1–3):**

1. **Mock parity harness** — Deterministic fake provider + scripted scenarios (`streaming_text`, `read_file_roundtrip`, `write_file_denied`, `plugin_tool_roundtrip`, `auto_compact_triggered`). claw-code proves you can gate merges on behavioral diffs without live API keys. Harness should ship `test/integration/` with the same scenario manifest pattern as `claw-code/rust/mock_parity_scenarios.json`.

2. **Headless lean binary** — `claw-analog` separates "agent for humans" from "agent for pipes." Harness: `cmd/harness-headless` with `--output-format json`, explicit caps (turns, bytes, glob hits), and no bash — ideal for CI and for another agent calling Harness.

3. **Sidecar retrieval** — Indexing and embeddings live in `claw-rag-service`; the agent only HTTP-calls `retrieve_context`. This mirrors `extensions/swe_distiller` + `cli/distiller`: **never** pull RAG or HTML pipelines into `core/`.

4. **Permission profile + enforcer** — Tools declare required mode; `PermissionEnforcer` returns structured `Denied { tool, active_mode, required_mode, reason }`. Map to Harness: `Tool.RequiredMode()` + `Approver` chain; return structured denials the LLM can read (not opaque strings).

5. **Event bus as automation API** — claw-code docs (`docs/g004-events-reports-contract.md`) insist: if a typed event exists, do not infer state from terminal text. Harness `Event` payloads should carry `schema_version`, monotonic `seq`, and optional fingerprint for dedupe (lane_events pattern).

**Adopt in Phase 5–6:**

6. **Approval tokens** — Scoped, expiring, single-use grants for policy exceptions (distill internal URL, run write in read-only session). Stronger than a boolean "user said yes."

7. **ToolSearch / deferred tools** — When built-in + extension tools exceed context budget, expose search/select tool (claw `ToolSearch`) instead of dumping all schemas every turn.

8. **External hook scripts** — claw plugins run `PreToolUse` / `PostToolUse` as configured commands. Harness equivalent: optional `HookRunner` that execs allowlisted hook binaries with JSON stdin (complements Go `Hook` interfaces for in-repo logic).

9. **Versioned reports** — For compaction summaries and architecture reviews, use claim kinds + confidence + content hash (`report_schema.v1`) so downstream automation does not parse markdown prose.

**Do not adopt (scope trap):**

- Full **lane lifecycle** (`lane.started` → `lane.merged`) unless building team-based SWE orchestration.
- **Green contract** / merge-ready levels — CI concern, not agent core.
- Chasing **40/40 tool stubs** — Harness composes fewer, sharper tools + `extensions/*`.

```text
claw-code lesson map → Harness modules

  mock-anthropic-service     →  test/mockprovider + httptest
  mock_parity_scenarios.json →  test/integration/scenarios.json
  claw-analog                →  cmd/harness-headless
  claw-rag-service           →  extensions/* sidecar OR future harness-rag
  permission_enforcer        →  core/approver + tools.RequiredMode()
  approval_tokens            →  core/approval/grant.go (Phase 6)
  lane_events / report_schema→  core/bus/events.go (schema_version, seq)
  ToolSearch                 →  core/tool/registry_search.go (Phase 5)
  plugins/hooks (subprocess) →  core/hook/runner.go (optional, allowlisted)
```

### 9. Lessons from Flue (Product Harness on pi-mono)

Flue validates that **“harness framework”** is a distinct product from **“coding agent CLI.”** It implements the same mental model as this document using pi-mono under the hood; Go Harness should match Flue’s **API shape** where possible while keeping pi-mono’s loop semantics in native Go.

**Adopt now (Phases 1–4):**

1. **Three-level identity** — `InstanceID` (customer/repo/conversation) → named harness config (model, sandbox, cwd) → session thread. HTTP: `POST /agents/{agent}/{instanceID}`. Avoids overloading one “session” blob with filesystem and tenancy concerns.

2. **Sandbox interface with a cheap default** — Virtual/restricted execution for high-volume tools; `Local` for CI (`flue run` / `local()` pattern) with **env allowlist**; `Remote` via `extensions/*` + connectors docs. Do not require a container per request.

3. **Headless binary + hosted server** — `cmd/harness-headless` mirrors `flue run` (production-shaped one-shot, JSON payload). `cmd/harness-server` mirrors `flue dev`/`build` HTTP surface for integrators.

4. **Roles without history pollution** — Subagent/reviewer instructions are turn-scoped system overlays (`call > session > harness`), not appended as fake user messages (Flue `roles.ts`).

5. **Tasks with depth limit** — Child work shares workspace sandbox but gets isolated message history and optional `cwd` (Flue `MAX_TASK_DEPTH = 4`). Map to `SpawnChild` / internal `task` tool.

6. **Structured results** — Schema-validated returns via `finish`/`give_up` tools, not prose JSON (Flue `result.ts`). Critical for `harness-headless` and HTTP agents.

7. **Compaction with overflow retry** — `deriveCompactionDefaults` from model metadata; on context overflow, compact then retry once (Flue `compaction.ts`). Wire into `TokenBudget` + default `SummaryCompactor`.

**Adopt in Phase 5–6:**

8. **Run registry** — Assign `runID` per invocation; expose status and event tail for automation (Flue `run-registry.ts`, OpenAPI envelopes).

9. **Connector recipes** — Document `extensions/swe_distiller`, Daytona, etc. as markdown install guides (Flue `flue add` pattern) even when wiring stays compile-time in `main.go`.

10. **MCP bootstrap in trusted code only** — Connect remote MCP in server bootstrap; pass `[]Tool` into registry; never auto-spawn stdio MCP from agent turns (Flue v1 scope).

**Do not adopt (stack mismatch):**

- Runtime dependency on `@earendil-works/pi-agent-core` (Harness reimplements the loop in Go).
- Cloudflare-specific DO/session store unless deploying Harness to Workers.
- Full Flue CLI build graph (`flue build --target cloudflare`) — optional later target, not v1.

```text
Flue lesson map → Harness modules

  FlueContext.init()           →  core.Harness + Sandbox + ToolRegistry
  instance URL <id>            →  core/agent/instance.go
  harness.session(name)        →  core/agent/session.go + SessionStore
  session.prompt(..., result)  →  core/agent/result.go
  session.task()               →  core/agent/spawn.go
  just-bash / local()          →  core/sandbox/virtual.go, local.go
  flue run                     →  cmd/harness-headless
  flue dev HTTP                →  cmd/harness-server
  compaction.ts                →  core/agent/compact.go + budget.go
  configureProvider            →  provider/* + layered Config
  flue add connectors          →  docs/connectors/*.md
```

### 10. Composable Backends & Cloud Portability (Capability Ports)

The deployed harness must not be married to one cloud. Rather than an "AWS adapter" and a "GCP adapter" that each reimplement everything, Harness defines **narrow, behavior-first ports** in `core/platform` (`BlobStore`, `KVStore`, `Secrets`, `Queue`, `Lock`, `Retriever`) plus a `Runtime` capability descriptor. The core depends only on these interfaces; vendor SDKs live in `platform/{gcp,aws,cf,modal,turbopuffer,local}`.

Three principles:

- **Split workloads, not clouds.** The orchestrator (stateful, steady) and the sandbox (ephemeral, untrusted, FS+subprocess) have opposite shapes and belong in different homes. Cloudflare can host the former but never the latter — encode that with `Runtime{Subprocess:false}` and degrade archaeology to the `Remote` sandbox tier.
- **Buy bespoke only for hard capabilities.** Modal (`Sandbox`) and turbopuffer (`Retriever`) solve genuinely hard problems — secure agent-code execution and object-store vector+BM25 search — that we don't want to operate ourselves. They map onto existing ports 1:1 (Modal's Go SDK ≈ our `Sandbox`; turbopuffer's namespace-per-prefix ≈ per-repo index). Commodity concerns (blobs, KV) use whatever the host cloud provides.
- **Never let a vendor be load-bearing.** Every port ships a self-host adapter (`platform/local`: filesystem blobs, BoltDB KV, pgvector). The managed provider is a swap, not a foundation. This bounds **vendor sprawl** — the real risk — to "one afternoon to replace," not "rewrite the core."

This is the same inversion applied to infrastructure: the core calls *out* to platform ports exactly as it calls out to `Provider`/`Tool`/`Analyzer`. Cloud choice becomes a wiring decision in `main.go`, not an architectural commitment.

### 11. The Capability Ladder (Designed for Agent Extension)

Because the primary extenders are AI agents, the framework offers a **ladder of extension surfaces** rather than a single "write a Go plugin" path (see *Metaengineering: A Framework Agents Can Extend*). Agents author on the bottom rungs — Markdown **skills**, sandboxed **Monty** Python, runtime **Starlark** tool defs — with **no wiring and no rebuild**, and with mistakes contained by sandboxing. Only proven, recurring capabilities (rule of three) graduate to a compiled Go `Tool` on the top rung.

This reconciles with §2 ("Why Interfaces, Not a Plugin Registry"): §2 governs the **compiled** tier, where interface satisfaction is compiler-checked and there is no runtime registry. The runtime tiers (Starlark/Monty) deliberately *do* allow dynamic registration — but **sandboxed**, so the safety argument holds without dynamic in-process Go loading (the thing pi-mono's `jiti` import model gets wrong for a static binary).

For the rare cold-path promotion, wiring is **explicit in `main.go`** (most legible, fully compile-checked, least machinery). Codegen (`go:generate`) is deferred until compiled-extension count makes that churn a *measured* pain — building it now would be the accretion every reference critique warns against. A `harness new <kind>` scaffold and a `harness doctor` validator provide deterministic, teaching feedback across all rungs.

### 12. Lessons from Deep Agents (Composition, ACP Completeness, Harbor Evals)

Deep Agents (LangChain's middleware-over-LangGraph harness) is the survey's strongest *organizational* reference. We adopt its composition lessons and — per the integration focus — its protocol and eval adapter shapes, while rejecting its three-framework Python substrate (see *What Deep Agents Proves Works*).

**Adopt now:**

1. **Filesystem-as-context-overflow → content-addressable cache.** Offload summarized history and oversized tool results (Deep Agents evicts >20K-token results) to the cache, leaving a `sha256` handle. This is the concrete payoff of our content-addressable cache tenet (`OffloadCompactor`, §6).
2. **Path-routed composite backend.** A `CompositeBackend`-style router over the `Sandbox`/`Filesystem` interface (`/memories/` → store, workspace → sandbox) — the agent sees only tools; the backend decides where bytes live.
3. **Two-tier eval scoring.** `.success()` correctness assertions gate merges; `.expect()` efficiency metrics (step count, tool-call shape) are logged but never fail. Pairs with our deterministic mock-parity harness: parity tests gate (fast, offline), behavioral evals observe (statistical).

**ACP frontend — do the half Deep Agents skipped.** Their `AgentServerACP` is the right *shape* (thin adapter implementing `acp.Agent`, translating stream events → `session/update`, HITL interrupts → `session/request_permission`, model-switch reusing checkpointer+`thread_id`) but covers only the prompt-turn core. Because ACP's value is that **the editor is the environment**, our `frontend/acp` must:

- **Bridge the client's `fs/*` and `terminal/*` capabilities** when present — route file/shell ops through the *client's* filesystem/terminal (a `Sandbox` backed by ACP host capabilities), not the agent's own disk, so the editor owns the diff/permission surface. (Deep Agents brought its own backend — "only using half the protocol.")
- **Model the full tool-status lifecycle** (`pending → in_progress → completed/failed`), not just `pending → completed` — our typed `LoopState`/events map cleanly.
- **Emit the reasoning stream** (`agent_thought_chunk`) for thinking models.
- **Support real session lifecycle** (`session/load`, fork/resume/close) — our `SessionStore` + `InstanceID` + branchable SQLite tree make this *structurally easier* than Deep Agents' uuid-in-memory sessions (a place we're better, not just at parity).

**Harbor evals — thin external adapter.** Run Terminal-Bench 2.0 (and later SWE-bench) without entangling the core: a `eval/harbor` package where a Go `BaseAgent`-equivalent wraps the harness and a `HarborSandbox` satisfies Harbor's environment contract. Copy the proven split — **agent on the orchestrator, only tool I/O in the container** (read/grep/glob via shell `exec`; write/edit via native upload/download to dodge `ARG_MAX`) — write an ATIF `trajectory.json`, and **let Harbor's in-container verifier produce the reward** (don't self-grade). Keep observability behind our event bus; an *optional* subscriber posts rewards anywhere — never require a SaaS (Deep Agents' LangSmith coupling is the anti-lesson).

**Do not adopt:** the LangGraph→create_agent→deepagents dependency depth, asyncio/GIL concurrency, LangSmith-required evals/tracing, "trust the LLM" default (our `Approver` is core and on by default), or claiming MCP at the product layer while the core lacks it.

---

## Learning Map

This project teaches through building. Each module exercises specific engineering skills.

| Skill Domain | Modules | Concepts |
|-------------|---------|----------|
| **Go Fundamentals** | protocol, transport | Structs, interfaces, error handling, testing |
| **Concurrency** | bus, agent/loop | Goroutines, channels, `select`, `errgroup`, `context` |
| **Systems Programming** | process, sandbox, cli/runner, cli/distiller, archive | Sandbox tiers, subprocess adapters, env allowlists, JIT archive streaming |
| **Cloud Portability** | platform/*, core/platform, core/retrieve | Ports-and-adapters, capability negotiation, composable backends (Modal, turbopuffer) |
| **Self-Extension / Metaengineering** | selfext/{skills,monty,starlark}, cmd/harness new\|doctor | Embedded interpreters, sandboxed subprocess, capability ladder, low-variance API as corpus |
| **Product/API design** | instance, session, result, harness-server | Multi-tenant instance IDs, structured prompt results, HTTP agents |
| **Protocol Design** | protocol, transport, acp/* | JSON-RPC, framing, capability negotiation, ACP fs/terminal bridging |
| **Evaluation** | eval/{scorer,parity,harbor} | Two-tier scoring, deterministic parity gates, container-based Harbor evals |
| **State Machines** | agent/state, agent/loop | State transitions, immutable snapshots |
| **API Design** | interfaces, provider/* | Interface design, composition, dependency injection |
| **Caching** | cache | Content-addressable storage, LRU, disk persistence |
| **UI Architecture** | tui/* | Elm architecture, component composition, rendering |
| **HTTP/Streaming** | provider/* | SSE parsing, chunked transfer, connection pooling |
| **Code Analysis** | analyzers/* | AST parsing, graph algorithms, symbol resolution |
| **Observability** | (phase 6) | Structured logging, tracing, metrics |
| **Database** | (phase 6) | SQLite, migrations, atomic writes |

---

## Anti-Patterns: What NOT to Do

*Derived from analysis of Claude Code (~512K LoC). See `docs/CLAUDE_CODE_CRITIQUE.md` for the full evaluation.*

These are explicit guardrails for Harness development. Each anti-pattern was observed in a production system and represents a trap that seems reasonable at first.

### ❌ God files

Claude Code's `QueryEngine.ts` is 46K lines. `Tool.ts` is 29K lines. `commands.ts` is 25K lines. These files are impossible to review, test, or refactor incrementally.

**Harness rule:** No file exceeds 500 lines. If it does, it needs to be decomposed.

### ❌ God state

Claude Code's `AppState` has ~100+ fields covering agent state, UI state, bridge state, plugin state, MCP state, companion state, and tungsten state. Every component can read everything.

**Harness rule:** `AgentState` has ~10 fields. UI state lives in the frontend. Connection state lives in connection managers. State is per-concern, not per-application.

### ❌ Tool context as universe carrier

Claude Code's `ToolUseContext` carries ~60 fields: settings, permissions, state store, model, messages, MCP connections, plugins, file history, analytics. Every tool is coupled to every product concern.

**Harness rule:** `Tool.Execute()` receives `context.Context` and `json.RawMessage`. Nothing else. Tools are independent of the application.

### ❌ Circular dependencies solved by lazy require

Claude Code has `const getTeamCreateTool = () => require(...)` scattered throughout to break circular imports. This indicates the dependency graph has no enforced layering.

**Harness rule:** Go modules enforce unidirectional dependencies at compile time. The core never imports extensions. If a cycle appears, the abstraction is wrong.

### ❌ Feature flags as architecture

Claude Code has 20+ feature flags (`PROACTIVE`, `KAIROS`, `COORDINATOR_MODE`, `VOICE_MODE`, `BUDDY`, `ULTRAPLAN`, etc.) creating code paths that may or may not exist at runtime. This is a combinatorial testing problem.

**Harness rule:** Features are Go modules. They're compiled in or they're not. No runtime conditionals for feature existence.

### ❌ Safety boundary in utility layer

Claude Code's permission system lives in `utils/permissions/` — a utility directory. Extensions can bypass it by calling tool execution directly.

**Harness rule:** The Approver is a core interface. The agent loop calls `Approver.Approve()` before `Tool.Execute()`. It cannot be bypassed. Safety is structural, not conventional.

### ❌ Analytics woven into agent logic

Claude Code has GrowthBook feature flags in the query engine, Datadog in the API client, OpenTelemetry spans around tool execution, first-party event logging in permission decisions.

**Harness rule:** The core emits events to the bus. An analytics extension subscribes. The core has zero knowledge of analytics providers.

### ❌ Frontend coupled to agent logic

Claude Code's `REPL.tsx` is ~5000 lines mixing React state, agent loop orchestration, permission handling, bridge management, and rendering. The TUI IS the application.

**Harness rule:** The frontend implements a 1-method interface (`Run()`). It reads state snapshots and emits user input. It has no access to the agent loop. Swap TUI for web UI without touching agent logic.

### ❌ React for terminal rendering

Claude Code maintains a ~15K-line custom Ink fork (layout engine, reconciler, event system, hit testing). This is extraordinary machinery for rendering text.

**Harness rule:** Bubble Tea's Elm architecture (Model → Update → View) achieves equivalent results. No virtual DOM, no Yoga flexbox, no React reconciler.

---

## References

### Architecture Inspiration
- [pi-mono](https://github.com/badlogic/pi-mono) — The TypeScript agent framework this design evolves from. Extension docs: [extensions.md](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md). Subprocess tool example: [truncated-tool.ts](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/examples/extensions/truncated-tool.ts).
- [extensions/swe_distiller/ARCHITECTURE.md](../extensions/swe_distiller/ARCHITECTURE.md) — Native extraction service wrapped by `cli/distiller`
- [claw-code](../claw-code/) — Rust agent port in this repo: mock parity harness, permission modes, typed events/reports, RAG sidecar, lean `claw-analog`. See `claw-code/rust/PARITY.md`, `claw-code/rust/MOCK_PARITY_HARNESS.md`, `claw-code/docs/g004-events-reports-contract.md`.
- [flue](../flue/) — Local reference copy (gitignored) of [withastro/flue](https://github.com/withastro/flue): headless harness on pi-mono, sandbox tiers, instance/harness/session API, structured `prompt` results, `flue run` / HTTP deploy. Read `flue/README.md`, `flue/packages/runtime/src/session.ts`, `flue/packages/runtime/src/compaction.ts`, `flue/packages/runtime/src/result.ts`.
- [Deep Agents](https://github.com/langchain-ai/deepagents) — LangChain's middleware-over-LangGraph harness; the survey's strongest *organizational* reference (composition, pluggable backends, FS-as-overflow, thin ACP/Harbor adapters). Local checkout at `../deepagents`. See `docs/DEEPAGENTS_DEEP_DIVE.md` and `docs/DEEPAGENTS_CRITIQUE.md`.
- [Claude Code](https://github.com/anthropics/claude-code) — Anthropic's production agentic CLI. Studied for permission system, tool orchestration, compaction, and startup patterns. See `docs/CLAUDE_CODE_CRITIQUE.md` and `docs/CLAUDE_DEEP_DIVE.md`.
- [Aider](https://github.com/Aider-AI/aider) — Repository map and AST-based code analysis patterns
- [OpenAI Codex CLI](https://github.com/openai/codex) — Agent-client protocol design

### Go
- [Effective Go](https://go.dev/doc/effective_go)
- [Uber Go Style Guide](https://github.com/uber-go/guide/blob/master/style.md)
- [100 Go Mistakes](https://100go.co/)
- [Concurrency in Go](https://www.oreilly.com/library/view/concurrency-in-go/9781491941294/)
- [gopls](https://pkg.go.dev/golang.org/x/tools/gopls) · [golangci-lint](https://golangci-lint.run/) — the one-right-way feedback substrate for agent-authored Go

### Self-Extension & Agent Ergonomics
- [Use boring languages with LLMs](https://jry.io/writing/use-boring-languages-with-llms/) (Jacob Young) — low-variance, strong-convention ecosystems produce reliable agent output; the external grounding for the Go bet and the capability ladder.
- [docs/PLAN.md](./PLAN.md) — authoritative self-extension model (Starlark/Monty/Go tiers, rule of three, recursive self-build) that the Metaengineering section reconciles into this plan.
- [go.starlark.net](https://pkg.go.dev/go.starlark.net/starlark) — embedding Starlark (Rung 2 tool defs); [Starlark spec](https://github.com/bazelbuild/starlark/blob/master/spec.md).
- [Monty](https://github.com/pydantic/monty) — sandboxed Python subset via subprocess (Rung 1 hot-path code).

### Protocol & Evals
- [Agent Client Protocol (ACP)](https://agentclientprotocol.com/protocol/overview) — JSON-RPC/stdio protocol where the editor *is* the environment; [prompt-turn lifecycle](https://agentclientprotocol.com/protocol/prompt-turn). Backs the `acp/*` frontend; bridge client `fs/*`+`terminal/*` (the half Deep Agents skipped).
- [Harbor](https://www.harborframework.com/docs) — container-based agent evaluation (Terminal-Bench 2.0 successor). Backs `eval/harbor`: agent on orchestrator, tool I/O in container, Harbor verifier produces the reward.

### TUI
- [Bubble Tea](https://github.com/charmbracelet/bubbletea)
- [Lip Gloss](https://github.com/charmbracelet/lipgloss)
- [Glamour](https://github.com/charmbracelet/glamour)

### Code Analysis
- [Tree-sitter](https://tree-sitter.github.io/tree-sitter/)
- [Zoekt](https://github.com/sourcegraph/zoekt)
- [ast-grep](https://ast-grep.github.io/)
- [Semgrep](https://semgrep.dev/)

### Systems
- [Designing Data-Intensive Applications](https://dataintensive.net/) (Kleppmann)
- [The Linux Programming Interface](https://man7.org/tlpi/) (Kerrisk)
- [Systems Performance](https://www.brendangregg.com/systems-performance-2nd-edition-book.html) (Gregg)

### Cloud & Composable Backends
- [Modal Sandboxes](https://modal.com/docs/guide/sandbox) — secure containers for agent code; gVisor exec, FS, snapshots, Go SDK. Backs the `Sandbox` Remote tier (`platform/modal`).
- [turbopuffer architecture](https://turbopuffer.com/architecture) — vector + BM25 + metadata search on object storage; namespace-per-prefix. Backs the `Retriever` port (`platform/turbopuffer`).
- [Ports & Adapters (Hexagonal Architecture)](https://alistair.cockburn.us/hexagonal-architecture/) (Cockburn) — the pattern behind `core/platform` + `platform/*`.
- [docs/AST_SERVICE_ARCHITECTURE.md](./AST_SERVICE_ARCHITECTURE.md) — object-store-backed AST/embeddings; retrieval tier that turbopuffer can serve.
