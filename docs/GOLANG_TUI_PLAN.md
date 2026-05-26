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
├── tools/                           # Built-in tool extensions
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
- **ChainCompactor**: extract memories THEN summarize THEN truncate

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

**Deliverable:** Full interactive TUI that renders agent state from immutable snapshots.

### Phase 5: CLI Plugins + Analyzers (Weeks 7–9)

The extensions that make this an **agent harness**, not just an agent.

| Module | What | Learn |
|--------|------|-------|
| `cli/runner` | Shared subprocess + truncation for native CLIs | `process.Runner`, argv-only spawn, output limits |
| `cli/distiller` | Go `Tool` adapter for `extensions/swe_distiller` | Polyglot extension pattern (pi `registerTool` → compile-time `Tool`) |
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
| HTTP agents | `harness-server`: `/agents/{name}/{instanceID}`, run registry | REST design, idempotent instance routing |
| Structured API results | `finish`/`give_up` on webhook agents | JSON Schema validation, CI-friendly responses |
| Configuration | TOML/YAML config, env vars, flags | `flag`, config layering; provider gateway (Flue `configureProvider`) |
| Testing | Integration tests, mock provider/tools | Table-driven tests, `httptest` |
| Connectors docs | `docs/connectors/*.md` install recipes | Extension onboarding without dynamic plugin load |
| Release | GoReleaser, cross-compilation | Build systems, CI/CD |

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

---

## Learning Map

This project teaches through building. Each module exercises specific engineering skills.

| Skill Domain | Modules | Concepts |
|-------------|---------|----------|
| **Go Fundamentals** | protocol, transport | Structs, interfaces, error handling, testing |
| **Concurrency** | bus, agent/loop | Goroutines, channels, `select`, `errgroup`, `context` |
| **Systems Programming** | process, sandbox, cli/runner, cli/distiller | Sandbox tiers, subprocess adapters, env allowlists |
| **Product/API design** | instance, session, result, harness-server | Multi-tenant instance IDs, structured prompt results, HTTP agents |
| **Protocol Design** | protocol, transport | JSON-RPC, framing, capability negotiation |
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
- [Claude Code](https://github.com/anthropics/claude-code) — Anthropic's production agentic CLI. Studied for permission system, tool orchestration, compaction, and startup patterns. See `docs/CLAUDE_CODE_CRITIQUE.md` and `docs/CLAUDE_DEEP_DIVE.md`.
- [Aider](https://github.com/Aider-AI/aider) — Repository map and AST-based code analysis patterns
- [OpenAI Codex CLI](https://github.com/openai/codex) — Agent-client protocol design

### Go
- [Effective Go](https://go.dev/doc/effective_go)
- [Uber Go Style Guide](https://github.com/uber-go/guide/blob/master/style.md)
- [100 Go Mistakes](https://100go.co/)
- [Concurrency in Go](https://www.oreilly.com/library/view/concurrency-in-go/9781491941294/)

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
