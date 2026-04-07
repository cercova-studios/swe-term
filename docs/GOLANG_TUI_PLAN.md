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
| Extension system | `pi.registerTool()`, `pi.on()` | Go interfaces (implicit satisfaction) |
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
└─────────────────────────────────────────────────────────────────┘
```

### The Inversion

pi-mono builds up: `pi-ai` is a library, `pi-agent-core` wraps it, `pi-coding-agent` wraps that, `pi-tui` renders it.

Harness inverts: the **core is the orchestrator**. Providers, tools, frontends, and analyzers all plug into it. Nothing wraps the core — the core calls out to extensions via interfaces.

This means:
- Swap `OpenAI` for `Ollama` by changing one line
- Replace the TUI with a web frontend without touching agent logic
- Add a new tool by implementing a 3-method interface
- Integrate an AST analyzer as a specialized tool

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
```

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
│   │   ├── session.go               # Session lifecycle, persistence
│   │   ├── budget.go                # Token budget tracking, compaction triggers
│   │   ├── compact.go               # Compaction interface + memory extraction hooks
│   │   └── spawn.go                 # Child agent spawning (multi-agent)
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
├── cli/                             # CLI tool plugins
│   ├── go.mod                       # module github.com/user/harness/cli
│   ├── plugin.go                    # CLI plugin interface + runner
│   ├── xsearch/
│   │   └── xsearch.go              # X API search — implements Tool
│   ├── webmd/
│   │   └── webmd.go                 # Web→Markdown (like defuddle) — implements Tool
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
│   ├── harness/                     # Main agent CLI
│   │   └── main.go                  # Wire core + providers + tools + TUI → run
│   └── harness-server/              # Headless RPC server
│       └── main.go                  # Wire core + providers + tools → gRPC/HTTP
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
    └── examples/
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
    "github.com/user/harness/cli/webmd"
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
            webmd.New(),
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

**Verification:**
```bash
echo '{"prompt": "Hello"}' | go run ./cmd/harness/
# Streams response tokens to stdout
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
| `cli/xsearch` | X API search as a tool | OAuth, API clients, pagination |
| `cli/webmd` | Web→Markdown extractor | HTML parsing, readability algorithms |
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
| Persistence | Session save/restore, SQLite history | Atomic writes, migrations |
| Configuration | TOML/YAML config, env vars, flags | `flag`, config layering |
| Testing | Integration tests, mock provider/tools | Table-driven tests, `httptest` |
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

---

## Learning Map

This project teaches through building. Each module exercises specific engineering skills.

| Skill Domain | Modules | Concepts |
|-------------|---------|----------|
| **Go Fundamentals** | protocol, transport | Structs, interfaces, error handling, testing |
| **Concurrency** | bus, agent/loop | Goroutines, channels, `select`, `errgroup`, `context` |
| **Systems Programming** | process, transport | Subprocess mgmt, pipes, signals, IPC |
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
- [pi-mono](https://github.com/badlogic/pi-mono) — The TypeScript agent framework this design evolves from
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
