# swe-term: Master Build Plan

> A polyglot coding agent built for learning Go, Rust, and agent architecture.
> Porting the architecture of [pi-mono](https://github.com/badlogic/pi-mono) — without its gaps.

---

## Design Thesis

Three languages, three strengths, zero overlap:

| Language | Role | Why |
|----------|------|-----|
| **Go** | Orchestration, agent loop, providers, session, CLI | Fast compile, goroutines, single binary, simple enough for LLMs to self-modify |
| **Rust** | TUI renderer, tokenizer | Performance-critical rendering and byte-level text processing |
| **Starlark** | Agent-authored tool definitions | Python syntax LLMs already know, safe execution, Go-native via `go.starlark.net` |
| **Monty** | Agent-generated runtime code | Real Python subset (async, dataclasses), sandboxed, via `monty-cli` subprocess |

### Self-Extension Model

```
Hot path:  Agent needs capability NOW → writes Python → Monty executes → throwaway
Cold path: Agent notices recurring gap → writes Go tool → `go build` (~3s) → restart → permanent
```

Rule of three: first time ad-hoc (Monty), second time notice, third time promote to Go.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTENDS                               │
│  ┌──────────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ TUI (Rust)   │  │ RPC      │  │ Print    │  │ Web (fut) │  │
│  │ ratatui      │  │ Server   │  │ stdout   │  │           │  │
│  └──────┬───────┘  └────┬─────┘  └────┬─────┘  └─────┬─────┘  │
│         └───────────────┴──────┬──────┴───────────────┘        │
│                          Frontend interface (Go)                │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────┴────────────────────────────────┐
│                        CORE (Go, zero external deps)            │
│                                                                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌─────────────┐ │
│  │ Agent     │  │ Event     │  │ Session   │  │ Config      │ │
│  │ Loop      │  │ Bus       │  │ Store     │  │             │ │
│  │           │  │           │  │           │  │ Single      │ │
│  │ - turns   │  │ - typed   │  │ - SQLite  │  │ struct,     │ │
│  │ - tools   │  │ - chans   │  │ - tree    │  │ layered     │ │
│  │ - steer   │  │ - ctx     │  │ - compact │  │ precedence  │ │
│  │ - follow  │  │ - backpr  │  │ - branch  │  │             │ │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └──────┬──────┘ │
│        └──────────────┴──────┬───────┴───────────────┘         │
│                              │                                  │
│                     Protocol Types                              │
│                     Messages, content blocks, tool calls        │
│                     context.Context throughout                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                    Provider / Tool / Hook interfaces
                               │
┌──────────────────────────────┴──────────────────────────────────┐
│                       EXTENSIONS                                │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LLM Providers (Go)                                     │   │
│  │  OpenAI │ Anthropic │ Google │ Ollama │ OpenRouter       │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Tools (Go)                                              │   │
│  │  Read │ Write │ Edit │ Bash │ Grep │ Find │ Ls           │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Hooks (Go)                                              │   │
│  │  PreToolUse │ PostToolUse │ PrePrompt │ OnCompact │ ...  │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Self-Extension Layer                                    │   │
│  │  Starlark (tool defs) │ Monty (runtime code)            │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Native Components (Rust, via FFI or subprocess)         │   │
│  │  swe-tui (ratatui) │ swe-tokenizer (tiktoken)           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## pi-mono Gap Analysis (Don't Port These)

Gaps identified from deep analysis of pi-mono's codebase. These inform what to do differently, not what to copy.

| # | pi-mono Gap | swe-term Fix |
|---|-------------|--------------|
| 1 | `AgentSession` is 3268-line god object | Decompose: `Session`, `Compaction`, `ToolRegistry`, `RetryPolicy`, `HookRunner` |
| 2 | Extension system requires JS dynamic `import()` | Starlark for tool defs (Go-native), hooks as Go interfaces |
| 3 | Provider compat via URL substring detection (13 flags) | Clean `Provider` interface, per-provider implementations |
| 4 | JSONL append-only session with tree walking (no index) | SQLite via `modernc.org/sqlite` (pure Go, no CGo) |
| 5 | Token estimation uses `chars/4` heuristic | `tiktoken-go` or Rust tokenizer via subprocess |
| 6 | No interfaces between layers (all concrete classes) | Go implicit interfaces at consumer site |
| 7 | `AbortSignal` pattern everywhere | `context.Context` (native Go) |
| 8 | `Promise.all` for parallel tools | `errgroup` + goroutines |
| 9 | Convention-based error contracts (JSDoc) | Go `error` return values (enforced by compiler) |
| 10 | Config scattered across 5+ sources | Single `Config` struct, layered precedence: defaults → file → env → flags |
| 11 | Custom TUI renderer (2900 lines) | Rust `ratatui` (or Go Bubble Tea for phase 1) |
| 12 | `any` types despite rules against them | `json.RawMessage` for dynamic data, typed structs everywhere else |

---

## Phases

### Phase 1: Core Agent Loop (Go)

**Goal:** Send a prompt to an LLM, get a streaming response, execute tool calls, loop until done. Stdout as the frontend.

**ACP Foundation:** Protocol types in `internal/protocol/` MUST align with [Agent Client Protocol](https://agentclientprotocol.com/) field names and semantics. This is free now and prevents a full refactor when we add ACP compliance in Phase 5. Use JSON-RPC 2.0 framing for all IPC. Don't implement: `initialize` handshake, capability negotiation, auth, `session/load` — those come later.

**Modules:**

```
swe-term/
├── go.work
├── cmd/
│   └── swe-term/
│       └── main.go              # CLI entry point
├── internal/
│   ├── protocol/
│   │   ├── messages.go          # UserMessage, AssistantMessage, ToolResultMessage
│   │   ├── content.go           # ContentBlock (text, image, resource — ACP-aligned)
│   │   ├── events.go            # session/update types (agent_message_chunk, tool_call, tool_call_update)
│   │   ├── jsonrpc.go           # JSON-RPC 2.0 request/response/notification framing
│   │   ├── toolcall.go          # ToolCall with ACP status lifecycle (pending/in_progress/completed/failed)
│   │   └── usage.go             # Token usage, cost calculation
│   ├── agent/
│   │   ├── loop.go              # Main agent loop: prompt → stream → tools → repeat
│   │   ├── state.go             # AgentState (immutable snapshots)
│   │   └── queue.go             # Steering + follow-up channels
│   ├── provider/
│   │   ├── provider.go          # Provider interface
│   │   ├── registry.go          # Model registry
│   │   ├── anthropic.go         # Anthropic Messages API
│   │   └── openai.go            # OpenAI Chat Completions
│   ├── tool/
│   │   ├── tool.go              # Tool interface
│   │   ├── registry.go          # Tool registry
│   │   ├── read.go              # Read file
│   │   ├── write.go             # Write file
│   │   ├── edit.go              # Edit file (search/replace)
│   │   ├── bash.go              # Shell execution
│   │   ├── grep.go              # Grep/ripgrep
│   │   └── find.go              # Find files (glob)
│   ├── session/
│   │   ├── store.go             # SessionStore interface
│   │   ├── sqlite.go            # SQLite implementation
│   │   ├── memory.go            # In-memory implementation (tests)
│   │   └── tree.go              # Branch/fork/navigate logic
│   ├── config/
│   │   └── config.go            # Single Config struct, layered loading
│   └── hook/
│       ├── hook.go              # Hook interface (PreToolUse, PostToolUse, etc.)
│       └── runner.go            # HookRunner — dispatches events to registered hooks
└── pkg/
    └── jsonschema/
        └── schema.go            # JSON Schema for tool parameters
```

**Core Interfaces (Phase 1):**

```go
// Provider streams LLM completions
type Provider interface {
    Stream(ctx context.Context, model Model, msgs []Message, tools []ToolDef) (<-chan StreamEvent, error)
}

// Tool executes actions in the environment
type Tool interface {
    Name() string
    Description() string
    Schema() json.RawMessage
    Execute(ctx context.Context, id string, args json.RawMessage) (*ToolResult, error)
}

// SessionStore persists conversation state
type SessionStore interface {
    Append(entry SessionEntry) error
    Branch(fromID string) error
    GetBranch(leafID string) ([]SessionEntry, error)
    GetContext() ([]Message, error)
}

// Hook intercepts agent lifecycle events
type Hook interface {
    Name() string
    On(event HookEvent) (HookResult, error)
}
```

**Verification:**
```bash
echo "list the files in the current directory" | go run ./cmd/swe-term/
# Agent calls bash tool, returns file listing, stops
```

**What you'll learn:**
- Go project structure, modules, workspace
- Interfaces, structs, error handling
- `context.Context` for cancellation/timeout
- Goroutines + channels for streaming
- HTTP clients + SSE parsing (provider)
- `os/exec` for bash tool
- `modernc.org/sqlite` for session persistence

---

### Phase 2: Hooks + Compaction (Go)

**Goal:** Lifecycle hooks for extensibility. Context compaction when approaching window limits.

**Modules:**

```
internal/
├── hook/
│   ├── hook.go                  # Hook interface
│   ├── runner.go                # Dispatches events, collects results
│   └── builtin/
│       └── compaction.go        # Auto-compaction hook
├── compaction/
│   ├── compaction.go            # Summarize old context via LLM
│   ├── tokencount.go            # Token counting (tiktoken-go or Rust subprocess)
│   └── cutpoint.go              # Find valid cut points in message history
```

**Hook Events (modeled after Claude Code hooks + pi-mono):**

```go
type HookEvent struct {
    Type      string          // "pre_tool_use", "post_tool_use", "pre_prompt", "on_compact", ...
    ToolName  string          // for tool events
    Args      json.RawMessage // tool arguments
    Result    json.RawMessage // tool result (post only)
    Messages  []Message       // current context (for on_compact)
}

type HookResult struct {
    Block   bool              // prevent tool execution
    Reason  string            // why blocked
    Replace json.RawMessage   // replace args or result
}
```

**What you'll learn:**
- Observer pattern in Go
- `errgroup` for parallel hook execution
- Token counting strategies
- LLM-as-a-service for summarization

---

### Phase 3: Starlark Self-Extension (Go + Starlark)

**Goal:** Agent can define new tools at runtime by writing Starlark scripts.

**Modules:**

```
internal/
├── starlark/
│   ├── loader.go                # Load .star files, register as tools
│   ├── runtime.go               # Starlark execution environment
│   └── bridge.go                # Expose Go functions to Starlark (file I/O, HTTP, etc.)
```

**How it works:**

1. Agent writes a `.star` file defining a tool:
```python
# tools/parse_yaml.star
def execute(args):
    content = read_file(args["path"])
    return {"parsed": yaml_parse(content)}

tool = {
    "name": "parse_yaml",
    "description": "Parse a YAML file and return structured data",
    "schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"]
    }
}
```

2. Go host loads it, registers as a `Tool`, agent can now call `parse_yaml`

**What you'll learn:**
- Embedding scripting languages in Go (`go.starlark.net`)
- Sandboxed execution
- Dynamic tool registration
- Bridge functions between host and guest

---

### Phase 4: Monty Runtime Code (Go + Monty subprocess)

**Goal:** Agent can write and execute Python programs for complex one-shot tasks.

**Modules:**

```
internal/
├── monty/
│   ├── executor.go              # Spawn monty-cli, send code, handle suspension
│   ├── bridge.go                # External functions the Python code can call
│   └── sandbox.go               # Resource limits, timeout enforcement
```

**How it works:**

```go
result, err := monty.Execute(ctx, monty.Request{
    Code: `
data = fetch(url)
parsed = json_parse(data)
return [item["name"] for item in parsed["results"]]
    `,
    Inputs: map[string]any{"url": targetURL},
    ExternalFunctions: map[string]monty.Func{
        "fetch":      httpFetchBridge,
        "json_parse": jsonParseBridge,
    },
    Limits: monty.Limits{MaxDuration: 5 * time.Second},
})
```

**What you'll learn:**
- Subprocess management from Go
- JSON-based IPC
- Suspension/resume protocol
- Resource limit enforcement

---

### Phase 5: TUI (Rust)

**Goal:** Full terminal UI replacing stdout, built with `ratatui`.

**Modules:**

```
crates/
└── swe-tui/
    ├── Cargo.toml
    ├── src/
    │   ├── main.rs              # Entry point, event loop
    │   ├── app.rs               # App state machine
    │   ├── ui.rs                # Layout and rendering
    │   ├── input.rs             # Key handling, editor
    │   ├── protocol.rs          # JSON protocol with Go host
    │   ├── components/
    │   │   ├── chat.rs          # Message display
    │   │   ├── editor.rs        # Multi-line input
    │   │   ├── status.rs        # Footer/status bar
    │   │   ├── tool_output.rs   # Tool result rendering
    │   │   └── markdown.rs      # Markdown rendering
    │   └── theme.rs             # Theming system
```

**Communication with Go host:**
- Rust TUI spawned as subprocess by Go
- JSON-RPC over stdin/stdout
- Go sends events (stream deltas, tool results), Rust renders
- Rust sends user input back to Go

**What you'll learn:**
- Rust ownership, borrowing, lifetimes
- `ratatui` immediate-mode rendering
- `crossterm` terminal manipulation
- Async Rust with `tokio`
- IPC protocol design

---

### Phase 6: Tokenizer (Rust)

**Goal:** Accurate token counting for compaction and cost estimation.

**Modules:**

```
crates/
└── swe-tokenizer/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs               # Core tokenizer
    │   ├── main.rs              # CLI: echo "text" | swe-tokenizer --model gpt-4
    │   └── models.rs            # Model-specific BPE vocabularies
```

**Integration:** Go calls `swe-tokenizer` as subprocess or via CGo (compile as `staticlib`).

---

### Phase 7: Recursive Self-Improvement

**Goal:** Agent can modify its own Go source, rebuild, and restart with new capabilities.

**Modules:**

```
internal/
├── selfbuild/
│   ├── builder.go               # go build orchestration
│   ├── state.go                 # Serialize/restore agent state across restart
│   └── hook.go                  # RecursiveBuildHook — the restart lifecycle
```

**The loop:**
1. Agent identifies recurring capability gap
2. Agent writes new Go tool (`.go` file in `internal/tool/`)
3. `RecursiveBuildHook` serializes current session state to SQLite
4. `go build ./cmd/swe-term/` (~2-5 seconds)
5. Current process `exec()`s the new binary with `--resume <session-id>`
6. New binary loads session state, continues where it left off

**What you'll learn:**
- `syscall.Exec` for process replacement
- State serialization/deserialization
- Build system integration
- Graceful shutdown and resume

---

## Key Dependencies

### Go
| Package | Purpose | URL |
|---------|---------|-----|
| `go.starlark.net` | Starlark interpreter | https://github.com/google/starlark-go |
| `modernc.org/sqlite` | Pure Go SQLite (no CGo) | https://gitlab.com/cznic/sqlite |
| `github.com/tiktoken-go/tokenizer` | Token counting | https://github.com/tiktoken-go/tokenizer |

### Rust
| Crate | Purpose | URL |
|-------|---------|-----|
| `ratatui` | TUI framework | https://github.com/ratatui/ratatui |
| `crossterm` | Terminal backend | https://github.com/crossterm-rs/crossterm |
| `tokio` | Async runtime | https://github.com/tokio-rs/tokio |
| `serde` / `serde_json` | Serialization | https://github.com/serde-rs/serde |
| `tiktoken-rs` | Tokenizer | https://github.com/zurawiki/tiktoken-rs |

### External
| Tool | Purpose | URL |
|------|---------|-----|
| `monty-cli` | Python sandbox | https://github.com/pydantic/monty |

---

## Learning Resources

### Go (start here)

**Fundamentals:**
- [A Tour of Go](https://go.dev/tour/) — interactive intro, do this first
- [Effective Go](https://go.dev/doc/effective_go) — idiomatic patterns
- [Go by Example](https://gobyexample.com/) — practical cookbook
- [Learn Go with Tests](https://quii.gitbook.io/learn-go-with-tests/) — TDD approach

**Intermediate:**
- [100 Go Mistakes](https://100go.co/) — common pitfalls (read during Phase 1-2)
- [Concurrency in Go](https://www.oreilly.com/library/view/concurrency-in-go/9781491941294/) — goroutines, channels, patterns
- [Uber Go Style Guide](https://github.com/uber-go/guide/blob/master/style.md) — production conventions

**Reference codebases to study:**
- [charm/bubbletea](https://github.com/charmbracelet/bubbletea) — Elm architecture in Go (TUI patterns)
- [ollama/ollama](https://github.com/ollama/ollama) — LLM provider implementation in Go
- [sourcegraph/zoekt](https://github.com/sourcegraph/zoekt) — code search in Go
- [google/starlark-go](https://github.com/google/starlark-go) — embedding Starlark (study for Phase 3)

### Rust (start at Phase 5)

**Fundamentals:**
- [The Rust Book](https://doc.rust-lang.org/book/) — chapters 1-10 before writing code
- [Rustlings](https://rustlings.cool/) — exercises for ownership/borrowing
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/) — practical patterns

**TUI-specific:**
- [Ratatui Book](https://ratatui.rs/introduction/) — official guide
- [Ratatui examples](https://github.com/ratatui/ratatui/tree/main/examples) — study before Phase 5
- [Async Rust Book](https://rust-lang.github.io/async-book/) — for tokio integration

**Reference codebases:**
- [gitui](https://github.com/extrawurst/gitui) — full TUI app in Rust (ratatui)
- [bat](https://github.com/sharkdp/bat) — terminal rendering, syntax highlighting
- [delta](https://github.com/dandavison/delta) — terminal diff rendering

### Starlark

- [Starlark Language Spec](https://github.com/bazelbuild/starlark/blob/master/spec.md) — the language
- [go.starlark.net docs](https://pkg.go.dev/go.starlark.net/starlark) — Go embedding API
- [Bazel Starlark Guide](https://bazel.build/rules/language) — practical usage patterns

### Monty

- [Monty README](https://github.com/pydantic/monty) — API and usage
- [Pydantic blog post](https://pydantic.dev/articles/pydantic-monty) — design rationale
- [Talk Python episode #541](https://talkpython.fm/episodes/show/541/monty-python-in-rust-for-ai) — deep dive interview

### Agent Architecture & Protocols

- [Agent Client Protocol (ACP)](https://agentclientprotocol.com/) — the protocol we align with for agent↔client communication
- [ACP OpenAPI Schema](https://agentclientprotocol.com/api-reference/openapi.json) — machine-readable protocol types (use to generate Go structs)
- [ACP Rust SDK](https://agentclientprotocol.com/libraries/rust.md) — reference for the Rust TUI side
- [pi-mono source](https://github.com/badlogic/pi-mono) — the reference implementation we're porting from
- [Claude Code hooks docs](https://code.claude.com/docs/en/hooks) — hook lifecycle patterns
- [Anthropic tool use docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — tool calling protocol
- [OpenAI function calling](https://platform.openai.com/docs/guides/function-calling) — OpenAI's approach

### Systems & Architecture

- [Designing Data-Intensive Applications](https://dataintensive.net/) — Kleppmann (session storage, state machines)
- [The Linux Programming Interface](https://man7.org/tlpi/) — Kerrisk (process management, signals)

---

## Phase 1 Checklist

Start here. Each item is a single PR-sized unit of work.

- [ ] **1.1** Initialize Go workspace: `go.work`, `cmd/swe-term/main.go`, `internal/` skeleton
- [ ] **1.2** Define protocol types aligned with ACP: `ContentBlock`, `ToolCall` (with status lifecycle), `SessionUpdate`, JSON-RPC 2.0 framing
- [ ] **1.3** Implement `Provider` interface + Anthropic provider (streaming SSE)
- [ ] **1.4** Implement basic agent loop: prompt → stream → collect response → print
- [ ] **1.5** Add tool calling: parse tool calls from response, dispatch, return results, loop
- [ ] **1.6** Implement `Read` tool (read file, return contents)
- [ ] **1.7** Implement `Bash` tool (execute command, capture output)
- [ ] **1.8** Implement `Write` and `Edit` tools
- [ ] **1.9** Implement `Grep`, `Find`, `Ls` tools
- [ ] **1.10** Implement `SessionStore` interface + SQLite implementation
- [ ] **1.11** Implement `Config` struct with layered loading (defaults → `~/.swe-term/config.json` → env → flags)
- [ ] **1.12** Implement system prompt builder (inject tool descriptions, context files)
- [ ] **1.13** Add `context.Context` propagation throughout (cancellation, timeout)
- [ ] **1.14** Add `errgroup` for parallel tool execution
- [ ] **1.15** Integration test: end-to-end prompt → tool use → response with mock provider

**Definition of done for Phase 1:**
```bash
export ANTHROPIC_API_KEY=sk-...
echo "Read the file go.mod and tell me what module this is" | go run ./cmd/swe-term/
# Agent calls read tool, reads go.mod, responds with module name
```
