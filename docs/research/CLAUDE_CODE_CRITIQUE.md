# Critical Evaluation of Claude Code as an Agentic Harness

> Evaluated against the Harness (swe-term) philosophy:
> zero-dependency core, interface-driven plugins, immutable state,
> goroutine-native concurrency, content-addressable caching, single binary.

---

## Verdict Summary

Claude Code is an impressively **feature-complete product** but a **poor architectural template** for a reusable agent harness. It was built as a monolithic application first and had modularity bolted on second. The 512K LoC codebase conflates product concerns (onboarding wizards, sticker commands, companion sprites) with agent infrastructure (tool execution, state machines, event streaming), making it impossible to extract the harness without carrying 400K lines of product.

**Do not port the architecture. Port only the lessons.**

---

## 1. Effectiveness

### What Claude Code does well

**Permission system is production-hardened.** The multi-layered permission model (bash classifier → rule matching → mode dispatch → UI prompt) is the most battle-tested part of the codebase. The `dangerousPatterns.ts`, `sedValidation.ts`, `pathValidation.ts` trio demonstrates what a real safety boundary looks like after thousands of edge cases.

**Tool orchestration is sophisticated.** `toolOrchestration.ts` partitions tool calls into concurrent-safe (read-only) and sequential (write) batches. Read-only tools run in parallel; write tools serialize. Context modifiers queue up and apply in order. This is a pattern worth studying.

**MCP integration is complete.** stdio, WebSocket, HTTP, in-process transports. OAuth auth. Elicitation. Resource reading. Channel permissions. This is the reference implementation.

### What Claude Code gets wrong

**The agent loop is not extractable.** `QueryEngine.ts` is 46K lines — a single file containing LLM streaming, tool execution, retry logic, token counting, thinking mode, speculative execution, prompt caching, and analytics. It cannot be used as a library. It cannot be tested in isolation. It cannot be replaced. This violates every principle in both pi-mono and Harness philosophy.

> **Lesson for Harness:** The agent loop MUST be a pure function: `func runLoop(ctx, provider, tools, state, bus) error`. No file should exceed ~1000 lines.

**No separation between agent infrastructure and product features.** The tool registry (`tools.ts`) mixes infrastructure tools (BashTool, FileReadTool) with product features (StickersCommand, BuddyCompanion, TungstenTool). There is no interface boundary — adding a tool means editing `tools.ts` directly and touching the same `Tool.ts` type definition that everything depends on.

> **Lesson for Harness:** Tools implement an interface. The registry is a data structure, not a 367-line file with conditional requires.

---

## 2. Efficiency

### Startup Optimization: Worth Porting

Claude Code's startup prefetch pattern is excellent:

```typescript
// main.tsx — fire before heavy imports
startMdmRawRead()        // subprocess for MDM settings
startKeychainPrefetch()  // async keychain reads
apiPreconnect()          // TCP preconnect to API
```

This is pure engineering pragmatism. Harness should do equivalent work: preconnect to LLM APIs, prefetch config, warm caches — all before the TUI mounts.

### Feature Flags as Dead Code Elimination: Study, Don't Copy

```typescript
const SleepTool = feature('PROACTIVE')
  ? require('./tools/SleepTool/SleepTool.js').SleepTool
  : null
```

This is Bun-specific build-time DCE. Clever, but it creates a codebase where **20+ features exist in a quantum superposition** — present in source, absent from builds, invisible in tests. The feature flag inventory (PROACTIVE, KAIROS, COORDINATOR_MODE, VOICE_MODE, BUDDY, ULTRAPLAN, TORCH, etc.) reveals a product developed by accretion, not design.

> **Do not port:** Feature-flag-driven architecture. In Harness, features either exist as compiled modules or they don't. Go's build tags serve the same purpose without the conditional-require spaghetti.

### Runtime Cost of React

Claude Code renders its TUI with React 19 + a custom Ink reconciler. The `REPL.tsx` screen is ~5000 lines. Every state change triggers a React render cycle — virtual DOM diffing, Yoga layout computation, ANSI output generation — for a terminal.

This is an extraordinary amount of machinery for rendering text. The custom Ink fork (`src/ink/`, ~80 files) is essentially a full terminal rendering engine: layout engine, event system, hit testing, focus management, cursor tracking, text selection.

> **Do not port:** React for terminal rendering. Bubble Tea's Elm architecture achieves the same result with ~10% of the complexity. The overhead of React reconciliation is unjustifiable for a terminal app.

### Token Budget Management: Port the Concept

`query/tokenBudget.ts` and the compaction system (`services/compact/`) implement real-world context window management. The multi-strategy approach (auto-compact, micro-compact, API-side compact, session-memory-compact) is battle-tested. The idea of extracting memories before compacting is important — naive truncation loses critical context.

> **Port the concept, not the code.** Harness needs a `TokenBudget` struct in core with a compaction interface that extensions can implement.

---

## 3. Simplicity

### AppState: The God Object

`AppStateStore.ts` defines `AppState` — a single type with **~100+ fields** covering:

- Settings, model selection, verbose mode
- Bridge connection state (7+ `replBridge*` fields)
- MCP state (clients, tools, commands, resources)
- Plugin state (enabled, disabled, commands, errors, installation status)
- Agent state (definitions, name registry, color, swarm view)
- Todo lists, notifications, elicitation queue
- File history, attribution, session hooks
- Tungsten session, WebBrowser state, companion reaction
- Speculation state (active/idle with mutable refs)
- Remote connection status, backgrounded task count
- Permission context, denial tracking
- And more...

This is a **monolithic state tree** that every component in the application can read. It mixes:
- **Agent state** (messages, tools, model) — belongs in the agent loop
- **UI state** (footer selection, expanded view, spinner tip) — belongs in the frontend
- **Connection state** (bridge, remote, MCP) — belongs in connection managers
- **Product state** (companion reaction, stickers, tungsten) — belongs nowhere near core

The `DeepImmutable<>` wrapper is a band-aid — it prevents mutation in TypeScript's type system but doesn't prevent the fundamental problem: everything knows about everything.

> **Do not port:** Monolithic state. Harness uses immutable `AgentState` snapshots with ~10 fields. UI state lives in the frontend. Connection state lives in connection managers. There is no god object.

### Circular Dependency Hell

The codebase is riddled with circular dependency workarounds:

```typescript
// Lazy require to break circular dependency: tools.ts -> TeamCreateTool -> ... -> tools.ts
const getTeamCreateTool = () =>
  require('./tools/TeamCreateTool/TeamCreateTool.js').TeamCreateTool

// Lazy require to avoid circular dependency: teammate.ts -> AppState.tsx -> ... -> main.tsx
const getTeammateUtils = () =>
  require('./utils/teammate.js')
```

These appear in `tools.ts`, `commands.ts`, `main.tsx`, `coordinatorMode.ts`, and throughout. They indicate a **dependency graph that has grown organically** with no enforced layering.

> **Do not port:** Any architecture that requires lazy requires to break cycles. Harness's Go modules enforce unidirectional dependencies at compile time. The core never imports extensions.

### 29K-line Type File

`Tool.ts` is ~29K lines defining the `Tool` type, `ToolUseContext`, `ToolPermissionContext`, and ~50 related types. This single file is imported by nearly everything. Any change to `Tool.ts` invalidates the entire build.

In Harness, the equivalent is 4 interface definitions totaling ~20 lines:

```go
type Provider interface { Stream(...) (<-chan Event, error); Models(...) ([]Model, error) }
type Tool interface { Name() string; Schema() json.RawMessage; Execute(...) (*Result, error) }
type Frontend interface { Run(...) error }
type Analyzer interface { Analyze(...) (*Analysis, error); Query(...) (*Result, error) }
```

> **Lesson:** The interface surface area is the most important design decision. Claude Code's `ToolUseContext` carries 60+ fields including settings, permissions, state store, model, messages, MCP connections, plugins, file history, and analytics — making every tool coupled to every product concern. Harness tools see `context.Context` and `json.RawMessage`. Nothing else.

---

## 4. Key Gaps

### No Content-Addressable Caching

Claude Code has no built-in caching layer. File reads, tool outputs, AST parses — all are recomputed on every invocation. The `fileStateCache.ts` and `fileReadCache.ts` are narrow, ad-hoc caches, not a general-purpose content-addressable store.

> **Gap for Harness to fill:** The `core/cache` module with `ContentKey(sha256) → value` is a fundamental advantage.

### No Provider Abstraction

Claude Code is hardwired to `@anthropic-ai/sdk`. There is no `Provider` interface. Switching models means switching the entire API client. Bedrock support (`utils/model/bedrock.ts`) is bolted on via model name mapping, not a separate provider.

> **Gap for Harness to fill:** The `Provider` interface is non-negotiable. Provider-agnostic core is a hard requirement.

### No Analyzer Concept

Claude Code has no equivalent of Harness's `Analyzer` interface. Code intelligence (if any) comes from the LLM calling GrepTool/GlobTool. There's no pre-processing of code structure, no repo maps, no dependency graphs injected into context automatically.

> **Gap for Harness to fill:** Analyzers that enrich context BEFORE the LLM sees it — not tools the LLM has to discover and call.

### No Formal State Machine

`query/transitions.ts` is:

```typescript
export type Terminal = any
export type Continue = any
```

Two `any` types. The "state machine" is aspirational, not implemented. The actual agent loop is an imperative `while` loop in `QueryEngine.ts` with nested conditionals, early returns, and catch blocks.

> **Gap for Harness to fill:** An explicit state machine with typed transitions. The agent loop should be a `switch` on state, not a 46K-line procedural waterfall.

---

## 5. Assumptions to NOT Port

### ❌ "React is appropriate for terminal UIs"

Claude Code invested ~15K lines in a custom Ink fork. The result is a capable but over-engineered rendering layer. Terminal UIs don't need virtual DOM diffing, Yoga flexbox, or a React reconciler. Bubble Tea's Elm architecture (Model → Update → View) achieves equivalent results with a fraction of the complexity.

### ❌ "Everything is a slash command"

Claude Code has ~70+ slash commands. Many are product features (`/stickers`, `/buddy`, `/mobile`, `/desktop`, `/chrome`) that pollute the namespace. The command system is a flat registry with no hierarchy or discovery mechanism beyond typeahead.

Harness should have a small set of core commands and let extensions register their own without modifying the core registry.

### ❌ "The tool type should carry the universe"

`ToolUseContext` in Claude Code is a superset of AppState, permissions, settings, model config, file history, MCP state, and analytics. Every tool receives the entire application context. This makes tools impossible to test in isolation and tightly couples them to the product.

Harness tools should receive `context.Context` (for cancellation), `json.RawMessage` (for arguments), and nothing else. If a tool needs file I/O, it uses the OS. If it needs settings, it reads config. It does not receive the entire application state.

### ❌ "Multi-agent requires product-level orchestration"

Claude Code's swarm system (`utils/swarm/`, ~30 files) implements tmux backends, iTerm2 setup, layout managers, permission sync, leader election, and teammate model selection. This is product-level orchestration built for a specific UX (split panes in the terminal).

Harness multi-agent should be simple: the `AgentTool` spawns a new agent loop with its own state. The parent communicates via the event bus. The frontend decides how to render it. No tmux, no layout managers, no pane backends.

### ❌ "Feature flags are architecture"

20+ feature flags means 20+ code paths that may or may not exist at runtime. This creates a combinatorial testing problem and makes the codebase impossible to reason about statically. In Harness, a feature is either a compiled module or it doesn't exist. Go build tags serve the same purpose without runtime conditionals.

### ❌ "Analytics and telemetry belong in the core"

Claude Code has analytics woven throughout: GrowthBook feature flags in the query engine, Datadog in the API client, OpenTelemetry spans around tool execution, first-party event logging in permission decisions. This creates coupling between the agent loop and Anthropic's infrastructure.

Harness emits events to the event bus. An analytics extension subscribes and exports wherever it wants. The core has zero knowledge of analytics providers.

### ❌ "The bridge (IDE integration) is a core concern"

Claude Code has ~30 files for bridge communication, 7+ `replBridge*` fields in AppState, and bridge-safe command filtering. The bridge is deeply coupled — the REPL checks bridge state, commands check bridge safety, permissions delegate to the IDE.

In Harness, the bridge is a `Frontend` implementation. It implements the same interface as the TUI. The core doesn't know it exists.

---

## 6. What to Actually Port

| From Claude Code | To Harness | Why |
|-----------------|------------|-----|
| Permission rule system | `core/approval` | Glob-based allow/deny rules are a good pattern |
| Tool concurrency partitioning | `core/agent/loop.go` | Read-only parallel, write sequential is correct |
| Startup prefetch pattern | `cmd/harness/main.go` | Preconnect, prefetch config in parallel |
| Compaction concept | `core/agent/compact.go` | Multi-strategy with memory extraction |
| Token budget tracking | `core/agent/budget.go` | Context window management is essential |
| Bash safety patterns | `ext/tools/terminal/safety.go` | Dangerous pattern detection, path validation |
| MCP client architecture | `ext/mcp/client.go` | Multi-transport, auth, elicitation |
| Structured diff rendering | `ext/tui/components/diff.go` | Good UX pattern |

---

## 7. Architectural Contrasts

```
Claude Code                          Harness
─────────────────────────────────    ─────────────────────────────────
512K LoC, ~1900 files                Target: <20K LoC core, <50 files
QueryEngine.ts: 46K lines           core/agent/loop.go: ~500 lines
Tool.ts: 29K lines                  core/interfaces.go: ~30 lines
AppState: ~100+ fields              AgentState: ~10 fields
React + custom Ink fork             Bubble Tea (Elm architecture)
Bun + TypeScript                    Go (single binary)
@anthropic-ai/sdk only              Provider interface (any LLM)
Feature flags (20+ runtime gates)   Go modules (compile-time)
Circular deps (lazy require)        Unidirectional (enforced by modules)
Monolithic state tree               Immutable snapshots per concern
No caching                          Content-addressable cache in core
No state machine                    Explicit typed state machine
No analyzer concept                 Analyzer interface (code intel)
Analytics in core                   Analytics as extension subscriber
Bridge in core                      Bridge as Frontend implementation
70+ commands in flat registry       Small core + extension commands
```

---

## 8. Final Assessment

Claude Code proves that you can build a **feature-complete agentic CLI** with this architecture. It also proves that you **shouldn't**. The 512K LoC codebase is a testament to accretive product development — each feature added to the monolith, each concern mixed with every other, until the result is impressive but unmaintainable.

The Harness philosophy is the antidote:

1. **The core is tiny** — agent loop, event bus, state machine, transport, cache. Nothing else.
2. **Everything is an extension** — providers, tools, frontends, analyzers, commands.
3. **Interfaces are the architecture** — 4 interfaces, not 29K lines of type definitions.
4. **State is per-concern** — agent state, UI state, connection state are separate.
5. **Dependencies flow one way** — extensions depend on core, never the reverse.

Build the harness. Not the monolith.
