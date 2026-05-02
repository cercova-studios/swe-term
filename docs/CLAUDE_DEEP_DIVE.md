# Claude Code Deep Dive — Architecture & Feature Analysis

> Analysis of the Claude Code source snapshot (`src/`, ~1,900 files, 512K+ LoC).
> Runtime: Bun. Language: TypeScript (strict). UI: React + Ink (terminal).

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Entrypoint & Bootstrap](#entrypoint--bootstrap)
3. [Tool System](#tool-system)
4. [Command System](#command-system)
5. [Query Engine & Agent Loop](#query-engine--agent-loop)
6. [Permission System](#permission-system)
7. [Multi-Agent / Swarm Architecture](#multi-agent--swarm-architecture)
8. [Skill System](#skill-system)
9. [Plugin System](#plugin-system)
10. [MCP Integration](#mcp-integration)
11. [Bridge System (IDE Integration)](#bridge-system-ide-integration)
12. [Terminal UI (Ink)](#terminal-ui-ink)
13. [Memory System](#memory-system)
14. [Session Management](#session-management)
15. [Context Compaction](#context-compaction)
16. [Voice Input](#voice-input)
17. [Remote & Teleport](#remote--teleport)
18. [Feature Flags & Dead Code Elimination](#feature-flags--dead-code-elimination)
19. [Observability & Analytics](#observability--analytics)
20. [Security Patterns](#security-patterns)
21. [Key Design Patterns](#key-design-patterns)
22. [Feature Flag Inventory (Unreleased)](#feature-flag-inventory-unreleased)

---

## System Overview

Claude Code is Anthropic's agentic CLI for software engineering. It allows Claude to read/write files, run shell commands, search codebases, manage tasks, coordinate sub-agents, and interact with external services — all from the terminal.

**Tech Stack:**

| Layer | Technology |
|-------|-----------|
| Runtime | Bun (not Node.js) |
| Language | TypeScript strict mode |
| Terminal UI | React 19 + Ink 5 (React reconciler for terminal) |
| CLI parsing | Commander.js with extra-typings |
| Schema | Zod v4 |
| Code search | ripgrep (via `execa`) |
| API | `@anthropic-ai/sdk` |
| Protocols | MCP SDK, LSP (`vscode-languageserver-protocol`) |
| Telemetry | OpenTelemetry + gRPC export |
| Feature flags | GrowthBook |
| Auth | OAuth 2.0, JWT, macOS Keychain |
| Networking | `undici`, `axios`, `ws` (WebSocket) |

**Scale:** ~1,900 source files. The `QueryEngine.ts` alone is ~46K lines. `Tool.ts` is ~29K lines. `commands.ts` is ~25K lines.

---

## Entrypoint & Bootstrap

**Path:** `src/entrypoints/cli.tsx` → `src/main.tsx`

Startup is heavily optimized for latency. Before any heavy module is loaded, side-effects fire:

```
startMdmRawRead()       ← MDM (managed device) settings prefetch
startKeychainPrefetch() ← macOS Keychain read (async, parallel)
apiPreconnect()         ← TCP preconnect to api.anthropic.com
```

`main.tsx` uses Commander.js to parse CLI args, then renders a React/Ink app. Multiple entrypoints exist:

| Entrypoint | Purpose |
|------------|---------|
| `cli.tsx` | Standard interactive CLI |
| `mcp.ts` | MCP server mode |
| `init.ts` | Project initialization |
| `sdk/` | Programmatic SDK types |

The bootstrap sequence (`src/bootstrap/state.ts`) initializes session ID, settings, and environment before the React tree mounts.

---

## Tool System

**Path:** `src/tools/`, `src/Tool.ts`, `src/tools.ts`

Every action Claude can take is a **Tool**. Tools are the core abstraction — they define schema, permissions, execution, and UI rendering.

### Tool Interface

Each tool is a static object conforming to the `Tool` type with:
- `name` — unique identifier (e.g., `Bash`, `Read`, `Edit`)
- `inputSchema` — JSON Schema for the LLM's tool-use arguments
- `isEnabled()` — runtime gate (can depend on env, settings, feature flags)
- Permission model — how approval is requested
- `execute()` — the actual execution logic
- `UI` — React component for rendering tool use/result in the TUI

### Tool Registry

`src/tools.ts` is the source of truth. `getAllBaseTools()` returns the exhaustive list:

| Tool | Purpose |
|------|---------|
| **BashTool** | Shell command execution with security validation |
| **FileReadTool** | Read files (text, images, PDFs, notebooks) |
| **FileWriteTool** | Create/overwrite files |
| **FileEditTool** | Partial string-replacement edits |
| **GlobTool** | File pattern matching (picomatch) |
| **GrepTool** | ripgrep-based content search |
| **WebFetchTool** | URL content fetching |
| **WebSearchTool** | Web search |
| **AgentTool** | Sub-agent spawning (the backbone of multi-agent) |
| **SkillTool** | Invoke registered skills |
| **MCPTool** | Call tools on MCP servers |
| **LSPTool** | Language Server Protocol queries (diagnostics, symbols) |
| **NotebookEditTool** | Jupyter notebook cell editing |
| **TaskCreateTool** | Create tracked tasks |
| **TaskGetTool** | Read task status |
| **TaskUpdateTool** | Update task progress |
| **TaskListTool** | List all tasks |
| **TaskStopTool** | Stop a running task/agent |
| **TaskOutputTool** | Display task output |
| **SendMessageTool** | Inter-agent messaging |
| **TeamCreateTool** | Spawn parallel team agents |
| **TeamDeleteTool** | Tear down team agents |
| **EnterPlanModeTool** | Switch to plan mode |
| **ExitPlanModeTool** | Exit plan mode |
| **EnterWorktreeTool** | Git worktree isolation |
| **ExitWorktreeTool** | Leave worktree |
| **AskUserQuestionTool** | Prompt user for input |
| **TodoWriteTool** | Write to a todo/task list |
| **ConfigTool** | Modify settings (internal-only) |
| **ToolSearchTool** | Deferred tool discovery (lazy loading) |
| **BriefTool** | Briefing / attachment upload |
| **SleepTool** | Wait in proactive mode |
| **CronCreate/Delete/List** | Scheduled triggers |
| **RemoteTriggerTool** | Remote agent triggers |
| **ListMcpResourcesTool** | List MCP server resources |
| **ReadMcpResourceTool** | Read MCP server resources |
| **PowerShellTool** | Windows PowerShell execution |
| **SyntheticOutputTool** | Structured JSON output |
| **EnterPlanModeTool** | Plan mode (think before acting) |

### Tool Assembly

`assembleToolPool()` merges built-in tools with MCP tools, deduplicates by name (built-ins win), and sorts for prompt-cache stability. The sort order is critical — built-in tools form a contiguous prefix so Anthropic's server-side cache breakpoints remain valid.

### Feature-Gated Tools

Many tools are behind feature flags and conditionally loaded:

```typescript
const SleepTool = feature('PROACTIVE') || feature('KAIROS')
  ? require('./tools/SleepTool/SleepTool.js').SleepTool : null

const WebBrowserTool = feature('WEB_BROWSER_TOOL')
  ? require('./tools/WebBrowserTool/WebBrowserTool.js').WebBrowserTool : null
```

These use Bun's dead-code elimination — when a feature flag is `false` at build time, the `require()` and all downstream code is stripped from the bundle.

---

## Command System

**Path:** `src/commands/`, `src/commands.ts`

Commands are user-facing slash commands (invoked with `/`). There are ~70+ commands spanning configuration, navigation, debugging, and workflow.

### Command Types

| Type | Behavior |
|------|----------|
| `prompt` | Expands to text content sent to the LLM (skills use this) |
| `local` | Executes locally, returns text output |
| `local-jsx` | Renders Ink JSX UI (interactive dialogs, settings) |

### Notable Commands

| Category | Commands |
|----------|----------|
| **Git** | `/commit`, `/commit-push-pr`, `/diff`, `/branch`, `/pr_comments` |
| **Context** | `/compact`, `/context`, `/clear`, `/files` |
| **Config** | `/config`, `/theme`, `/model`, `/vim`, `/keybindings`, `/effort`, `/output-style` |
| **Session** | `/resume`, `/session`, `/export`, `/share`, `/rename` |
| **Diagnostics** | `/doctor`, `/status`, `/cost`, `/usage`, `/stats`, `/heapdump` |
| **Plugins** | `/plugin`, `/reload-plugins`, `/mcp` |
| **Skills** | `/skills`, `/agents` |
| **Auth** | `/login`, `/logout` |
| **Workflow** | `/plan`, `/review`, `/security-review`, `/tasks` |
| **Features** | `/voice`, `/fast`, `/permissions`, `/hooks`, `/bridge` |
| **Misc** | `/help`, `/exit`, `/copy`, `/feedback`, `/stickers`, `/buddy` |

### Command Loading

Commands are loaded from multiple sources, merged in priority order:
1. **Built-in** — compiled into the binary
2. **Bundled skills** — ship with the CLI but are model-invocable
3. **Plugin commands** — from installed plugins
4. **MCP commands** — from MCP server prompts
5. **User skills** — from `~/.claude/skills/` or `.claude/skills/`

Remote-safe filtering (`REMOTE_SAFE_COMMANDS`) prevents local-only commands from executing over mobile/web bridges.

---

## Query Engine & Agent Loop

**Path:** `src/QueryEngine.ts`, `src/query/`

The QueryEngine is the heart of Claude Code (~46K lines). It implements the agent loop:

```
User prompt
  → Build context (system prompt, messages, tools)
  → Stream LLM response (via Anthropic SDK)
  → Parse tool calls from response
  → Execute tools (parallel or sequential)
  → Append results to conversation
  → Loop if more tool calls
  → Emit final response
```

### Query Configuration (`src/query/config.ts`)

Each query snapshots immutable config at entry:
- `sessionId`
- `gates.streamingToolExecution` — stream tool output as it executes
- `gates.emitToolUseSummaries` — generate summaries of tool uses
- `gates.isAnt` — internal Anthropic user
- `gates.fastModeEnabled` — fast mode (smaller model for simple tasks)

### Token Budget (`src/query/tokenBudget.ts`)

Manages context window constraints — determines when to compact, how much history to include, and when to warn about token limits.

### Transitions (`src/query/transitions.ts`)

State machine transitions for the query loop — handling tool calls, errors, rate limits, and stop conditions.

### Stop Hooks (`src/query/stopHooks.ts`)

Post-completion hooks that can trigger additional actions (e.g., memory extraction, session summary).

---

## Permission System

**Path:** `src/hooks/toolPermission/`, `src/utils/permissions/`

Every tool invocation passes through the permission system. This is the security boundary.

### Permission Modes

| Mode | Behavior |
|------|----------|
| `default` | Prompt user for approval on each destructive action |
| `plan` | Approve reads, require approval for writes |
| `bypassPermissions` | Auto-approve everything (requires explicit opt-in) |
| `auto` | AI-classified approval (bash classifier) |

### Permission Flow

```
Tool invocation
  → Check deny rules (blanket deny by tool name/pattern)
  → Check allow rules (pre-approved patterns)
  → Check permission mode
  → If auto-mode: run bashClassifier
  → If interactive: render PermissionPrompt UI
  → Log decision for audit
```

### Bash Classifier (`src/utils/permissions/bashClassifier.ts`)

In auto-mode, shell commands are classified as safe/unsafe using pattern matching:
- `readOnlyValidation.ts` — detect read-only commands
- `dangerousPatterns.ts` — detect destructive patterns
- `commandSemantics.ts` — understand command intent
- `sedValidation.ts` — special handling for sed edits
- `pathValidation.ts` — validate file paths are within workspace

### Permission Rules

Users can define allow/deny rules with glob patterns:
```
Allow: Bash(npm test)
Allow: Bash(git *)
Deny: Bash(rm -rf *)
Allow: Edit(src/**)
```

---

## Multi-Agent / Swarm Architecture

**Path:** `src/tools/AgentTool/`, `src/coordinator/`, `src/utils/swarm/`

Claude Code supports spawning sub-agents for parallel work. This is a sophisticated multi-agent system.

### AgentTool (`src/tools/AgentTool/`)

The `AgentTool` spawns a sub-agent with:
- Its own conversation history
- A subset of available tools
- An optional custom system prompt
- Color coding for visual distinction in the TUI

Key files:
- `runAgent.ts` — agent execution loop
- `forkSubagent.ts` — fork current context into a sub-agent
- `resumeAgent.ts` — resume a paused agent
- `agentMemory.ts` / `agentMemorySnapshot.ts` — memory for agents
- `agentColorManager.ts` — visual differentiation

### Built-in Agents

`src/tools/AgentTool/built-in/`:
- **planAgent** — planning and architecture
- **exploreAgent** — codebase exploration
- **verificationAgent** — verify changes work
- **generalPurposeAgent** — general tasks
- **claudeCodeGuideAgent** — help with Claude Code itself

### Coordinator Mode (`src/coordinator/coordinatorMode.ts`)

When `CLAUDE_CODE_COORDINATOR_MODE=true`, the main agent becomes a **coordinator** that:
- Only has access to `AgentTool`, `TaskStopTool`, `SendMessageTool`
- Delegates all file/shell work to spawned worker agents
- Workers get `Bash`, `FileRead`, `FileEdit` and other execution tools
- The coordinator orchestrates via task assignment and message passing

### Team Swarms (`src/utils/swarm/`)

`TeamCreateTool` enables parallel team work:
- Multiple agents run simultaneously in separate tmux panes or in-process
- A leader agent coordinates workers
- Permission syncing across the swarm
- `teammateLayoutManager.ts` — manages terminal layout for multi-agent
- `TmuxBackend.ts` / `InProcessBackend.ts` — execution backends

---

## Skill System

**Path:** `src/skills/`, `src/tools/SkillTool/`

Skills are reusable workflows that the model can invoke. They're a key extensibility mechanism.

### Skill Sources

| Source | Path | Description |
|--------|------|-------------|
| **Bundled** | `src/skills/bundled/` | Ship with the CLI |
| **User** | `~/.claude/skills/` | Global user skills |
| **Project** | `.claude/skills/` | Per-project skills |
| **Plugin** | Via plugin `skills/` dir | Plugin-provided skills |
| **MCP** | MCP server prompts | MCP-provided skills |

### Bundled Skills

Registered via `registerBundledSkill()`:

| Skill | Purpose |
|-------|---------|
| `verify` | Verify changes before committing |
| `batch` | Batch operations across files |
| `simplify` | Simplify complex code |
| `loop` | Iterative refinement |
| `remember` | Save to memory |
| `debug` | Debug issues |
| `stuck` | Help when stuck |
| `skillify` | Create new skills |
| `claudeApi` | Claude API usage (with per-language guides) |
| `claudeInChrome` | Chrome extension integration |
| `keybindings` | Keybinding configuration |
| `updateConfig` | Configuration management |
| `scheduleRemoteAgents` | Schedule remote agents |
| `loremIpsum` | Generate placeholder text |

### Skill File Format

Skills use YAML frontmatter:
```yaml
---
name: my-skill
description: What this skill does
whenToUse: When to invoke this skill
allowedTools: [Bash, Read, Edit]
model: claude-sonnet-4-20250514
---
Skill prompt content here...
```

### Skill Search (`src/services/skillSearch/`)

When the tool pool exceeds a threshold, `ToolSearchTool` enables deferred discovery — the LLM can search for relevant tools/skills rather than having all schemas in context.

---

## Plugin System

**Path:** `src/plugins/`, `src/utils/plugins/`

A full plugin lifecycle management system.

### Plugin Capabilities

Plugins can provide:
- **Commands** — new slash commands
- **Skills** — new model-invocable skills
- **Hooks** — event-based automation (PreToolUse, PostToolUse, Stop, etc.)
- **Agents** — custom agent definitions
- **MCP servers** — MCP server configurations
- **Output styles** — custom output formatting

### Plugin Management

| File | Purpose |
|------|---------|
| `pluginLoader.ts` | Load plugins from disk |
| `pluginInstallationHelpers.ts` | Install from marketplace |
| `pluginAutoupdate.ts` | Auto-update plugins |
| `pluginBlocklist.ts` | Block malicious plugins |
| `pluginPolicy.ts` | Policy enforcement |
| `validatePlugin.ts` | Validation |
| `reconciler.ts` | Reconcile installed vs expected |
| `dependencyResolver.ts` | Resolve plugin dependencies |
| `marketplaceManager.ts` | Official marketplace integration |
| `officialMarketplace.ts` | Browse/install from marketplace |
| `headlessPluginInstall.ts` | Non-interactive installation |

### DXT Format

`src/utils/dxt/` — Plugin packaging format (zip-based with manifest).

---

## MCP Integration

**Path:** `src/services/mcp/`

Full Model Context Protocol client implementation.

### MCP Features

- **Tool proxying** — MCP server tools appear as native Claude Code tools
- **Resource reading** — `ListMcpResources`, `ReadMcpResource` tools
- **OAuth authentication** — `src/services/mcp/auth.ts`
- **Elicitation** — interactive prompts from MCP servers
- **Channel management** — allowlists, permissions
- **Multiple transports** — stdio, WebSocket, HTTP, in-process

### MCP Connection Manager

`MCPConnectionManager.tsx` — React component that manages connections to all configured MCP servers. Handles:
- Connection lifecycle
- Reconnection with backoff
- Tool discovery and schema extraction
- Resource listing
- Notification handling

### Configuration

MCP servers configured in `~/.claude/settings.json` or `.claude/settings.json`:
```json
{
  "mcpServers": {
    "my-server": {
      "command": "npx",
      "args": ["my-mcp-server"],
      "env": { "API_KEY": "..." }
    }
  }
}
```

---

## Bridge System (IDE Integration)

**Path:** `src/bridge/`

Bidirectional communication between Claude Code CLI and IDE extensions (VS Code, JetBrains).

### Architecture

```
IDE Extension ←→ Bridge Protocol ←→ Claude Code CLI
```

### Components

| File | Purpose |
|------|---------|
| `bridgeMain.ts` | Main bridge loop |
| `bridgeMessaging.ts` | Message protocol (JSON over stdio/WebSocket) |
| `bridgePermissionCallbacks.ts` | Permission delegation to IDE |
| `replBridge.ts` | REPL session bridging |
| `jwtUtils.ts` | JWT-based authentication |
| `sessionRunner.ts` | Session lifecycle management |
| `bridgeConfig.ts` | Configuration |
| `bridgeDebug.ts` | Debug logging |
| `workSecret.ts` | Shared secret for auth |

### Remote Control

`src/remote/` — Remote session management for controlling Claude Code from mobile/web:
- `RemoteSessionManager.ts` — manages remote sessions
- `SessionsWebSocket.ts` — WebSocket transport
- `remotePermissionBridge.ts` — permission delegation over network

---

## Terminal UI (Ink)

**Path:** `src/ink/`, `src/components/`

A custom fork/extension of [Ink](https://github.com/vadimdemedes/ink), the React renderer for terminals.

### Custom Ink Internals

The `src/ink/` directory is essentially a full terminal rendering engine:

| Module | Purpose |
|--------|---------|
| `renderer.ts` | React reconciler for terminal |
| `layout/engine.ts` | Yoga-based layout engine |
| `layout/yoga.ts` | Yoga layout bindings |
| `render-node-to-output.ts` | Node → ANSI output |
| `render-to-screen.ts` | Screen rendering |
| `render-border.ts` | Box border rendering |
| `termio/` | Terminal I/O (CSI, OSC, SGR, DEC sequences) |
| `parse-keypress.ts` | Keyboard input parsing |
| `events/` | Event system (keyboard, mouse, paste, resize, focus) |
| `selection.ts` | Text selection |
| `cursor.ts` | Cursor management |
| `optimizer.ts` | Render optimization |
| `frame.ts` | Frame scheduling |
| `hit-test.ts` | Click target resolution |
| `focus.ts` | Focus management |

### UI Components (~140 components)

Major component areas:
- **Message rendering** — `src/components/messages/` (~30 message types)
- **Permission dialogs** — `src/components/permissions/` (per-tool approval UIs)
- **MCP UI** — `src/components/mcp/` (server management, tool lists)
- **Design system** — `src/components/design-system/` (Pane, Dialog, Tabs, etc.)
- **Diff views** — `src/components/diff/`, `src/components/StructuredDiff.tsx`
- **Settings** — `src/components/Settings/`
- **Agents** — `src/components/agents/` (agent creation wizard, management)
- **Prompt input** — `src/components/PromptInput/` (the main input area)
- **Spinner** — `src/components/Spinner/` (animated loading indicators)
- **Feedback** — `src/components/FeedbackSurvey/`

### Vim Mode

`src/vim/` — Full vim keybinding emulation:
- `motions.ts` — h, j, k, l, w, b, e, etc.
- `operators.ts` — d, c, y, p
- `textObjects.ts` — iw, aw, i", a", etc.
- `transitions.ts` — normal ↔ insert ↔ visual mode

---

## Memory System

**Path:** `src/memdir/`, `src/services/extractMemories/`, `src/services/SessionMemory/`

Persistent memory that persists across sessions.

### Memory Types

| Type | Scope | Path |
|------|-------|------|
| User memory | Global | `~/.claude/CLAUDE.md` |
| Project memory | Per-project | `.claude/CLAUDE.md` |
| Team memory | Shared team | `.claude/team-memory/` |
| Session memory | Per-session | Auto-extracted |

### Memory Operations

- **Auto-extraction** (`src/services/extractMemories/`) — After sessions, important learnings are automatically extracted to memory
- **Memory scanning** (`src/memdir/memoryScan.ts`) — Find relevant memories for context
- **Memory age** (`src/memdir/memoryAge.ts`) — Track memory freshness
- **Team memory sync** (`src/services/teamMemorySync/`) — Sync memories across team members with secret scanning guard

### Dream System (`src/services/autoDream/`)

Background memory consolidation:
- `autoDream.ts` — automatic dream scheduling
- `consolidationPrompt.ts` — memory consolidation prompt
- `consolidationLock.ts` — prevent concurrent consolidation
- `DreamTask` (`src/tasks/DreamTask/`) — background dream execution

---

## Session Management

**Path:** `src/utils/sessionStorage.ts`, `src/assistant/`, `src/history.ts`

### Session Storage

Sessions are persisted to disk:
- `sessionStorage.ts` — read/write session data
- `sessionDiscovery.ts` — find existing sessions
- `sessionHistory.ts` — session history management
- `sessionRestore.ts` — resume previous sessions

### Cross-Project Resume

`src/utils/crossProjectResume.ts` — Resume sessions that started in a different project directory.

### Teleport

`src/utils/teleport/` — "Teleport" sessions between environments:
- `gitBundle.ts` — bundle git state for transfer
- `environments.ts` — environment management
- `api.ts` — teleport API client

---

## Context Compaction

**Path:** `src/services/compact/`

When conversation context grows too large, automatic compaction kicks in.

### Compaction Strategies

| Strategy | File | Description |
|----------|------|-------------|
| **Auto-compact** | `autoCompact.ts` | Triggers based on token budget |
| **Micro-compact** | `microCompact.ts` | Lightweight summarization |
| **API micro-compact** | `apiMicrocompact.ts` | Server-side compaction |
| **Session memory compact** | `sessionMemoryCompact.ts` | Extract key memories before compact |
| **Time-based** | `timeBasedMCConfig.ts` | Time-based configuration |

### Compaction Flow

```
Token budget exceeded
  → Extract session memories (important learnings)
  → Group messages by topic
  → Summarize each group
  → Replace original messages with summaries
  → Inject memories into new context
```

---

## Voice Input

**Path:** `src/voice/`, `src/context/voice.tsx`, `src/services/voice.ts`

Voice mode (feature-gated behind `VOICE_MODE`):
- `voiceStreamSTT.ts` — streaming speech-to-text
- `voiceKeyterms.ts` — keyword detection
- `voiceModeEnabled.ts` — feature gate

---

## Remote & Teleport

### Remote Sessions (`src/remote/`)

Run Claude Code headlessly and control from mobile/web:
- `RemoteSessionManager.ts` — session lifecycle
- `SessionsWebSocket.ts` — WebSocket transport
- `sdkMessageAdapter.ts` — message format adaptation

### SSH Sessions (`src/ssh/`)

- `SSHSessionManager.ts` — manage SSH tunnels
- `createSSHSession.ts` — establish SSH connections

### Direct Connect (`src/server/`)

- `directConnectManager.ts` — direct peer connections
- `createDirectConnectSession.ts` — session establishment

---

## Feature Flags & Dead Code Elimination

**Path:** `src/_stubs/bun-bundle.ts`

Bun's build-time `feature()` function enables compile-time dead code elimination:

### Known Feature Flags

| Flag | Purpose |
|------|---------|
| `PROACTIVE` | Proactive agent mode (SleepTool, scheduling) |
| `KAIROS` | Advanced autonomous agent features |
| `BRIDGE_MODE` | IDE bridge support |
| `DAEMON` | Background daemon mode |
| `VOICE_MODE` | Voice input |
| `AGENT_TRIGGERS` | Cron-based agent triggers |
| `AGENT_TRIGGERS_REMOTE` | Remote trigger support |
| `MONITOR_TOOL` | MCP server monitoring |
| `COORDINATOR_MODE` | Multi-agent coordinator |
| `WEB_BROWSER_TOOL` | Web browser automation |
| `WORKFLOW_SCRIPTS` | Workflow scripting |
| `HISTORY_SNIP` | History editing/snipping |
| `CONTEXT_COLLAPSE` | Context inspection/collapse |
| `TERMINAL_PANEL` | Terminal capture tool |
| `UDS_INBOX` | Unix domain socket peer messaging |
| `FORK_SUBAGENT` | Fork current agent context |
| `BUDDY` | Companion sprite |
| `ULTRAPLAN` | Advanced planning |
| `TORCH` | Torch feature |
| `MCP_SKILLS` | MCP-based skills |
| `EXPERIMENTAL_SKILL_SEARCH` | AI-powered skill search |
| `OVERFLOW_TEST_TOOL` | Testing overflow handling |
| `CCR_REMOTE_SETUP` | Cloud remote setup |
| `KAIROS_BRIEF` | Brief mode for autonomous |
| `KAIROS_PUSH_NOTIFICATION` | Push notifications |
| `KAIROS_GITHUB_WEBHOOKS` | GitHub webhook subscriptions |

---

## Observability & Analytics

### Telemetry (`src/utils/telemetry/`)

- **OpenTelemetry** — traces, metrics, logs
- `perfettoTracing.ts` — Chrome Perfetto format export
- `sessionTracing.ts` — per-session trace propagation
- `bigqueryExporter.ts` — BigQuery telemetry export
- `pluginTelemetry.ts` — plugin performance tracking

### Analytics (`src/services/analytics/`)

- **GrowthBook** — feature flags and A/B testing
- **Datadog** — metrics and logging
- **First-party events** — custom event logging
- `sinkKillswitch.ts` — emergency analytics disable

### Cost Tracking (`src/cost-tracker.ts`, `src/costHook.ts`)

Real-time token usage and cost estimation per session.

---

## Security Patterns

### Bash Security (`src/tools/BashTool/bashSecurity.ts`)

- Command parsing and classification
- Dangerous pattern detection
- Path validation and sandboxing
- sed edit validation
- Git safety checks
- Destructive command warnings

### Permission Sandboxing

- File operations restricted to workspace by default
- Shell commands classified as read-only vs destructive
- MCP servers require explicit approval
- Plugin trust verification
- Team memory secret scanning

### Secure Storage (`src/utils/secureStorage/`)

- macOS Keychain integration
- Fallback plaintext storage (with warnings)
- Credential prefetch optimization

### Sandbox Runtime

`@anthropic-ai/sandbox-runtime` — container-based sandboxing for untrusted code execution (stubbed in open source).

---

## Key Design Patterns

### 1. React for Terminal UI

Claude Code renders its entire TUI using React with a custom Ink reconciler. This means:
- Components are React functional components with hooks
- State management via React context (`src/context/`)
- The entire application state (`src/state/AppState.tsx`) flows through React

### 2. Tool-as-Plugin Architecture

Every capability is a tool. Adding a new capability means implementing the Tool interface — no core changes needed.

### 3. Feature Flag Guard → Dead Code Elimination

```typescript
const SleepTool = feature('PROACTIVE')
  ? require('./tools/SleepTool/SleepTool.js').SleepTool
  : null
```

At build time, if `PROACTIVE` is false, the entire SleepTool module and its dependencies are eliminated from the bundle. This keeps the production binary lean while allowing internal development of unreleased features.

### 4. Lazy Module Loading

Heavy modules (OpenTelemetry, gRPC, analytics, some tools) are loaded via dynamic `import()` only when needed, keeping startup fast.

### 5. Parallel Prefetch at Startup

MDM settings, Keychain reads, API preconnect, and GrowthBook initialization happen in parallel before the heavy React tree mounts.

### 6. Prompt Cache Stability

Tool ordering in `assembleToolPool()` is carefully maintained so that the Anthropic API's server-side prompt cache remains valid across requests. Built-in tools form a contiguous sorted prefix, MCP tools follow.

### 7. Multi-Source Command/Skill Merging

Commands and skills from built-in, bundled, plugin, MCP, and user sources are merged with priority-based deduplication. This allows every extension point to contribute commands while maintaining a consistent namespace.

### 8. State Machine Query Engine

The query loop (`src/query/transitions.ts`) is modeled as a state machine with explicit transitions, making the complex agent loop testable and debuggable.

---

## Feature Flag Inventory (Unreleased)

Based on feature flag analysis, these capabilities are in development:

| Feature | Status | Evidence |
|---------|--------|----------|
| **Proactive mode** | Gated | SleepTool, cron scheduling, push notifications |
| **Kairos** | Gated | Autonomous agent with briefs, file sending, GitHub webhooks |
| **Web browser tool** | Gated | Full browser automation |
| **Coordinator mode** | Gated | Multi-agent orchestration with leader/worker topology |
| **Workflow scripts** | Gated | Reusable workflow definitions |
| **Voice mode** | Gated | Speech-to-text input |
| **Buddy companion** | Gated | Animated companion sprite in terminal |
| **History snipping** | Gated | Edit conversation history |
| **Context collapse** | Gated | Inspect and collapse context |
| **Terminal panel** | Gated | Terminal capture and monitoring |
| **Peer messaging** | Gated | Unix domain socket inter-agent communication |
| **Fork subagent** | Gated | Fork current conversation into sub-agent |
| **Ultraplan** | Gated | Advanced multi-step planning |
| **Remote agent triggers** | Gated | Trigger agents remotely |
| **MCP-based skills** | Gated | Skills provided by MCP servers |
| **Cloud remote setup** | Gated | Remote environment provisioning |

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Source files | ~1,900 |
| Lines of code | ~512,000+ |
| Tools | ~40+ (core) + unlimited via MCP |
| Slash commands | ~70+ |
| Bundled skills | ~15 |
| Feature flags | ~20+ |
| React components | ~140 |
| React hooks | ~80+ |
| Services | ~15 major subsystems |
| MCP transports | stdio, WebSocket, HTTP, in-process |
| Permission modes | 4 (default, plan, bypass, auto) |
| Agent backends | tmux, in-process |
| Memory types | user, project, team, session |
| Compaction strategies | 4 (auto, micro, API, session-memory) |

---

*Generated from static analysis of the Claude Code source snapshot (2026-03-31).*
