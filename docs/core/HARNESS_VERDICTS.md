# Harness Verdicts

swe-term's per-harness decisions: the verdict, what to port, and what not to port.
The compressed matrix is in [`FRAMEWORKS.md`](FRAMEWORKS.md). General, source-derived
knowledge about each harness (architecture, sizes, mechanisms) lives in the
`source-insights` library, not in this repository. Every verdict here was
revalidated against upstream on 2026-10-10; the pinned commit is in each header.

## pi

> Upstream: https://github.com/earendil-works/pi at `1402973a9` (1.3.0), revalidated 2026-10-18
> (previous pin `ea448f454`, 1.1.0). General architecture notes:
> source-insights entries `1f02f5710a14d39d5e970403` (2026-10-10) and `b1014b6151d898ae355ce69c` (pi-durable, 2026-10-18).

### Changes since `ea448f454` (154 commits, 2026-10-10 → 2026-10-18)
- **Experimental server stack removed** (`7472cda0d`): `pi-protocol`,
  `pi-server`, and `pi-client` are deleted, and coding-agent no longer depends
  on `chord`. pi's out-of-process story is again `--mode rpc` and the
  in-process SDK (`createAgentSession`, `packages/coding-agent/docs/sdk.md`).
- **`AgentSession` keeps growing**: 4,398 → 4,428 lines. The `agent-loop.ts`
  core is 949 lines.
- **`chord` reworked** (`0104fef1e`, protocol 3–4): objects addressed by
  reference across processes, typed principals, hot reload with migration at a
  safe point. It is now an application runtime, not a coding-agent dependency.
- **`pi-durable` matured** (1.3.0, ~25.6k lines of src). It is a second,
  separate harness, not coding-agent's runtime. See below.

### pi-durable
`packages/durable/README.md` ("Experimental. The API changes without notice")
plus `docs/spec.md` (5,612 lines). Its claim: conversations, model turns, tool
calls, and host state are committed to storage before anything is shown, and
reopening the storage resumes a turn the process died in.

Mechanisms (README line refs at `1402973a9`):
- **One atomic commit line.** Entries, typed JSON documents (`pi.agent`,
  `pi.live`, `pi.inbox`, `pi.usage`), and tasks change together in one commit,
  or not at all.
- **Everything is a checkpointed task.** `pi.generation` owns its `pi.tool`
  tasks; each step saves a checkpoint. Tasks can sleep on a stored deadline and
  wake after restart (l.132).
- **Tool replay policy** (l.267): a tool's intent is committed before
  `execute()`. After a crash it reruns only if declared `replay: "safe"`;
  otherwise the model gets an `interrupted` error result with the output
  committed so far.
- **Keyed nested calls** (l.318): a replay-safe caller that reruns gets its
  already-made nested calls back by key. Effectful replay-safe tools still need
  their own idempotency.
- **Deterministic code-mode replay** (l.358): virtual clock, seeded randomness,
  journaled results fed back in order; a divergent path, a changed sandbox
  fingerprint, or an expired timeout fails the call instead of re-executing.
- **Crash-loop guard** (l.206): crashes that cut a task off are counted, and the
  task is parked as `restart_limit` after 5 (`settings.restartLimit`), with an
  explicit `retryTask`.
- **Idempotent submission** (l.120): the same `requestId` returns the existing
  submission.
- **Failure isolation** (l.809): a throwing tool gets an error result, a throwing
  `beforeTool` hook blocks the call, a throwing commit callback rolls back.
- **Storage**: memory, JSONL, SQLite, with conformance suites. **One process owns
  a storage; there is no cross-process locking** (l.774).
- **Telemetry** (l.80): an adapter boundary only; it "currently emits no spans or
  metrics."

**Verdict.** pi-durable is the strongest independent statement in this survey of
the contracts swe-term already wants: commit before show, intent before
effect, interruption as an explicit result, and an explicit replay policy per
tool. It confirms the swe-term ordering of journal → effect → receipt. Do not
port it as a runtime: it is experimental TypeScript, single-process by
construction, and adds a task scheduler (2,368 lines) whose generality swe-term
does not need for a single-session loop. Port the contracts.


### Verdict Changes in the 2026-10-10 Refresh
| Verdict | Change | Evidence |
|---------|--------|----------|
| God object (gap #1) | **Strengthened.** `AgentSession` grew 3,060 → 4,398 lines. | `packages/coding-agent/src/core/agent-session.ts` |
| URL sniffing (gap #3) | **Partly changed.** Upstream now has a real `Provider` interface with one module per provider — the fix swe-term planned. But `detectCompat()` URL sniffing remains and its list grew. Verdict: port the interface, still drop the sniffing. | `packages/ai/src/models.ts:150`; `packages/ai/src/api/openai-completions.ts:1593` |
| `chars/4` (gap #5) | **Corrected, not changed.** The trigger anchors on the last provider-reported usage; `chars/4` applies only to trailing messages. This was already true at `ff3e9df5f`. Gap still holds, with a smaller error surface. | `core/compaction/compaction.ts:196,298` |
| No interfaces between layers (gap #6) | **Partly changed.** `Provider` and `TUI` are interfaces; the agent core takes `streamFn` by injection (`setDefaultStreamFn`) and exposes `beforeToolCall`/`afterToolCall` hooks. `AgentSession` is still a concrete class. | `packages/agent/src/stream-fn.ts`; `packages/agent/src/types.ts:326,341`; `packages/tui/src/tui.ts:506` |
| `Promise.all` (gap #8) | **Minor.** Tool execution now has an explicit `sequential`/`parallel` policy; parallel still collects via `Promise.all`. | `packages/agent/src/agent-loop.ts:519,646` |
| `AbortController` sprawl (gap #7) | **Unchanged in kind.** Five singletons became four plus a `Set` for bash runs. | `agent-session.ts:403–420` |
| JSONL storage (gap #4) | **Strengthened.** Coding agent is still JSONL-only; upstream's new `pi-durable` adds SQLite storage — independent confirmation of the SQLite fix. | `packages/durable/src/storage/sqlite/` |
| Layering / scope | **Qualified.** The core four packages remain clean, but the repo is now 14 packages with an experimental durable/remote stack (`chord`, `pi-protocol`, `pi-server`, `pi-client`, `pi-durable`, `pi-env`). Port the core four only. 2026-10-18: `pi-protocol`/`pi-server`/`pi-client` were deleted (`7472cda0d`); 11 packages remain. | `packages/*/package.json` |
| MCP | **New port input.** MCP is built in (`pi-mcp` + bundled `mcp` extension), and `pi-codemode` runs tools from a QuickJS sandbox. Both are references for swe-term's MCP `Tool` source and sandboxed-execution tier. | `packages/mcp`, `packages/codemode`, `packages/coding-agent/src/extensions/` |

### Verdict Summary
pi-mono is the **best architectural template** of the agents surveyed — which is precisely why
swe-term chose to port it. Where Claude Code is a 512K-LoC monolith and Codex is a 120-crate
fortress, pi-mono is a **small, cleanly-layered set of packages**: `pi-ai` (providers) →
`pi-agent-core` (agent) → `pi-coding-agent` (coding) + `pi-tui` (UI). The layering is correct, the
scope is sane, and the agent core is genuinely provider- and domain-agnostic (Flue proves this by
building on `pi-agent-core` + `pi-ai` directly).

But pi-mono is a **Node/TypeScript hacker's tool**, and its implementation choices reflect that:
a **4,428-line** `AgentSession` god object, dynamic-`import()` extensions, URL-substring provider
sniffing as a fallback, `chars/4` estimation for the un-reported tail of compaction decisions, a JSONL tree with only
an in-memory index, `AbortController` sprawl, and `any` types in the hot path. None of these are
fatal — they're the difference between "great architecture, pragmatic implementation" and
"production harness."

**Port the layering and the loop. Fix the twelve gaps. The result is swe-term.**

### Assumptions to NOT Port
#### ❌ "One orchestrator object can own everything"

`AgentSession` is the anti-pattern. Decompose by concern; the loop is a function over interfaces.

#### ❌ "Extensions should be dynamically-loaded TS modules"

Great for a Node hacker tool; impossible for a static single binary and a security surface in-process.
Use Starlark + compiled Go hooks + sandboxed Monty.

#### ❌ "Provider differences can be sniffed from URLs"

Fragile. Declare capabilities; implement per provider behind an interface.

#### ❌ "`chars/4` is good enough for compaction decisions"

It's good enough until you're at the context limit and it's wrong. Use a real tokenizer.

#### ❌ "Append-only JSONL with tree-walking is sufficient storage"

Correct model, wrong engine at scale. Keep the tree, add an index (SQLite).

#### ❌ "`any` is an acceptable escape hatch"

Each `any` is an un-typed boundary. Use `json.RawMessage` for dynamic data and types everywhere else.

#### ❌ "Hand-roll the TUI"

Undifferentiated work. Borrow ratatui / Bubble Tea.

### What to Actually Port
| From pi-mono | To Harness | Why |
|-------------|------------|-----|
| Package layering (ai → agent-core → coding → ui) | The whole module plan | Correct dependency direction |
| Provider-agnostic agent core | `core/agent` depends only on interfaces | Domain-free loop |
| `Provider` interface (`pi-ai/src/models.ts:150`) + `Model.compat` flags | `Provider` interface + model registry entry struct | Declare, don't sniff (skip `detectCompat()`) |
| Agent loop shape (stream → tools → loop) | `core/agent/loop.go` | The cycle is right |
| JSONL tree *model* (id/parentId branching) | `session/tree.go` over SQLite | Branching is correct; add an index |
| Compaction-at-a-cut-point concept | `core/agent/compact.go` | Summarize old, preserve recent |
| Interactive/print/json/rpc mode split | `Frontend` implementations | Frontends are interchangeable |
| Zero-build iteration *spirit* | Starlark hot-reload of tool defs | Keep the DX, lose the dynamic import |
| `beforeToolCall` / `afterToolCall` loop hooks | `Hook` interface | Pre/Post tool-use is a good model |
| `pi-mcp` client + `codemode` nested tool calls | MCP `Tool` source; sandboxed (Monty) execution tier | Built-in MCP and out-of-context tool composition now have a reference |
| pi-durable: tool intent committed before `execute()`; `replay: "safe"` or an `interrupted` result (2026-10-18) | Tool manifest gains a replay policy; journal row before effect; restart emits `interrupted`, never a silent rerun | Crash semantics become a declared per-tool contract (Invariant 9) |
| pi-durable: keyed nested calls; deterministic code-mode replay with fingerprint check (2026-10-18) | Monty tier journals tool results; rerun fails on divergence | Code mode can resume without re-running effects |
| pi-durable: crash-loop guard parks a task after N cut-offs (2026-10-18) | Session resume counts crashes; park with an explicit retry | Prevents resume-crash loops |
| pi-durable: idempotent `requestId` on submit (2026-10-18) | SDK `submit` takes a request ID | Client retries after reconnect do not double-submit |
| pi SDK `createAgentSession` (`coding-agent/docs/sdk.md`) | Python and TypeScript SDK client ergonomics (ARCHITECTURE.md §9) | The API shape to match, over the protocol instead of in process |

## Codex

> Upstream: https://github.com/openai/codex at `cc7ba3360`, revalidated 2026-10-10. General architecture notes:
> source-insights entry `ae0b5877a3df5404758accb9`.


### Verdict Changes (2026-10-10 revalidation)
| Verdict | Change | Why (evidence) |
|---------|--------|----------------|
| Port the protocol seam | **Strengthened** | Codex dropped its third façade (`codex mcp-server`, `531f3836a`); the app-server is now the only external surface, and `Submission` became core-internal (`core/src/session/submission.rs`). |
| Do not embed V8 | **Refined** | Codex itself moved V8 out of the agent process into a `codex-code-mode-host` binary over gRPC/stdio (`aa46f2deb`, `2dc0791c7`). That validates our subprocess (Monty) direction; it does not make V8 lighter. |
| "Two of everything" | **Partly resolved upstream** | `execpolicy-legacy` (`656a2d090`), legacy remote compaction (`1ac689cc7`), `core-skills` (`45f8cafa4`), and the `Op::UserTurn` alias are deleted. Multi-agent v1+v2 and the `ConversationManager = ThreadManager` alias (`core/src/lib.rs`) remain. |
| One DB, not four | **Strengthened** | SQLite files grew from 4 to 8 (`state/src/sqlite.rs`, `ext/agent-message-board/src/local/pool.rs`). |
| No LLM in the approval hot path | **Strengthened** | Guardian now has two implementations (`ext/guardian-reviewer`, ~16K-line `ext/guardian-v2` async scorer) plus a user override op. |
| Port the 10 hook events | **Updated** | Now 12 (`SessionEnd`, `Interrupt`) plus an `McpTool` handler (`hooks/src/lib.rs`). |
| Responses-API monoculture, shared-mutable session, agent identity | **Unchanged** | `WireApi` still has only `Responses`; the new `amazon-bedrock-runtime` provider uses Bedrock's OpenAI-compatible endpoint. `Mutex<SessionState>` remains (the I/O split, `e4711f2a3`, did not change it). |

### Verdict Summary
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

### Assumptions to NOT Port
#### ❌ "The OpenAI Responses API is the universal model protocol"

Codex bets the whole core on it, including server-side compaction. A provider-agnostic harness
cannot make this bet. Responses-specific capabilities belong *behind* the `Provider` interface.

#### ❌ "Ship V8 to get programmatic tool orchestration"

Code mode is a good idea executed with a heavy hammer. Get the batching benefit with a lightweight
embedded interpreter (Starlark) or a sandboxed subprocess (Monty), not a browser JS engine. Codex
now runs V8 in a separate host process rather than in the agent binary — which fixes isolation but
not weight.

#### ❌ "Shared mutable session state behind a lock"

Works in lint-enforced Rust; a liability in Go. Prefer immutable snapshots threaded through the loop.

#### ❌ "Four databases, two policy engines, v1+v2 of everything"

Accretive product growth. The Harness answer is one DB, one policy mechanism, and compile-time
deletion of superseded code.

#### ❌ "A second LLM (Guardian) in the approval path by default"

Guardian auto-reviews `on-request` approvals with a model call (fail-closed, timeout, circuit
breaker). It's clever, but it puts a latency- and cost-bearing model in the hot path of *running a
command* — and since June it has grown a second, asynchronous background scorer
(`ext/guardian-v2`, ~16K lines) beside the synchronous reviewer. For Harness, approval should be deterministic (rules + sandbox); an LLM reviewer, if any,
is an *optional extension subscriber*, not core.

#### ❌ "Agent identity needs Ed25519 keys and a backend JWKS"

`agent-identity` (Ed25519 + JWT + backend registration) is infrastructure for OpenAI's hosted
multi-agent/cloud story. A local-first harness does not need cryptographic agent identity to spawn
a subagent.

#### ❌ "Surfaces should be 450K lines"

The TUI crate is enormous (~450K lines, up from ~290K; `chat_composer.rs` ~12.7K lines). The protocol seam is excellent; the
*frontend* is a monolith. Harness should keep the TUI thin (Bubble Tea / ratatui) precisely
*because* the protocol seam lets it stay thin.

### What to Actually Port
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

## Claude Code

> Upstream: https://code.claude.com/docs (closed source; public docs and changelog) at `v2.1.296`, revalidated 2026-10-10. General architecture notes:
> source-insights entry `0944512937b0ba37b6a4e197`.

### Verdict changes (2026-10-10)

1. *Permission system* (§1): strengthened, and its mechanism corrected. Auto mode is now the default
   starting mode (v2.1.283) and uses a model classifier, not only pattern-matching validators.
2. *Port table* (§6): "Bash safety patterns" stays, and an OS-level sandbox row is added. Public evidence
   now shows deterministic egress/credential controls (`sandbox.network.strictAllowlist`,
   `sandbox.credentials`) carrying the safety load beside the classifier.

All other verdicts are unchanged: no public evidence bears on them.

### Verdict Summary
Claude Code is an impressively **feature-complete product** but a **poor architectural template** for a reusable agent harness. It was built as a monolithic application first and had modularity bolted on second. The 512K LoC codebase conflates product concerns (onboarding wizards, sticker commands, companion sprites) with agent infrastructure (tool execution, state machines, event streaming), making it impossible to extract the harness without carrying 400K lines of product.

**Do not port the architecture. Port only the lessons.**

### Assumptions to NOT Port
#### ❌ "React is appropriate for terminal UIs"

Claude Code invested ~15K lines in a custom Ink fork. The result is a capable but over-engineered rendering layer. Terminal UIs don't need virtual DOM diffing, Yoga flexbox, or a React reconciler. Bubble Tea's Elm architecture (Model → Update → View) achieves equivalent results with a fraction of the complexity.

#### ❌ "Everything is a slash command"

Claude Code has ~70+ slash commands. Many are product features (`/stickers`, `/buddy`, `/mobile`, `/desktop`, `/chrome`) that pollute the namespace. The command system is a flat registry with no hierarchy or discovery mechanism beyond typeahead.

Harness should have a small set of core commands and let extensions register their own without modifying the core registry.

#### ❌ "The tool type should carry the universe"

`ToolUseContext` in Claude Code is a superset of AppState, permissions, settings, model config, file history, MCP state, and analytics. Every tool receives the entire application context. This makes tools impossible to test in isolation and tightly couples them to the product.

Harness tools should receive `context.Context` (for cancellation), `json.RawMessage` (for arguments), and nothing else. If a tool needs file I/O, it uses the OS. If it needs settings, it reads config. It does not receive the entire application state.

#### ❌ "Multi-agent requires product-level orchestration"

Claude Code's swarm system (`utils/swarm/`, ~30 files) implements tmux backends, iTerm2 setup, layout managers, permission sync, leader election, and teammate model selection. This is product-level orchestration built for a specific UX (split panes in the terminal).

Harness multi-agent should be simple: the `AgentTool` spawns a new agent loop with its own state. The parent communicates via the event bus. The frontend decides how to render it. No tmux, no layout managers, no pane backends.

#### ❌ "Feature flags are architecture"

20+ feature flags means 20+ code paths that may or may not exist at runtime. This creates a combinatorial testing problem and makes the codebase impossible to reason about statically. In Harness, a feature is either a compiled module or it doesn't exist. Go build tags serve the same purpose without runtime conditionals.

#### ❌ "Analytics and telemetry belong in the core"

Claude Code has analytics woven throughout: GrowthBook feature flags in the query engine, Datadog in the API client, OpenTelemetry spans around tool execution, first-party event logging in permission decisions. This creates coupling between the agent loop and Anthropic's infrastructure.

Harness emits events to the event bus. An analytics extension subscribes and exports wherever it wants. The core has zero knowledge of analytics providers.

#### ❌ "The bridge (IDE integration) is a core concern"

Claude Code has ~30 files for bridge communication, 7+ `replBridge*` fields in AppState, and bridge-safe command filtering. The bridge is deeply coupled — the REPL checks bridge state, commands check bridge safety, permissions delegate to the IDE.

In Harness, the bridge is a `Frontend` implementation. It implements the same interface as the TUI. The core doesn't know it exists.

### What to Actually Port
| From Claude Code | To Harness | Why |
|-----------------|------------|-----|
| Permission rule system | `core/approval` | Glob-based allow/deny rules are a good pattern |
| Tool concurrency partitioning | `core/agent/loop.go` | Read-only parallel, write sequential is correct |
| Startup prefetch pattern | `cmd/harness/main.go` | Preconnect, prefetch config in parallel |
| Compaction concept | `core/agent/compact.go` | Multi-strategy with memory extraction |
| Token budget tracking | `core/agent/budget.go` | Context window management is essential |
| Bash safety patterns | `ext/tools/terminal/safety.go` | Dangerous pattern detection, path validation |
| OS sandbox + default-deny egress (added 2026-10-10) | approval/sandbox boundary | Seatbelt/bubblewrap isolation, strict host allowlist, credential masking (https://code.claude.com/docs/en/sandboxing) |
| MCP client architecture | `ext/mcp/client.go` | Multi-transport, auth, elicitation |
| Structured diff rendering | `ext/tui/components/diff.go` | Good UX pattern |

## opencode

> Upstream: https://github.com/anomalyco/opencode at `501b15af26` (v2.0.32), revalidated 2026-10-18
> (previous pin `823ac87f58`, v2.0.26). General architecture notes:
> source-insights entries `fb8c4bcd0bc7329b73ca9fff` (2026-10-10) and `3e59a50483011038620159f3` (2026-10-18 refresh).

### Changes since `823ac87f58` (293 commits, 2026-10-10 → 2026-10-18)
Most commits are TUI dialog redesign, console marketing, and provider quirks.
The harness-relevant ones:

- **Bounded retry policy** (`core/src/model-retry.ts`, new): exponential 2s
  capped at 10s per gap, 10 retries, jitter; provider `retry-after` honored
  but capped at 15 min; transport timeouts get a separate cap of 3 retries.
  Failed attempts never contribute to returned output.
- **Output-limit outcome** (`session/runner/step.ts:41,282`,
  `session/runner/llm.ts:290`): `finish === "length"` is its own
  `OutputLimit` outcome; the runner continues, and fails the step after three
  consecutive length-finished assistant messages.
- **Incomplete tool input fails explicitly** (`publish-llm-event.ts:406`
  `failIncompleteTools`): a call whose arguments never finished streaming is
  failed, not executed.
- **Shell exit after capture** (`core/src/shell.ts`, #56321): exit and timeout
  race on one `Deferred`; the exit is reported only after output is captured.
- **Line-ending-preserving edit** (`tool/plugin/edit.ts:104–200`): match on
  the LF view the model saw, map offsets back, and reuse each region line's
  original ending via LCS alignment, so mixed CRLF/LF files keep untouched bytes.
- **Product mass keeps growing**: experimental voice session endpoint
  (`core/src/voice.ts`, #55951), iOS voice delegations, device pairing.
- **Unchanged**: tool calls still fork per call (`step.ts:133`
  `Effect.forkScoped`); still no OS sandbox in `packages/core/src` (no
  Seatbelt/bubblewrap/Landlock hits — #55817 only *tolerates* denials from an
  external sandbox); plugins still in-process. Runner is now 363 lines
  (`index.ts` 42 + `step.ts` 321).


### Verdict Summary
opencode V2 is the **best-structured mainstream harness in this survey** and
the one whose *shape* is closest to swe-term's plan. It ships the protocol
seam swe-term wants (core behind a typed `HttpApi`, every UI a client), an
event-sourced session that the loop only reads from, permission as ordered
data with last-match semantics, and a loop split into files of a few hundred
lines. There is no 9k-line session object.

It is also a **product platform wearing a harness**. The app UI alone is
~154k lines. `@opencode/ai` carries 49 protocol files including video and
speech. Plugins are npm packages installed and run in-process. There is no OS
sandbox. Effect is load-bearing everywhere, so the architecture cannot be
read, or ported, without the framework.

**Steal the seams:** history-only `drain`, durable inbox with steer vs
next-turn, typed `once|always|reject|corrected` replies, `opaque`-not-guess
shell scanning, tool filtering ≠ authorization, explicit `truncated`/
`timeout` output, checkpoint compaction with verbatim recent tail, subagent
depth default 1. **Refuse the platform:** in-process npm plugins, a
framework-as-architecture, unbounded parallel tool fan-out, a second
provider SDK, and a sandbox story that is a permission prompt.

### Assumptions to NOT Port
#### ❌ "Precise permission prompts are a sandbox"
`shell/scan.ts` is excellent prompt input. It is not confinement. swe-term
keeps fail-closed argv sandboxing; port the scanner's `opaque` contract only.

#### ❌ "Every tool call should start the moment it streams"
`step.ts:126–137` forks per call. swe-term needs an effect declaration
(read/write/exec) and schedules writes serially.

#### ❌ "Plugins are npm packages loaded in-process"
`plugin/module.ts:104–109`. swe-term plugins are interfaces in-process for
first-party code and sidecars over a protocol for third-party code.

#### ❌ "Own the provider SDK and also keep the other one"
`packages/ai` + `core/src/aisdk.ts`. swe-term keeps one `Provider` port.

#### ❌ "A framework can be the architecture"
Effect makes opencode's invariants enforceable *inside* Effect. swe-term
encodes the same invariants in Go types and `context.Context`.

### What to Actually Port
| Pattern | opencode source | swe-term shape |
|---|---|---|
| History-only continuation | `session/runner/index.ts` `drain` | `Loop.Drain(sessionID)` reads `SessionStore`, takes no messages |
| Durable inbox, steer vs next-turn | `session/inbox.ts`, `Promotable` | Inbox table; `steer` amends active turn, else queue |
| Evented execution lifecycle + claims | `session-event.ts`, `execution/restart.ts` | `execution.{started,succeeded,failed,interrupted}` rows + claim attempt for crash recovery |
| Last-match ruleset, default ask, empty = deny | `permission.ts:19,87–97` | `[]Rule{Action,Resource,Effect}`; pure `Evaluate` func |
| Typed replies incl. corrected | `schema/permission.ts:42`, `CorrectedError` | `Reply{Once,Always,Reject,Corrected(msg)}` |
| Scanner returns `opaque` with reason | `shell/scan.ts` | Use `mvdan.cc/sh` parse; unparseable ⇒ ask |
| Visibility ≠ authorization | `tool/AGENTS.md` | Registry filters schemas; tools call `Permission.Assert` |
| Per-request tool snapshot | `tool/AGENTS.md` "Registration" | Immutable `ToolSet` value per request |
| Edit fallback tiers | `tool/plugin/edit.ts:49–54,161–167` | exact → Unicode-normalized → trailing-ws; report tier used |
| Checkpoint compaction + verbatim tail | `session/compaction.ts:45,160–173,228` | Summary checkpoint updated in place; protected recent tail |
| Bounded subscriber, loud overflow | `server/event-feed.ts` | Per-client buffered chan; overflow closes stream with error |
| Subagent depth default 1, background notify | `tool/plugin/subagent.ts:118–132` | Already planned; confirms default |
| Bounded retry, capped `retry-after`, timeout sub-budget (2026-10-18) | `core/src/model-retry.ts` | `Provider` retry wrapper; each attempt a journal row; no partial output from failed attempts |
| `OutputLimit` outcome + 3-strike exhaustion (2026-10-18) | `step.ts:282`, `llm.ts:290` | Distinct loop outcome; continuation counted from `SessionStore`, not memory |
| Incomplete tool input ⇒ failed, never run (2026-10-18) | `publish-llm-event.ts:406` | Invariant 9: explicit `failed` outcome with repair hint |
| Exit reported only after output captured (2026-10-18) | `core/src/shell.ts` #56321 | Shell tool returns one terminal record: status, exit, captured tail |
| LF-view matching with per-line ending reuse (2026-10-18) | `tool/plugin/edit.ts:104–200` | Edit tool tier 0; property test on mixed-ending files |

### Where swe-term must beat opencode V2
opencode is the bar for the shape. swe-term wins only on axes opencode has
structurally declined, each with a check that makes the claim falsifiable.
`PLAN.md` "Competitive bar" maps these to phases.

| Axis | opencode V2 (`501b15af26`) | swe-term target | Check |
|---|---|---|---|
| Confinement | Permission prompts only | OS sandbox + default-deny egress under the permission layer | Escape fixtures: write outside root, network, credential read all fail |
| Tool scheduling | Fork every call as it streams | `EffectDeclaration`-driven: reads parallel, writes/exec serial | Race fixture: two writes to one file never interleave |
| Extensions | In-process npm plugins | First-party in-process; third-party sidecars over protocol | Fitness test: no third-party code in core process |
| Definition of done | Model stops calling tools | Obligations + receipts + V&V rung gate | `ReceiptGate` blocks exit with an open obligation |
| Context | Read/grep on demand | Analyzer `ContextPacket` (blast radius, coupling) before the model decides | Benchmark vs. no-packet baseline |
| Footprint | Bun/TS + Effect, ~154k-line app | Single static Go binary, framework-free core | RSS and cold-start budget in CI |
| Loop size | 363 lines | Stay under it | `cmd/drift` line ratchet on loop package |

## oh-my-pi

> Upstream: https://github.com/can1357/oh-my-pi at `fd27203d27`, revalidated 2026-10-10. General architecture notes:
> source-insights entry `0d0705d0d8abaa22bc64af45`.


### Verdict Summary
oh-my-pi is the **highest-quality tool surface** in this survey and the
**clearest demonstration of how a pi fork dies of product success**. The
layering is still pi. The pass-rate story is real: edit format, summarized
reads, and in-process grep move the needle more than another model wrapper.
The Rust natives are serious. The catalog/dialect split is cleaner than
upstream. JSONL + SQLite metadata is the session design swe-term already
chose.

It is also a **kitchen sink with a 13,598-line `AgentSession`**, an in-process
N-API ABI that is the opposite of swe-term's sidecar rule, eval kernels that
can invoke the full tool registry (including bash and subagents), and a
Bun-locked platform. snapcompact is clever research, not a default compaction
strategy. 92 provider rule files and 23 web-search backends are product, not
harness.

**Steal hashline, per-model tool contracts, summarized reads, ToolSession,
JSONL+SQLite, prompt-as-files, and the idea of Rust engines. Implement those
engines as sidecars. Do not steal N-API-as-plugin-surface, the god object, or
the IDE-complete core repo.**

### Verdict changes since 17.3.5
No verdict flipped. Four got stronger evidence, and one port item was added.

- **God object — strengthened.** `AgentSession` grew 9,428 → 13,598 lines
  (`packages/coding-agent/src/session/agent-session.ts`).
- **Eval → full tool registry — severity raised.** `browser` and `computer`
  are now reachable *only* through eval (`64508e1259`), and Codex Code Mode
  routes every non-keep-set tool through the eval bridge
  (`session/code-mode.ts`). Eval is now the tool gateway, not an escape hatch.
- **N-API as native ABI — strengthened.** The in-process surface widened to
  edit apply (`crates/pi-edit`), git/jj (`crates/pi-vcs`), and text prediction
  (`crates/pi-predict`). Non-vendor Rust grew ~171k → ~302k lines.
- **Hashline in Rust — confirmed by omp itself.** omp moved its edit engine
  out of TS into `crates/pi-edit` (`c478e41a04`). That supports "edit engine in
  Rust"; the sidecar-vs-in-process disagreement is unchanged.
- **Added port item: notes-backed context windows.** A summarizer-free
  rollover (`docs/compaction.md`, `54cae2c717`) is a new mechanical
  compaction strategy worth taking.
- **Session-index gap narrowed.** Titles and recaps are now indexed in
  `history.db` (`session/session-index.ts`); transcript replay is still a
  JSONL scan.

### Assumptions to NOT Port
#### ❌ "Native code should load in-process via N-API"

Wrong plugin surface for a Go harness. Sidecar + stdin/stdout/json.

#### ❌ "The eval kernel should call every agent tool"

Powerful, unbounded. Closed host RPC or nothing.

#### ❌ "The coding-agent package is where features go"

That is how you get 72k lines of tools/ (42k at 17.3.5) and a dead upstream.

#### ❌ "Vision-model compaction is a general strategy"

Keep mechanical elision. Leave snapcompact as a paper.

#### ❌ "One edit format for all models" *or* "hashline for all models"

omp's own exclusions (kimi/mimo/minimax/deepseek/stepfun classes →
replace, `utils/edit-mode.ts`) and a fifth mode (`sloppy`) prove the table
must be data. Do not pick a single grammar and pray.

### What to Actually Port
| From omp | To Harness | Why |
|----------|------------|-----|
| Hashline / snapshot-verified apply | Edit sidecar (Rust) + Go adapter | First-attempt edits, fail closed |
| Per-model edit variant table | Model registry field | Tool contract is adapter data |
| Summarized read + structured grep | Default tool behavior + caps | Token economy |
| `ToolSession` / `ToolContext` | Tool constructors take context | Policy and model flow in |
| JSONL transcript + SQLite metadata | Already in PLAN | Confirm; do it |
| Prompts as files, stable prefix | Prompt templates on disk | Cache + reviewability |
| Mechanical compaction (shake, elision) | `compact.go` strategies | Cheap context wins |
| Notes-backed rollover (`context_notes`/`new_context`) | Compaction strategy | Summarizer-free; agent-owned bounded notebook |
| Worker/subcommand re-entry | `st worker <engine>` | One user-facing binary |
| Harness metrics (tool retry, edit fail) | Event schema | You cannot improve what you do not log |
| LSP as an adapter, not the loop | Optional sidecar | IDE knowledge without core mass |
| Inband dialects for tool-less models | Provider compat flags | Keep pi's compat instinct, data-driven |

## Prime Agent

> Upstream: https://github.com/PrimeIntellect-ai/prime-agent at `2ee7e62`, revalidated 2026-10-10. General architecture notes:
> source-insights entry `67cb29023017f3020a13d8ef`.


### Verdict changes (2026-10-10)
| Verdict | 0.7.2 | 0.10.0 | Why |
|---------|-------|--------|-----|
| God-object session class | **Avoid** — 11,288-ln `AgentSession` | **Resolved upstream**; lesson kept | Rust `SessionEngine` is 971 ln in a 61-file module; largest non-generated file is 2,235 ln (`crates/pa-daemon/src/acp/daemon.rs`) |
| IPython-as-the-loop | Avoid — Jupyter/ZMQ stack | **Avoid (unchanged), cost lower** | Kernel is now `rlm.repl` over NDJSON; `ipython` is still the only LLM tool (`session_engine/engine.rs:432–443`) |
| Three-process daemon as v1 | Avoid | **Avoid (unchanged)**, now client + supervisor + worker | Catalog subprocess removed (`crates/pa-daemon/src/main.rs:15–43`) |
| Dual harness writers | Avoid — "will desync" | **Avoid (unchanged), risk reduced** | Writers now share a cross-process lock (`97b4a03`); two schema owners remain |
| Default depth 1 is "correct for v1" | Port depth 1 | **Port depth 1; note Prime moved to 2** | `9e49b73`; `DEFAULT_RLM_MAX_DEPTH = 2` |
| Admission-not-await | Port | **Port, strengthened** | `rlm.collect(timeout_ms)` adds a bounded typed join |
| Protocol versioning | Port | **Port, strengthened** | Schema 16 → 30 and a full language rewrite without a protocol bump |
| Extensions next to a privileged kernel | Gap | **Removed upstream** | `f54551d` deleted user extensions |

### Verdict Summary
Prime Agent is the most serious **long-running-agent** design in this survey. It
answers a question pi-mono left open: *what happens when the human disconnects,
the task spans hours, and the agent needs children, memory, and a programming
environment that outlives a turn?* The answers — detachable workers, a
capability-gated local protocol, a rollbackable harness, typed host requests,
admission-not-await subagents — are real engineering, not slideware.

It was also a **warning about gravitational collapse**. At 0.7.2 `AgentSession`
was an **11,288-line** god object, and it reached 15,101 lines before the Rust
port (`39bc99a^`). The port dissolved it: the largest non-generated Rust file is
now 2,235 lines. The lesson stands — Prime needed a full rewrite to escape the
class. The product still replaces a multi-tool LLM surface with a Python REPL that
is explicitly **not a sandbox** (`README.md:87`). Default RLM depth moved from 1 to
2 (`9e49b73`): the authors loosened the cap, but still cap it.

**Steal the execution/UI split, the protocol versioning, the harness
snapshot/rollback, and the host-request idea. Do not steal REPL-as-the-only-tool
or grow a god session class (Prime had to rewrite to escape one). swe-term should remain a Go loop over
interfaces; Prime is a product built *on* that loop, not a replacement for it.**

### Assumptions to NOT Port
#### ❌ "The model should program the harness in Python"

A research-agent bet. swe-term's default surface is typed tools + optional
code-exec, not a REPL that *is* the product.

#### ❌ "Lifecycle isolation is isolation"

Workers/kernels are crash domains. They are not sandboxes. Do not market
process splits as security.

#### ❌ "One session object can absorb RLM, goals, refine, messaging, and the kernel"

It cannot. Prime is the existence proof: the class reached 15,101 lines and was
only split by a full Rust rewrite.

#### ❌ "Deep recursive subagents are the architecture"

Prime's default is now 2 (`9e49b73`), still capped. Treat multi-agent as a tree with a hard cap, not as unbounded
recursion.

#### ❌ "Harness state can have two writers"

Pick one owner. In swe-term that owner is Go.

### What to Actually Port
| From Prime Agent | To Harness | Why |
|------------------|------------|-----|
| Client ≠ runtime | `Frontend` over protocol | Detach, headless, TUI share one loop |
| Protocol version + schema revision + capabilities | ACP/JSON-RPC evolution | Additive features without silent breaks |
| Host-request closed RPC | Sidecar → core adapter | Privileged ops stay in Go |
| Subagent admission handles + bounded collect + depth cap | Job IDs, `collect(timeout)`, max depth 1 default | Parallelism without await-all |
| Spawn ledger separate from transcript | Subagent table in SQLite | Topology recoverable after compact |
| Harness snapshot + rollback | Versioned memory records | Agent-editable ops manual, reversible |
| Catalog/index without loading runtimes | Session index queries | `st agents` stays cheap |
| Rewrite behind a frozen wire/disk contract | Golden wire/disk fixtures | Prime ported TS → Rust with byte parity and no protocol bump |
| Autonomous bounds + quality gates | Optional continuation policy | Overnight work without pretending success |
| steer vs follow-up inbox | Already in pi loop; keep it | Interrupt vs queue is the right pair |

## DeepSeek Harness

> Upstream: https://github.com/deepseek-ai/deepseek-harness at `d743267388`, revalidated 2026-10-10. General architecture notes:
> source-insights entry `c2cb0429e9433eeda41fd7d0`.


### Verdict Changes (2026-10-10)
| Verdict | Before | Now | Evidence |
|---------|--------|-----|----------|
| Dispatch-time reconstruction invariant | dsh enforces it at runtime | dsh **removed** the runtime check; reconstruction is by construction (derive, freeze, brand) plus a test. Still the top steal, but swe-term keeps the runtime check dsh dropped. | `f028f25667`; `packages/core/agent-loop/src/agent.ts:688–700`; `packages/core/agent-loop/tests/request-reconstruction.spec.ts` |
| `JSON.stringify` equality footgun | Live concern | Moot: that code was deleted with the invariant | `f028f25667` |
| `SESSION_FORMAT_VERSION = 0` / "migrate or abandon" | Open risk | Resolved toward **migrate**: format 4, adjacent migration chain, V4 recorded as released | `packages/core/session/src/types.ts:70–89`; `docs/session-format-status.md` |
| Package explosion | 226 packages | **Worse**: 340 manifests, ~1.10M TS/TSX lines | `packages/*/*/package.json` |
| In-process `eval` self-modification | Research demo in `dsh-tool-cordis` | Narrowed: `cordis_define` moved to `dsh-cordis-host-runner` with human-approved client activation, web-app bundle only. "Do not port" stands. | `packages/extensions/cordis-host-runner/src/index.ts:1–5`; `packages/bundle/web-app/cordis.patch.yml` |
| Invariant-companion bureaucracy | Every package owns `./invariant` | Removed; other gates (per-file 100% coverage, doc budgets, bilingual docs) remain | `f028f25667`; `vitest.config.ts:377–387`; `AGENTS.md:173` |

Unchanged: seams, waterfall phases, turn/step vocabulary, fail-closed sandbox,
profiles-as-patches, `AgentFactory` dependency direction, "do not port Cordis".

### Verdict Summary
DeepSeek Harness is the **best written harness architecture** in this survey and
the **worst template to copy at face value**. The loop is 705 lines. The session
log is the only model history. Every loop request is derived from the log,
deep-frozen, and branded; a test proves reconstruction (the runtime invariant
was removed in `f028f25667`). Capability seams mean swapping the filesystem backend
moves Bash and LSP with it. Sandbox missing → fail closed. Profiles are ordered
patches. ACP is a thin adapter over `ctx.agents`.

It takes **340 packages**, a vendored actor/DI framework, per-file 100%
coverage, generated catalogs, bilingual doc budgets, and a Typert type-graph
gateway to hold that architecture in TypeScript. That is an org-scale machine
for a product that still advertises breaking changes, though it now migrates
session logs (`SESSION_FORMAT_VERSION = 4`).

**Import the log-centric loop contract and the seam taxonomy. Do not import
Cordis, the package explosion, or the gate factory. A Go core can be an order
of magnitude smaller while keeping the ideas that actually make agents
correct.**

### Assumptions to NOT Port
#### ❌ "Everything is a plugin, including the loop"

The loop should be a function. Plugins attach at named hooks. Making the driver
itself a YAML row is cute and makes the event schema the *actual* core anyway.

#### ❌ "A DI/actor framework is required for replaceability"

Go interfaces + a registry are enough. Cordis solves TypeScript's lack of a
boring module system.

#### ❌ "100% coverage and generated catalogs keep architecture honest"

They keep *this repo* honest at lab scale. They will not keep swe-term small.

#### ❌ "The agent should mount new plugins into its own process"

Inspect, yes. `eval` of model-authored JS, no.

#### ❌ "Host and client are two Cordis trees"

One core, many frontends. The browser is a frontend, not a second runtime.

### What to Actually Port
| From dsh | To Harness | Why |
|----------|------------|-----|
| `deriveMessages()` + frozen branded request (dsh dropped its runtime check) | `SessionStore.Messages()` checked before `Provider.Stream` | Ghost context becomes a failed test |
| Turn/step durable events | Protocol event types | Resume, ACP, telemetry share units |
| Waterfall hook phases | `Hook` with must-continue | Extension without loop forks |
| Seam triple (def/provider/consumer) | Go interfaces | Swap sandbox world, not tools |
| Profile = ordered patches | Headless vs TUI as config layers | One binary, multiple compositions |
| Fail-closed sandbox argv wrap | `Sandbox` adapter | Missing confinement is an error |
| `AgentFactory` vs protocol adapters | ACP talks to core, not loop internals | Dependency direction |
| Branded ids | Distinct Go types | SessionId ≠ string |
| `steer` / `followup` / `inject` | Keep from pi; dsh names them cleanly | Three inbox lanes are enough |
| SQLite schema version ≠ log version | Two version numbers | Physical layout vs event semantics |

## Flue

> Upstream: https://github.com/withastro/flue at `6583259`, revalidated 2026-10-10. General architecture notes:
> source-insights entry `774b982199ed94df84c939c9`.

### Verdict changes (2026-10-10)

(details inline, marked *Updated*):
1. *Ephemeral sessions on Node* — softened. Node now has a durable `sqlite()` store and
   Postgres/libSQL/MySQL/MongoDB/Redis adapters; the default is still in-memory.
2. *Durability is Cloudflare-coupled* — largely withdrawn. 2.x defines one submission
   contract with identical recovery decisions on both targets; Cloudflare keeps an operational
   edge (structural single owner vs Node leases).
3. *`just-bash` sandbox* — narrowed. It is no longer the default; there is no implicit sandbox.
4. *Pre-1.0 instability* — restated. Flue is `2.x`, but 2.0 was a breaking redesign with a
   reset-only persisted schema, so the instability verdict stands on new evidence.
5. *What to port* — roles row replaced (roles are gone); accepted-work contract and durable
   tools added.
Unchanged: borrowed core, MCP client-only, build-step overhead, no CAS/analyzer, inherited
provider layer.

### Verdict Summary
Flue is the most **conceptually interesting** agent in this survey because it asks a different
question. Claude Code, Codex, and pi all answer *"how do I build a great terminal coding agent?"*
Flue asks *"what if the agent harness were a deployable framework, like Next.js for agents?"* That
reframing — **agent = a directory compiled into a server artifact, headless by default, deployable
to Node or the edge** — is genuinely novel and worth taking seriously.

But as an *architectural template for swe-term*, Flue is the weakest fit of the four. It
**churns hard** (0.7.0 → 1.0 betas → a breaking 2.0 redesign in under two months; now `2.2.2`), it is a **thin framework over
pi-mono's engine** (so its agent-loop ideas are really pi's — Flue's `Session` literally wraps pi's
`Agent`), its best durability ergonomics are still **Cloudflare-shaped** (one Durable Object per
conversation), and its built-in sandboxes are **emulation (`just-bash`) or none (`local()`)** —
real isolation means a remote provider. Flue is a
deployment-and-packaging layer, not an agent core.

**Steal the framework framing and the headless/deployment model. Do not look to Flue for the core —
that's pi-mono's job, which swe-term already ports.**

### Assumptions to NOT Port
#### ❌ "An in-process virtual bash is a sandbox"

`just-bash` is a convenience, not a security boundary. Real isolation needs OS mechanisms (Codex's
model) or remote containers. Don't conflate emulation with isolation.

#### ❌ "Durability is opt-in per deployment"

*Updated 2026-10-10:* Flue fixed the contract but kept an in-memory default on Node. swe-term's
persistence must be on by default (local SQLite) regardless of environment.

#### ❌ "Agents need a bundler/build pipeline"

True for multi-runtime TypeScript; false for a Go single binary. swe-term's build is `go build`.

#### ❌ "Runtime-agnostic" while best-on-one-vendor

The aspiration is right; the reality is Cloudflare-favored. A zero-lock-in harness should be equally
capable everywhere, which for swe-term means *one binary, one local data dir, no platform services*.

#### ❌ "MCP client-only (HTTP, no stdio, no OAuth) is enough"

It's a pre-1.0 gap. A serious harness is both MCP client and server, multi-transport.

#### ❌ "Borrow the engine, ship the framework"

Reasonable for Flue's goals; the opposite of swe-term's. swe-term's value *is* the engine. Borrowing
pi's loop wholesale would defeat the learning purpose and the "without its gaps" thesis.

### What to Actually Port
| From Flue | To Harness | Why |
|-----------|------------|-----|
| Headless-first framing | `Frontend` interface design | Print/RPC frontends are first-class, not afterthoughts |
| "Agent as a deployable artifact" | Phase 7 recursive self-build | swe-term compiles to *itself* instead of a server |
| Primitive taxonomy (session/role/task/sandbox/skill/tool/MCP) | feature checklist | Confirms the surface area a harness needs beyond the loop |
| Subagents via a framework `task` tool + catalog in the system prompt | `AgentTool` semantics | Delegate runs in fresh context, own durable transcript; only final answer returns |
| Accepted-work contract: durable admission → exactly one terminal outcome | session/turn state machine | Crash recovery from durable evidence, not in-memory state |
| Tool-batch repair: unresolved calls → explicit unknown-outcome error | tool execution recovery | Never silently re-run a side effect after a crash |
| Durable tools with `step.do` checkpoints | long-running tool contract | Exactly-once recorded, at-least-once executed |
| Host-side tool execution, schema-only to model | tool secret hygiene | Credentials never enter the context |
| Typed results via schema (`prompt({ result })`) | structured-output tool | Agents as typed functions, not just chat |
| `dev` / `run` watch-and-reload DX | dev ergonomics | Fast iteration loop for tool/skill authoring |

## Deep Agents

> Upstream: https://github.com/langchain-ai/deepagents at `9f0e39d`, revalidated 2026-10-10. General architecture notes:
> source-insights entry `6486e54b9235e8725341506e`.


### Verdict changes since 0.6.7 (2026-06-06)
Evidence: upstream changelogs at the commit above.
Only these verdicts moved; everything else stands as written.

| Verdict | Was | Now | Why |
|---|---|---|---|
| Harbor adapter pattern (§5, §8.6) | Steal `DeepAgentsWrapper(BaseAgent)` + `HarborSandbox`; agent on orchestrator, tool I/O in container | **Revised.** Steal "ship the agent as a standard graph artifact that the eval host runs in the container" | Both classes are gone; Harbor's built-in `langgraph` agent hosts `bare`/`dcode`/`tau3` graphs with `LocalShellBackend` in the container (`libs/evals/deepagents_harbor/langgraph_project/`, `.github/workflows/_harbor_run.yml:962-968`) |
| Benchmarks reflect the SDK, not the product (§5) | CLI-agent mode in Harbor disabled | **Withdrawn.** | `harbor.yml` defaults `agent_impl` to `dcode`; unified evals run `bare` and `dcode` (`.github/workflows/harbor.yml:117`, `libs/evals/UNIFIED_EVALS.md`) |
| Benchmark coverage "sampled" (§5) | Terminal-Bench 2.0 the only full sandboxed benchmark | **Softened.** | Unified battery adds harbor-index, τ³-bench, Context-Bench, DRBench with pass@K/avg@K. SWE-bench still not wired |
| ACP completeness (§4) | No session/load, no reasoning stream, status `pending → completed` only | **Softened.** Core gap (no client `fs/*`/`terminal/*`, MCP unused) unchanged | Opt-in `session/load`, `agent_thought_chunk`, and `failed` status landed in `deepagents-acp` 0.0.10–0.0.11 (`libs/acp/CHANGELOG.md`) |
| Observability coupling (§3, §7) | LangSmith-shaped evals | **Strengthened.** | `langsmith>=0.14.4` is now a *core* runtime dependency of `deepagents` (`libs/deepagents/pyproject.toml`), and runs carry `ls_integration` metadata (`graph.py:1004`) |
| Local-shell safety (§1, §7) | Default local backend has no isolation | **Softened, not reversed.** | `FilesystemBackend`/`LocalShellBackend` default to path-anchored `virtual_mode=True` and permissions can raise HITL interrupts (0.7.0, commit `540a0fa`; 0.6.8, `a090162`). Shell commands are still not process-isolated |

Unchanged after re-check: middleware-composition verdict, backend protocol, filesystem-as-overflow,
isolated sub-agents, two-tier scoring, approximate tokens, MCP-not-in-core, HITL untested, and the
dependency-depth verdict.

### Verdict Summary
Deep Agents is the **best-engineered "harness as composition" design** in this survey, and the
single most useful one to learn *organizing principles* from — while being almost the **opposite** of
swe-term's deployment philosophy at the substrate level.

Its thesis is sharp and correct: **a harness is a curated bundle of middleware over a generic agent
loop.** It does not write an agent loop at all. `create_deep_agent` assembles ~12 `AgentMiddleware`
(filesystem, sub-agents, summarization, skills, memory, HITL, …; `write_todos` became opt-in in 0.7.0) and hands them to LangChain's
`create_agent`, which compiles a LangGraph `CompiledStateGraph`. Capabilities are *layers*, not
forks. That is exactly the modularity swe-term wants — and Deep Agents demonstrates it works at the
~136-eval, multi-provider, ACP-and-Harbor-integrated scale.

But as an *architectural template for a single Go binary*, Deep Agents is a cautionary tale of
**dependency depth**. The "core" is three frameworks deep (LangGraph → `create_agent` → deepagents)
before you reach a line of Deep Agents' own logic. Its state model, streaming, checkpointing,
persistence, and even its agent loop are LangGraph's. Its strongest features (delta-reduced
checkpoints, pluggable stores, interrupt-based HITL) are **LangGraph features Deep Agents configures**,
not invents. You cannot port Deep Agents; you can only port LangGraph's ideas, re-expressed.

**Steal the middleware-composition model, the pluggable-backend protocol, the two-tier eval scoring,
the "agent-as-acp.Agent" adapter pattern, and the "agent as a standard artifact the eval host runs"
Harbor pattern.** Do not adopt the
three-framework dependency stack, the Python/asyncio concurrency model, or the LangSmith-coupled
observability as your core.**

### Assumptions to NOT Port
#### ❌ "The agent loop is someone else's problem"
True for a framework standing on LangGraph; false for swe-term, whose *reason to exist* is a clean,
readable, zero-dependency loop. Borrowing a loop means inheriting a dependency tree.

#### ❌ "Trust the LLM; enforce only at the sandbox"
The default local shell backend has no process isolation (path anchoring arrived in 0.7.0, but
commands are unconfined). swe-term wants a layered guard *before* the sandbox, on by default.

#### ❌ "Observability = LangSmith"
Don't bake a SaaS into the harness or the test suite. Keep it behind a logging/tracing interface.
Deep Agents moved the other way: `langsmith` became a core SDK dependency after 0.6.7.

#### ❌ "Approximate tokens are good enough for budget decisions"
Acceptable for a heuristic ceiling; risky as the basis for eviction. Prefer a real tokenizer or at
least a provider-aware estimate.

#### ❌ "An ACP agent can bring its own filesystem"
Under ACP, route file/terminal ops through the *client's* capabilities. Bringing your own backend
ignores the protocol's purpose.

#### ❌ "MCP support" at the product layer counts as core support
Be honest about where MCP lives. If swe-term claims MCP, put it in the core (or clearly label it a
plugin).

### What to Actually Port
The highest-value lessons in the survey for swe-term's *organization* (less for its substrate):

1. **Middleware composition over a generic loop** — capabilities are plugins, never forks. Deep
   Agents proves this scales to ~12 layers + profiles, and 0.7.0 lets a same-named user middleware
   replace a default instead of forking the stack.
2. **One backend protocol, many implementations** — state / disk / store / composite / sandbox all
   behind `read_file`/`write_file`. Route by path prefix (`CompositeBackend`).
3. **Filesystem as context-overflow substrate** — offload summarized history and oversized tool
   results to (content-addressable) files; pass handles, not bytes.
4. **Isolated-context sub-agents; any agent is a valid sub-agent** — delegation and composition share
   one seam.
5. **Two-tier eval scoring** — correctness hard-fails, efficiency is logged. Copy this exactly.
6. **Keep protocol and eval frameworks out of the core** — the `acp.Agent` adapter still implements
   the other system's interface; the Harbor side now ships a standard graph artifact that Harbor's
   generic agent hosts. swe-term's ACP frontend and Harbor entry point should be equally thin and
   external.
7. **Delta-reduced, channel-based state** — the O(N) checkpoint idea is worth re-implementing in Go's
   state model.

## Joern

> Upstream: https://github.com/joernio/joern at `84ac957ce (verdict); library entry at b38192263`, revalidated 2026-10-10. General architecture notes:
> source-insights entry `e0a84c79ca4bdb91fa333614`.

### Verdict Summary
**Do not rewrite Joern in Rust.** A faithful port is a multi-year compiler-team
project whose hardest pieces are *not even Joern’s* (Eclipse CDT, Soot, Ghidra,
Roslyn, Graal). `cpg-rs` already shows the cheap part — serde types for the
schema — and stops there for a reason. AppThreat’s `chen` forked Joern and is
**still on the JVM**. Joern 4 already rewrote the graph store (OverflowDB →
flatgraph) *in Scala* for 40% RAM; that is the rewrite that was worth it, and
they did it.

**Do rebirth the CPG *idea* as a swe-term Analyzer**, in Rust, at 5–10% of
Joern’s power and ~90% of agent value: tree-sitter tags + import/call graph +
SCIP ingest + SQLite, behind the existing `swe_distiller` sidecar pattern.
Optionally **wrap** stock `joern-parse` later for security-review taint — with
a **closed query pack**, never the unsandboxed Scala HTTP interpreter.

| Option | Effectiveness for swe-term | Effort |
|--------|----------------------------|--------|
| Full Joern → Rust | **Negative ROI** | Multi-year, team |
| Wrap Joern JVM as always-on sidecar | **Poor** (startup, RAM, Scala API, no incrementality) | Weeks, operational pain forever |
| Wrap Joern for opt-in security mode, canned queries | **Fair**, later | Days–weeks |
| Rebirth: `extensions/swe_graph` (CPG-lite) | **High** | Matches Phase 5 Analyzer plan |
| Point LLM at `joern --server` | **Forbidden** (RCE) | — |

### Assumptions to NOT Port
#### ❌ “Rewrite it in Rust and we get Joern without the JVM”

You get weggli-plus-graph, or you wrap the same native parsers. Pick one and
name it honestly.

#### ❌ “The agent can write CPGQL/Scala”

Unsandboxed interpreter. Closed ops only.

#### ❌ “We’ll run full CPG before every turn”

Minutes + tens of GB. Analyzer must be incremental and cheap, or async
precompute (the AST_SERVICE doc’s Cloud Run story) — not a blocking sidecar
on the TUI path.

#### ❌ “Joern replaces tree-sitter, SCIP, ast-grep, and Semgrep”

Those four already partition the space (syntax, precise refs, structural
pattern, SAST). Joern overlaps all of them badly for agent UX and wins only
at taint.

#### ❌ “atom/chen is the shortcut”

Still JVM, still a Joern-shaped fork. Don’t inherit a second analysis OS.

### What to Actually Take
| From Joern | To swe-term | How |
|------------|-------------|-----|
| CPG as *vocabulary* (METHOD/CALL/REACHING_DEF) | Analyzer result types | JSON schema, not Scala classes |
| Overlays / passes | Pipeline stages in `swe_graph` | AST tags → edges → optional SCIP |
| Fuzzy parse without build | tree-sitter default | SCIP optional upgrade |
| Library semantics as data | Later taint pack | YAML, not hardcoded |
| Packaged queries | Semgrep + optional Joern query ids | Model never authors the engine language |
| Export IR for later | SQLite + optional GraphSON | Content-addressed artifacts |
| `joern-parse` CLI split from REPL | Sidecar has no REPL | `swe_graph` is parse+query, not a shell |
