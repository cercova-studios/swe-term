# opencode Deep Dive — Architecture & Feature Analysis

> Analysis of [`anomalyco/opencode`](https://github.com/anomalyco/opencode) —
> an open-source coding agent shipped as a **local HTTP server with many
> clients** (TUI, web, desktop, ACP). Runtime: **Bun** (`packageManager:
> bun@1.4.2`, root `package.json`). Language: TypeScript on **Effect**.
> License: MIT (`packages/ai/package.json`, root `LICENSE`).
>
> **Validated against a local source checkout** at `opencode/` version
> **2.0.26** (root `package.json`), HEAD `823ac87f58` (2026-10-10). All paths
> below are relative to `opencode/` unless stated otherwise. This is the **V2**
> codebase: older docs and blog posts describing `packages/opencode/` (V1) do
> not match this tree.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Monorepo Layout](#monorepo-layout)
3. [Client/Server Split and Protocol](#clientserver-split-and-protocol)
4. [Agent Loop](#agent-loop)
5. [Provider Abstraction](#provider-abstraction)
6. [Tools and Edit Format](#tools-and-edit-format)
7. [Permissions and Sandboxing](#permissions-and-sandboxing)
8. [Sessions, Storage, Compaction](#sessions-storage-compaction)
9. [Subagents](#subagents)
10. [Plugins, MCP, CodeMode, LSP](#plugins-mcp-codemode-lsp)
11. [TUI and Frontends](#tui-and-frontends)
12. [Config and Instructions](#config-and-instructions)
13. [Why It Matters for swe-term](#why-it-matters-for-swe-term)
14. [Summary Statistics](#summary-statistics)

---

## System Overview

opencode V2 is a **server-first** harness. The agent runtime lives in
`packages/core`, is exposed by `packages/server` over an Effect `HttpApi`
defined in `packages/protocol`, and every frontend — terminal UI, browser app,
desktop shell, ACP adapter — is a client of that API.

Three architectural bets define V2:

1. **Effect everywhere.** Services are `Context.Service` classes composed as
   layers (`packages/core/src/session/runner/index.ts`,
   `packages/core/src/plugin/hooks.ts`). Errors are typed unions
   (`RunError` in `session/runner/index.ts`). The repo enforces Effect idioms
   with ast-grep rules (`lint:effect-patterns` in root `package.json`).
2. **Event-sourced sessions.** Session state is a stream of durable events
   (`packages/schema/src/session-event.ts`, 723 lines) projected into SQLite
   tables (`packages/core/src/session/projector.ts`).
3. **Everything is a plugin.** Built-in tools, providers, system prompts, and
   skills are registered through the same plugin surface as third-party code
   (`packages/core/src/tool/AGENTS.md`, `packages/core/src/plugin/provider/`).

Dependency direction is written down as a rule: Schema → Core and Protocol →
Server; clients may depend on Schema and Protocol but never Core or Server
(`AGENTS.md`, top section).

---

## Monorepo Layout

Turbo + Bun workspace (`turbo.json`, `bun.lock`). Non-test TS/TSX line counts
per package (measured with `find … | wc -l`, excluding `node_modules` and
files containing `test` in the path):

| Package | LoC | Role |
|---|---:|---|
| `packages/app` | ~154k | Web/desktop application UI |
| `packages/core` | ~69.5k | Agent runtime: sessions, tools, permissions, plugins, DB |
| `packages/tui` | ~62k | Terminal UI (OpenTUI + Solid) |
| `packages/gui-extensions` | ~58k | GUI extension surface |
| `packages/console` | ~43.5k | Hosted console (product) |
| `packages/ui`, `session-ui` | ~36k, ~22.7k | Shared UI components |
| `packages/ai` | ~32k | Own provider/protocol layer (`@opencode/ai`) |
| `packages/client` | ~17k | Generated client (`bun run generate`, `AGENTS.md`) |
| `packages/cli` | ~14.7k | Binary entry, server process, ACP, run modes |
| `packages/codemode` | ~10.9k | Confined JS-like interpreter for tool calling |
| `packages/schema` | ~5.8k | Shared schemas (events, permissions, agents) |
| `packages/protocol` | ~4.4k | `HttpApi` groups |
| `packages/server` | ~4.2k | HTTP handlers, auth, SSE feed |
| `packages/plugin` | ~3.4k | Plugin SDK |

Product mass (app, console, stats, gui-extensions, desktop, enterprise) is
larger than the harness. The **harness proper** is core + ai + schema +
protocol + server + cli: roughly 130k lines.

Largest core-path files (non-test):

| File | Lines |
|---|---:|
| `packages/core/src/shell/scan.ts` | 2,409 |
| `packages/ai/src/protocols/anthropic-messages.ts` | 1,676 |
| `packages/core/src/github-copilot/responses/openai-responses-language-model.ts` | 1,636 |
| `packages/ai/src/protocols/open-responses.ts` | 1,507 |
| `packages/core/src/git.ts` | 1,403 |
| `packages/ai/src/protocols/openai-chat.ts` | 1,316 |
| `packages/core/src/database/v1-migration.bun.ts` | 1,135 |
| `packages/core/src/aisdk.ts` | 1,078 |
| `packages/core/src/session/compaction.ts` | 963 |
| `packages/core/src/bus.ts` | 908 |
| `packages/protocol/src/groups/session.ts` | 906 |
| `packages/core/src/mcp/index.ts` | 846 |
| `packages/core/src/session/projector.ts` | 810 |

Notably there is **no god-object session file**. The session directory
(`packages/core/src/session/`) is split into ~40 files, the largest being
`compaction.ts` (963) and `projector.ts` (810). The runner itself
(`session/runner/`) is 4 small files: `llm.ts` 375, `step.ts` 325,
`index.ts` 42, `max-steps.ts` 16.

---

## Client/Server Split and Protocol

- **API definition:** `packages/protocol/src/api.ts` plus 30 groups in
  `packages/protocol/src/groups/` (session, message, permission, pty, fs,
  mcp, worktree, …). OpenAPI is generated to `packages/protocol/openapi.json`;
  `V2_HTTP_API_AUDIT.md` reports **139 endpoints** under review for a stable
  V2 commitment.
- **Server:** `packages/server/src/handlers/` mirrors each group;
  `process.ts` binds hostname/port (`listen`, line 142).
- **Events:** Server-Sent Events. `packages/server/src/event-feed.ts` frames
  each event as `data: <json>\n\n` (`frame`), with a per-subscriber bounded
  queue (`SubscriberCapacity = 4_096`) that fails the subscriber with
  `SubscriberOverflowError` rather than dropping silently.
- **Auth:** HTTP basic with a password, plus pairing-link session tokens
  (`packages/server/src/auth.ts`, `pairing.ts`). Remote access goes through a
  tunnel (`packages/cli/src/services/remote-tunnel.ts`).
- **Background service:** the CLI elects and manages a long-lived local
  server (`opencode service status` / `get password`, `AGENTS.md` "Live V2 TUI
  Testing"; `packages/cli/src/server-process.ts`,
  `packages/cli/src/services/service-registration.ts`). A client whose version
  differs may replace the managed server.
- **Generated client:** `packages/client`, regenerated from the API; manual
  edits are forbidden (`AGENTS.md`).

---

## Agent Loop

Entry: `SessionRunner.Interface.drain` (`packages/core/src/session/runner/index.ts`)
— "Runs one local continuation from already-recorded Session history." The
runner never takes messages as arguments; it reads durable history and
returns `DrainResult` = `Complete | Moved | Reloaded`.

Per step (`session/runner/step.ts`):

1. A filesystem snapshot is captured concurrently with the provider request
   (`snapshots.capture().pipe(Effect.forkScoped)`, line 80).
2. Provider stream events are published as they arrive.
3. **Each `tool-call` event forks its executor immediately** (lines 126–137):
   `executeTool → toolOutput.truncate → publisher.toolExecution`, wrapped in
   `uninterruptibleMask` and `forkScoped`. There is no read/write partition;
   ordering of file writes is delegated to a keyed mutex in
   `packages/core/src/file-mutation.ts` (uses `effect/keyed-mutex.ts`).
4. Context-overflow provider errors before any output are captured and
   routed to compaction rather than surfaced (lines 117–123).

Step limits are per-agent (`agent.info.steps`, `runner/llm.ts:223`). On the
limit, tools are disabled (`step.ts:97`) and `MAX_STEPS_PROMPT`
(`runner/max-steps.ts`) forces a text-only summary.

Retries: `session/runner/retry.ts` (115 lines). Restart after crash:
`session/execution/restart.ts`, backed by the `execution_claim_attempts`
migration (`database/migration/20260811161259_execution_claim_attempts.ts`).

---

## Provider Abstraction

Two layers coexist:

- **`packages/ai`** (`@opencode/ai`, ~32k lines): opencode's own Effect-native
  model client with **49 protocol files** in `packages/ai/src/protocols/`
  (`anthropic-messages.ts`, `openai-responses.ts`, `bedrock-converse.ts`,
  `gemini.ts`, plus image/speech/video/transcription protocols). HTTP is
  recorded/replayed via `packages/http-recorder`.
- **Vercel AI SDK bridge:** `packages/core/src/aisdk.ts` (1,078 lines) and
  `aisdk-native.ts`; `packages/core/package.json` still pins `@ai-sdk/*`
  packages (e.g. `@ai-sdk/provider` 3.0.8).
- **Providers as plugins:** 36 files in `packages/core/src/plugin/provider/`
  (anthropic via `opencode.ts`, `openai.ts`, `amazon-bedrock.ts`,
  `github-copilot.ts`, `ollama.ts`, …). The catalog comes from models.dev
  (`packages/core/src/models-dev.ts`, `models-dev/snapshot.txt`).
- **Per-model system prompts:** `packages/core/src/plugin/system-prompt/`
  (`anthropic.txt`, `gpt.txt`, `gpt-astra.txt`, `kimi.txt`, …).

---

## Tools and Edit Format

Built-ins live in `packages/core/src/tool/plugin/` and register through the
same `Tool.Service.transform` path as plugins and MCP
(`packages/core/src/tool/AGENTS.md`, "Registration"):

`read`, `write`, `edit`, `patch`, `glob`, `grep`, `shell`, `webfetch`,
`websearch`, `question`, `skill`, `subagent`, `mcp-resource`, `opencode`
(meta/self tool), `file-diff`.

**Edit format — two tools, both string-anchored:**

- `edit.ts` (230 lines): `oldString`/`newString`/`replaceAll`. Must match
  exactly once unless `replaceAll`. Matching falls back in tiers: exact →
  Unicode-normalized (smart quotes, dashes, NBSP; `normalizeForMatch`,
  lines 49–54) → trailing-whitespace line match (lines 163–167). CRLF is
  preserved (lines 161–162).
- `patch.ts` (363 lines) with model-facing grammar in `tool/patch.txt`:
  the Codex-style `*** Begin Patch / *** Update File / @@ / *** End Patch`
  envelope with Add/Delete/Update/Move.
- `edit`, `write`, and `patch` share one permission action, `edit`
  (`tool/AGENTS.md`, "Permissions").

**Registry semantics** (`tool/AGENTS.md`): Location-scoped; latest valid
registration for a name wins; disposal reveals the overridden definition;
each model request snapshots the effective tool set. Tool filtering is
**catalog visibility, not authorization** — leaves enforce permission.

**Output bounding:** generic truncation runs in the runner after execution
(`toolOutput.truncate` in `step.ts`; `packages/core/src/tool-output.ts`).
Shell stores full output in a file and returns a bounded tail with the path
plus explicit `truncated` / `timeout` fields (`tool/plugin/shell.ts`, lines
42–68).

---

## Permissions and Sandboxing

- **Model:** ordered rules `{action, resource, effect: allow|deny|ask}`
  (`packages/schema/src/permission.ts`, lines 56–66). `evaluate` uses
  **last-match-wins** over flattened rulesets, defaulting to `ask`
  (`packages/core/src/permission.ts`, lines 87–97). Any `deny` across the
  requested resources short-circuits (line 166, 175).
- **Layering:** agent permissions merged with session permissions (line 162);
  an agent with no permissions gets `* * deny` (line 19). User "always"
  replies persist as saved rules (`permission/saved.ts`, `permission/sql.ts`).
- **Replies:** `once | always | reject` (`schema/src/permission.ts:42`); a
  decline with feedback is a typed `CorrectedError` the model can continue
  from (`tool/AGENTS.md`, "Construction").
- **Hooks:** plugins can observe/alter evaluation via
  `hooks.trigger("permission", "evaluate", …)` (`permission.ts:179`).
- **Shell command analysis:** `packages/core/src/shell/scan.ts` (2,409
  lines — the largest core file) parses bash into per-command resources for
  permission prefixes, and returns `opaque` with a reason
  (`command-substitution`, `heredoc`, `dynamic-execution`, …) instead of
  guessing. The comment is explicit: "Opaque describes a parsing limitation,
  not a permission decision."
- **No OS sandbox.** A search for seatbelt/landlock/bwrap/sandbox in
  `packages/core/src/shell*`, `environment/`, `tool/plugin/shell.ts`, and
  `persistent-pty.ts` returns nothing. `environment/driver.ts` abstracts the
  process spawner (`local.ts`, `memory.ts`, `unavailable.ts`) — an execution
  seam, not confinement.
- **CodeMode is the confinement story** for programmatic tool use: a pure
  interpreter with no network/fs/process access except host tools
  (`packages/codemode/README.md`). Tools default into CodeMode unless
  registered with `codemode: false` (`tool/AGENTS.md`).

---

## Sessions, Storage, Compaction

- **Storage:** SQLite via Drizzle (`packages/core/src/database/`), with
  Bun/Node/workerd drivers (`sqlite.bun.ts`, `sqlite.node.ts`,
  `sqlite.workerd.ts`). **49 migrations** in `database/migration/`, plus a
  1,135-line V1 importer (`v1-migration.bun.ts`).
- **Event sourcing:** durable events such as `session.created`,
  `session.forked`, `session.inbox.enqueued`, `session.execution.started`
  /`succeeded`/`failed`/`interrupted` (`schema/src/session-event.ts`) are
  projected into `session`, `session_message`, `session_pending`,
  `session_inbox`, `instruction_entry`/`blob`/`state` tables
  (`core/src/session/sql.ts`, `session/projector.ts`). Migration names show
  the path: `20260604172448_event_sourced_session_input.ts`,
  `20260622170816_reset_v2_session_state.ts`.
- **Inbox:** user input queued while a run is active is a durable inbox with
  steer vs next-turn promotion (`session/inbox.ts` 550 lines;
  `Promotable` in `runner/index.ts`).
- **Snapshots/revert:** git-backed shadow repository
  (`core/src/snapshot.ts` uses `Git.Service`, lines 76–98);
  `session/revert.ts`, `session/diff.ts`.
- **Compaction** (`session/compaction.ts`, 963 lines): triggers `auto` or
  `overflow` (line 55). Produces a structured summary **plus verbatim recent
  context** (`keep`, line 45), or a provider-native compaction
  (`providerContext`, line 71). Overflow shrinks by stepped factors
  (line 228). Summaries are updated incrementally as one checkpoint with
  "newer history always takes precedence" rules (lines 160–173) and a
  template-conformance retry (line 179). The prompt requires preserving exact
  paths, symbols, commands, and errors (lines 140–143).

---

## Subagents

`tool/plugin/subagent.ts` (311 lines):

- Child sessions linked by `parentID`. Depth is computed by walking parents
  and capped by `experimental.subagent_depth`, **default 1** (lines 118–132).
- `background: true` returns immediately with the child session ID; the
  parent is notified on completion (lines 19–23, 44–61;
  `session/subagent-completion.ts`, `session/subagent-job.ts`).
- Agents (`core/src/agent.ts`) are named profiles; default is `build`
  (line 13). Each carries its own permission ruleset and optional `steps`.

---

## Plugins, MCP, CodeMode, LSP

- **Plugin host:** `core/src/plugin/host.ts` (689 lines). Plugins are npm
  packages resolved/installed at runtime (`plugin/module.ts` uses
  `Npm.Service.add`, lines 104–109) and run **in-process**. Per-plugin KV
  storage via `PluginHost.storage` (line 609).
- **Hook domains:** `aisdk`, `session`, `permission`, `shell`, `tool`
  (`plugin/hooks.ts`). Only `tool.execute.before` may fail (rejects a call);
  all others are typed `never`-failure (comment above `Failures`).
- **MCP:** `core/src/mcp/` (`client.ts`, `stdio.ts`, `oauth.ts`,
  `index.ts` 846 lines). MCP tools install as one stable transform whose
  source updates on list changes, so later plugin overrides keep precedence
  (`tool/AGENTS.md`). MCP tools also flow into CodeMode
  (`plugin/mcp-codemode-defaults.ts`).
- **CodeMode:** `packages/codemode` (~10.9k lines) — a JS-subset interpreter
  where tools are the only side-effect channel; host limits time, call count,
  and output size (`packages/codemode/README.md`). Credits Cloudflare's code
  mode.
- **LSP: absent in V2 core.** `core/src/file-mutation.ts:131`:
  `// TODO: Notify LSP and collect diagnostics after LSP runtime exists.`
- **Skills:** `core/src/skill.ts`, `tool/plugin/skill.ts`;
  `session.skill.activated` event in `schema/src/session-event.ts`.

---

## TUI and Frontends

- **TUI:** `packages/tui` on OpenTUI + Solid (`@opentui/core`,
  `@opentui/solid`, `solid-js` in `packages/tui/package.json`). Largest file:
  `src/routes/session/index.tsx` (3,640 lines). Fixture-driven storybook in
  `src/feature-plugins/system/storybook` (`AGENTS.md`).
- **Web/desktop:** `packages/app`, `packages/desktop`, `packages/web`.
- **Headless:** `packages/cli/src/run/noninteractive.ts` (859 lines).
- **ACP adapter:** `packages/cli/src/acp/` (20 files: `agent.ts`,
  `permission.ts`, `replay.ts`, `compaction.ts`, …) translating the server
  API to Agent Client Protocol.
- **Persistent PTY daemon:** `core/src/persistent-pty/daemon.ts` (690 lines).

---

## Config and Instructions

- **Files:** `opencode.json` / `opencode.jsonc`
  (`core/src/config/discovery.ts:11`); directories `.opencode`, `.agents`,
  and **`.claude`** are all scanned (line 41) — deliberate cross-harness
  compatibility.
- **Instructions:** global `AGENTS.md` under the config dir
  (`config/plugin/instruction.ts:38`); project AGENTS.md discovery is session
  time, not boot time (`core/src/instance.ts:125–129`). Instruction state is
  persisted and versioned (`instruction_entry`, `instruction_blob`,
  `instruction_state` tables; `session.instructions.updated` event).
- **Normalization/variables/watch:** `config/normalize.ts` (808 lines),
  `config/variable.ts`, `config/watch.ts`.
- **Managed policy:** `core/src/managed-policy.ts`, `packages/enterprise`.

---

## Why It Matters for swe-term

opencode V2 is the **closest production analogue to swe-term's intended
shape**: protocol-first core, every frontend a client, event log as state,
small loop files, permission as ordered data. It is also the clearest case of
a harness absorbing a product: ~154k lines of app UI, 49 AI protocols
including video and speech, in-process npm plugins, and a framework (Effect)
that every contributor must speak.

---

## Summary Statistics

| Metric | Value | Source |
|---|---|---|
| Version / HEAD | 2.0.26 / `823ac87f58` | root `package.json`, `git log` |
| Workspace packages | 35 | `packages/` |
| Core LoC (non-test) | ~69.5k | `packages/core` |
| Core runtime deps | 39 | `packages/core/package.json` |
| HTTP endpoints | 139 | `V2_HTTP_API_AUDIT.md` |
| AI protocol files | 49 | `packages/ai/src/protocols/` |
| Provider plugins | 36 | `packages/core/src/plugin/provider/` |
| Built-in tool files | 15 | `packages/core/src/tool/plugin/` |
| DB migrations | 49 | `packages/core/src/database/migration/` |
| Subagent depth default | 1 | `tool/plugin/subagent.ts:129` |
| OS sandbox | none | see Permissions section |
| LSP | not yet in V2 | `core/src/file-mutation.ts:131` |
