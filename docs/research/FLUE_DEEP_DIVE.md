# Flue Deep Dive — Architecture & Feature Analysis

> Analysis of [`withastro/flue`](https://github.com/withastro/flue) — Fred K. Schott's
> ("the Astro creator") **agent harness framework**. Positioned as "The Agent Harness Framework.
> Not another SDK." (`README.md`). Runtime targets: Node.js and Cloudflare Workers. Language:
> TypeScript. Still built on **`@earendil-works/pi-agent-core` + `@earendil-works/pi-ai`**
> (`^0.87.1`, `packages/runtime/package.json`).
>
> **Validated against a local source checkout** at `/Users/rohit/sandbox/swe-term/flue`
> (HEAD `6583259`, "Version Packages (#794)", 2026-09-28; all packages `2.2.2`, license
> Apache-2.0) on **2026-10-10**. Previous validation: `0.7.0` at upstream `bb9245f01`
> (2026-06-06). 588 commits and a breaking 2.0 redesign separate the two; the earlier
> gvision consumer-app section is now historical (see [A Real App: gvision](#a-real-app-gvision-historical)).

---

## Changes since 2026-06-06 (0.7.0)

1. **2.0 redesign: the agent *is* a function composed from hooks.** `'use agent'` modules export
   a capitalized function that calls `useModel`, `useSandbox`, `useTool`, `useSkill`,
   `useSubagent`, `useMcpConnection`, `usePersistentState`, … and returns its instructions
   (`packages/runtime/src/hooks/`, `README.md`). `FlueContext`/`ctx.init()`/`defineAgent` are gone
   (`apps/docs/src/content/docs/guide/migration.md`). Released as `v2.0.0` in `5485efe2`
   (2026-07-31); the hooks + Vite rewrite landed via `b814b82b` (nightly, 2026-07-24).
2. **Vite owns build and dev.** `flue build`/`flue dev` removed; new `@flue/vite` plugin, so it is
   `vite dev` / `vite build` (`migration.md` §"Build and dev commands", `packages/vite/`).
   `defineConfig` moved to `@flue/runtime/config`.
3. **Durability became a uniform submission contract.** Every input is a durably admitted
   *submission* that reaches exactly one terminal outcome, with lease-based recovery on Node and
   wake-driven recovery on Cloudflare (`apps/docs/src/content/docs/guide/durability.md`). Landed
   in `ea351cad` "durable execution, pluggable persistence" (2026-06-08).
4. **Pluggable persistence on Node.** `src/db.ts` selects a store: built-in `sqlite()` on
   `node:sqlite` (`@flue/runtime/node`), or adapter packages `@flue/postgres`, `libsql`, `mysql`,
   `mongodb`, `redis` (`packages/`, `guide/database.md`). Contract test suites ship in
   `@flue/runtime/test-utils`.
5. **Runs and workflows removed; canonical conversation streams replace them.** `RunStore`,
   `RunRegistry`, `defineWorkflow`, `invoke()` and run events are gone (`migration.md`
   §"Workflows are removed", §"Database"). Events stream over the Durable Streams protocol;
   named sessions were removed (`f5f2928a`, 2026-06-09).
6. **Durable tools.** `defineTool({ durable: true })` with `step.do(name, fn)` checkpoints
   (`guide/durability.md` §"Durable tools and `step.do`").
7. **No implicit sandbox.** An agent without `useSandbox()` has no filesystem or file/bash tools;
   `just-bash` is opt-in via `bash(...)`. Remote sandboxes (E2B, Daytona, Modal, Vercel, Cloudflare,
   …) arrive as `flue add sandbox <name>` blueprints (`migration.md` §"Sandboxes", `blueprints/`).
8. **Channels.** ~18 connector packages (`@flue/slack`, `teams`, `discord`, `github`, …) mounted
   via `createChannelRouter` (`packages/runtime/src/runtime/channel-routes.ts`, `guide/channels.md`).
9. **Client SDK is conversation-scoped**, and **`@flue/react`** adds React hooks (`d82f5d39`).
   `createFlueClient()` now returns `send`/`read`/`wait`/`abort`/`history`/`observe` for one
   conversation (`packages/sdk/src/client.ts`).
10. **Tool schemas unified on valibot.** `defineTool({ input: v.object(...) })`; pi-ai's `Type` is no
    longer re-exported from `@flue/runtime` (`packages/runtime/src/tool-types.ts`,
    `packages/runtime/src/index.ts`).

---

## Table of Contents

1. [System Overview](#system-overview)
2. [The Framework-vs-SDK Thesis](#the-framework-vs-sdk-thesis)
3. [Package Layout](#package-layout)
4. [Engine: It Wraps pi, It Doesn't Rebuild It](#engine-it-wraps-pi-it-doesnt-rebuild-it)
5. [Core Primitives](#core-primitives)
6. [The Agent Function & Hooks](#the-agent-function--hooks)
7. [Conversations, Submissions & Dispatch](#conversations-submissions--dispatch)
8. [Tools & Typed Results](#tools--typed-results)
9. [Subagents](#subagents)
10. [Sandboxes](#sandboxes)
11. [Skills](#skills)
12. [MCP Integration](#mcp-integration)
13. [The Durability Model](#the-durability-model)
14. [The Build System (Agent-as-Artifact)](#the-build-system-agent-as-artifact)
15. [Deployment Targets](#deployment-targets)
16. [Providers & Models](#providers--models)
17. [The Client SDK (`@flue/sdk`)](#the-client-sdk-fluesdk)
18. [A Real App: gvision (historical)](#a-real-app-gvision-historical)
19. [Relationship to pi-mono](#relationship-to-pi-mono)
20. [Key Design Patterns](#key-design-patterns)
21. [Summary Statistics](#summary-statistics)

---

## System Overview

Flue is a serious attempt to make **"the agent harness" a deployable framework** rather than a
library you wire up yourself. Its pitch borrows the web-framework playbook — in 2.x quite
literally: an agent is a React-style **function component** built from hooks, and the app is a
**Vite** project. `vite build` produces a server artifact for Node or Cloudflare Workers.

It is explicitly **headless**. There is no TUI. An agent is a function exposed as an HTTP service
(**Hono**), reachable through explicit routes in `app.ts`, channels (Slack, GitHub, …), schedules,
or `dispatch()` (`guide/routing.md`, `guide/channels.md`, `guide/schedules.md`).

**Tech Stack (from `packages/runtime/package.json`):**

| Layer | Technology |
|-------|-----------|
| Language | TypeScript (Node ≥ 22.19.0) |
| Runtime targets | Node.js, Cloudflare Workers |
| Agent engine | `@earendil-works/pi-agent-core` `^0.87.1` (pi's `Agent`, loop, steering) |
| Provider/model layer | `@earendil-works/pi-ai` `^0.87.1` |
| HTTP | `hono` |
| Tool + result schema | `valibot` |
| MCP | `@modelcontextprotocol/client` `2.0.0` |
| Event streaming | Durable Streams protocol (`@durable-streams/client` in `@flue/sdk`) |
| Virtual sandbox | `just-bash` (opt-in, wrapped by `bash(...)`) |
| Build/dev | Vite + `@flue/vite` (`@cloudflare/vite-plugin` on Cloudflare) |
| Config | optional `flue.config.ts` (`defineConfig` from `@flue/runtime/config`) |

---

## The Framework-vs-SDK Thesis

Flue's central claim is unchanged: existing agent tooling ships as **SDKs**, and the missing layer
is a **framework** (opinionated structure + build + deploy). In 2.x:

- **The agent is the function.** Hooks compose capabilities; the return value is the instructions.
- **A build step.** `vite build` compiles the project into a deployable artifact.
- **Web-framework DX.** `vite dev` / `vite build`, plus `flue run <module>` for one-shot local runs.

---

## Package Layout

The monorepo (pnpm + turbo) now ships **~28 packages** under `packages/*/`, all `2.2.2`:

| Package | Role |
|---------|------|
| **`@flue/runtime`** | Harness, sessions, hooks, tools, sandbox plumbing, Hono runtime, MCP client, compaction, durability coordinators. Subpaths include `./config`, `./routing`, `./tool`, `./telemetry`, `./node`, `./cloudflare`, `./cloudflare/workers-ai`, `./test-utils`. |
| **`@flue/vite`** | Vite plugin: dev server and production build for both targets. |
| **`@flue/cli`** | `flue` binary: `init`, `run`, `add`/`update` (blueprints), `docs` (`packages/cli/src/commands/`). No longer builds. |
| **`@flue/sdk`** | Conversation-scoped client for deployed agents. |
| **`@flue/react`** | React hooks over the SDK's `observe()` layer (`guide/react.md`). |
| **`@flue/opentelemetry`** | OTel tracing adapter (present at `bb9245f01`; missed by the 0.7.0 doc). |
| **Database adapters** | `@flue/postgres`, `@flue/libsql`, `@flue/mysql`, `@flue/mongodb`, `@flue/redis`. |
| **Channel connectors** | `@flue/slack`, `teams`, `discord`, `github`, `google-chat`, `linear`, `notion`, `telegram`, `whatsapp`, `twilio`, `stripe`, `shopify`, `zendesk`, `intercom`, `messenger`, `resend`, `salesforce-marketing-cloud`. |

The deprecated `@flue/connectors` stub is gone. Sandbox providers are **not** packages: they are
Markdown blueprints in `blueprints/` that `flue add sandbox <name>` hands to your coding agent.

---

## Engine: It Wraps pi, It Doesn't Rebuild It

Still true in 2.2.2: **Flue does not implement an agent loop.** `packages/runtime/src/session.ts`
(now ~6,100 lines) imports `Agent` from `@earendil-works/pi-agent-core` and constructs it:

```ts
this.agentLoop = new Agent({
  initialState: { systemPrompt, model: this.config.model, tools, messages: previousMessages,
                  thinkingLevel: this.config.thinkingLevel ?? 'medium' },
  streamFn: this.emitTurnRequestAndStream,
  toolExecution: 'parallel',
  steeringMode: 'all',
  followUpMode: 'all',
  sessionId: this.affinityKey,
  ...(options.rerender ? { prepareNextTurnWithContext: (turn) => this.prepareRerenderTurn(turn) } : {}),
});
// turns: this.agentLoop.prompt(...), this.agentLoop.continue(), this.agentLoop.steer(...)
```

What changed is how much Flue drives *through* pi's hooks: it supplies its own `streamFn` (to
record each turn request), uses pi's steering queue to join mid-turn deliveries, uses
`continue()` for crash recovery, and uses `prepareNextTurnWithContext` to **re-render the agent
function between turns** so tools and prompt reflect committed state. The loop is still pi's; the
surrounding machinery (submissions, recovery, canonical conversation records) is much larger.

---

## Core Primitives

| Primitive | What it is |
|-----------|-----------|
| **Agent** | An exported capitalized function in a `'use agent'` module; hooks declare capabilities, return value is instructions. Statics such as `durability` and `initialData` configure it. |
| **Hooks** | `useModel`, `useSandbox`, `useTool`, `useSkill`, `useSubagent`, `useMcpConnection`, `useInstruction`, `usePersistentState`, `useInitialData`, `useDelivery`, `useDataWriter`, lifecycle hooks (`useAgentStart/Finish`, `useResponseStart/Finish`) (`packages/runtime/src/index.ts`). |
| **Conversation / instance** | One agent instance per id; an append-only canonical record stream is its source of truth (`guide/database.md`). |
| **Submission** | One durably admitted input with exactly one terminal outcome (`guide/durability.md`). |
| **Sandbox** | Explicit via `useSandbox`: `bash(...)` virtual, `local()` host, or remote adapter. |
| **Tools** | `defineTool` with valibot `input`/`output`; optional `durable: true` + `step.do`. |
| **Skills** | Agent-Skills-format Markdown, imported as modules and mounted with `useSkill` / `defineSkill`. |
| **Subagents** | `useSubagent` / `defineSubagent` / built-in `GeneralSubagent`, invoked by a framework `task` tool. |
| **MCP** | `useMcpConnection` / `defineMcpConnection` / `createMcpConnection`. |
| **Channels** | Verified inbound webhooks routed with `createChannelRouter`. |

`roles/<name>.md` personas no longer appear in source (no `roles/` references under
`packages/runtime/src` or `packages/vite/src`); instructions come from the agent's return value
and `useInstruction`, and specialization comes from subagents.

---

## The Agent Function & Hooks

From `README.md`:

```ts
'use agent';
import { useModel, useSandbox, useSkill, useTool } from '@flue/runtime';
import { local } from '@flue/runtime/node';
import triage from '../skills/triage/SKILL.md';

export function Triage() {
  useModel('anthropic/claude-sonnet-4-6');
  useSandbox(local());
  useSkill(triage);
  useTool(openIssue);
  return `Triage a bug report end-to-end ...`;
}
```

Hooks may be conditional: an agent can gain or lose a tool, MCP server, or sandbox based on
persistent state, and the model is told about the change (`guide/mcp.md`, `guide/sandboxes.md`
§"Conditional attachment"). The re-render between turns is what makes this work.

---

## Conversations, Submissions & Dispatch

- **`dispatch(Agent, { id, message, initialData?, uid? })`** — fire-and-forget at durable
  admission; resolves with a `DispatchReceipt` carrying `submissionId`.
- **`init(Agent, { id })`** — an awaited handle with `dispatch`, `read(receipt)`, `abort()`.
- **`uid` send conditions** — continue-or-create, continue-only, or create-only.
- One submission runs at a time per conversation; arrivals during a turn join at a turn boundary
  or queue (`guide/durability.md` §"Submissions").
- Standalone scripts boot the runtime with `start()` from `@flue/runtime/node`.

(All from `migration.md` §"Dispatch and conditional sends" and §"Workflows are removed".)

---

## Tools & Typed Results

The 0.7.0 "dual schema stack" (pi-ai `Type` for params, valibot for results) is gone. Tools are
`defineTool({ name, description, input: v.object(...), run })` with valibot schemas
(`packages/runtime/src/tool-types.ts` imports `valibot`; `index.ts` no longer exports `Type`).

The host-side boundary is unchanged: `run` executes in the host process with access to secrets;
the model sees only the schema.

Typed results still inject a **`finish`/`give_up`** tool pair (`packages/runtime/src/result.ts`,
`FINISH_TOOL_NAME`, `GIVE_UP_TOOL_NAME`) with a follow-up prompt if the model ends a turn without
calling either.

Sandbox-backed tools are now composable factories — `createReadTool`, `createWriteTool`,
`createEditTool`, `createBashTool`, `createGrepTool`, `createGlobTool` (`index.ts`) — that a
`SandboxFactory` can add, drop, or swap.

**Durable tools:** `durable: true` gives `run` a `step` object; each `step.do(name, fn)` records its
result, so recovery replays completed steps and resumes from the first unfinished one
(exactly-once recorded, at-least-once executed; `guide/durability.md`).

---

## Subagents

Declared with `useSubagent({ name, description, agent })`. Delegation is model-driven: every agent
has a framework `task` tool, and declared delegates appear in an "Available Agents" system-prompt
section (`guide/subagents.md`). A delegate runs in its own fresh context and its own durable
conversation stream; only its final answer returns. On recovery, an interrupted `task` call
reattaches to the child transcript and resumes it, recursively (`guide/durability.md`
§"Delegated tasks").

---

## Sandboxes

| Sandbox | Mechanism | Isolation |
|---------|-----------|-----------|
| **None (default)** | No `useSandbox()` → no filesystem, no file/bash tools. | N/A |
| **Virtual** | `bash(() => new Bash({ fs: new InMemoryFs() }))` — `just-bash` emulation, rebuilt fresh per initialization. | Emulation, not isolation. |
| **`local()`** (Node) | Real fs + host shell; process-group kill; env allowlist (`PATH`, `HOME`, …) with per-variable opt-in. | None by design. |
| **Remote adapters** | Blueprints for E2B, Daytona, Modal, Vercel, Cloudflare Sandbox/Computer, boxd, islo, mirage, exe.dev (`blueprints/sandbox--*.md`); `createSandbox({ id })` keyed on instance id gives durable workspaces. | Provider isolation. |

(`guide/sandboxes.md`, `migration.md` §"Sandboxes".) Workspace durability is explicitly separate
from conversation durability.

---

## Skills

Skills follow the Agent Skills format (frontmatter + `instructions`, optional supporting files
loaded only when read), are imported as `.md` modules, and mounted with `useSkill`
(`guide/skills.md`). Progressive disclosure: only name/description are listed until activation.

---

## MCP Integration

`useMcpConnection({ name, url, auth, optional? })` connects to a **remote** server, namespaces tools
as `mcp__<server>__<tool>`, reuses the connection while the instance is live, and may be
conditional (`guide/mcp.md`). Built on `@modelcontextprotocol/client` `2.0.0`.

Still client-only for remote servers: no stdio spawning is documented, and *"Flue never stores or
manages your tokens. It is your responsibility to own any OAuth flow, token storage, and refresh
token logic."* (`guide/mcp.md`). `auth` accepts a bearer token or a provider you supply.

---

## The Durability Model

The 0.7.0 run/session split is gone. 2.x has one model (`guide/durability.md`):

- **Submission contract:** admitted before model work; exactly one terminal outcome (`completed`,
  `failed`, `aborted`) written as `submission_settled` in the canonical stream.
- **Recovery classifies durable evidence:** requeue un-persisted input, settle finished work,
  continue from partial text, repair tool batches (unresolved ordinary calls get an explicit
  unknown-outcome error, not a re-run), retry transient provider errors, compact on overflow.
- **Budgets:** default 10 attempts and 1 hour per submission, set per agent with a `durability`
  static; deadlines enforced preemptively.
- **Not durable by design:** sandbox files, local promises, code outside the agent, external side
  effects.

| Target | Owner of a conversation | Recovery trigger | Storage |
|--------|------------------------|------------------|---------|
| **Node** | In-process coordinator with heartbeated leases | Startup reconciliation + periodic lease scans | `src/db.ts`: in-memory default, `sqlite()` file, or Postgres/libSQL/MySQL/MongoDB/Redis adapters |
| **Cloudflare** | One Durable Object per conversation (structural) | Wake-on-start + self-renewing alarm | DO SQLite; `db.ts` rejected |

Node caveat in the docs: one live owner per conversation; multi-replica deployments must route each
conversation to one owner.

---

## The Build System (Agent-as-Artifact)

- **`vite.config.ts`** with `flue()` from `@flue/vite` (before `cloudflare()` on that target).
- **`app.ts` is required** and mounts routes explicitly; the old auto-router is gone (`guide/routing.md`).
- **`vite dev` / `vite build`** replace `flue dev` / `flue build`. Node output runs as
  `node dist/server.mjs`; Cloudflare uses generated `.flue-vite/` + `.flue-vite.wrangler.jsonc`.
- **`flue run <module> -m "<message>" [--id] [--data] [--json]`** executes one agent module
  in-process (no server, never loads `app.ts`).
- **`flue init`** (interactive scaffold), **`flue add` / `flue update`** (blueprints for sandboxes,
  databases, channels, tooling), **`flue docs`** (offline docs). `flue logs` is gone.

(`migration.md` §"Build and dev commands", §"CLI"; `packages/cli/src/commands/`.) There is still
no `flue deploy`; you deploy the Vite output.

---

## Deployment Targets

| Target | Build | Durable state | Notes |
|--------|-------|---------------|-------|
| **Node.js** | `vite build` → `dist/server.mjs` | `db.ts` adapter (in-memory default) | Also documented: GitHub Actions, GitLab CI, Render (`README.md`) |
| **Cloudflare** | `vite build` with `@cloudflare/vite-plugin` | DO SQLite per conversation | `wrangler deploy` |

Durability is no longer "strongest on Cloudflare" in contract terms: the recovery decisions are
identical on both targets (`guide/durability.md` §"Recovery by target"). Node durability still
depends on choosing a durable `db.ts`.

---

## Providers & Models

- Models are `provider/model-id` strings set with `useModel(...)` or per call.
- `setProvider(provider)` registers a provider; built-in providers load via
  `registerBuiltinProviderModule` (`packages/runtime/src/runtime/providers.ts`).
  `registerProvider`/`configureProvider` from 0.7.0 are not in the current file.
- Cloudflare Workers AI and an Anthropic binding path live under `packages/runtime/src/cloudflare/`
  (`workers-ai-provider.ts`, `anthropic-binding-request.ts`).
- pi bumped to `0.87.1` for new models (`89032c55`, 2026-09-24).

### Compaction

Still Flue's own `compaction.ts`: `isContextOverflow` from pi-ai, summarization through
`getRuntimeModels().completeSimple(...)`, chars/4 fallback estimator. Recovery also compacts and
retries on a context-overflow response (`guide/durability.md`).

### Observability

`observe()` event subscribers, `instrument()`, and the `@flue/opentelemetry` adapter; Braintrust and
Sentry via blueprints (`README.md`, `index.ts`).

---

## The Client SDK (`@flue/sdk`)

`createFlueClient(options)` is now **one client per conversation URL** (`packages/sdk/src/client.ts`):

- `send(...)` — admit a message; `read(target)` — wait for and return the reply; `wait(admission)`.
- `abort()` — durable abort of unsettled submissions.
- `history()` — point-in-time snapshot; `observe()` — snapshot then live updates (SSE, falling back
  to Durable Streams long-poll).

The 0.7.0 deployment-wide `agents.invoke` / `runs` / `admin` surface is gone (`migration.md`
§"Agent SDK").

---

## A Real App: gvision (historical)

The 0.7.0 doc confirmed the public API against `/home/rohit/sandbox/gvision`. That app is not
present on this machine (`/Users/rohit/sandbox/gvision` does not exist), and its API
(`FlueContext`, `ctx.init`, `roles/companion.md`, `Type`-based `ToolDef`, `flue()` from
`@flue/runtime/app`) is entirely superseded by 2.x. Treat that evidence as describing 0.7.0 only.

---

## Relationship to pi-mono

Unchanged in substance: **Flue is a framework wrapper around pi-mono's lower layers.**

- `Agent`, the loop, steering, and `ThinkingLevel` come from `@earendil-works/pi-agent-core`.
- Models, `isContextOverflow`, `completeSimple`, and usage shapes come from
  `@earendil-works/pi-ai` (imported across ~19 runtime files, e.g. `session.ts`, `compaction.ts`,
  `usage.ts`, `runtime/providers.ts`).

What grew is Flue's own share: a durable submission/recovery engine, canonical conversation records,
hook-based composition, and re-render-per-turn — all wrapped around pi's loop rather than replacing it.

---

## Key Design Patterns

### 1. Agent = function component → server artifact
Hooks declare capabilities; Vite compiles the project to a deployable server.

### 2. Headless-first
No TUI. Inputs come from HTTP, channels, schedules, or `dispatch()`.

### 3. Build on pi, don't rebuild
pi owns the loop; Flue drives it via `streamFn`, steering, `continue()`, and `prepareNextTurnWithContext`.

### 4. Explicit, swappable sandbox
No implicit workspace. Virtual → `local()` → remote adapters, with workspace durability separate
from conversation durability.

### 5. Accepted-work contract
Durable admission, exactly-one terminal outcome, evidence-based recovery, conservative tool-batch
repair, `step.do` checkpoints. Uniform across targets.

### 6. Host-side tool execution, schema-only to the agent
Tools run in the host; the model sees only the valibot schema. `local()` withholds host env by default.

### 7. Integrations as blueprints, not packages
Sandboxes and some tooling ship as Markdown instructions for a coding agent (`flue add`).

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Language | TypeScript (Node ≥ 22.19.0) |
| Version | `2.2.2` (HEAD `6583259`, 2026-09-28), Apache-2.0 |
| Packages | ~28: runtime, vite, cli, sdk, react, opentelemetry, 5 DB adapters, 17 channel connectors |
| Engine | `@earendil-works/pi-agent-core` + `pi-ai` `^0.87.1` |
| Agent loop | delegated to pi's `Agent` (`toolExecution: 'parallel'`, steering `'all'`) |
| HTTP runtime | Hono; explicit routes in required `app.ts` |
| Agent API | `'use agent'` function + hooks; `durability` / `initialData` statics |
| Invocation | `dispatch()`, `init()` handle, channels, schedules, HTTP |
| Tool + result schema | valibot (`defineTool`, `finish`/`give_up`) |
| Durable tools | `durable: true` + `step.do` |
| Sandboxes | none by default; `bash(...)` virtual; `local()`; remote adapters via blueprints |
| Persistence | Node `db.ts` (memory, `sqlite()`, Postgres/libSQL/MySQL/MongoDB/Redis); Cloudflare DO SQLite |
| Durability | submission contract; Node leases, Cloudflare DO wakes |
| Streaming | Durable Streams protocol + SSE |
| MCP | remote client only (`useMcpConnection`); app owns OAuth |
| Build | Vite + `@flue/vite`; packages built with `tsdown` |
| CLI | `init`, `run`, `add`, `update`, `docs` (no `dev`/`build`/`logs`/`deploy`) |
| Client SDK | conversation-scoped `createFlueClient()`; `@flue/react` |
| Headless | Yes |

---

*Validated 2026-10-10 against `/Users/rohit/sandbox/swe-term/flue` at `6583259` (`2.2.2`). File and
symbol references are from that checkout; claims sourced from `apps/docs/` guides are Flue's own
documentation, not independently exercised.*
