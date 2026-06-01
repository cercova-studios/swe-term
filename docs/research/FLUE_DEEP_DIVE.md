# Flue Deep Dive — Architecture & Feature Analysis

> Analysis of [`withastro/flue`](https://github.com/withastro/flue) — Fred K. Schott's
> ("the Astro creator") **agent harness framework**. Positioned as "Claude Code, but 100%
> headless and programmable. No TUI. No GUI. Just TypeScript."
> Runtime targets: Node.js and Cloudflare Workers. Language: TypeScript.
> Built on **`@earendil-works/pi-agent-core` + `@earendil-works/pi-ai`** (the renamed pi-mono
> lower layers).
>
> **Validated against a local source checkout** at `/home/rohit/sandbox/swe-term/flue`
> (version `0.7.0`, license Apache-2.0, marked *Experimental*). A real consumer app —
> `/home/rohit/sandbox/gvision` — is also used to confirm the public API surface. An earlier
> draft of this doc was sourced from DeepWiki and got several specifics wrong; those are
> corrected here against the actual code.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [The Framework-vs-SDK Thesis](#the-framework-vs-sdk-thesis)
3. [Package Layout](#package-layout)
4. [Engine: It Wraps pi, It Doesn't Rebuild It](#engine-it-wraps-pi-it-doesnt-rebuild-it)
5. [Core Primitives](#core-primitives)
6. [The Agent Handler & `FlueContext`](#the-agent-handler--fluecontext)
7. [Harness & Sessions](#harness--sessions)
8. [Tools & Typed Results (the dual schema stack)](#tools--typed-results-the-dual-schema-stack)
9. [Subagents: Roles and Tasks](#subagents-roles-and-tasks)
10. [Sandboxes](#sandboxes)
11. [Skills](#skills)
12. [MCP Integration](#mcp-integration)
13. [Runs vs Sessions: the durability model](#runs-vs-sessions-the-durability-model)
14. [The Build System (Agent-as-Artifact)](#the-build-system-agent-as-artifact)
15. [Deployment Targets](#deployment-targets)
16. [Providers & Models](#providers--models)
17. [The Client SDK (`@flue/sdk`)](#the-client-sdk-fluesdk)
18. [A Real App: gvision](#a-real-app-gvision)
19. [Relationship to pi-mono](#relationship-to-pi-mono)
20. [Key Design Patterns](#key-design-patterns)
21. [Summary Statistics](#summary-statistics)

---

## System Overview

Flue is a serious attempt to make **"the agent harness" a deployable framework** rather than a
library you wire up yourself. Its pitch borrows the web-framework playbook (think Astro or
Next.js, applied to agents): you write agents as source files under a `.flue/` directory, run
`flue build`, and get a self-contained server artifact you can deploy to Node or Cloudflare
Workers.

It is explicitly **headless**. There is no TUI, no human-operator assumption. An agent is a handler
function plus an initialized runtime, exposed as an HTTP service (built on **Hono**). This is a
deliberate inversion of Claude Code / Codex / pi (all of which center a human at a terminal).

**Tech Stack (from `packages/runtime/package.json`):**

| Layer | Technology |
|-------|-----------|
| Language | TypeScript (Node ≥ 22.18) |
| Runtime targets | Node.js, Cloudflare Workers |
| Agent engine | `@earendil-works/pi-agent-core` (pi's `Agent`, loop, `ThinkingLevel`) |
| Provider/model layer | `@earendil-works/pi-ai` (model registry, providers, `Type`, usage) |
| HTTP | `hono` + `hono-openapi` (`@hono/node-server` on Node) |
| Tool schema | pi-ai `Type` (TypeBox lineage) for tool params |
| Typed results | `valibot` + `@valibot/to-json-schema` |
| MCP | `@modelcontextprotocol/sdk` (client) |
| Default sandbox | `just-bash` (in-process emulated shell) |
| Cloudflare | `@cloudflare/sandbox`, `@cloudflare/shell`, `@cloudflare/codemode`, Durable Objects |
| Package build | `tsdown` (publishing); `esbuild` / `wrangler` (user-agent builds) |
| Config | `flue.config.ts` (`defineConfig` from `@flue/cli/config`) |

---

## The Framework-vs-SDK Thesis

Flue's central claim is that existing agent tooling ships as **SDKs** (libraries you assemble) and
that the missing layer is a **framework** (opinionated structure + build + deploy). Concretely:

- **Convention over wiring.** Agents, roles, skills, and `app.ts` live in a known `.flue/`
  directory.
- **A build step.** `flue build` compiles the project into a deployable artifact — agents are
  *compiled*, not just imported.
- **Web-framework DX.** `flue dev` (watch mode), `flue run` (one-shot), `flue build` (artifact)
  mirror `next dev` / `next build`.

The tagline "if you know Claude Code / Codex / OpenCode / Pi, you know how to build agents with
Flue" reframes harness engineering as a teachable discipline with a standard framework.

---

## Package Layout

The monorepo (pnpm + turbo) ships **four** packages under `packages/*/` (all version `0.7.0`
except the deprecated stub):

| Package | Version | Role |
|---------|---------|------|
| **`@flue/runtime`** | `0.7.0` | The runtime a built Flue app depends on: agent harness, sessions, tools, sandbox plumbing, the Hono HTTP runtime, MCP client, compaction, providers. Subpath exports: `./app`, `./node`, `./cloudflare`, `./client`, `./sandbox`, `./internal`. |
| **`@flue/cli`** | `0.7.0` | The `flue` binary + build/dev tooling. Bundles user agents (esbuild on Node), merges `wrangler` config on Cloudflare, parses agent files, loads env files. Exports `defineConfig` from `@flue/cli/config`. |
| **`@flue/sdk`** | `0.7.0` | A **client SDK** for calling *deployed* Flue servers over HTTP (`createFlueClient()`). **Not** the old runtime — legacy `@flue/sdk/*` subpaths now throw migration errors. |
| **`@flue/connectors`** | `0.3.7` | **Deprecated private stub.** Connectors are no longer an npm package; install them with `flue add <name>` (which fetches Markdown from `flueframework.com`). |

> **Correction vs the earlier draft:** there are four packages, not two. The old draft also called
> `@flue/sdk` "a migration placeholder for `@flue/runtime`" — backwards. `@flue/sdk` is the
> *client*; the *runtime* used to be published under the `@flue/sdk` name and moved to
> `@flue/runtime` + `@flue/cli`.

---

## Engine: It Wraps pi, It Doesn't Rebuild It

The single most important architectural fact: **Flue does not implement an agent loop.** It
constructs pi's `Agent` and delegates each turn to it.

`packages/runtime/src/session.ts`:

```ts
import { Agent } from '@earendil-works/pi-agent-core';
// ...
this.harness = new Agent({
  initialState: {
    systemPrompt,
    model: this.config.model,
    tools,
    messages: previousMessages,
    thinkingLevel: this.config.thinkingLevel ?? 'medium',
  },
  getApiKey: (provider) => this.getProviderApiKey(provider),
  onPayload: (payload, model) => this.applyProviderPayloadOverrides(payload, model),
  toolExecution: 'parallel',
  sessionId: options.affinityKey,
});
// a turn is just:
await this.harness.prompt(args.promptText, args.images);
```

So pi-agent-core owns the stream → tools → loop cycle (with `toolExecution: 'parallel'`), and pi-ai
owns providers, model metadata, the `Type` schema export, and usage accounting. **Everything Flue
adds is *around* that core**: session/run persistence, compaction, an event/SSE layer, skills and
roles, sandbox adapters, typed-result tools, the build system, and the HTTP runtime.

This is the same `pi-agent-core` + `pi-ai` pair that swe-term ports — which makes Flue a direct
study in *what a harness needs beyond the loop*.

---

## Core Primitives

| Primitive | What it is |
|-----------|-----------|
| **Agent** | A handler function (`agents/<name>.ts`, default export) plus a model + harness, deployed as an HTTP endpoint. |
| **`FlueContext`** | The argument every agent handler receives: `{ id, runId, payload, env, req, log, init }`. |
| **Harness** | The `FlueHarness` returned by `ctx.init()` — owns the sandbox and named sessions. (Impl class `Harness`.) |
| **Session** | A stateful conversation (`FlueSession`) with message history, tool outputs, and compaction. (Impl class `Session`, wraps pi's `Agent`.) |
| **Run** | A single HTTP invocation of an agent, with its own event stream and result — distinct from a session (see [Runs vs Sessions](#runs-vs-sessions-the-durability-model)). |
| **Sandbox** | The execution environment: default in-process `just-bash` + `InMemoryFs`, `local()` (Node host), or Cloudflare containers. |
| **Tools** | Typed actions (`ToolDef`); params via pi-ai `Type`; implementation runs host-side, the agent sees only the schema. |
| **Skills** | Reusable Markdown "sub-programs" invoked via `session.skill()`. |
| **Roles** | Markdown personas (`roles/<name>.md`, frontmatter + body) applied as system-prompt overlays. |
| **MCP** | `connectMcpServer()` — a client adapter to remote MCP servers. |
| **Typed result** | `prompt(..., { result: schema })` returns validated output via a valibot `finish`/`give_up` tool pair. |

---

## The Agent Handler & `FlueContext`

An agent is a module under `.flue/agents/<name>.ts` whose **default export** is an async handler.
The handler receives a `FlueContext` (`packages/runtime/src/types.ts`):

```ts
export interface FlueContext<TPayload = any, TEnv = Record<string, any>> {
  readonly id: string;        // agent instance id (also the durable-object key on CF)
  readonly runId: string;     // this specific HTTP invocation
  readonly payload: TPayload; // parsed request body
  readonly env: TEnv;         // secrets / bindings (host-side only)
  readonly req: Request | undefined; // the raw Fetch Request
  readonly log: FlueLogger;   // structured log → emitted as run events
  init(options: AgentInit): Promise<FlueHarness>;
}
```

> **Correction vs the earlier draft:** `init()` is **not** a top-level export — it's a method on
> `FlueContext`. The context also carries `runId`, `req`, and a structured `log`, not just
> `{ init, payload, env, log, id }`.

An agent may also export `export const triggers = { webhook: true }` to expose itself as a public
HTTP webhook in production (the build statically parses this; `FLUE_MODE=local` / `flue run`
bypasses the gate for local testing).

---

## Harness & Sessions

`ctx.init(options: AgentInit)` configures the runtime and returns a **`FlueHarness`**. `AgentInit`
(real fields, `types.ts`):

```ts
export interface AgentInit {
  name?: string;
  cwd?: string;
  sandbox?: false | SandboxFactory | BashFactory; // omit/false ⇒ default in-memory sandbox
  persist?: SessionStore;
  model: ModelConfig;        // REQUIRED, e.g. 'anthropic/claude-sonnet-4-6'
  role?: string;             // a roles/<name>.md persona
  thinkingLevel?: ThinkingLevel;
  tools?: ToolDef[];
  compaction?: /* compaction config */;
}
```

`FlueHarness` and `FlueSession` are the **public interfaces** (impl classes `Harness` / `Session`):

```ts
export interface FlueHarness {
  readonly name: string;
  session(name?: string, options?: SessionOptions): Promise<FlueSession>;
  readonly sessions: FlueSessions;
  shell(command: string, options?: ShellOptions): CallHandle<ShellResult>;
  readonly fs: FlueFs;
}

export interface FlueSession {
  readonly name: string;
  prompt(message, options?): CallHandle<PromptResponse | PromptResultResponse>;
  shell(command, options?): CallHandle<ShellResult>;
  readonly fs: FlueFs;
  skill(name, options?): CallHandle<...>;
  task(prompt, options?): CallHandle<...>;
  compact(): Promise<void>;
  delete(): Promise<void>;
}
```

Two things worth noting:

- **`CallHandle`** — `prompt`/`skill`/`task`/`shell` return an *abortable* handle (`abort.ts`), not
  a bare promise. Long operations can be cancelled.
- **Harness-level `shell` and `fs`** exist alongside the session-level ones, so an orchestrator can
  touch the sandbox without a conversation.

Multiple harnesses can share **one sandbox** (e.g. a control harness and a project harness over the
same filesystem) — a clean separation of "the agent that orchestrates" from "the agent that works
in the repo."

### Session persistence

- **Node:** `InMemorySessionStore` — *"Sessions persist for the lifetime of the process."* Pass a
  custom `SessionStore` via `persist` for real durability.
- **Cloudflare:** the generated entry defaults to a **Durable-Object SQLite** store
  (`CREATE TABLE IF NOT EXISTS flue_sessions (id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at INTEGER NOT NULL)`),
  so history survives across requests. (A simpler DO `setState`-map store exists in
  `cloudflare/session-store.ts` but is described as "usually not needed".)

---

## Tools & Typed Results (the dual schema stack)

Flue uses **two** schema systems, which the earlier draft conflated into "valibot":

| Use | Library | How you get it |
|-----|---------|----------------|
| **Tool parameters** | pi-ai **`Type`** (TypeBox lineage; `TSchema`) | `import { Type } from '@flue/runtime'` (re-exported: `export { Type } from '@earendil-works/pi-ai'`) |
| **Typed `result` output** | **valibot** | `import * as v from 'valibot'`; pass as `prompt(msg, { result: schema })` |

A `ToolDef` carries a `name`, `description`, `parameters` (a `Type.Object({...})`), and an
`execute(args, signal)` function. Crucially, the **tool implementation runs in the host process**
(it can read `ctx.env` / `process.env`), while the **agent only ever sees the schema** — secrets
stay host-side and never enter the prompt or filesystem context. This host-side-execution model is
a clean security boundary a headless framework can enforce more strictly than a terminal agent.

Typed results work by injecting a valibot-backed **`finish`/`give_up` tool pair** (`result.ts`):
the model "returns" by calling `finish` with arguments matching the schema, and Flue runs
`valibot.safeParse` to enforce refinements before resolving the typed value. Agents become
functions with typed return values, not just chat.

Built-in tools (`createTools`, `BUILTIN_TOOL_NAMES` in `agent.ts`): `bash` and the file tools
(`read`, `write`, `edit`, `grep`, `glob`) operating against the active sandbox.

---

## Subagents: Roles and Tasks

- **Roles** — Markdown personas at `roles/<name>.md` with YAML frontmatter (`description`, and may
  override model / `thinkingLevel`) and a system-prompt body. Selected per call via
  `init({ role })` or per session, applied as a **system-prompt overlay** without forking state.
- **Tasks** — `session.task(prompt)` runs a focused, one-shot **child agent in a detached
  session**: it shares the parent's sandbox/filesystem but keeps its own message history. This is
  the delegation primitive: spin up a specialist, let it work in the same workspace, collect its
  result (optionally typed via `result`).

---

## Sandboxes

Sandboxes are where Flue's "agents take real action safely" promise lives. Three real flavors:

| Sandbox | Mechanism | Isolation |
|---------|-----------|-----------|
| **Default (in-memory)** | `createDefaultEnv()` wires the **`just-bash`** `Bash` emulator over an **`InMemoryFs`**. Selected by omitting `sandbox` (or `sandbox: false`). | **Emulation, not isolation** — runs in-process; no real filesystem, no container. Lightweight, fast, cheap. |
| **`local()`** (Node) | `local()` from `@flue/runtime/node` → real `fs` + `child_process`, env **allowlist** (only explicitly-listed vars passed through to avoid secret leakage). | **None** — intended for environments where the runner itself is the jail (a CI job). |
| **Cloudflare** | `getShellSandbox()` (Workspace + codemode) and `cfSandboxToSessionEnv()` over `@cloudflare/sandbox` containers. | Real container isolation. |

> **Corrections vs the earlier draft:** there is **no Flue export literally named `just-bash()`** —
> `just-bash` is the npm package that powers the *default* sandbox. And **`getVirtualSandbox()` was
> removed** — it now throws, telling you the default in-memory sandbox is already what you wanted.

---

## Skills

Skills are reusable "sub-programs" defined as **Markdown files**, invoked via
`session.skill('name', { args })`. Their bodies are read at call time, so editing a skill between
invocations takes effect immediately. This mirrors the Claude-Code/Codex skill model (Markdown +
frontmatter) but fits Flue's "agent is a directory" philosophy: skills are just files the build
packages into the artifact.

---

## MCP Integration

MCP is a **runtime tool adapter** built on `@modelcontextprotocol/sdk`: `connectMcpServer(name,
options)` connects an agent to a remote MCP server and exposes its tools. It defaults to modern
**streamable HTTP**, with an opt-in for legacy **SSE** (`transport: 'sse'`).

Per the project's own docs: Flue does **not** auto-detect transports, does **not** spawn local
stdio MCP servers, and does **not** handle OAuth callbacks. So it is an MCP *client* for remote HTTP
servers, not a full MCP host. The value proposition is secret hygiene — connect to authenticated
services via MCP so credentials live in env vars, not in prompts.

---

## Runs vs Sessions: the durability model

A subtlety the earlier draft missed: Flue tracks **runs** (HTTP invocations) separately from
**sessions** (conversations).

- **`RunStore`** holds per-run events + result.
  - Node: `InMemoryRunStore`.
  - Cloudflare: `createDurableRunStore` → Durable-Object **SQLite**.
- **`RunRegistry`** is a cross-deployment **pointer index** that maps a `runId` to the instance
  serving it.
  - Node: `InMemoryRunRegistry`.
  - Cloudflare: the auto-injected **`FlueRegistry` Durable Object** (`extends DurableObject`) plus a
    client.

Runs expose a `RunRecord` (`runId`, `instanceId`, `agentName`, `status`, …), an SSE stream
(`/runs/:runId/stream`), and `flue logs` tailing. On Cloudflare, webhook handlers keep running via
a `runFiber` so work survives the request lifecycle. This is **run/event durability** (resume the
*stream and result* of an invocation), distinct from **session continuity** (resume the
*conversation*). swe-term needs both, and Flue's split is a useful reference.

---

## The Build System (Agent-as-Artifact)

The defining idea: **an agent is a directory compiled into a deployable server artifact.**

- **`flue.config.ts`** — project config via `defineConfig` (from **`@flue/cli/config`**, not the
  runtime). Fields: `target` (`'node' | 'cloudflare'`), optional `root`, optional `output`.
- **Agent files** — `.flue/agents/<name>.ts`, with `.flue/roles/`, `.flue/app.ts`, skills, etc.
- **`flue build`** — compiles `.flue/` into an artifact in `./dist` for production deploys.
- **`flue dev`** — long-running watch-mode dev server; edits trigger rebuild + reload.
- **`flue run <agent>`** — one-shot: build, start a temporary server, invoke once, stream output,
  shut down (designed for CI; sets `FLUE_MODE=local`).
- **`flue init` / `flue add` / `flue logs`** — scaffold a project, install a connector (Markdown
  fetched from `flueframework.com`), tail run logs.

> **Correction vs the earlier draft:** there is **no `flue deploy` command**. The CLI verbs are
> `dev`, `run`, `build`, `init`, `add`, `logs`. You deploy by running `wrangler deploy` after
> `flue build` (Cloudflare) or by running the `dist/server.mjs` (Node).

### What actually bundles the agent

- The **packages themselves** (`@flue/runtime`, `@flue/cli`, `@flue/sdk`) build with **`tsdown`**.
- **User agents on the Node target** are bundled by **`esbuild`** into a single self-contained
  `dist/server.mjs` (`build-plugin-node.ts`: *"Node has no platform-provided bundler — esbuild is
  our final-output step"*).
- **User agents on the Cloudflare target** are **not** esbuild-bundled by Flue: the CLI writes an
  `_entry.ts` and **`wrangler` bundles it** (`build-plugin-cloudflare.ts`: `bundle = 'none'`).

> **Correction vs the earlier draft:** user-agent builds use esbuild/wrangler, not tsdown (tsdown
> only builds the published packages).

---

## Deployment Targets

| Target | Build output | Session state | Run state | Notes |
|--------|-------------|---------------|-----------|-------|
| **Node.js** | bundled `dist/server.mjs` (esbuild) | `InMemorySessionStore` (process lifetime; custom store optional) | `InMemoryRunStore` / `InMemoryRunRegistry` | run `node dist/server.mjs` |
| **Cloudflare** | `_entry.ts` bundled by `wrangler` | DO SQLite `flue_sessions` | DO SQLite run store + `FlueRegistry` DO | `wrangler deploy` after `flue build` |

**Durable execution** is strongest on Cloudflare: Durable Objects persist session and run state
across requests, and each agent **instance** is keyed by the URL `<id>` to its own DO. On Node,
durability is opt-in (bring a `SessionStore`).

---

## Providers & Models

Flue leans on **`pi-ai`** for the provider/model layer and adds a thin registration surface
(`runtime/src/runtime/providers.ts`):

- Models are `provider/model-id` strings (e.g. `anthropic/claude-sonnet-4-6`, `openai/gpt-5.5`).
- **`registerProvider(name, registration)`** — a URL-prefix provider registry.
- **`configureProvider(provider, settings)`** — transport overrides (baseUrl, apiKey, headers) keyed
  by pi provider slug.
- **`registerApiProvider`** — a re-export of pi-ai's own registration.
- API keys come from env vars (`ANTHROPIC_API_KEY`, …) or from an optional **`app.ts`** at the
  source root.
- **Cloudflare Workers AI:** models route through **`env.AI.run()`** instead of HTTP, via a pi-ai
  provider registered under the `cloudflare-ai-binding` API (`workers-ai-provider.ts`). The AI
  binding is captured at registration; model metadata is hydrated from pi-ai's catalog.

### Usage accounting

`response.usage` is a `PromptUsage` that **structurally mirrors pi-ai's `Usage`**
(`fromProviderUsage` normalizes it):

```ts
export interface PromptUsage {
  input: number; output: number; cacheRead: number; cacheWrite: number; totalTokens: number;
  cost: { input: number; output: number; cacheRead: number; cacheWrite: number; total: number };
}
```

### Compaction

Flue has its **own** `compaction.ts` (it does not import pi's compaction). It detects overflow with
pi-ai's `isContextOverflow` and summarizes with pi-ai's `completeSimple`. Token counting uses
`calculateContextTokens(usage)` from real provider usage when available, falling back to an
`estimateTokens` **chars/4** heuristic; the threshold is ~96% of the context window minus a reserve.

---

## The Client SDK (`@flue/sdk`)

`@flue/sdk` is the **caller-side** library for talking to a *deployed* Flue server —
`createFlueClient({ ... })` returns:

- `agents.invoke(...)` — call an agent **sync**, as a **webhook**, or as a **stream**.
- `runs.{ get, events, stream }` — fetch a run, its events, or an SSE stream.
- `admin.{ agents, instances, runs }` — inspect a deployment.

(The runtime also exposes a read-only **`admin()`** Hono sub-app with an OpenAPI surface for the
same inspection.) Legacy `@flue/sdk/app`, `/client`, `/sandbox`, `/node`, `/cloudflare`, `/config`
subpaths now throw migration errors pointing to `@flue/runtime` / `@flue/cli`.

---

## A Real App: gvision

`/home/rohit/sandbox/gvision` is a real Flue app (an agentic backend for Meta RayBans driven by
Gemini Live) that confirms the public API in practice.

**`.flue/app.ts`** — the Cloudflare Workers entry uses `@flue/runtime/app`'s `flue()` and routes
the OpenAI-Responses-shaped request to an agent instance keyed by session id:

```ts
import { flue } from '@flue/runtime/app';
// ...maps POST /v1/responses → /agents/gvision/<sessionId>...
return flue().fetch(forwarded, env, ctx);
```

**`.flue/agents/gvision.ts`** — the handler shows the real `FlueContext` destructuring and the
`init → session → prompt` flow:

```ts
export const triggers = { webhook: true };

export default async function ({ init, payload, env, log, id }: FlueContext<ResponsesPayload, Env>) {
  const harness = await init({
    model: 'anthropic/claude-sonnet-4-6',
    role: 'companion',
    tools: buildTools(env),
  });
  const session = await harness.session();
  const response = await session.prompt(promptText);
  log.info('gvision.complete', {
    inputTokens: response.usage.input,
    outputTokens: response.usage.output,
    cost: response.usage.cost.total,   // ← pi-ai Usage shape, surfaced verbatim
  });
  return responsesShape({ /* ... */ text: response.text, model: response.model.id });
}
```

**`.flue/tools/save-note.ts`** — a custom tool, params via `Type` re-exported from `@flue/runtime`,
`execute(args, signal)` runs host-side with a `GITHUB_TOKEN` from `env`:

```ts
import { Type, type ToolDef } from '@flue/runtime';
export function createSaveNoteTool(opts): ToolDef {
  return {
    name: 'save_note',
    description: '...',
    parameters: Type.Object({ title: Type.String(), content: Type.String(), /* ... */ }),
    execute: async (args, signal) => { /* GitHub API calls; secret never reaches the model */ },
  };
}
```

**`.flue/roles/companion.md`** — a role file: YAML frontmatter (`description:`) + a system-prompt
body that documents the available tools and even references the default sandbox by name (*"Real
Linux is not available here by default; this is just-bash."*).

This one app independently confirms: the four-package split, the `@flue/runtime` import surface,
`FlueContext`, `init`/`session`/`prompt`, the `Type`-based `ToolDef` with `execute(args, signal)`,
the pi-ai `Usage` shape on `response.usage`, the `provider/model` id format, Markdown roles, and the
Cloudflare `wrangler deploy` workflow.

---

## Relationship to pi-mono

Flue is **not** a from-scratch agent — it is a **framework wrapper around pi-mono's lower layers**:

- The agent loop, `Agent`, and `ThinkingLevel` come from **`@earendil-works/pi-agent-core`**
  (Flue's `Session` constructs the `Agent` and calls `.prompt()`; it does not reimplement the loop).
- The provider/model registry, the `Type` schema, and the `Usage` accounting come from
  **`@earendil-works/pi-ai`**.

So the lineage is: **pi-mono** (badlogic) provides the engine; **Flue** (Schott/Astro) provides the
framework, build system, sandbox abstraction, run/session durability, HTTP runtime, client SDK, and
deployment targets on top. This makes Flue a precise study in *what a harness needs beyond the agent
loop* — exactly the surface area swe-term must also provide (sessions, runs, sandboxes, deployment,
frontends) around its ported pi-core.

> Note the scope detail: the **local pi-mono checkout** publishes `@mariozechner/*` (v0.65.2), while
> **Flue depends on the renamed `@earendil-works/*`** packages. Same code lineage, two scopes.

---

## Key Design Patterns

### 1. Agent = compiled directory → server artifact
Convention-based `.flue/` layout compiled (esbuild on Node, wrangler on Cloudflare) into a
deployable server. Web-framework DX (`dev`/`run`/`build`).

### 2. Headless-first
No TUI, no human-in-the-loop assumption. An agent is a handler you call from a server, a cron, or a
CI job. The opposite of terminal-centric agents.

### 3. Build on pi, don't rebuild
Reuse `pi-agent-core` (`Agent`, loop) + `pi-ai` (providers, `Type`, usage); spend effort on
framework concerns. A pragmatic "stand on the shoulders" choice — and the clearest evidence that the
pi-core is reusable.

### 4. Sandbox as a first-class, swappable primitive
Default in-process `just-bash` emulation → `local()` (host, CI-jailed) → Cloudflare containers (real
isolation). The same agent code runs against different isolation guarantees by swapping the sandbox.

### 5. Runs and sessions are different durable objects
Run/event durability (resume an invocation's stream + result) is separate from session continuity
(resume a conversation). Both are backed by Durable-Object SQLite on Cloudflare.

### 6. Host-side tool execution, schema-only to the agent
Tools run in the host process with `env` access; the model sees only the `Type` schema. Secrets
never enter the context.

### 7. Two schemas for two jobs
pi-ai `Type` (TypeBox lineage) for tool params; valibot for typed `result` output (via an injected
`finish`/`give_up` tool pair).

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Language | TypeScript (Node ≥ 22.18) |
| Version / stage | `0.7.0`, Apache-2.0, *Experimental* |
| Packages | 4: `@flue/runtime`, `@flue/cli`, `@flue/sdk` (client) + deprecated `@flue/connectors` |
| Engine | `@earendil-works/pi-agent-core` (`Agent`, loop) + `@earendil-works/pi-ai` (providers, `Type`, usage) |
| Agent loop | **delegated to pi's `Agent`** (`toolExecution: 'parallel'`); Flue does not reimplement it |
| HTTP runtime | Hono (`@hono/node-server` on Node) |
| Entry point | `ctx.init(options)` (method on `FlueContext`) → `FlueHarness` |
| Context | `{ id, runId, payload, env, req, log, init }` |
| Session API | `prompt`, `skill`, `task`, `shell`, `compact`, `delete`, `fs` (all `CallHandle`/abortable) |
| Tool schema | pi-ai `Type` (re-exported from `@flue/runtime`) |
| Typed results | valibot (`prompt(..., { result })` → `finish`/`give_up` pair) |
| Sandboxes | default in-process `just-bash` + `InMemoryFs`; `local()` (Node host); Cloudflare containers; `getVirtualSandbox()` removed |
| Session store | Node `InMemorySessionStore`; Cloudflare DO SQLite (`flue_sessions`) |
| Run model | `RunStore` + `RunRegistry`; Cloudflare `FlueRegistry` Durable Object; SSE `/runs/:id/stream` |
| MCP | client only: `connectMcpServer()`, streamable HTTP (default) / SSE; no stdio, no OAuth |
| Build | packages via `tsdown`; user agents via `esbuild` (Node) / `wrangler` (Cloudflare) |
| CLI | `dev`, `run`, `build`, `init`, `add`, `logs` (no `deploy`) |
| Config | `flue.config.ts` via `defineConfig` from `@flue/cli/config`; `target: 'node' \| 'cloudflare'` |
| Compaction | Flue's own `compaction.ts` (pi `completeSimple` + `isContextOverflow`; chars/4 fallback) |
| Providers | `registerProvider` / `configureProvider`; Cloudflare Workers AI via `env.AI.run()` |
| Client SDK | `@flue/sdk` `createFlueClient()` — `agents.invoke`, `runs`, `admin` |
| Headless | Yes — no TUI/GUI by design |

---

*Validated against the local Flue source checkout at `/home/rohit/sandbox/swe-term/flue` (`0.7.0`)
and the real consumer app at `/home/rohit/sandbox/gvision`. File and symbol references are from that
checkout; Flue is Experimental and specifics change frequently.*
