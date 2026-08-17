# DeepSeek Harness Deep Dive — Architecture & Feature Analysis

> Analysis of [`deepseek-ai/deepseek-harness`](https://github.com/deepseek-ai/deepseek-harness)
> (`dsh`) — DeepSeek's **plugin-based agent harness**. Runtime: Node.js.
> Language: TypeScript (plus a Python SDK and a C11 Landlock launcher).
> Framework: vendored [Cordis](https://github.com/cordiverse/cordis). License: MIT.
> Status: **developer preview** (`0.1.0-rc.5`); breaking changes expected.
>
> **Validated against a local source checkout** at `deepseek-harness/`, HEAD
> `47f9438` (2026-08-13). Package count and line counts below are from that tree.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Cordis: Everything Is a Plugin](#cordis-everything-is-a-plugin)
3. [Profiles, Bundles, and Boot](#profiles-bundles-and-boot)
4. [Core Packages](#core-packages)
5. [The Agent Loop: Turn and Step](#the-agent-loop-turn-and-step)
6. [Session Log as Source of Truth](#session-log-as-source-of-truth)
7. [Capability Seams](#capability-seams)
8. [Sandbox](#sandbox)
9. [ACP, JSON-RPC, Web, Headless](#acp-json-rpc-web-headless)
10. [Self-Modification](#self-modification)
11. [The Complexity Machine](#the-complexity-machine)
12. [Key Design Patterns](#key-design-patterns)
13. [Why It Matters for swe-term](#why-it-matters-for-swe-term)
14. [Summary Statistics](#summary-statistics)

---

## System Overview

DeepSeek Harness is not "a coding agent with plugins." It is a **composition
machine whose default composition happens to be a coding agent**. The README's
one-liner is accurate: *everything is a plugin*, including the model adapter,
the tool registry, the session log, and **the agent loop itself**.

A running `dsh` is an ordered stack of Cordis config rows. `web` and `headless`
are named **profiles**. Each profile stacks **bundles**. Bundles are patch files
plus the packages they mount. The user can replace any row by id.

That is a different bet from pi-mono (layered packages, concrete classes) and
from Claude Code (a product monolith). dsh bets that **replaceability at
boot** is the harness feature, and that mechanical invariants can keep a
226-package tree honest.

It is currently unreleased in the compatibility sense:
`SESSION_FORMAT_VERSION = 0` means old logs are rejected, not migrated
(`packages/core/session/src/types.ts:56`).

---

## Cordis: Everything Is a Plugin

[Cordis](https://github.com/cordiverse/cordis) (vendored under `vendor/`,
rescoped to `@deepseek-ai/cordis`) provides:

- **`Context` (`ctx`)** — proxied DI container. Services hang off properties
  (`ctx.sessions`, `ctx.llm`, `ctx.tools`, `ctx.agents`).
- **Fibers** — each config row is a fiber with validated config, injected
  services, and reversible **effects**. Unload unwinds registrations.
- **Events** — `emit` / `parallel` / `serial` / `bail` / **`waterfall`**.
  Waterfall listeners **must** call `next()` or they short-circuit the chain.

The agent loop is a waterfall consumer at `agent/pre-step`, `agent/request`,
`llm/stream`, and `tools/pre-execute` → `execute` → `post-execute`. New
behavior is supposed to attach there, not fork `agent-loop`. Changing the loop
requires updating `docs/architecture.md` (standing order in root `AGENTS.md`).

Honest boundary: boot glue (`packages/boot/app-boot`) and vendored Cordis are
**not** YAML-replaceable. "Everything is a plugin" means product behavior, not
the loader.

---

## Profiles, Bundles, and Boot

```114:117:deepseek-harness/packages/boot/app-boot/src/profile.ts
export const PROFILE_TEMPLATES: Record<string, readonly string[]> = {
  web: ['@deepseek-ai/dsh-base', '@deepseek-ai/dsh-web-app'],
  headless: ['@deepseek-ai/dsh-base', '@deepseek-ai/dsh-headless'],
}
```

Layer order onto an empty entry list:

1. Each bundle's `cordis.patch.yml` (from `dsh.bundle` in `package.json`)
2. Profile `cordis.patch.yml`
3. Home-level patch
4. `--patch` overlays

`dsh-base` is the product spine: model adapters, tools, persistence, sandbox,
approvals, settings, credentials, telemetry. `dsh-web-app` adds the browser
app; `dsh-headless` adds a one-shot runner and disables HMR.

Boot (`packages/boot/app-boot/src/index.ts` `boot()`): construct `Context`,
install `Loader`, mount root include, await activation, **fail loud** on
misconfiguration. `dsh --profile web --dump-config` prints the tree a machine
actually boots.

This is Flue's "harness as artifact" idea without Cloudflare: composition is a
config stack you can dump, patch, and reason about.

---

## Core Packages

From `docs/architecture.md`:

| Package | Owns | `ctx` key |
|---------|------|-----------|
| `core/session` | Append-only `SessionEvent` log | `ctx.sessions` |
| `core/system-prompt` | Prompt sections + tool schemas | `ctx.systemPrompt` |
| `core/tools` | Scoped registry + guarded execute | `ctx.tools` |
| `core/agent` | `Agent` interface, live registry | `ctx.agents` |
| `core/agent-loop` | Default driver (`ReactLoopAgent`) | `ctx.agentLoop` |
| `llm/llm` | Stream vocabulary + adapter seam | `ctx.llm` |

`dsh-agent` exposes `AgentFactory` so ACP/UI call `ctx.agents.create()` without
importing the loop package (`packages/core/agent/src/index.ts:183–214`). Only
one factory may register. The loop is structurally replaceable; the **durable
event vocabulary is not**.

The default driver is small by this survey's standards: `agent.ts` is **496
lines**. That is the dividend of pushing everything else into plugins.

---

## The Agent Loop: Turn and Step

A **step** is one model request plus the tools it calls. A **turn** is zero or
more steps. The turn opens before the first input is claimed and closes once
nothing is owed.

Inbox (same names Prime uses, cleaner implementation):

```122:132:deepseek-harness/packages/core/agent-loop/src/agent.ts
  followup(input: UserMessage): void {
    this.send(input, 'next-turn', true)
  }

  steer(input: UserMessage): void {
    this.send(input, 'next-step', true)
  }

  inject(input: UserMessage): void {
    this.send(input, 'next-step', false)
  }
```

`followup` wakes the next turn; `steer` wakes the next step; `inject` lands in
the next step **without** waking (context for the next real message).

Flow (durable events vs live waterfalls):

```
turn/start
  claim inbox + assemble prompt/tools
  -> agent/pre-step          (waterfall: reject | enter)
     step/start
     user/message*
     agent/request -> llm/stream -> assistant/chunk* -> assistant/message
     tool/call* -> tools/pre-execute -> execute -> post-execute -> tool/result*
     step/end
  -> agent/turn-stopping     (serial, no next())
turn/end
```

Tool scheduling honors per-call concurrency mode with a bounded rolling pool
(`packages/core/agent-loop/src/tool-calls.ts`). Abort drains started calls and
synthesizes results for unstarted ones — a real cancellation policy, not
`Promise.all` hope.

---

## Session Log as Source of Truth

This is dsh's load-bearing correctness claim.

**Model-visible means logged.** Anything that reaches a model request must be
reconstructable from the session log. A new model-visible input requires a new
`SessionEventMap` member. Runtime invariant on `llm/stream`:

```21:42:deepseek-harness/packages/core/agent-loop/src/invariant.ts
  ctx.on('llm/stream', (options: GenerateOptions, next) => {
    // ...
    const expected = session.deriveMessages()
    if (JSON.stringify(options.messages) !== JSON.stringify(expected)) {
      fail(`llm request for session "${String(session.id)}" diverges from the dispatch-time durable derivation (log-reconstruction desync)`)
    }
```

Loop-built requests are frozen and branded. The check also folds
`request/header` and compares model/system/tools. Ghost context is a failed
invariant, not a late-night heisenbug.

`deriveMessages()` projects history from the **ordered surface** (compaction
uses `replace` to shadow old nodes without deleting durable bytes)
(`packages/core/session/src/index.ts:708+`). Fork, resume, transcripts,
telemetry, and UI all derive from this stream.

`SESSION_FORMAT_VERSION = 0` (structural compatibility). SQLite backends have a
separate `STORAGE_SQLITE_SCHEMA_VERSION = 1` (`PRAGMA user_version`). Event
vocabulary growth uses `ignorable: true` on unknown types rather than bumping
the format for every new event.

Opaque ids are branded (`SessionId`, `CallId`) via `dsh-brand`.

---

## Capability Seams

A **seam** is three roles, always:

1. **Service Definition** — interface + events on `ctx.<name>`
2. **Service Provider** — backend
3. **Consumer** — usually a model-facing tool

One role is not a seam. The payoff: pointing `fs` + `subprocess` + `shell` at
the same sandbox backend moves Bash, PTY, and LSP together. E2B is a different
*execution world* (shared SDK handle); local Landlock is the same-world
argv-wrapper. Subagent providers range from in-process child to ACP delegation
behind one interface (`docs/architecture.md` capability table).

Key seams: `fs`, `subprocess`, `sandbox`, `shell`, `llm`, `tools`, `session`,
`subagent`, `compaction`, `skill`, `lsp`, `web`.

This is Go's interface story written in TypeScript plugins. swe-term can take
the taxonomy without Cordis.

---

## Sandbox

`dsh-sandbox` wraps **exact subprocess argv** under a host-path file policy.
Containers/microVMs replace the surrounding seam rather than forking every
tool.

`dsh-sandbox-local` selects a platform chain and **fails closed**:

> Missing or unusable confinement fails closed rather than returning the
> original argv.
> (`packages/sandbox/sandbox-local/src/index.ts:1–6`)

| Platform | Runner |
|----------|--------|
| Linux | bwrap, then Landlock (`@deepseek-ai/node-addon-landlock-run`, ~300 ln C11, musl-static, self-restrict-then-exec) |
| macOS | Seatbelt |
| Windows | ACL restricted-token runner |

Policy plugins interpret `read-only` / `workspace-write` / `danger-full-access`.
Escalation goes through approval markers. E2B is a POC remote world, not the
default.

This is the strongest local sandbox story in the survey after Codex's
execution policy — and it is behind an interface, which Codex's is not in the
same "swap the world" sense.

---

## ACP, JSON-RPC, Web, Headless

- **ACP** (`packages/acp/acp`) — automation-only Agent Client Protocol over
  stdio JSON-RPC (`@agentclientprotocol/sdk`). Creates agents via `ctx.agents`,
  auto-approves permission waterfalls for trusted programmatic clients.
- **JSON-RPC SDK** — streams `session.event` notifications (full log
  envelopes). Python SDK launches bundled `dsh-jsonrpc-agent`.
- **Headless profile** — one-shot task, no HTTP, no browser.
- **Web profile** — host transport + browser Cordis tree (large
  `cordis.patch.yml`).

Same `dsh-base` spine. Frontends are bundles, not forks of the loop. That
matches swe-term's Frontend interface more closely than Prime's TUI-shaped
interactive-mode.ts.

---

## Self-Modification

Not a separate `self-modification/` package in this checkout. The surface is
`dsh-tool-cordis`: inspect the live plugin catalog, `cordis_define` immutable
JS function-body packages (no TS transform), `cordis_run` / `stop` / `undefine`
to mount per session. A pre-step waterfall can inject `@pluginId` context.

This is a research demo (`pnpm run demo:cordis`), not a hardened loader.
Dynamic JS eval in-process with Cordis fiber lifecycle. Interesting as a
"harness that can see itself"; dangerous as a default.

---

## The Complexity Machine

| Mechanism | Scale |
|-----------|-------|
| npm workspaces under `packages/` | **226** |
| Package groups | ~49 |
| Approx TS/TSX lines (`packages/` + `vendor/` + `native/` + `apps/`) | ~527k |
| Generated catalogs | tools, config, persistence, Cordis API, module graph, scoped events, … |
| Coverage gate | **per-file 100%** on `packages/*/*/src` |
| Docs | word budgets, bilingual pairing, `doc-sync`, export JSDoc verification |
| Invariants | every package owns `./invariant` |
| Vendored Cordis | pinned SHAs + logged local modifications |

Pre-release stance (root `AGENTS.md`): prefer the correct foundation over
compatibility shims; no external consumers yet. That is a luxury swe-term will
not have once it ships, and a cost it should not copy *before* it ships.

---

## Key Design Patterns

1. **Log is the model history.** Derive, don't shadow.
2. **Invariant at the LLM boundary.** Check reconstruction on every stream.
3. **Waterfalls, not subclasses.** Extension points are named phases.
4. **Seam = definition + provider + consumer.** Swap worlds, not tools.
5. **Profile = ordered patches.** Headless vs web is composition.
6. **Fail closed / fail loud.** Missing sandbox and missing config do not
   silently degrade.
7. **Turn/step vocabulary.** Auditable unit of work for ACP and telemetry.
8. **Branded ids.** Session and call ids are types, not strings.

---

## Why It Matters for swe-term

dsh is the most explicit **harness architecture** in the survey — more so than
pi-mono, which is a layered product, and more so than Claude Code, which is a
monolith with extension points. The ideas that change agent *correctness*
(log-derived requests, named extension phases, capability seams, fail-closed
sandbox) fit a small Go core. The ideas that change *org process* (226
packages, 100% coverage, Typert, bilingual doc gates) do not.

If swe-term takes one sentence from dsh, it should be: **the bytes that hit
the provider are a pure function of the session log, and we assert that at
dispatch time.**

---

## Summary Statistics

| Metric | Value (local 0.1.0-rc.5) |
|--------|--------------------------|
| Package manifests under `packages/` | 226 |
| Default loop (`agent.ts`) | **496** lines |
| Session store (`session/src/index.ts`) | 1,157 lines |
| Loop invariant | 63 lines, load-bearing |
| Session format version | 0 (no migration) |
| SQLite schema version | 1 |
| Profiles | `web`, `headless` |
| Sandbox default | fail-closed local chain |
| ACP | first-class automation server |
| Status | developer preview, breaking OK |
