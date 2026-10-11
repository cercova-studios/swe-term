# DeepSeek Harness Deep Dive — Architecture & Feature Analysis

> Analysis of [`deepseek-ai/deepseek-harness`](https://github.com/deepseek-ai/deepseek-harness)
> (`dsh`) — DeepSeek's **plugin-based agent harness**. Runtime: Node.js.
> Language: TypeScript (plus a Python SDK and a C11 Landlock launcher).
> Framework: vendored [Cordis](https://github.com/cordiverse/cordis). License: MIT.
> Status: **developer preview** (`0.2.1-alpha.2`); breaking changes expected
> (`README.md` "Developer preview").
>
> **Validated against a local source checkout** at `deepseek-harness/`, HEAD
> `d743267388` (committed 2026-10-09; refreshed 2026-10-10). Package count and
> line counts below are from that tree. Previous validation: `c15f2f4d2`
> (2026-08-16).

---

## Changes since 2026-08-16 (c15f2f4d2)

~9,000 commits; package manifests 226 → **340**, TS/TSX lines ~529k → ~1.10M.

1. **Runtime invariant plugins deleted.** `@deepseek-ai/dsh-invariants`, every
   `./invariant` companion, and `packages/core/agent-loop/src/invariant.ts` are
   gone (`f028f25667`, `docs/upgrade-guide/v0.2.0-rc.2/remove-runtime-invariants/`).
   Request reconstructability is now enforced by construction plus tests (see
   [Session Log](#session-log-as-source-of-truth)).
2. **Session format left 0 and shipped.** `SESSION_FORMAT_VERSION = 4`
   (`packages/core/session/src/types.ts:89`) with an adjacent migration chain;
   V4 is recorded as released in `dsh-v0.2.0-rc.2` (`docs/session-format-status.md`).
   "Old logs are rejected" no longer holds.
3. **Five profiles, not two.** `acp`, `web`, `headless`, `sdk`, `sdk-minimal`
   (`packages/boot/app-boot/src/profile.ts:179`); new bundles `acp-app`,
   `sdk-app`, `sdk-minimal` under `packages/bundle/`.
4. **ACP became a standard v1 server with forwarded permissions.**
   `session/request_permission` goes to the client instead of auto-approval
   (`511181684c`, `packages/acp/acp/README.md`).
5. **Remote execution retired.** E2B and the `code-runtime` group were removed
   (`c49db8bc8c`); a new `ptc-runtime/` group runs model-written programs that
   call host functions (`packages/ptc-runtime/README.md`, `dsh-ptc-runtime-node`
   mounted by `packages/bundle/base/cordis.patch.yml`).
6. **Self-modification split and gated.** `dsh-tool-cordis` now only exposes
   `cordis_inspect_list` / `cordis_inspect_query`; `cordis_define` moved to
   `dsh-cordis-host-runner` with human-approved client activation
   (`packages/extensions/cordis-host-runner/src/index.ts:1–5`), mounted only by
   the web-app bundle.
7. **New package groups:** `browser-use`, `computer-use`, `ssh`, `deliverables`,
   `document`, `telemetry`, and a 44-package `experimental/` tree (agent teams,
   Claude Code / Codex hook compatibility, Python PTC). Removed: `hooks`,
   `e2b`, `code-runtime`, `examples`, `runtime-diagnostics`.
8. **Landlock launcher folded into `@deepseek-ai/node-addon-system`**
   (`native/system/packages/entry/src/main.c`), replacing
   `node-addon-landlock-run`.

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
340-package tree honest.

It now carries compatibility obligations: `SESSION_FORMAT_VERSION = 4`
(`packages/core/session/src/types.ts:89`) with a migration chain for historical
formats, and V4 is recorded as published (`docs/session-format-status.md`).

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

```179:195:deepseek-harness/packages/boot/app-boot/src/profile.ts
export const PROFILE_TEMPLATES: Record<string, ProfileTemplate> = {
  acp: { bundles: ['@deepseek-ai/dsh-base', '@deepseek-ai/dsh-acp-app'] },
  web: { bundles: ['@deepseek-ai/dsh-base', '@deepseek-ai/dsh-web-app'] },
  headless: { bundles: ['@deepseek-ai/dsh-base', '@deepseek-ai/dsh-headless'] },
  sdk: { bundles: ['@deepseek-ai/dsh-base', '@deepseek-ai/dsh-sdk-app'] },
  'sdk-minimal': { bundles: ['@deepseek-ai/dsh-sdk-minimal'] },
}
```
(condensed; each entry spans three lines in source)

Layer order onto an empty entry list:

1. Each bundle's `cordis.patch.yml` (from `dsh.bundle` in `package.json`)
2. Profile `cordis.patch.yml`
3. Home-level patch
4. `--patch` overlays

`dsh-base` is the product spine: model adapters, tools, persistence, sandbox,
approvals, settings, credentials, telemetry. `dsh-web-app` adds the browser
app; `dsh-headless` adds a one-shot runner and disables HMR; `dsh-acp-app` and
`dsh-sdk-app` front the same spine. `sdk-minimal` is the only profile that does
not stack on `dsh-base`.

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
| `core/scope` | Per-agent scoped-registration primitive | library, no key |
| `llm/llm` | Stream vocabulary + adapter seam | `ctx.llm` |

`dsh-agent` exposes `AgentFactory` so ACP/UI call `ctx.agents.create()` without
importing the loop package (`packages/core/agent/src/index.ts:171`,
`setFactory` at `:358`). Only
one factory may register. The loop is structurally replaceable; the **durable
event vocabulary is not**.

The default driver is still small by this survey's standards: `agent.ts` is
**705 lines** (was 496 at the previous validation). That is the dividend of
pushing everything else into plugins.

---

## The Agent Loop: Turn and Step

A **step** is one model request plus the tools it calls. A **turn** is zero or
more steps. The turn opens before the first input is claimed and closes once
nothing is owed.

Inbox (same names Prime uses, cleaner implementation):

```163:173:deepseek-harness/packages/core/agent-loop/src/agent.ts
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
`SessionEventMap` member.

The previous dispatch-time runtime check (`agent-loop/src/invariant.ts`, a
`JSON.stringify` compare on `llm/stream`) was **deleted** with all runtime
invariant plugins in `f028f25667`. The property is now enforced by
construction: the loop builds each request from `session.deriveMessages()` at
the step boundary, deep-freezes it, and brands it with `markAgentLoopRequest`:

```688:700:deepseek-harness/packages/core/agent-loop/src/agent.ts
    const boundaryMessages = session.deriveMessages()
    // ... deepFreeze each message ...
    const request = markAgentLoopRequest(Object.freeze({
      ...header.config,
      messages: boundaryMessages,
      toolHistory: session.toolHistory(),
      ...
```

`llm/stream` listeners receive the frozen request and must read, never
rewrite, it (`packages/llm/llm/src/index.ts:62–74`). Hand-built calls carry no
brand. The equality check survives only as a test
(`packages/core/agent-loop/tests/request-reconstruction.spec.ts`). Ghost
context is now a failed test or a mutation throw, not a failed runtime
assertion.

`deriveMessages()` projects history from the **ordered surface** (compaction
uses `replace` to shadow old nodes without deleting durable bytes)
(`packages/core/session/src/index.ts:984`). Fork, resume, transcripts,
telemetry, and UI all derive from this stream.

`SESSION_FORMAT_VERSION = 4`: a single monotonic integer, bumped only when a
discriminator cannot prevent unsafe interpretation; readers translate older
headers through an adjacent migration chain (`packages/core/session/src/types.ts:70–89`).
SQLite backends keep a separate `STORAGE_SQLITE_SCHEMA_VERSION = 1`
(`packages/storage/storage-sqlite/src/schema.ts:20`). Event vocabulary growth
uses `ignorable: true` on unknown types rather than bumping the format for
every new event.

Opaque ids are branded (`SessionId`, `CallId`) via `dsh-brand`.

---

## Capability Seams

A **seam** is three roles, always:

1. **Service Definition** — interface + events on `ctx.<name>`
2. **Service Provider** — backend
3. **Consumer** — usually a model-facing tool

One role is not a seam. The payoff: pointing `fs` + `subprocess` + `shell` at
the same sandbox backend moves Bash, PTY, and LSP together. Local Landlock is the same-world argv-wrapper; the E2B remote world that
illustrated the other case was retired in `c49db8bc8c`. Subagent providers range from in-process child to ACP delegation
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
| Linux | bwrap, then Landlock (`landlock-run` in `@deepseek-ai/node-addon-system`, `native/system/`, ~540 ln C total, self-restrict-then-exec) |
| macOS | Seatbelt |
| Windows | ACL restricted-token runner (`packages/sandbox/sandbox-windows-acl`) |

Policy plugins interpret `read-only` / `workspace-write` / `danger-full-access`.
Escalation goes through approval markers. There is no remote execution world
in tree since E2B was retired (`c49db8bc8c`).

This is the strongest local sandbox story in the survey after Codex's
execution policy — and it is behind an interface, which Codex's is not in the
same "swap the world" sense.

---

## ACP, JSON-RPC, Web, Headless

- **ACP** (`packages/acp/acp`, own `acp` profile) — automation-only Agent
  Client Protocol over stdio JSON-RPC. Exposes only the standard ACP v1
  surface; permission prompts go to the client as `session/request_permission`
  with one-shot allow/reject choices (`packages/acp/acp/README.md`,
  `511181684c`). Creates agents via `ctx.agents`.
- **JSON-RPC SDK** — streams `session.event` notifications (full log
  envelopes). Python SDK launches bundled `dsh-jsonrpc-agent`.
- **Headless profile** — one-shot task, no HTTP, no browser.
- **SDK profiles** — `sdk` (on `dsh-base`) and `sdk-minimal` (standalone).
- **Web profile** — host transport + browser Cordis tree (large
  `cordis.patch.yml`).

Same `dsh-base` spine. Frontends are bundles, not forks of the loop. That
matches swe-term's Frontend interface more closely than Prime's TUI-shaped
interactive-mode.ts.

---

## Self-Modification

Not a separate `self-modification/` package in this checkout. The surface is
now split:

- `dsh-tool-cordis` — read-only: `cordis_inspect_list`, `cordis_inspect_query`.
- `dsh-cordis-host-runner` — "immutable package definitions, one active run per
  Plugin, human-approved Client activation" (`packages/extensions/cordis-host-runner/src/index.ts:1–5`);
  `cordis_define` validation lives there (`:172–175`).

Both are mounted only by `packages/bundle/web-app/cordis.patch.yml`, not by
`dsh-base`, and `pnpm run demo:cordis` no longer exists. It is still dynamic
JS in-process with Cordis fiber lifecycle; the human activation gate narrows,
but does not remove, the risk.

---

## The Complexity Machine

| Mechanism | Scale |
|-----------|-------|
| `package.json` manifests under `packages/` | **340** (226 at `c15f2f4d2`) |
| Package groups | 52 (49) |
| `experimental/` packages | 44 |
| Approx TS/TSX lines (`packages/` + `vendor/` + `native/` + `apps/`, tracked) | ~1.10M (~529k) |
| Generated catalogs | tools, config, persistence, Cordis API, module graph, scoped events, … |
| Coverage gate | **per-file 100%** on `packages/*/*/src` |
| Docs | word budgets, bilingual pairing, `doc-sync`, export JSDoc verification |
| Invariants | per-package `./invariant` companions **removed** (`f028f25667`) |
| Upgrade guides | per-release migration notes (`docs/upgrade-guide/`) |
| Vendored Cordis | pinned SHAs + logged local modifications |

The pre-release stance has shifted: `docs/session-format-status.md` states that
an alpha/beta/RC publication "establishes released Session-format
obligations", and each release now ships upgrade guides. dsh is paying the
migration cost the previous version of this document predicted.

---

## Key Design Patterns

1. **Log is the model history.** Derive, don't shadow.
2. **Reconstruction by construction.** Build the request from the log, freeze
   and brand it; prove equality in tests (the runtime check was removed).
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
sandbox) fit a small Go core. The ideas that change *org process* (340
packages, 100% coverage, Typert, bilingual doc gates) do not.

If swe-term takes one sentence from dsh, it should be: **the bytes that hit
the provider are a pure function of the session log.** dsh itself now
guarantees that by construction and tests rather than a dispatch-time
assertion; in Go the dispatch-time check is cheap enough to keep.

---

## Summary Statistics

| Metric | Value (local 0.2.1-alpha.2) |
|--------|-----------------------------|
| Package manifests under `packages/` | 340 |
| Default loop (`agent.ts`) | **705** lines |
| Session store (`session/src/index.ts`) | 1,452 lines |
| Loop invariant | removed (`f028f25667`); test-only |
| Session format version | 4 (released; migration chain) |
| SQLite schema version | 1 |
| Profiles | `acp`, `web`, `headless`, `sdk`, `sdk-minimal` |
| Sandbox default | fail-closed local chain; no remote world |
| ACP | standard v1 automation server |
| Status | developer preview, breaking OK |
