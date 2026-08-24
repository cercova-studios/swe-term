# Prime Agent Deep Dive — Architecture & Feature Analysis

> Analysis of [`PrimeIntellect-ai/prime-agent`](https://github.com/PrimeIntellect-ai/prime-agent)
> — Prime Intellect's **self-improving RLM agent**. Runtime: Node.js. Language: TypeScript
> (host) + Python (IPython kernel). Distribution: versioned installer + npm workspaces.
> License: MIT.
>
> **Validated against a local source checkout** at `prime-agent/` version **0.7.2**
> (`package.json`), HEAD `2c34b82` (2026-08-16). Line counts below are exact from that
> checkout. The product is a hard fork of [`pi-mono`](https://github.com/badlogic/pi-mono)
> republished as `@earendil-works/pi-*`, then grown into a daemon-backed, IPython-first
> control plane.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Lineage vs pi-mono](#lineage-vs-pi-mono)
3. [Package Layout](#package-layout)
4. [The RLM Bet: One LLM Tool](#the-rlm-bet-one-llm-tool)
5. [Host-Request Bridge](#host-request-bridge)
6. [Subagents: `rlm()` Admission, Not Awaits](#subagents-rlm-admission-not-awaits)
7. [Continual Harness and `/refine`](#continual-harness-and-refine)
8. [Daemon: Client / Supervisor / Worker](#daemon-client--supervisor--worker)
9. [Persistence](#persistence)
10. [Long-Running Work](#long-running-work)
11. [Skills](#skills)
12. [Security and Trust](#security-and-trust)
13. [Key Design Patterns](#key-design-patterns)
14. [Why It Matters for swe-term](#why-it-matters-for-swe-term)
15. [Summary Statistics](#summary-statistics)

---

## System Overview

Prime Agent is a coding/research agent built for **work that outlives a chat window**.
The README names two abstractions:

- **Recursive Language Model (RLM)** — context is a Python variable space; tools and
  child agents are function calls inside a persistent IPython REPL.
- **Continual Harness** — supplemental prompts, memories, skill descriptions, and
  subagent specs live as durable, reviewable state that `/refine` can update without
  rewriting the immutable base system prompt.

The architectural split is sharper than the marketing:

| Layer | Owns |
|-------|------|
| **Client** (TUI / print / JSON / RPC) | Rendering, keyboard, local UI prefs. Does **not** own execution. |
| **Daemon supervisor** | Discovery, routing, attachments, worker health, cross-agent messages. |
| **Session worker** | One root `AgentSessionRuntime`, scheduler, kernels, RLM descendants. |
| **IPython kernel** | Model-facing control environment. Typed host requests bounce authority back to TypeScript. |
| **Storage** | Session JSONL trees, harness JSON, spawn ledger, kernel dill snapshots. |

Workers and kernels are separate processes for **lifecycle isolation**, not security
sandboxes. They run with the user's OS permissions (`README.md` warning block).

The execution path is the same whether the prompt comes from a human, a heartbeat, a
cron job, a goal continuation, autonomous mode, or another agent.

---

## Lineage vs pi-mono

Prime Agent **is** pi-mono's layering with a philosophical pivot.

Preserved:

- Package direction: `pi-ai` → `pi-agent-core` → `pi-coding-agent` + `pi-tui`
- `Agent` + `runAgentLoop()` in `packages/agent`
- JSONL session trees with `id`/`parentId`
- jiti-loaded TypeScript extensions (`packages/coding-agent/src/core/extensions/loader.ts`)
- Interactive / print / RPC modes

Changed:

- npm scope `@mariozechner/*` → `@earendil-works/pi-*` (loader still aliases the old
  names at `loader.ts:62–72`)
- Product config: `piConfig.name = "prime-agent"`, `configDir = ".prime/agent"`
- **Default LLM-visible tool set collapsed to `ipython` only**
- Daemon-backed resident workers; TUI attaches over a local JSONL unix-socket protocol
- Continual harness, RLM spawn, agent-to-agent messaging, goals, autonomous mode

The loop itself is still pi's. Turn execution calls `this.agent.prompt(...)` on the
pi `Agent` (`packages/coding-agent/src/core/agent-session.ts`). Prime's originality is
**around** the loop: what the model is allowed to call, where execution lives, and
what state survives a disconnect.

---

## Package Layout

Four TypeScript workspaces plus a Python runtime:

| Path | npm name | Role |
|------|----------|------|
| `packages/ai` | `@earendil-works/pi-ai` | Provider abstraction, generated model catalog |
| `packages/agent` | `@earendil-works/pi-agent-core` | Domain-agnostic agent loop (~986 ln `agent-loop.ts`) |
| `packages/coding-agent` | `@earendil-works/pi-coding-agent` | Product: session, RLM, daemon, TUI glue (~119k ln src) |
| `packages/tui` | `@earendil-works/pi-tui` | Differential terminal renderer |
| `prime-agent-runtime/` | Python wheel | Kernel shim: `rlm`, `host_request`, harness CRUD |

Build order matches pi: tui → ai → agent → coding-agent (`package.json` `build` script).

**Prime-original directories** (not in vanilla pi):

- `packages/coding-agent/src/core/kernel/` — ZMQ IPython manager
- `packages/coding-agent/src/core/refinement/` — continual harness
- `packages/coding-agent/src/modes/daemon/` — supervisor, worker, catalog, protocol
- `packages/coding-agent/src/core/rlm-runtime.ts`, `rlm-ledger.ts`
- `prime-agent-runtime/`

---

## The RLM Bet: One LLM Tool

This is the load-bearing design choice. The built-in tool registry is:

```46:47:prime-agent/packages/coding-agent/src/core/tools/index.ts
export type ToolName = "ipython";
export const allToolNames: Set<ToolName> = new Set(["ipython"]);
```

Bash, edit, read, and write still exist as **Python skills** invoked *inside* the
kernel, and as SDK exports for extensions. They are not native multi-tool LLM calls
in the default product. Goals force-activate ipython
(`agent-session.ts:2037–2052`).

The thesis: instead of teaching the model 20 tool schemas, give it a persistent
REPL. File ops, shell, skills, subagents, and context management are code. Python
state (imports, parsed results, child handles) survives tool calls **and**
compaction. Kernel namespace is dill-snapshotted (`kernel/state-snapshot.ts`).

Tradeoff, stated plainly: the model must be good at writing Python, the host must
run a Jupyter kernel (ZMQ, `ipykernel`, uv bootstrap), and every capability that
needs authoritative host state must round-trip through `host_request`. That is a
large operational surface for a "coding agent."

---

## Host-Request Bridge

Python is the programming surface; TypeScript remains the authority.

- Kernel comm target: `host.request` (`packages/coding-agent/src/core/kernel/index.ts`)
- Python: `rlm.host_request(type, **payload)` in `prime-agent-runtime/src/rlm/__init__.py`
- TypeScript registers branded handlers per request type (`rlm.run`, `goal.*`,
  `refine.*`, `agent_message.*`, `compact`, …)

Credentials, provider calls, transcript writes, worker routing, and scheduling stay
out of Python. The kernel can *request* a child spawn; the host *creates* the child
`AgentSession`. Forged handlers are rejected via branded capability wrappers
(`kernel/index.ts:114–137`).

This is the cleanest Prime idea: **a typed RPC from an execution environment into
the harness**, rather than stuffing every privileged operation into a model-visible
tool schema.

---

## Subagents: `rlm()` Admission, Not Awaits

```python
handle = await rlm("Review the authentication flow", name="auth-reviewer")
```

The call returns immediately after **task admission** with
`{rlm_child_id, name, session_dir, model}`. It never waits for the child's answer
(`prime-agent-runtime/src/rlm/__init__.py`, `packages/coding-agent/src/core/prompts/rlm.ts`).
Results arrive through `agent_message` or files.

Host path (`agent-session.ts` `_handleRlmRun`):

1. Validate kwargs (`name`, `model` only).
2. Enforce depth: `if (this._rlmDepth >= this._rlmMaxDepth)`.
3. Create `sub-<8hex>` under the parent RLM session dir.
4. Spawn via `AgentSessionRuntime.createRlmSubagentRuntime`.

Default max depth is **1** (`_resolveRlmMaxDepth()`, `agent-session.ts:1577+`):
persisted chat state → inherited → global settings → env `RLM_MAX_DEPTH` → 1.
Even Prime treats deep recursion as dangerous.

Spawn topology is recorded in an append-only ledger
(`packages/coding-agent/src/modes/daemon/rlm-ledger.ts`) **separate** from session
headers — the parent can recover the child roster after compaction, kernel restart,
or parent restore.

---

## Continual Harness and `/refine`

Editable supplemental state, explicitly **not** the base system prompt
(`packages/coding-agent/src/core/refinement/refinement.ts:123–138`).

Kinds: `prompt | memory | skill | subagent` (`refinement.ts:30–31`).

Storage:

| Scope | Path |
|-------|------|
| Global | `~/.prime/agent/harness/harness_state.json` + `refinements.jsonl` |
| Local | `<session-artifacts>/<id>/harness/harness_state.json` |

`/refine` reviews the current trajectory, proposes structured create/update/delete
edits, applies them atomically (temp file + rename, `refinement.ts:350–358`), and
records a session custom entry (`REFINEMENT_CUSTOM_TYPE`). `/refine --rollback <id>`
builds inverse edits from stored `before` snapshots (`refinement.ts:804–835`).

Auto-refine can fire on a turn interval or post-compaction
(`agent-session.ts:195–196`). The reviewer still decides whether to apply.

Python `harness.py` mirrors the same state model into the kernel. That dual-writer
is a consistency tax (see the critique).

---

## Daemon: Client / Supervisor / Worker

Transport: local JSONL over a unix domain socket
(`packages/coding-agent/src/modes/daemon/daemon-protocol.ts:43–50`).

Versioning is unusually adult for a CLI:

```52:64:prime-agent/packages/coding-agent/src/modes/daemon/daemon-protocol.ts
export const DAEMON_PROTOCOL_NAME = "prime-agent.daemon";
export const DAEMON_PROTOCOL_VERSION = 7;
export const DAEMON_COMMAND_ENVELOPE_MIN_PROTOCOL_VERSION = 7;
// ...
export const DAEMON_SCHEMA_REVISION = 16;
export const DAEMON_SCHEMA_ID = "protocol-7-schema-16-1bcb9e7f1a49";
```

Protocol version vs schema revision vs capability sets
(`DaemonClientCapability` / `DaemonServerCapability`) lets optional features
degrade instead of hard-breaking attach. `AGENTS.md` requires classifying every
wire change as compatible, capability-gated, or incompatible.

Roles:

| Process | Job |
|---------|-----|
| Client | `attach` / `reattach` / `prompt` / `detach` |
| Supervisor (`daemon-supervisor.ts`, **5,236 ln**) | Socket server, worker fleet, prompt admission, idle eviction, recovery |
| Worker (`daemon-mode.ts`, **7,064 ln**) | Owns `AgentSessionRuntime`, executes daemon commands |
| Catalog (`daemon-catalog-process.ts`) | List/resolve/delete sessions **without loading agents** |

Crash recovery uses JSONL journals (`WorkerRecoveryJournal`,
`CommandRecoveryJournal`) plus worker lifecycle states
`starting | ready | recovering | stopping | failed`.

This is distributed-systems engineering inside a local agent CLI. It is how
"the terminal disconnected but the agent kept working" is actually implemented.

---

## Persistence

No SQLite in the core product.

| Artifact | Format |
|----------|--------|
| Session transcript tree | Append-only JSONL, `CURRENT_SESSION_VERSION = 3` |
| Harness state | JSON (global + per-session) |
| Refinement history | JSONL |
| RLM spawn ledger | JSONL |
| Cron / heartbeats | JSON via `getCronJobsPath()` |
| Kernel namespace | dill + manifest |
| Agent logs | `~/.prime/agent/logs/agent.jsonl` |

Session files live under `~/.prime/agent/sessions/<uuid>.jsonl`. The tree model
(`id`/`parentId`) is pi's. Prime adds artifact dirs for harness, RLM child
metadata, and scheduled jobs.

---

## Long-Running Work

| Feature | Mechanism |
|---------|-----------|
| Compaction | Branch summaries + token threshold; **kernel state survives** |
| Goals | IPython `goal` skill + `goal.*` host handlers; `/goal` slash |
| Heartbeats | Cron jobs with `source: "heartbeat" \| "rlm_heartbeat"` |
| Schedules | `AgentCronScheduler` — once / cron / interval |
| Autonomous | Host injects continuations; optional **quality gates** as shell commands (`autonomous.ts`, default `maxTurns: 12`) |
| Agent messaging | Discover, send, observe; family-reach rules + rate limits |

Busy-session delivery: `steer` (interrupt into the current step) vs `follow_up`
(queue for the next turn) — the same vocabulary DeepSeek Harness uses, reached
independently.

---

## Skills

Discovery scans skill dirs for `SKILL.md` plus optional `pyproject.toml`
(`packages/coding-agent/src/core/skills.ts`).

| Kind | Detection | Invocation |
|------|-----------|------------|
| Markdown | No `pyproject.toml` | Prompt / context only |
| Python | `pyproject.toml` + `src/<name>/__init__.py` | Pre-imported into the kernel; `await <import>(...)` |

Only skill **metadata** is in the startup prompt. Full `SKILL.md` loads on match.
Built-in skills include `edit`, `goal`, `refine`, `agent-message`, `agent-observe`,
`compact`, `rlm-heartbeat`, `skill-creator`.

Python-backed skills are a superset of instruction-only skills and may themselves
call `rlm(...)`.

---

## Security and Trust

The product is honest:

> Worker and kernel processes improve lifecycle isolation and recovery; they are
> **not** a security sandbox.

`@anthropic-ai/sandbox-runtime` is a **devDependency** used only in
`packages/coding-agent/examples/extensions/sandbox/` — opt-in OS-level bash
sandboxing, not the default path. The default RLM runtime executes
model-generated Python with the worker's user permissions.

The host-request branded-handler check is a real control against a confused
kernel, not a sandbox.

---

## Key Design Patterns

1. **UI does not own execution.** Attach is a client of a protocol.
2. **One model-facing tool, many host operations.** Privileged work is typed RPC.
3. **Admission ≠ completion.** `rlm()` returns a handle; answers are messages.
4. **Immutable base prompt, mutable harness.** Refinement is evidence-backed and
   rollbackable.
5. **Capability-gated wire protocol.** Optional features degrade; schema revision
   is not a silent break.
6. **Catalog process.** Listing sessions must not mean loading them.
7. **Spawn ledger vs session log.** Topology is a first-class artifact.

---

## Why It Matters for swe-term

Prime Agent is the strongest existence proof that **pi's loop can grow a
long-running, detachable, multi-agent product** without replacing the loop. That
is encouraging for swe-term's port.

What it does *not* prove is that IPython-as-the-only-tool is the right programming
model for a Go single-binary harness. The daemon protocol, harness snapshots,
host-request idea, and catalog split are the portable pieces. The kernel control
plane is the tax.

---

## Summary Statistics

| Metric | Value (local 0.7.2) |
|--------|---------------------|
| TS packages | 4 (`ai`, `agent`, `coding-agent`, `tui`) |
| `agent-session.ts` | **11,288** lines |
| `interactive-mode.ts` | **10,024** lines |
| `daemon-mode.ts` | **7,064** lines |
| `daemon-supervisor.ts` | **5,236** lines |
| `agent-loop.ts` (pi-core) | 986 lines |
| Default LLM tools | 1 (`ipython`) |
| Daemon protocol | v7 / schema revision 16 |
| Default RLM max depth | 1 |
| Session store | JSONL tree, no SQLite |
| Sandbox default | None (example extension only) |
