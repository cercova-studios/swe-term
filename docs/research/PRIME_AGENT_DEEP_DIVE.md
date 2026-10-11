# Prime Agent Deep Dive — Architecture & Feature Analysis

> Analysis of [`PrimeIntellect-ai/prime-agent`](https://github.com/PrimeIntellect-ai/prime-agent)
> — Prime Intellect's **self-improving RLM agent**. Runtime: native Rust binaries
> (`pa-cli`, `pa-daemon`). Language: Rust (host) + Python (CPython REPL runtime).
> Distribution: versioned installer (`install-rust.sh`, `install.ps1`) with an npm
> launcher kept for upgrades from the TypeScript era. License: MIT.
>
> **Validated against a local source checkout** at `prime-agent/` version **0.10.0**
> (`Cargo.toml` `[workspace.package]`), HEAD `2ee7e62` (2026-10-10). Line counts below
> are exact from that checkout and exclude `*test*` files and `tests/` dirs unless
> stated. Previous validation: 0.7.2, `2c34b82` (2026-08-16), when the product was a
> TypeScript hard fork of [`pi-mono`](https://github.com/badlogic/pi-mono). On
> 2026-09-29 the whole product was ported to Rust (`39bc99a`, "Port Prime Agent to
> Rust (#2524)", 2,797 files, +668k/−416k lines). No product `.ts` source remains; the only `.ts` file is the differential-test driver `crates/pa-ai/tests/differential/ts_driver.ts`.

---

## Changes since 0.7.2 (2026-08-16)

548 commits. The most significant:

1. **TypeScript → Rust port.** `packages/{ai,agent,coding-agent,tui}` are gone,
   replaced by a 9-crate Cargo workspace (`Cargo.toml` `members`). Commit `39bc99a`
   (2026-09-29). Byte-for-byte TS wire/disk parity was a port goal (`Cargo.toml`
   `serde_json` `preserve_order` comment; `crates/pa-daemon/src/ts_era.rs`).
2. **IPython/ZMQ kernel replaced by a minimal CPython REPL runtime.** `python -m
   rlm.repl`, NDJSON over stdin / a private dup of fd 1, protocol version 3
   (`prime-agent-runtime/src/rlm/repl.md`, `crates/pa-core/src/kernel/protocol.rs`
   `REPL_PROTOCOL_VERSION = 3`). Added in `61eb647` (2026-08-27). The runtime wheel
   now depends only on `mcp` and `tyro`; `dill` is a dev dependency, imported lazily
   for snapshots (`prime-agent-runtime/pyproject.toml`, `repl.md` §Snapshot).
3. **Default RLM max depth raised from 1 to 2.** `9e49b73` (2026-08-24); now
   `DEFAULT_RLM_MAX_DEPTH: u32 = 2` (`crates/pa-daemon/src/rlm_children.rs:34`).
4. **Subagent API grew from one call to a family.** `rlm.spawn`, `collect` (typed
   results, optional bounded wait), `list_subagents`, `delete_subagent`, `rename`,
   `progress_note`, `create_session`, `find_models`; spawn `kind="decision"` for a
   Decision API child (`prime-agent-runtime/src/rlm/__init__.py:179–507`).
5. **Factory: declarative subagent state machines.** New harness kind `factory`
   (`crates/pa-core/src/refinement/mod.rs:10`), validated/compiled/executed in the
   kernel (`prime-agent-runtime/src/rlm/factory.py`), host bridge in
   `crates/pa-core/src/session_engine/factory_host.rs`. Rust port `e387021`
   (2026-10-02).
6. **User extensions removed.** jiti-loaded TS extensions deleted in `f54551d`
   (2026-09-30, −6,778 lines). A package manager for `npm:`/git/local *resource*
   sources remains, explicitly not executing session code
   (`crates/pa-core/src/packages/mod.rs:1–4`).
7. **Catalog subprocess and command recovery journal removed.** `pa-daemon` is now
   `supervisor | worker` only (`crates/pa-daemon/src/main.rs:15–43`); catalog resolve
   lives in-supervisor (`crates/pa-daemon/src/session_catalog.rs`).
   `CommandRecoveryJournal` deleted in `121e59d`; only `WorkerRecoveryJournal`
   remains (`crates/pa-daemon/src/journal.rs:174`).
8. **ACP stdio mode on the daemon path.** `crates/pa-daemon/src/acp/` (5,541 ln);
   single daemon path with the in-process transport deleted in `15929ad`
   (2026-10-03).
9. **Daemon schema revision 16 → 30**, protocol version still 7
   (`crates/pa-types/src/daemon/mod.rs:12–17`).
10. **`AgentSession` god object dissolved into modules.** The TS class had reached
    15,101 lines just before the port (`git show 39bc99a^:packages/coding-agent/src/core/agent-session.ts`).
    Its Rust successor is `SessionEngine` (`crates/pa-core/src/session_engine/engine.rs`,
    971 ln) inside a 61-file, 28,338-line module; no non-generated Rust file exceeds
    2,235 lines.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Lineage vs pi-mono](#lineage-vs-pi-mono)
3. [Crate Layout](#crate-layout)
4. [The RLM Bet: One LLM Tool](#the-rlm-bet-one-llm-tool)
5. [Host-Request Bridge](#host-request-bridge)
6. [Subagents: Admission, Collect, Factory](#subagents-admission-collect-factory)
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
  child agents are function calls inside a persistent Python REPL.
- **Continual Harness** — supplemental prompts, memories, skill descriptions,
  subagent specs, and factory machines live as durable, reviewable state that
  `/refine` can update without rewriting the immutable base system prompt.

The architectural split:

| Layer | Owns |
|-------|------|
| **Client** (`pa-cli`: TUI / print / JSON / RPC / ACP) | Rendering, keyboard, local UI prefs. Does **not** own execution. |
| **Daemon supervisor** (`pa-daemon supervisor`) | Socket, worker fleet, routing, session catalog, agent roster, cross-agent messages. |
| **Session worker** (`pa-daemon worker`) | One root session engine, scheduler, kernels, RLM descendants. |
| **Python REPL runtime** (`rlm.repl`) | Model-facing control environment. Typed host requests bounce authority back to Rust. |
| **Storage** | Session JSONL trees, harness JSON, spawn ledger, kernel dill snapshots. |

Workers and kernels are separate processes for **lifecycle isolation**, not security
sandboxes. They run with the user's OS permissions (`README.md:87`).

The execution path is the same whether the prompt comes from a human, a heartbeat, a
cron job, a goal continuation, autonomous mode, a factory state, or another agent.

---

## Lineage vs pi-mono

Prime Agent **was** pi-mono's layering with a philosophical pivot. Since `39bc99a`
it is a Rust reimplementation of that layering; the pi lineage survives in crate
boundaries and on-disk formats, not in shared code.

Preserved:

- Layer direction: `pa-ai` → `pa-agent` → `pa-core` → `pa-daemon` / `pa-tui` / `pa-cli`
  (mirrors pi's `ai` → `agent-core` → `coding-agent` + `tui`)
- An agent loop module (`crates/pa-agent/src/agent_loop/`, 1,498 ln)
- JSONL session trees with `id`/`parentId`, `CURRENT_SESSION_VERSION = 3`
  (`crates/pa-core/src/session/mod.rs:19`)
- Interactive / print / RPC modes

Changed since the fork:

- **Language and runtime:** Rust binaries instead of Node; TS-era compatibility shims
  only where on-disk/wire state must be read (`crates/pa-daemon/src/ts_era.rs`)
- **Extensions removed** (`f54551d`); pi's jiti extension loader no longer exists
- Default LLM-visible tool set is `ipython` only
- Daemon-backed resident workers; clients attach over a local JSONL unix-socket protocol
- Continual harness, RLM spawn/collect, factory, agent messaging, goals, autonomous mode

---

## Crate Layout

Nine Rust crates plus a Python runtime (`Cargo.toml` `members`):

| Path | Role | Rust ln (all, incl. tests) |
|------|------|------:|
| `crates/pa-types` | Wire/domain types, daemon protocol constants | 13,153 |
| `crates/pa-telemetry` | Usage telemetry | 4,941 |
| `crates/pa-ai` | Provider abstraction, generated model catalog | 38,027 |
| `crates/pa-models` | Model selection | 4,702 |
| `crates/pa-agent` | Domain-agnostic agent loop | 6,944 |
| `crates/pa-core` | Session engine, kernel manager, tools, refinement, cron, MCP | 135,007 |
| `crates/pa-daemon` | Supervisor, worker, ACP, catalog, ledgers | 157,867 |
| `crates/pa-tui` | Terminal UI | 137,088 |
| `crates/pa-cli` | Entry point, print/headless modes, login, update | 71,256 |
| `prime-agent-runtime/` | Python wheel: `rlm` REPL, host requests, harness, factory, MCP, bash | — |

Rust total: 568,985 lines; 378,314 excluding test files.

**Prime-original areas** (not in pi):

- `crates/pa-core/src/kernel/` — REPL manager, provisioner, snapshots (10,135 ln)
- `crates/pa-core/src/refinement/` — continual harness
- `crates/pa-core/src/session_engine/` — session, RLM host, goals, factory host
- `crates/pa-daemon/src/{supervisor,worker,acp,rlm_ledger}`
- `prime-agent-runtime/`

---

## The RLM Bet: One LLM Tool

This is still the load-bearing design choice. The session engine injects the
`ipython` tool when the configured tool list lacks it
(`crates/pa-core/src/session_engine/engine.rs:432–443`). `bash` and `edit` tool
definitions exist in `crates/pa-core/src/tools/{bash,edit}.rs` and are exported from
`pa-core` (`lib.rs:40–51`), but no daemon or CLI path registers them as LLM tools.
Bash and edit are used by the model as **Python skills inside the REPL**
(`prime-agent-runtime/src/rlm/bash.py`, `skills/edit`).

The thesis: instead of teaching the model many tool schemas, give it a persistent
REPL. File ops, shell, skills, subagents, MCP, and context management are code.
Python state survives tool calls **and** compaction. The namespace is dill-snapshotted
on request (`repl.md` §Snapshot / restore; `crates/pa-core/src/kernel/state_snapshot.rs`).

What changed is the kernel itself. The Jupyter stack (ipykernel + ZMQ) was replaced
by a ~one-module CPython REPL runtime (`prime-agent-runtime/src/rlm/repl.py`, `61eb647`)
speaking NDJSON on stdin and a private dup of fd 1. Requests: `execute`, `interrupt`,
`host_reply`, `snapshot`, `restore`, `list_names`, `mcp_status`, `bash_activity`,
`shutdown`. The handshake requires an exact protocol-version match
(`crates/pa-core/src/kernel/protocol.rs:12–14`).

Tradeoff, restated: the model must still be good at Python and the host must still
provision a Python environment (`crates/pa-core/src/kernel/provisioner.rs`, 1,784 ln),
but the ZMQ/Jupyter operational surface is gone.

---

## Host-Request Bridge

Python is the programming surface; Rust remains the authority.

- Python: `rlm.host_request(type, payload)` (`prime-agent-runtime/src/rlm/__init__.py:153`)
- Frames: `host_request` events out, `host_reply` requests back (`repl.md`)
- Rust routes by exact type string through a per-session handler map; an
  unregistered type fails with `host request type "…" is not available in this
  session` (`crates/pa-core/src/kernel/manager/host_requests.rs:70–85`)
- Handlers implement `HostRequestHandler` (`crates/pa-core/src/kernel/shared.rs:246`).
  The TS branded-wrapper check is gone; closure over the map is the control.
- Registered families include `rlm.*` (`session_engine/rlm_host.rs`), `goal.*` and
  `rlm_heartbeat.*` (`session_engine/host_requests.rs:76–305`), plus factory, refine,
  and agent-message handlers.

Credentials, provider calls, transcript writes, worker routing, and scheduling stay
out of Python. The kernel *requests* a child; the host *creates* it. Each request is
tagged with the triggering cell's source (`cellSourceCode`, `host_requests.rs:87–98`).

This remains the cleanest Prime idea: **a closed, typed RPC from an execution
environment into the harness**.

---

## Subagents: Admission, Collect, Factory

```python
handle = await rlm.spawn("Review the authentication flow", name="auth-reviewer")
results = await rlm.collect(handle, timeout_ms=60_000)
```

`spawn` returns once the task is **admitted** (`__init__.py:179–210`). Kwargs are
`name` (required, unique among siblings), `model`, `thinking`, and `kind`
(`"decision"` only). The wire type stays `rlm.run` for cross-version compatibility
(`__init__.py:207`).

New since 0.7.2: `collect(targets, timeout_ms=0)` returns typed `RLMChildResult`s.
`0` is a non-blocking snapshot; a positive value blocks only that kernel call and
never steers the parent (`__init__.py:416–440`). Admission-not-await is preserved;
`collect` adds an explicit, bounded join. Also new: `list_subagents`,
`delete_subagent`, `rename`, `progress_note`, `create_session`, `find_models`.

Depth policy (`crates/pa-daemon/src/agent_engine/session_engine_impl.rs:840–861`):
persisted chat state → inherited → global settings → env `RLM_MAX_DEPTH` → default
**2** (`rlm_children.rs:34`, changed from 1 in `9e49b73`). `set_rlm_max_depth` is a
daemon command (`crates/pa-daemon/src/rlm_surface.rs:71`).

Spawn topology is still recorded in an append-only ledger
(`crates/pa-daemon/src/rlm_ledger/`) separate from session headers. The supervisor
now also serves an event-driven agent roster from that ledger (`8d5722e`, `1d2e91d`).

**Factory** (`prime-agent-runtime/src/rlm/factory.py`): a harness entry of kind
`factory` stores a declarative state machine of subagent states — entry states,
guarded transitions, bounded re-entry (`max_entries`). A DAG form compiles to the
machine form. The executor admits states via `rlm.spawn`, settles via `rlm.collect`,
cancels via `rlm.delete_subagent`. Run state lives in kernel memory and **does not
survive a kernel restart**; children are supervisor-owned and keep running
(`factory.py` module docstring).

---

## Continual Harness and `/refine`

Editable supplemental state, explicitly not the base system prompt.

Kinds: `prompt | memory | skill | subagent | factory`
(`crates/pa-core/src/refinement/mod.rs:10`).

Storage:

| Scope | Path |
|-------|------|
| Global | `~/.prime/agent/harness/harness_state.json` + `refinements.jsonl` |
| Local | `<session-artifacts>/<id>/harness/harness_state.json` |

`/refine` reviews the trajectory, proposes edits, and applies them
(`crates/pa-core/src/refinement/planner.rs`: "proposal parsing, edit validation,
application, and rollback"). Each applied outcome is persisted with a
`rollback_of` link (`refinement/mod.rs:325–336`). Auto-refine is triggered from
`session_engine/auto_refine_trigger.rs` and post-compaction
(`crates/pa-daemon/src/compact_autorefine.rs`).

Python `harness.py` still writes the same JSON file (`os.replace` at
`harness.py:648`; `create_memory`/`update_memory`/`delete_memory` at `:914–940`).
Since `97b4a03` (2026-10-07) both writers share a cross-process lock directory with
an owner file (`crates/pa-core/src/platform/lock_dir.rs:1–3`; `harness.py:471–485`).
Two writers remain, but they are now serialized.

---

## Daemon: Client / Supervisor / Worker

Transport: local JSONL over a unix domain socket (`crates/pa-daemon/src/socket.rs`,
`framing.rs`).

Versioning (`crates/pa-types/src/daemon/mod.rs:12–25`):

```rust
pub const DAEMON_PROTOCOL_NAME: &str = "prime-agent.daemon";
pub const DAEMON_PROTOCOL_VERSION: u64 = 7;
pub const DAEMON_SCHEMA_REVISION: u64 = 30;
pub const DAEMON_SCHEMA_ID: &str = "protocol-7-schema-30-8e4b17c2a9f5";
pub type DaemonClientCapability = String;
pub type DaemonServerCapability = String;
```

`DAEMON_COMMAND_ENVELOPE_MIN_PROTOCOL_VERSION` equals the protocol version
(`crates/pa-daemon/src/protocol.rs:19`). Capabilities are open strings on the wire
(closed unions in TS, comment at `mod.rs:23`). Schema revision moved 16 → 30 across
the port with no protocol bump: the capability-gating discipline held.

Roles:

| Process | Job |
|---------|-----|
| Client (`pa-cli`) | `attach` / `reattach` / `prompt` / `detach`; ACP stdio mode |
| Supervisor (`crates/pa-daemon/src/supervisor/`, 14 files, 8,725 ln) | Accept loop, worker lifecycle, routing, launch budget, catalog resolve, roster |
| Worker (`crates/pa-daemon/src/worker/`, 18 files, 9,973 ln) | Owns the session engine, executes daemon commands |

The separate **catalog process is gone**: `pa-daemon` accepts only `supervisor` or
`worker` (`main.rs:15–43`). Listing/resolving saved sessions is done in the
supervisor without loading runtimes (`session_catalog.rs`; metadata append without
full transcript parses, `c37f5eb`).

Crash recovery uses `WorkerRecoveryJournal` (`journal.rs:174`) plus worker lifecycle
`Starting | Ready | Recovering | Stopping | Failed`
(`crates/pa-types/src/daemon/worker.rs:12–18`). `CommandRecoveryJournal` was deleted
as unused (`121e59d`); the per-append session lease guard was dropped (`92dc7e5`).

**ACP** (`crates/pa-daemon/src/acp/`): JSON-RPC stdio mode served over a daemon
session — a second client protocol on the same supervisor/worker path (`15929ad`,
`da8a80a`).

---

## Persistence

No SQLite in the product (no `rusqlite`/`sqlite` in any crate `Cargo.toml`).

| Artifact | Format |
|----------|--------|
| Session transcript tree | Append-only JSONL, `CURRENT_SESSION_VERSION = 3` |
| Harness state | JSON (global + per-session), lock-dir guarded |
| Refinement history | JSONL |
| RLM spawn ledger | JSONL |
| Cron / heartbeats | JSON (`AgentCronJobStore`, `crates/pa-core/src/cron/store`) |
| Kernel namespace | dill + manifest |

On-disk formats were kept byte-compatible with the TS era so upgrades read old
sessions (`Cargo.toml` `preserve_order` comment; `c174739` "support TypeScript
updates to Rust").

---

## Long-Running Work

| Feature | Mechanism |
|---------|-----------|
| Compaction | Branch summaries + token threshold; kernel state survives (`session_engine/compaction*.rs`) |
| Goals | `goal` skill + `goal.*` host handlers (`session_engine/host_requests.rs:76–96`) |
| Heartbeats | `rlm_heartbeat.*` handlers over the cron store (`host_requests.rs:211–305`) |
| Schedules | `crates/pa-core/src/cron/`, `crates/pa-daemon/src/scheduled_jobs/` |
| Autonomous | Host-injected continuations, `DEFAULT_MAX_TURNS = 12` (`crates/pa-core/src/autonomous/mod.rs:23`) |
| Factory | Declarative subagent state machines (see Subagents) |
| Agent messaging | `crates/pa-daemon/src/agent_messaging*`, broadcast/ingest |

Busy-session delivery still distinguishes `steer` vs `follow_up`
(`crates/pa-types/src/daemon/plane.rs:26–27`).

---

## Skills

Bundled skills (`skills/`): `agent-message`, `agent-observe`, `attach-image`,
`compact`, `decision-api`, `edit`, `factory`, `goal`, `mcp`, `prime-intellect`,
`refine`, `rlm-heartbeat`, `skill-creator`, `websearch`.

Skills are Markdown (`SKILL.md`) or Python packages pre-imported into the REPL.
Python skills that fail to import are reported to the model on the next turn
(`session_engine/engine.rs:402–415`). Only metadata enters the startup prompt.

MCP servers run **inside the kernel** (`prime-agent-runtime/src/rlm/mcp.py`); the
host opens configured servers in the background after kernel prewarm
(`engine.rs:452–470`).

---

## Security and Trust

The README is still honest (`README.md:87`):

> Its worker and kernel processes improve lifecycle isolation and recovery; they are
> **not** a security sandbox.

The 0.7.2 opt-in sandbox example extension is gone with the extension system
(`f54551d`); `examples/` contains only `catch`. Model-generated Python runs with the
worker's user permissions.

Hardening that did land: the closed host-request map, removal of user-executed
extension code, log redaction of session ids (`c6926c5`), and a contributor trust
gate in CI (`069014c`).

---

## Key Design Patterns

1. **UI does not own execution.** TUI, print, RPC, and ACP are clients of one daemon path.
2. **One model-facing tool, many host operations.** Privileged work is typed RPC.
3. **Admission ≠ completion.** `spawn` returns a handle; `collect` is an explicit, bounded join.
4. **Immutable base prompt, mutable harness.** Refinement is rollbackable.
5. **Capability-gated wire protocol.** 14 schema revisions and a language rewrite without a protocol bump.
6. **Catalog without loading runtimes.** Now in-supervisor rather than a subprocess.
7. **Spawn ledger vs session log.** Topology is a first-class artifact, and the roster reads it.
8. **Rewrite behind a frozen wire/disk contract.** The Rust port kept byte parity with TS state.

---

## Why It Matters for swe-term

Prime Agent is now also the strongest existence proof that **a pi-shaped harness can
be rewritten into a native systems language behind a stable wire and disk
contract**. The daemon protocol, host-request bridge, harness rollback, and spawn
ledger all survived the rewrite unchanged in shape.

It still does not prove that a Python REPL as the only tool is the right programming
model for a Go single-binary harness. The kernel got cheaper (no Jupyter), not
optional.

---

## Summary Statistics

| Metric | 0.7.2 (`2c34b82`) | 0.10.0 (`2ee7e62`) |
|--------|------|------|
| Host language | TypeScript (4 npm workspaces) | Rust (9 crates) |
| Rust lines (all / non-test) | — | 568,985 / 378,314 |
| Session orchestrator | `agent-session.ts` 11,288 ln (15,101 just before port) | `SessionEngine` 971 ln; `session_engine/` 61 files, 28,338 ln |
| Largest non-generated source file | `agent-session.ts` 11,288 | `acp/daemon.rs` 2,235 |
| Agent loop | `agent-loop.ts` 986 ln | `pa-agent/src/agent_loop/` 1,498 ln |
| Default LLM tools | 1 (`ipython`) | 1 (`ipython`) |
| Kernel | ipykernel over ZMQ | CPython `rlm.repl`, NDJSON, protocol v3 |
| Daemon protocol | v7 / schema 16 | v7 / schema 30 |
| Daemon processes | client, supervisor, worker, catalog | client, supervisor, worker |
| Default RLM max depth | 1 | 2 |
| Harness kinds | 4 | 5 (+`factory`) |
| User extensions | jiti TS | Removed |
| Session store | JSONL tree, no SQLite | Same |
| Sandbox default | None (example extension only) | None |
