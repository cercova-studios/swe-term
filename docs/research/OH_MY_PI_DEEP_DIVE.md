# oh-my-pi Deep Dive — Architecture & Feature Analysis

> Analysis of [`can1357/oh-my-pi`](https://github.com/can1357/oh-my-pi) (`omp`) —
> a **pi-mono fork with the IDE wired in**. Runtime: **Bun**. Languages:
> TypeScript + **~80k LoC Rust** (N-API). Distribution: curl installer, Homebrew,
> npm/bun, Nix, compiled bun binary. License: see repo `LICENSE`.
>
> **Validated against a local source checkout** at `oh-my-pi/` version **17.3.5**,
> HEAD `37eee71978` (2026-08-16). Fork delta list:
> `docs/porting-from-pi-mono.md` (upstream marker `b21b42d`).

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Relationship to pi-mono](#relationship-to-pi-mono)
3. [Package and Crate Layout](#package-and-crate-layout)
4. [Rust Natives: In-Process, Not Sidecars](#rust-natives-in-process-not-sidecars)
5. [Tool Surface](#tool-surface)
6. [Hashline and Per-Model Edit Formats](#hashline-and-per-model-edit-formats)
7. [Eval Kernels and the Tool Bridge](#eval-kernels-and-the-tool-bridge)
8. [Agent Loop, Session, Compaction](#agent-loop-session-compaction)
9. [Catalog, Dialects, Prompts](#catalog-dialects-prompts)
10. [Observability and Distribution](#observability-and-distribution)
11. [Key Design Patterns](#key-design-patterns)
12. [Why It Matters for swe-term](#why-it-matters-for-swe-term)
13. [Summary Statistics](#summary-statistics)

---

## System Overview

oh-my-pi keeps pi-mono's *layer names* and replaces the *platform*.

The README's claim is product, not architecture: **60+ providers, 31 built-in
tools, 14 LSP ops, 28 DAP ops, ~80k lines of Rust core.** The architectural
claim underneath: a coding agent should expose **IDE-grade tools** (LSP rename
that hits barrel files, a real debugger, grep that does not shell out) and
**model-tuned contracts** (edit format selected per weights, not one
`str_replace` for everyone).

The blog post they point at — ["The Harness Problem"](https://blog.can.ac/2026/02/12/the-harness-problem/) —
is the thesis: harness quality (edit format, read summarization, search) moves
pass rates more than swapping the model. Measured lifts in the README (Grok
Code Fast 1 **6.7% → 68.3%** when the edit format stops eating the model) are
the existence proof.

omp is the **tool-quality** extremum in this survey. Prime is the
**long-running** extremum. dsh is the **harness-correctness** extremum.

---

## Relationship to pi-mono

Preserved:

- `pi-ai` → `pi-agent-core` → `pi-coding-agent` + `pi-tui`
- `createAgentSession()` → interactive / print / RPC
- JSONL session trees under a per-cwd directory
- Extension hooks (loaded with Bun `import()` instead of jiti)

Intentionally diverged (`docs/porting-from-pi-mono.md`):

| Area | pi-mono | omp |
|------|---------|-----|
| Runtime | Node | **Bun** (`bun:sqlite`, embeds, `$` shell) |
| Tool factories | `createTool(cwd)` | `BUILTIN_TOOLS[name](ToolSession)` |
| Auth | `auth.json` | **`agent.db`** SQLite, multi-credential round-robin |
| Natives | shell-out to rg/bash | **N-API Rust addon** |
| Scope | `@mariozechner/*` | `@oh-my-pi/*` |

Upstream merges are manual. Drift is structural. Treat omp as a **sibling** of
pi, not a drop-in upgrade.

---

## Package and Crate Layout

**TypeScript workspaces** (17 under `packages/`): `ai`, `agent`, `coding-agent`,
`tui`, plus omp-only `catalog`, `hashline`, `snapcompact`, `mnemopi`, `omptype`,
`wire`, `natives`, `stats`, `collab-web`, `browser-relay`, `metaharness`,
`typescript-edit-benchmark`, `utils`.

**Rust crates** (8 under `crates/`):

| Crate | Role (README claims) |
|-------|----------------------|
| `pi-shell` | Embedded bash (vendored `brush-core`), PTY, in-process coreutils (~38k) |
| `pi-natives` | N-API cdylib aggregating hot paths (~25k) |
| `pi-walker` | Ignore-aware parallel walker + scan cache |
| `pi-iso` | Worktree isolation (APFS/btrfs/zfs/overlay) |
| `pi-ast` | tree-sitter + ast-grep summaries |
| `pi-voice` | Audio / WebRTC |
| `pi-builtins` | In-process CLI utilities |

`crates/pi-natives/src/lib.rs` registers grep, glob, fd, workspace, ast,
snapcompact, shell, pty, desktop, highlight, diff, tokens, html/pdf, etc.

---

## Rust Natives: In-Process, Not Sidecars

Architecture (`docs/natives-architecture.md`):

```
JS (packages/natives) → N-API → Rust modules
```

- Platform-tagged `.node` selection (AVX2 vs baseline, optional npm leaf packages)
- Bazel via `scripts/bazel-natives.ts`; napi-rs generates `index.d.ts`
- Compiled bun binary: addon gzip-embedded, extracted to `~/.omp/natives/<version>/`
- Post-load: `__ompInstallTokioRuntime()` on the libuv pool

This is **not** swe-term's intended shape. swe-term wants heavy engines as
**sidecar binaries behind thin Go adapters**. omp wants grep/walk/PTY in the
same address space as the agent loop for latency.

Sidecars still exist for *other* jobs: Python/Ruby/Julia kernels, JS eval
worker, LSP/DAP children, browser/computer workers. The split is: **hot path =
N-API; long-lived protocol servers = child processes.**

---

## Tool Surface

Registry (`packages/coding-agent/src/tools/index.ts:416–446`):

`read`, `security_scan`, `bash`, `edit`, `ast_grep`, `ast_edit`, `ask`,
`debug`, `eval`, `github`, `glob`, `grep`, `lsp`, `inspect_image`, `browser`,
`computer`, `checkpoint`, `rewind`, `task`, `hub`, `todo`, `web_search`,
`write`, plus memory/skill tools (`memory_edit`, `retain`, `recall`, `reflect`,
`learn`, `manage_skill`). Hidden: `think`, `yield`, `goal`.

Factories take a **`ToolSession`**, not a cwd string. Activation/gating lives
on the session (`createIf` for optional tools). That is a cleaner tool
lifecycle than pi's `createTool(cwd)`.

**read:** files, dirs, archives, SQLite, URLs, internal schemes (`pr://`,
`agent://`, …); **structural summaries**; hashline `#TAG` snapshots; line
selectors (`docs/tools/read.md`).

**grep:** in-process `@oh-my-pi/pi-natives` grep (Rust regex → PCRE2 fallback →
literal recovery); hashline anchors on matches (`docs/tools/grep.md`).

**lsp:** 14 actions including `rename` / `rename_file` through
`workspace/willRenameFiles` so re-exports update (`docs/tools/lsp.md`).

**debug:** 28 DAP actions — launch, attach, breakpoints, stepping, memory
(`docs/tools/debug.md`). Most agents still print-debug. omp drives lldb/dlv/debugpy.

coding-agent `tools/` alone is on the order of **~42k lines**. That is product
mass, not loop mass.

---

## Hashline and Per-Model Edit Formats

Default edit mode is **`hashline`** (`packages/coding-agent/src/utils/edit-mode.ts:3–5`):
content-hash anchors instead of brittle line numbers. Stale anchors reject
instead of silently applying in the wrong place.

Modes: `hashline` | `apply_patch` | `patch` | `replace`.

Selection is **data-driven**:

1. `settings.getEditVariantForModel(activeModel)` (`edit.modelVariants`)
2. Hardcoded exclusions (Kimi, Mimo, DeepSeek V4 Flash, Step 3.7 Flash →
   `replace`) at `edit-mode.ts:16–21`
3. Default `hashline`

Each mode swaps grammar, prompt file, and schema
(`packages/coding-agent/src/edit/index.ts`). This is the harness-problem thesis
in code: **the tool contract is part of the model adapter**, not a universal
XML.

---

## Eval Kernels and the Tool Bridge

`eval` is a first-class tool (`py` / `js` / `rb` / `jl`), exclusive per
session. Persistent kernels. Either kernel can call back into the agent's
tools.

Python path (`packages/coding-agent/src/eval/py/tool-bridge.ts:1–8`):

> HTTP loopback bridge … POSTs to `/v1/tool` over a 127.0.0.1 loopback
> socket; the host resolves the request against the `ToolSession` … same
> `callSessionTool` implementation the JS bridge uses.

JS path: Bun worker re-enters `cli.ts` via `__omp_worker_js_eval`, then
`eval/js/tool-bridge.ts` calls `session.getToolByName()`.

Prelude helpers: `tool.read(...)`, `agent()`, `completion()`, `parallel()`.
Bridge calls inherit parent **approval** semantics. Python env is allowlisted
and API keys denylisted before spawn — mitigations, not a sandbox.

Trust boundary = parent session minus stripped secrets. Feature-by-design
(README §01), and a larger blast radius than Prime's closed `host_request`
set.

---

## Agent Loop, Session, Compaction

Still a god object:

```1:14:oh-my-pi/packages/coding-agent/src/session/agent-session.ts
 * AgentSession - Core abstraction for agent lifecycle and session management.
 * ...
 * - Agent state access
 * - Event subscription with automatic session persistence
 * - Model and thinking level management
 * - Compaction (manual and auto)
 * - Bash execution
 * - Session switching and branching
```

| File | Lines |
|------|------:|
| `agent-session.ts` | **9,428** |
| `agent-loop.ts` | 2,935 |
| `agent.ts` | 1,748 |

The loop is more extracted than Prime's (Prime's `agent-loop.ts` is still
~986 of *pi-core*; omp grew the core loop to 2.9k). Orchestration still
concentrates in `AgentSession`.

**Sessions:** append-only JSONL at
`~/.omp/agent/sessions/<encoded-cwd>/<timestamp>_<id>.jsonl`.

**SQLite (`bun:sqlite`):** `~/.omp/agent/agent.db` (settings, auth, usage);
`~/.omp/stats.db` (omp-stats). Transcripts stay JSONL. This is the Codex-like
split (JSONL + SQLite metadata) pi lacked.

**Compaction** (`docs/compaction.md`):

1. **context-full** — LLM summary (default)
2. **shake** — mechanical elision to `artifact://`
3. **snapcompact** — deterministic PNG frames via Rust, vision-model readback
4. **handoff** — provider-native streaming compaction

Plus cheap wins: useless-result elision, superseded-read pruning. snapcompact
is a research strategy (vision billing, provider-specific); shake + elision
are the portable bits.

---

## Catalog, Dialects, Prompts

**68 providers** in `packages/catalog/src/provider-models/descriptors.ts`
(README says 60+; code is ahead). Roles: `default`, `smol`, `slow`, `plan`,
`commit`, `vision`, `designer`, `task`, `advisor`, `tiny`. Custom models in
`~/.omp/agent/models.yml`.

**Inband tool dialects** (`packages/ai/src/dialect/factory.ts`): anthropic,
gemini, harmony, kimi, glm, qwen3, … Models without native tool APIs get
prompt-rendered tools via `renderInbandToolPrompt()`. Catalog lives in
`pi-catalog`, not in `pi-ai` barrels — a cleaner split than pi's generated
`models.generated.ts` megazord.

**Prompts are files.** AGENTS.md forbids inline prompt strings. Tools load
`prompts/tools/<name>.md` via `import … with { type: "text" }`. Handlebars for
dynamic bits. Date/cwd moved to a per-request reminder so the prefix stays
cache-stable (`docs/system-prompt-customization.md`).

---

## Observability and Distribution

`omp stats` serves a localhost dashboard from incremental JSONL →
`stats.db` (tokens/s, cache rate, errors, cost, TTFT). `scripts/session-stats/`
does corpus analytics (tool/edit aggregates, LLM-assisted waste audit).
Agent-core has OTel hooks.

**Worker-host re-entry:** `cli.ts` dispatches `__omp_worker_*` selectors
*before* the command registry, then `declareWorkerHostEntry()`. Compiled
binary, npm bundle, and source share **one** `Bun.main`. Workers spawn
`new Worker(workerHostEntry(), { argv: ["__omp_worker_…"] })`
(`packages/utils/src/worker-host.ts`). This is how a single compiled artifact
hosts stats sync, JS eval, and computer workers without extra entrypoints.

Install: `curl -fsSL https://omp.sh/install | sh`, brew, bun global, Nix flake
(`programs.omp` Home Manager module), Windows irm, mise.

---

## Key Design Patterns

1. **ToolSession, not cwd.** Tools close over session capability.
2. **Per-model tool contracts.** Edit mode is adapter data.
3. **Summarize reads, anchor grep.** Token economy + edit chain.
4. **Hot path in native code.** Grep/walk/PTY are not `child_process`.
5. **JSONL transcript + SQLite metadata.** Two stores, clear jobs.
6. **Prompts as artifacts.** No string-built system prompts in TS.
7. **One binary, argv-selected workers.** Distribution stays simple.
8. **Measure the harness.** Stats DB and edit-format benches are product
   features.

---

## Why It Matters for swe-term

oh-my-pi is the best argument that **swe-term's sidecar plan is directionally
right and N-API-as-platform is the wrong coupling**. Grep, walk, AST, and
tokenizers belong in Rust. They should not share a process with the approval
plane, and they should not be the plugin ABI.

It is also the best argument that **tool quality is the product**. Hashline,
summarized reads, and per-model edit variants will move eval scores more than
another provider wrapper. Those belong in swe-term as extensions with
regression benches, not as core-loop complexity.

---

## Summary Statistics

| Metric | Value (local 17.3.5) |
|--------|----------------------|
| TS packages | 17 |
| Rust crates | 8 |
| `agent-session.ts` | **9,428** lines |
| `agent-loop.ts` | 2,935 lines |
| Builtin tool factories | 29 + hidden 3 |
| LSP actions | 14 |
| DAP actions | 28 |
| Catalog providers | 68 |
| Default edit mode | hashline |
| Session transcript | JSONL |
| Auth/settings | `bun:sqlite` `agent.db` |
| Native coupling | in-process N-API |
| Runtime | Bun ≥ 1.3.14 |
