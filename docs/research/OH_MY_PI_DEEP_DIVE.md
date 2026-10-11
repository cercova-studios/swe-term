# oh-my-pi Deep Dive — Architecture & Feature Analysis

> Analysis of [`can1357/oh-my-pi`](https://github.com/can1357/oh-my-pi) (`omp`) —
> a **pi-mono fork with the IDE wired in**. Runtime: **Bun**. Languages:
> TypeScript + **~302k LoC Rust** excluding `crates/vendor/` (N-API; README
> still says ~80k). Distribution: curl installer, Homebrew,
> npm/bun, Nix, compiled bun binary. License: see repo `LICENSE`.
>
> **Validated against a local source checkout** at `oh-my-pi/` version **18.8.9**,
> HEAD `fd27203d27` (validated 2026-10-10). Previous validation: 17.3.5,
> upstream `316c11868` (2026-08-16); 10,850 commits between. Fork delta list:
> `docs/porting-from-pi-mono.md` (upstream marker `b21b42d`).

---

## Changes since 17.3.5 (2026-08-16)

1. **Edit engine moved to Rust.** `crates/pi-edit` (~30k ln) owns parsing,
   matching, streaming preview, and atomic apply for every edit mode; the
   `packages/hashline` TS package was deleted. Commit `c478e41a04`.
2. **New edit mode `sloppy`.** `EditMode` is now
   `replace | patch | hashline | apply_patch | sloppy`
   (`packages/tui/src/tools/edit.ts:51`, `crates/pi-edit/src/modes/sloppy`).
3. **Browser and computer folded into eval.** They are no longer builtin
   tools; eval reaches them instead. Commit `64508e1259`. `inspect_image` was
   replaced by `read` image questions (`ecc97e9c66`); `hub` was deprecated for
   `wait` (`f89a6db15e`).
4. **Codex Code Mode.** For `code_mode_only` models, the direct tool surface
   collapses to a keep-set and every other tool is reached through the eval
   bridge (`packages/coding-agent/src/session/code-mode.ts`, `1087ad09c1`).
5. **Eval re-engineered; Ruby/Julia removed.** Kernels are now `py` and `js`
   only, with a `WorkPool` bridge (`05bb0c1989`).
6. **In-process VCS.** `crates/pi-vcs` unifies git (gitoxide) and Jujutsu
   (jj-lib) behind N-API, replacing CLI wrappers (`2a7db5855d`).
7. **Virtual filesystem for the embedded shell.** `crates/pi-vfs` lets shell
   coreutils see `scheme://` paths without host materialization (`8b984d7a5a`).
8. **Catalog moved to a KDL policy engine.** Provider data is authored in
   92 `packages/catalog/src/compat/rules/providers/*.kdl` files; `descriptors.ts`
   keeps only factories (`dd57045396`).
9. **Notes-backed context windows (experimental).** `context_notes` +
   `new_context` tools roll the window over without a summarizer
   (`docs/compaction.md`, `54cae2c717`).
10. **`AgentSession` grew 44%:** 9,428 → **13,598** lines.

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
tools, 14 LSP ops, 28 DAP ops, ~80k lines of Rust core** (`README.md:27`,
unchanged at 18.8.9; the Rust figure is now ~4× stale). The architectural
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

**TypeScript workspaces** (16 under `packages/`): `ai`, `agent`, `coding-agent`,
`tui`, plus omp-only `catalog`, `snapcompact`, `mnemopi`, `omptype`,
`wire`, `natives`, `stats`, `collab-web`, `browser-relay`, `metaharness`,
`typescript-edit-benchmark`, `utils`. (`hashline` was removed in `c478e41a04`.)

**Rust crates** (12 under `crates/`, excluding `vendor/`; line counts measured
at `fd27203d27`):

| Crate | Role | Lines |
|-------|------|------:|
| `pi-builtins` | In-process CLI utilities | ~98k |
| `pi-natives` | N-API cdylib aggregating hot paths | ~82k |
| `pi-shell` | Embedded bash (vendored `brush-core`), PTY | ~42k |
| `pi-edit` | Edit engine for all modes, streaming sessions (new) | ~30k |
| `pi-vcs` | git (gitoxide) + jj (jj-lib), in-process (new) | ~13k |
| `pi-predict` | Prompt-composer word completion (new) | ~8k |
| `pi-walker` | Ignore-aware parallel walker + scan cache | ~8k |
| `pi-vfs` | Injectable async FS for shell coreutils (new) | ~6k |
| `pi-ast` | tree-sitter + ast-grep summaries | ~5k |
| `pi-iso` | Worktree isolation (APFS/btrfs/zfs/overlay) | ~5k |
| `pi-voice` | Audio / WebRTC | ~4k |
| `pi-diff` | jsdiff-compatible Myers diff, no FFI (new) | ~1k |

Non-vendor Rust grew from ~171k to ~302k lines since `316c11868`.
`crates/pi-natives/src/lib.rs` now registers ~45 modules, including grep,
glob, fd, workspace, ast, edit, vcs, predict, snapcompact, shell, pty, iso,
desktop, highlight, diff, tokens, html/pdf, spelling, and `tty_writer`.

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

Since 17.3.5 the in-process surface has *widened*: edit application
(`pi-edit`), git/jj (`pi-vcs`), and text prediction (`pi-predict`, served by a
machine-global Bun-worker daemon over a unix socket) all bind through N-API.

This is **not** swe-term's intended shape. swe-term wants heavy engines as
**sidecar binaries behind thin Go adapters**. omp wants grep/walk/PTY in the
same address space as the agent loop for latency.

Sidecars still exist for *other* jobs: Python/Ruby/Julia kernels, JS eval
worker, LSP/DAP children, browser/computer workers. The split is: **hot path =
N-API; long-lived protocol servers = child processes.**

---

## Tool Surface

Registry (`packages/coding-agent/src/tools/index.ts:552–589`):

`read`, `security_scan`, `bash`, `edit`, `ast_grep`, `ast_edit`, `ask`,
`debug`, `ida`, `eval`, `github`, `glob`, `grep`, `find`, `lsp`,
`checkpoint`, `rewind`, `context_notes`, `new_context`, `task`, `wait`,
`todo`, `web_search`, `write`, plus memory/skill tools (`memory_edit`,
`retain`, `recall`, `reflect`, `learn`, `manage_skill`). Hidden: `think`,
`yield`, `goal`.

Removed from the builtin map since 17.3.5: `browser` and `computer` (folded
into eval, `64508e1259`), `inspect_image` (`ecc97e9c66`), `hub`
(`f89a6db15e`).

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

coding-agent `tools/` alone is **~72.5k lines** (non-test `.ts`; was ~42k). That is product
mass, not loop mass.

---

## Hashline and Per-Model Edit Formats

Default edit mode is **`hashline`** (`cfgEditMode`, resolved in
`packages/coding-agent/src/utils/edit-mode.ts`): content-hash anchors instead
of brittle line numbers. Stale anchors reject instead of silently applying in
the wrong place.

Modes: `hashline` | `apply_patch` | `patch` | `replace` | `sloppy`
(`packages/tui/src/tools/edit.ts:51`).

Selection is **data-driven** (`resolveEditMode`):

1. `editVariantForModel()` — first `edit.modelVariants` pattern that occurs in
   the model id
2. `PI_EDIT_VARIANT` env pins the mode exactly
3. Hardcoded exclusions when the setting is `hashline` and
   `PI_STRICT_EDIT_MODE` is off: model classes kimi, mimo, minimax, deepseek,
   stepfun, plus codex-spark and GLM flash 5.3 → `replace`
4. Default `hashline`

The **engine** is now Rust: `crates/pi-edit` implements every mode's parse,
match, preview, and stage, with a streaming `Session` that turns tool-call
argument deltas into progressive previews and a final atomic apply through a
host-owned writer (`crates/pi-edit/src/lib.rs`). The TS side
(`packages/coding-agent/src/edit/`, ~1.7k lines) keeps schemas, prompts, LSP
writethrough, and auto-repair. This is the harness-problem thesis
in code: **the tool contract is part of the model adapter**, not a universal
XML.

---

## Eval Kernels and the Tool Bridge

`eval` is a first-class tool (`py` / `js`; Ruby and Julia were removed in
`05bb0c1989`), exclusive per session. Persistent kernels. Either kernel can
call back into the agent's tools. The bridge set grew: agent, completion,
judgment, budget, handle, and `WorkPool` bridges
(`packages/coding-agent/src/eval/*-bridge.ts`).

Python path (`packages/coding-agent/src/eval/py/tool-bridge.ts:1–8`):

> HTTP loopback bridge … POSTs to `/v1/tool` over a 127.0.0.1 loopback
> socket; the host resolves the request against the `ToolSession` … same
> `callSessionTool` implementation the JS bridge uses.

JS path: Bun worker re-enters `cli.ts` via `__omp_worker_js_eval`, then
`eval/js/tool-bridge.ts` calls `session.getToolByName()`.

Prelude helpers: `tool.read(...)`, `agent()`, `completion()`, `parallel()`.

Eval is now also a **gateway**, not only a tool:

- `browser` and `computer` are reached through eval, not the builtin map
  (`64508e1259`).
- **Code Mode** (`packages/coding-agent/src/session/code-mode.ts`) collapses
  the direct surface for `code_mode_only` models to `eval`, `ask`, `todo`,
  `yield`, `think`, checkpoint/rewind, and routes every other session tool
  through the eval bridge, mirroring codex-rs `ToolMode::CodeModeOnly`.
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
| File | 17.3.5 | 18.8.9 |
|------|------:|------:|
| `agent-session.ts` | 9,428 | **13,598** |
| `agent-loop.ts` | 2,935 | 4,028 |
| `agent.ts` | 1,748 | 2,242 |

The loop is more extracted than Prime's (Prime's `agent-loop.ts` is still
~986 of *pi-core*; omp grew the core loop to 4k). Orchestration still
concentrates in `AgentSession`, and the `session/` directory has fanned out to
~100 sibling modules (speculation, prewalk, TTSR, retry chains, advisors)
around it.

**Sessions:** append-only JSONL at
`~/.omp/agent/sessions/<encoded-cwd>/<timestamp>_<id>.jsonl` by default
(`FileSessionStorage`). The transcript sits behind a `SessionStorage`
interface with file, memory, SQL, and Redis backends
(`session/session-storage.ts`, `sql-session-storage.ts`,
`redis-session-storage.ts`). The SQL and Redis backends predate 17.3.5
(`cacf996f90`, `47bd4fd766`, 2026-05-28); the earlier version of this doc
omitted them.

**SQLite (`bun:sqlite`):** `agent.db` (settings, auth, usage);
`stats.db` (omp-stats); `history.db`, which now also holds per-session
`session_titles` and an idle-recap journal so listings stop content-scanning
JSONL (`session/session-index.ts`). Transcripts stay JSONL. This is the
Codex-like split (JSONL + SQLite metadata) pi lacked.

**Compaction** (`docs/compaction.md`):

1. **context-full** — LLM summary (default)
2. **shake** — mechanical elision to `artifact://`
3. **snapcompact** — deterministic PNG frames via Rust, vision-model readback
4. **handoff** — provider-native streaming compaction

Plus cheap wins: useless-result elision, superseded-read pruning. snapcompact
is a research strategy (vision billing, provider-specific); shake + elision
are the portable bits. snapcompact is still opt-in via `compaction.strategy`
(`packages/agent/CHANGELOG.md`).

New since 17.3.5: **notes-backed context windows** (experimental,
`compaction.experimentalContextManagement`). The model keeps a ≤16 KiB
notebook via `context_notes`; `new_context` rolls the window at a safe tool
boundary, keeping recent complete tool units plus the notebook, with **no
summarization call**. Originals stay reachable via `history://current/full`
(`docs/compaction.md` § Experimental notes-backed context windows,
`54cae2c717`).

---

## Catalog, Dialects, Prompts

**92 provider rule files** in
`packages/catalog/src/compat/rules/providers/*.kdl` (README says 60+; was 68
TS descriptors at 17.3.5). Provider data moved from TS into a KDL policy
engine (`dd57045396`); `provider-models/descriptors.ts` keeps only the
runtime model-manager factories. Roles: `default`, `smol`, `slow`, `plan`,
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

| Metric | Value (local 18.8.9) |
|--------|----------------------|
| TS packages | 16 |
| Rust crates (non-vendor) | 12, ~302k lines |
| `agent-session.ts` | **13,598** lines |
| `agent-loop.ts` | 4,028 lines |
| Builtin tool factories | 30 + hidden 3 |
| LSP / DAP actions | 14 / 28 (README; not recounted) |
| Catalog provider rules | 92 KDL files |
| Edit modes | 5; default hashline; engine in Rust `pi-edit` |
| Eval kernels | py, js |
| Session transcript | JSONL default; SQL/Redis/memory backends |
| Auth/settings/index | `bun:sqlite` `agent.db`, `history.db`, `stats.db` |
| Native coupling | in-process N-API (wider: edit, vcs, predict) |
| Runtime | Bun (`engines.bun >=1.3.14`; root `packageManager bun@>=1.4`) |
