# Critical Evaluation of oh-my-pi as an Agentic Harness

> Evaluated against the Harness (swe-term) philosophy:
> zero-dependency core, interface-driven plugins, immutable state,
> goroutine-native concurrency, content-addressable caching, single binary.
>
> **Validated against the local omp source** at `oh-my-pi/` (`17.3.5`).
> omp is a pi-mono fork. Loop/god-object critiques of pi still apply; this
> document focuses on what the fork *added* and which of those additions
> swe-term should take as sidecars vs refuse as core.

---

## Verdict Summary

oh-my-pi is the **highest-quality tool surface** in this survey and the
**clearest demonstration of how a pi fork dies of product success**. The
layering is still pi. The pass-rate story is real: edit format, summarized
reads, and in-process grep move the needle more than another model wrapper.
The Rust natives are serious. The catalog/dialect split is cleaner than
upstream. JSONL + SQLite metadata is the session design swe-term already
chose.

It is also a **kitchen sink with a 9,428-line `AgentSession`**, an in-process
N-API ABI that is the opposite of swe-term's sidecar rule, eval kernels that
can invoke the full tool registry (including bash and subagents), and a
Bun-locked platform. snapcompact is clever research, not a default compaction
strategy. 68 providers and 23 web-search backends are product, not harness.

**Steal hashline, per-model tool contracts, summarized reads, ToolSession,
JSONL+SQLite, prompt-as-files, and the idea of Rust engines. Implement those
engines as sidecars. Do not steal N-API-as-plugin-surface, the god object, or
the IDE-complete core repo.**

---

## 1. Effectiveness

### What omp does well

**The harness is the model adapter.** Per-model edit variants
(`edit-mode.ts`, `settings.getEditVariantForModel`) treat the tool wire format
as part of talking to a given set of weights. The README's Grok Code Fast
lift is the punchline: same model, different edit grammar, order-of-magnitude
pass rate. pi-mono and Claude Code under-invest here.

> **Lesson for Harness:** `Tool` schemas should be **model-parameterized**. A
> registry entry can select `hashline` vs `apply_patch` vs `replace` from the
> model id. Keep the executor one implementation; swap the prompt+grammar.

**Hashline is the right edit primitive.** Content-hash anchors fail closed on
staleness. Line-number `str_replace` fails open. For a harness that wants
first-attempt edits, this belongs in the edit sidecar (Rust) with a thin Go
adapter — exactly swe-term's extension story.

> **Lesson for Harness:** Put hashline (or equivalent snapshot-verified apply)
> in `extensions/`, with a golden edit-format bench. Do not bake it into the
> loop.

**Summarized reads + structured grep.** Dumping files into context is how
agents burn the window. omp's read tool returns snippets, hashline tags, and
selectors; grep returns a grouped tree with anchors. That is token
engineering as a tool concern, which is where it belongs.

> **Lesson for Harness:** Default `Read`/`Grep` tools cap and summarize.
> Full-file dumps are an explicit selector, not the happy path. Pre-LLM
> `Analyzer` can go further (repo map); omp shows the cheap version.

**ToolSession factory.** `BUILTIN_TOOLS[name](session)` plus `createIf` for
optional tools is a better lifecycle than pi's `createTool(cwd)`. Tools close
over approvals, telemetry, mutation versions, and model id.

> **Lesson for Harness:** `Tool` constructors take a `ToolContext` (session
> id, policy, cwd, model). Not a naked filesystem.

**JSONL transcripts + SQLite metadata.** Auth, settings, usage in `agent.db`;
conversation in JSONL; stats in `stats.db`. Jobs are separated. This matches
PLAN (SQLite session index) and Codex's split, and fixes pi's
index-free JSONL walk for *some* queries (not for transcript replay — still a
scan).

**Prompts as files, cache-stable prefixes.** No inline prompt strings.
Date/cwd in a per-request reminder. That is how you stop breaking prefix
cache on every turn.

> **Lesson for Harness:** System prompt templates live on disk. The loop
> fills a small, stable envelope. Do not `fmt.Sprintf` a novel each turn.

**Measure the harness.** `omp stats`, session-stats audits, typescript edit
benchmarks. Most agents fly blind. omp treats tool quality as something you
graph.

> **Lesson for Harness:** Log tool success/retry/token cost. A stats sidecar
> is optional; the event schema should make it possible.

**Worker re-entry for a single artifact.** One compiled bun binary, argv
selectors for workers. swe-term's single-binary goal has the same problem
once sidecars exist: either many binaries or one binary with subcommands
(`st worker grep`). omp's pattern is worth copying at the CLI layer.

### What omp gets wrong (for *our* purposes)

**N-API as the native ABI fights the Harness plan.** Grep in-process is
fast. It also couples the agent process to a platform matrix (AVX2 vs
baseline, musl libstdc++, gzip-embedded `.node` extraction). swe-term
already decided: **Go owns the loop; Rust is a sidecar.** omp is the
existence proof that the engines are worth writing *and* that in-process is
the coupling we refused.

> **Harness fix:** `extensions/swe_distiller` is the template. Spawn
> `st-grep` / `st-edit` / tokenizer. Never N-API into the core.

**Eval → full tool registry.** Loopback `tool.bash()` / `agent()` from Python
is delightful and is the lethal trifecta with a welcome mat. Prime's
`host_request` enumerates privileged ops. omp re-exports the whole surface
and inherits approvals — better than nothing, still a confused-deputy
highway.

> **Harness fix:** If code-exec exists, it gets a closed host RPC (Prime) or
> no host tools at all. Default deny re-entry.

**9,428-line AgentSession.** Smaller than Prime's 11k, larger than pi's 3k,
same disease. LSP writethrough, eval tracking, compaction triggers, bash,
branching — one type.

**Product kitchen sink in the core repo.** Collab/relay, computer/browser,
23 web-search backends, advisor, vibe, memory backends, GitHub-as-FS,
security_scan, metaharness. Each is a reasonable *extension*. Together they
make the fork unmergeable with upstream and unportable as an architecture.

**snapcompact as a compaction strategy.** Rendering the transcript to PNG for
a vision model is a research paper, not an operational default. Provider
billing, nondeterminism, and "did the model actually read the pixels" are
failure modes. Keep **shake** (mechanical elision to artifacts) and
useless-result pruning.

**Bun lock-in.** Fine for omp. Fatal as a swe-term assumption. Do not take
`Bun.file`, `bun:sqlite`, or `$` shell idioms as design.

---

## 2. Efficiency

In-process grep/walk/PTY are **faster** than sidecars on the hot path.
Sidecars are **cheaper to isolate, restart, and swap**. swe-term prefers the
second curve unless measurement says spawn overhead dominates — and for grep
over a repo, a long-lived sidecar amortizes spawn.

Hashline + summarized reads are **token-efficient**. That is the efficiency
that matters for LLM cost. Native grep is the efficiency that matters for
UX. Do not conflate them: the first is a tool contract, the second is an
engine placement.

68 providers in core is **catalog mass**. Generated catalogs (pi, Prime,
omp) all suffer this. Keep the registry data-driven and out of the loop
package.

---

## 3. Simplicity

omp kept pi's package *names* so the graph looks simple and then put an IDE
in `coding-agent/src/tools`. 31 tools × prompts × dialects × LSP/DAP is not
a small core. The honest module plan for swe-term:

- Core: loop, tools interface, session, policy
- Extensions: hashline edit, grep sidecar, LSP adapter, eval sidecar
- Not in v1: browser, computer, collab, 23 search backends, snapcompact

---

## 4. Key Gaps (relative to Harness goals)

| Gap | omp | Harness |
|-----|-----|---------|
| Native engines | In-process N-API | Sidecar binaries |
| Session index | SQLite for auth/stats; JSONL scan for transcripts | SQLite for topology + metadata |
| God object | 9.4k `AgentSession` | Split types |
| Sandbox | Approvals; eval is not confined | Policy + sandbox adapter |
| Dynamic extensions | Bun `import()` | Starlark + compiled Go |
| Analyzer | LSP/ast tools at request time | Pre-LLM enrichment interface |
| Content-addressable cache | Partial (walker scan cache in Rust) | `core/cache` as a layer |

---

## 5. Assumptions to NOT Port

### ❌ "Native code should load in-process via N-API"

Wrong plugin surface for a Go harness. Sidecar + stdin/stdout/json.

### ❌ "The eval kernel should call every agent tool"

Powerful, unbounded. Closed host RPC or nothing.

### ❌ "The coding-agent package is where features go"

That is how you get 42k lines of tools/ and a dead upstream.

### ❌ "Vision-model compaction is a general strategy"

Keep mechanical elision. Leave snapcompact as a paper.

### ❌ "One edit format for all models" *or* "hashline for all models"

omp's own exclusions (Kimi/Mimo/DeepSeek flash → replace) prove the table
must be data. Do not pick a single grammar and pray.

---

## 6. What to Actually Port

| From omp | To Harness | Why |
|----------|------------|-----|
| Hashline / snapshot-verified apply | Edit sidecar (Rust) + Go adapter | First-attempt edits, fail closed |
| Per-model edit variant table | Model registry field | Tool contract is adapter data |
| Summarized read + structured grep | Default tool behavior + caps | Token economy |
| `ToolSession` / `ToolContext` | Tool constructors take context | Policy and model flow in |
| JSONL transcript + SQLite metadata | Already in PLAN | Confirm; do it |
| Prompts as files, stable prefix | Prompt templates on disk | Cache + reviewability |
| Mechanical compaction (shake, elision) | `compact.go` strategies | Cheap context wins |
| Worker/subcommand re-entry | `st worker <engine>` | One user-facing binary |
| Harness metrics (tool retry, edit fail) | Event schema | You cannot improve what you do not log |
| LSP as an adapter, not the loop | Optional sidecar | IDE knowledge without core mass |
| Inband dialects for tool-less models | Provider compat flags | Keep pi's compat instinct, data-driven |

---

## 7. Architectural Contrasts

```
oh-my-pi                                 Harness (swe-term)
────────────────────────────────────     ─────────────────────────────────
Bun + in-process N-API Rust              Go core + Rust sidecar binaries
pi layers + IDE product mass             pi layers, features as extensions
AgentSession 9,428 ln                    Split Session/Compaction/Registry
31 tools in coding-agent                 Few core tools; rest extensions
Hashline default, per-model table        Same idea, sidecar + registry
Eval kernel → full ToolSession           Closed host RPC or no re-entry
JSONL + agent.db + stats.db              SQLite SSOT + optional JSONL export
snapcompact vision compaction            Mechanical elision first
68-provider catalog in-tree              Generated/registry, not loop
Single bun compile + worker argv         Single st binary + worker subcmd
```

---

## 8. Final Assessment

oh-my-pi is what happens when a talented fork treats **the harness as the
model's IDE**. That is the right product instinct and the wrong packaging
instinct for swe-term.

Take the IDE *capabilities* (edit that lands, search that is fast, LSP when
needed) as **sidecars and extensions**. Take the IDE *process* (N-API, 31
tools in core, eval that is a superuser) as a cautionary tale. The Go core
stays the pi loop with dsh's log invariant and Prime's protocol-shaped
frontends. omp supplies the tool-quality backlog, not the architecture.
