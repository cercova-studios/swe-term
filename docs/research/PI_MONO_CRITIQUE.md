# Critical Evaluation of pi-mono as an Agentic Harness

> Evaluated against the Harness (swe-term) philosophy:
> zero-dependency core, interface-driven plugins, immutable state,
> goroutine-native concurrency, content-addressable caching, single binary.
>
> pi-mono is not just *a* reference — it is *the* reference swe-term ports from
> (`../core/PLAN.md`). This critique expands the PLAN's 12-row gap analysis into a
> Harness-aligned assessment: what to keep, what to fix, what to discard.
>
> **Validated 2026-10-10** against `earendil-works/pi` (formerly `badlogic/pi-mono`) at commit
> `ea448f454`, packages `1.1.0`, scope `@earendil-works/*`. Previous validation: `0.65.2`
> (`@mariozechner/*`), 2026-06-06. Evidence paths are in `PI_MONO_DEEP_DIVE.md`.

---

## Verdict Changes in the 2026-10-10 Refresh

| Verdict | Change | Evidence |
|---------|--------|----------|
| God object (gap #1) | **Strengthened.** `AgentSession` grew 3,060 → 4,398 lines. | `packages/coding-agent/src/core/agent-session.ts` |
| URL sniffing (gap #3) | **Partly changed.** Upstream now has a real `Provider` interface with one module per provider — the fix swe-term planned. But `detectCompat()` URL sniffing remains and its list grew. Verdict: port the interface, still drop the sniffing. | `packages/ai/src/models.ts:150`; `packages/ai/src/api/openai-completions.ts:1593` |
| `chars/4` (gap #5) | **Corrected, not changed.** The trigger anchors on the last provider-reported usage; `chars/4` applies only to trailing messages. This was already true at `ff3e9df5f`. Gap still holds, with a smaller error surface. | `core/compaction/compaction.ts:196,298` |
| No interfaces between layers (gap #6) | **Partly changed.** `Provider` and `TUI` are interfaces; the agent core takes `streamFn` by injection (`setDefaultStreamFn`) and exposes `beforeToolCall`/`afterToolCall` hooks. `AgentSession` is still a concrete class. | `packages/agent/src/stream-fn.ts`; `packages/agent/src/types.ts:326,341`; `packages/tui/src/tui.ts:506` |
| `Promise.all` (gap #8) | **Minor.** Tool execution now has an explicit `sequential`/`parallel` policy; parallel still collects via `Promise.all`. | `packages/agent/src/agent-loop.ts:519,646` |
| `AbortController` sprawl (gap #7) | **Unchanged in kind.** Five singletons became four plus a `Set` for bash runs. | `agent-session.ts:403–420` |
| JSONL storage (gap #4) | **Strengthened.** Coding agent is still JSONL-only; upstream's new `pi-durable` adds SQLite storage — independent confirmation of the SQLite fix. | `packages/durable/src/storage/sqlite/` |
| Layering / scope | **Qualified.** The core four packages remain clean, but the repo is now 14 packages with an experimental durable/remote stack (`chord`, `pi-protocol`, `pi-server`, `pi-client`, `pi-durable`, `pi-env`). Port the core four only. | `packages/*/package.json` |
| MCP | **New port input.** MCP is built in (`pi-mcp` + bundled `mcp` extension), and `pi-codemode` runs tools from a QuickJS sandbox. Both are references for swe-term's MCP `Tool` source and sandboxed-execution tier. | `packages/mcp`, `packages/codemode`, `packages/coding-agent/src/extensions/` |

---

## Verdict Summary

pi-mono is the **best architectural template** of the agents surveyed — which is precisely why
swe-term chose to port it. Where Claude Code is a 512K-LoC monolith and Codex is a 120-crate
fortress, pi-mono is a **small, cleanly-layered set of packages**: `pi-ai` (providers) →
`pi-agent-core` (agent) → `pi-coding-agent` (coding) + `pi-tui` (UI). The layering is correct, the
scope is sane, and the agent core is genuinely provider- and domain-agnostic (Flue proves this by
building on `pi-agent-core` + `pi-ai` directly).

But pi-mono is a **Node/TypeScript hacker's tool**, and its implementation choices reflect that:
a **4,398-line** `AgentSession` god object, dynamic-`import()` extensions, URL-substring provider
sniffing as a fallback, `chars/4` estimation for the un-reported tail of compaction decisions, a JSONL tree with only
an in-memory index, `AbortController` sprawl, and `any` types in the hot path. None of these are
fatal — they're the difference between "great architecture, pragmatic implementation" and
"production harness."

**Port the layering and the loop. Fix the twelve gaps. The result is swe-term.**

---

## 1. Effectiveness

### What pi-mono does well

**The layering is the lesson.** `pi-ai` → `pi-agent-core` → `pi-coding-agent` is exactly the
dependency direction the Harness wants: providers at the bottom, a domain-agnostic agent in the
middle, the coding specialization on top, UI to the side. This is the separation Claude Code never
made and Codex achieves only with 120 crates. pi-mono achieves it with a four-package core
(`pi-ai` → `pi-agent-core` → `pi-coding-agent` + `pi-tui`). As of 1.1.0 the repo has 14 packages:
`pi-web-ui`, `pi-mom`, and pods are gone; `pi-mcp`, `pi-codemode`, `pi-telemetry`, and `pi-evals`
were added; and an experimental durable/remote stack (`pi-durable`, `pi-env`, `chord`,
`pi-protocol`, `pi-server`, `pi-client`) sits beside the core and does not depend on
`pi-agent-core`.

> **Lesson for Harness:** This *is* the Harness module plan. `core` (loop/bus/state/protocol) →
> `provider` extensions → `tool` extensions → `frontend`. Keep pi-mono's direction; enforce it with
> Go modules instead of npm-workspace convention.

**Provider-agnostic agent core.** `pi-agent-core` knows nothing about files or bash. The coding
tools live one layer up. That clean cut is *why* Flue could reuse it headlessly and *why* swe-term
can re-implement just that layer in Go.

> **Lesson for Harness:** The agent loop must depend only on `Provider`/`Tool` interfaces, never on
> concrete coding tools. pi-mono validates this works in practice.

**Capability flags over provider branching.** `Model.compat` (`supportsStore`,
`supportsReasoningEffort`, `maxTokensField`, …) is the right instinct: declare what a model
supports, don't sprinkle `if provider === 'anthropic'` across the codebase.

> **Lesson for Harness:** Model capabilities belong in a typed struct on the model registry entry,
> consulted by the provider implementation. Keep this.

**Zero-build extensions are great DX.** `jiti` loads a `.ts` file and its `pi.registerTool()` calls
make tools live *without a reload*. For human iteration speed this is excellent.

### What pi-mono gets wrong

**`AgentSession` is a 4,398-line god object** (PLAN gap #1; it was 3,060 in 0.65.2). One class owns lifecycle, state,
persistence, model/thinking management, manual *and* automatic compaction, bash execution, session
branching, retry, and extension integration (`ExtensionRunner`). It cannot be unit-tested in
pieces; every concern is reachable from every other. This is the same failure mode as Claude Code's
`AppState`, just smaller.

> **Harness fix (already in PLAN):** Decompose into `Session`, `Compaction`, `ToolRegistry`,
> `RetryPolicy`, `HookRunner`. No single type owns the universe. The agent loop is a function, not a
> method on a god object.

**URL-substring provider detection** (PLAN gap #3). When `Model.compat` is unset, `pi-ai`'s
`detectCompat()` (`packages/ai/src/api/openai-completions.ts:1593`) infers behavior by matching
the provider id or base URL against strings like `api.z.ai`, `api.moonshot.`, `openrouter.ai`,
`cerebras.ai`, `deepseek.com`, `api.x.ai`, and `chutes.ai`. This is fragile (URLs change, proxies
obscure them) and is the kind of implicit coupling that breaks silently. Upstream has since added a
proper `Provider` interface (`packages/ai/src/models.ts:150`) with one module per provider, but kept
the sniffing as the fallback for OpenAI-compatible endpoints.

> **Harness fix:** A clean `Provider` interface with per-provider implementations. Capability is
> declared, never sniffed. If a model needs special handling, that lives in its provider impl, not
> in a URL regex.

---

## 2. Efficiency

### `chars/4` token estimation for compaction (PLAN gap #5)

pi-ai reports *real* usage for accounting. The decision of *when* to compact anchors on the last
provider-reported usage and applies a `chars/4` heuristic only to messages after it
(`estimateContextTokens()`, `core/compaction/compaction.ts:196`) — and to the whole history before
the first response. That bounds the error, but it is still unbounded for a large tool result or
pasted file in the tail. That's fine until it isn't: under- or over-estimating near the context
limit either wastes window or triggers a mid-turn overflow. For a harness that wants predictable
compaction, an approximation in the trigger is a real bug surface.

> **Harness fix:** `tiktoken-go` or a Rust tokenizer subprocess for accurate pre-flight counts.
> Compaction triggers on real token math.

### On-disk-index-free JSONL tree (PLAN gap #4)

The append-only per-session `{timestamp}_{id}.jsonl` with `id`/`parentId` is elegant and
human-readable, and branching is free. There's an **in-memory `byId` map** built on load
(`_buildIndex()`), but **no on-disk index** — opening a session means reading the whole file and
rebuilding that map, and branch/path resolution walks the `parentId` chain. For short sessions this
is invisible; for long-lived, deeply-branched sessions it's O(n) work on every load and navigation.

> **Harness fix:** SQLite via `modernc.org/sqlite` (pure Go, no CGo). Keep the tree *model* (it's
> correct), but back it with indexed storage so branch/fork/resume are queries, not scans. Codex
> independently reached the same conclusion (JSONL + SQLite metadata).

### `AbortController` sprawl (PLAN gap #7)

`AgentSession` holds **four** single-slot `AbortController`s (`_compactionAbortController`,
`_autoCompactionAbortController`, `_branchSummaryAbortController`, `_retryAbortController`) plus a
`_bashAbortControllers: Set<AbortController>` (`agent-session.ts:403–420`) — plus the `Agent`'s own
run controller — one per cancellable operation. This works but is manual bookkeeping: each controller must be created, threaded, and
torn down by hand.

> **Harness fix:** `context.Context`. One cancellation primitive, composable, idiomatic, and
> automatically propagated through the call tree. Child contexts replace per-operation controllers.

### `Promise.all` for parallel tools (PLAN gap #8)

Tool execution now has an explicit `toolExecution: "sequential" | "parallel"` policy, and a tool can
force sequential mode (`packages/agent/src/agent-loop.ts:519`). Parallel results are still gathered
with `Promise.all` (`agent-loop.ts:646`): correct but unstructured — no shared cancellation
semantics, no partial-failure policy beyond what the caller writes.

> **Harness fix:** `errgroup` + goroutines, with `context` cancellation and first-error propagation
> built in.

---

## 3. Simplicity

### Dynamic-`import()` extensions (PLAN gap #2)

The `jiti`-based extension system is the best *DX* feature and the worst *architecture* feature for
a portable harness. It requires a JS runtime that can dynamically evaluate arbitrary TypeScript at
runtime — exactly the capability a single static binary doesn't (and shouldn't) have. It also means
extension code runs with full Node privileges in-process.

> **Harness fix:** Two-tier self-extension. **Starlark** (`go.starlark.net`) for sandboxed tool
> definitions the agent can author at runtime, and **Go interfaces** for hooks compiled into the
> binary. The "hot path" (throwaway runtime code) goes to a sandboxed Monty subprocess. This keeps
> the live-iteration spirit without dynamic in-process code loading.

### `any` types in TypeScript (PLAN gap #12)

`Model<any>`, `_scopedModels: Array<{ model: Model<any> }>` — `any` leaks into the model layer
despite the project's strict-TS goals. Each `any` is a hole in the type system at exactly the place
(model/provider polymorphism) where types matter most.

> **Harness fix:** `json.RawMessage` for genuinely dynamic data (tool args, provider-specific
> blobs) and typed structs everywhere else. Go's interface system handles the polymorphism that
> pushed pi-mono to `any`.

### Convention-based error contracts

TypeScript error handling is convention (JSDoc, thrown values) rather than enforced. A caller can
forget to handle a rejection and the compiler won't object.

> **Harness fix:** Go's `error` return values, enforced by the compiler. Errors are values, not
> exceptions, and an unhandled one is a visible lint/vet failure.

### Custom TUI renderer (PLAN gap #11)

`pi-tui`'s hand-rolled differential renderer (**~19,500 lines across 46 files**, up from ~10,300 /
25 in 0.65.2) is impressive but
is undifferentiated infrastructure — every line is a line you maintain instead of borrowing.

> **Harness fix:** `ratatui` (Rust) or Bubble Tea (Go). Don't build a terminal diff engine; use a
> mature one and spend the complexity budget on the agent.

---

## 4. Key Gaps (relative to Harness goals)

### No interfaces between layers (PLAN gap #6)

pi-mono's layers are *packages*, and at the top the boundary is still a concrete class: frontends
depend on `AgentSession`, not on an abstraction of it. Lower boundaries have improved since 0.65.2 —
`Provider` and `TUI` are interfaces, the agent core takes its `streamFn` by injection
(`setDefaultStreamFn()`, `packages/agent/src/stream-fn.ts`), and tool interception is a pair of
loop hooks (`beforeToolCall`/`afterToolCall`). Substituting the session store or the orchestrator is
still harder than it should be.

> **Harness fix:** Go's implicit, consumer-side interfaces. The core defines `Provider`, `Tool`,
> `SessionStore`, `Hook`; extensions satisfy them structurally. Boundaries are interfaces, not
> classes.

### Config scattered across 5+ sources (PLAN gap #10)

Configuration is spread across multiple files/env vars without a single typed precedence model.

> **Harness fix:** One `Config` struct, layered precedence (defaults → file → env → flags), loaded
> once. (Notably, Codex *does* do layered config well — that's the model, minus its 3.8K-line
> sprawl.)

### No content-addressable cache

Like every other agent surveyed, pi-mono recomputes file reads, tool outputs, and any analysis on
each invocation.

> **Harness fix:** `core/cache` with `ContentKey(sha256) → value`. This is a Harness
> differentiator none of the references provide.

### No analyzer concept

Code intelligence is whatever the model gets by calling tools. No pre-LLM enrichment (repo map,
dependency graph) is injected into context.

> **Harness fix:** The `Analyzer` interface — enrich context before the LLM sees it.

---

## 5. Assumptions to NOT Port

### ❌ "One orchestrator object can own everything"

`AgentSession` is the anti-pattern. Decompose by concern; the loop is a function over interfaces.

### ❌ "Extensions should be dynamically-loaded TS modules"

Great for a Node hacker tool; impossible for a static single binary and a security surface in-process.
Use Starlark + compiled Go hooks + sandboxed Monty.

### ❌ "Provider differences can be sniffed from URLs"

Fragile. Declare capabilities; implement per provider behind an interface.

### ❌ "`chars/4` is good enough for compaction decisions"

It's good enough until you're at the context limit and it's wrong. Use a real tokenizer.

### ❌ "Append-only JSONL with tree-walking is sufficient storage"

Correct model, wrong engine at scale. Keep the tree, add an index (SQLite).

### ❌ "`any` is an acceptable escape hatch"

Each `any` is an un-typed boundary. Use `json.RawMessage` for dynamic data and types everywhere else.

### ❌ "Hand-roll the TUI"

Undifferentiated work. Borrow ratatui / Bubble Tea.

---

## 6. What to Actually Port

| From pi-mono | To Harness | Why |
|-------------|------------|-----|
| Package layering (ai → agent-core → coding → ui) | The whole module plan | Correct dependency direction |
| Provider-agnostic agent core | `core/agent` depends only on interfaces | Domain-free loop |
| `Provider` interface (`pi-ai/src/models.ts:150`) + `Model.compat` flags | `Provider` interface + model registry entry struct | Declare, don't sniff (skip `detectCompat()`) |
| Agent loop shape (stream → tools → loop) | `core/agent/loop.go` | The cycle is right |
| JSONL tree *model* (id/parentId branching) | `session/tree.go` over SQLite | Branching is correct; add an index |
| Compaction-at-a-cut-point concept | `core/agent/compact.go` | Summarize old, preserve recent |
| Interactive/print/json/rpc mode split | `Frontend` implementations | Frontends are interchangeable |
| Zero-build iteration *spirit* | Starlark hot-reload of tool defs | Keep the DX, lose the dynamic import |
| `beforeToolCall` / `afterToolCall` loop hooks | `Hook` interface | Pre/Post tool-use is a good model |
| `pi-mcp` client + `codemode` nested tool calls | MCP `Tool` source; sandboxed (Monty) execution tier | Built-in MCP and out-of-context tool composition now have a reference |

---

## 7. Architectural Contrasts

```
pi-mono                                Harness (swe-term)
─────────────────────────────────     ─────────────────────────────────
TypeScript / Node.js                   Go (+ Rust TUI/tokenizer)
14 npm packages (@earendil-works/*)   core + extensions (Go modules)
AgentSession: 4,398 ln (god object)    Session/Compaction/ToolRegistry/… split
Provider/TUI ifaces; AgentSession class Implicit interfaces between layers
Provider iface + URL sniffing fallback Provider interface, per-provider impl
last usage + chars/4 tail estimate     tiktoken-go / Rust tokenizer subprocess
JSONL tree, in-memory index only       SQLite (modernc) with tree schema
AbortController (4 + a Set)           context.Context (one primitive)
Promise.all for parallel tools         errgroup + goroutines
dynamic import() extensions (jiti)     Starlark tool defs + Go hooks + Monty
any types in model layer               json.RawMessage + typed structs
custom pi-tui renderer (~19,500 ln)    ratatui / Bubble Tea
config across 5+ sources               one Config struct, layered precedence
no content-addressable cache           core/cache (sha256 → value)
no analyzer concept                    Analyzer interface (pre-LLM enrichment)
```

---

## 8. Final Assessment

pi-mono earns its place as swe-term's reference. Its **architecture is right where it matters
most**: the core-four layering, the provider-agnostic core, the clean agent loop, the branchable session
model, the interchangeable frontends. These are the bones swe-term keeps.

What swe-term changes is **implementation, not architecture**. The twelve gaps are all the same
shape: pragmatic Node/TypeScript choices that don't survive the move to a zero-dependency,
single-binary, statically-typed harness. Upstream has since closed part of gap #3 itself (a real
`Provider` interface) and moved its experimental durable stack to SQLite, which supports both fixes;
it has also grown the god object. The god object becomes composed services. Dynamic imports
become Starlark + Go interfaces. URL sniffing becomes a `Provider` interface. `chars/4` becomes a
real tokenizer. JSONL-walking becomes SQLite. `AbortController` becomes `context.Context`. `any`
becomes `json.RawMessage` + types. The custom renderer becomes ratatui.

The relationship is unusually clean: **pi-mono is the spec, swe-term is the implementation, and the
gap analysis is the diff.** Unlike Claude Code ("port the lessons, not the architecture") or Codex
("steal the seam, leave the mass"), the verdict here is simply: **port the architecture, fix the
twelve gaps.** That's the entire project.
