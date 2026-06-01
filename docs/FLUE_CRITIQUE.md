# Critical Evaluation of Flue as an Agentic Harness

> Evaluated against the Harness (swe-term) philosophy:
> zero-dependency core, interface-driven plugins, immutable state,
> goroutine-native concurrency, content-addressable caching, single binary.
>
> **Validated against the local Flue source** at `/home/rohit/sandbox/swe-term/flue` (`0.7.0`,
> *Experimental*) and the real consumer app at `/home/rohit/sandbox/gvision`. The central claim
> below — that Flue's core is pi's — is now **code-confirmed**: `packages/runtime/src/session.ts`
> constructs pi-agent-core's `Agent` and delegates each turn to `agent.prompt()`.

---

## Verdict Summary

Flue is the most **conceptually interesting** agent in this survey because it asks a different
question. Claude Code, Codex, and pi all answer *"how do I build a great terminal coding agent?"*
Flue asks *"what if the agent harness were a deployable framework, like Next.js for agents?"* That
reframing — **agent = a directory compiled into a server artifact, headless by default, deployable
to Node or the edge** — is genuinely novel and worth taking seriously.

But as an *architectural template for swe-term*, Flue is the weakest fit of the four. It is
**Experimental and pre-1.0** (`0.7.0`, frequent breaking changes), it is a **thin framework over
pi-mono's engine** (so its agent-loop ideas are really pi's — Flue's `Session` literally wraps pi's
`Agent`), its flagship durability story is **tightly coupled to Cloudflare Durable Objects**, and
its default sandbox (`just-bash`) is an **in-process emulation with no real isolation**. Flue is a
deployment-and-packaging layer, not an agent core.

**Steal the framework framing and the headless/deployment model. Do not look to Flue for the core —
that's pi-mono's job, which swe-term already ports.**

---

## 1. Effectiveness

### What Flue does well

**It names the right problem: harness ≠ SDK.** Flue's thesis — that the missing layer is a
*framework* (convention, build, deploy), not another SDK — is correct and under-served. Every other
agent here is either a product (Claude Code, Codex) or a library (pi-mono). Flue is the first to say
"agents need a `build`/`run`/`deploy` story" and ship one.

> **Lesson for Harness:** swe-term is a single-binary CLI, not a deploy-to-edge framework — but the
> insight that *the harness is a first-class artifact with its own lifecycle* is worth absorbing.
> The PLAN's Phase 7 (recursive self-build: agent writes Go → `go build` → exec new binary) is
> swe-term's version of "agent as a compiled artifact." Flue compiles to a server; swe-term compiles
> to itself.

**Headless-first is a clean stance.** By refusing a TUI/human-operator assumption, Flue forces a
clean separation between "the agent logic" and "who's driving it." A handler function that runs in a
server, a cron, or a CI job is a more composable unit than a REPL.

> **Lesson for Harness:** This is the same payoff as Codex's protocol seam, reached from the other
> direction. swe-term's `Frontend` interface should make the *print/RPC* frontends genuinely
> first-class, not afterthoughts to the TUI. A headless agent you can script is the foundation;
> the TUI is one frontend.

**Host-side tool execution with schema-only exposure is a strong security boundary.** Tools run in
the host process with `process.env`; the model sees only the schema. Secrets never enter the prompt
or filesystem context. A headless framework can enforce this more strictly than a terminal agent.

> **Lesson for Harness:** Keep secrets out of tool *arguments* and context. Go tools should read
> credentials from the environment/config at execution time, never receive them through the model.

**First-classing sessions / subagents / sandboxes as framework primitives is the right taxonomy.**
Flue's primitive list (harness, session, role, task, sandbox, tool, skill, MCP) is almost exactly
the surface area swe-term must also provide *around* its ported pi-core. Flue is a useful checklist
of "what a harness needs beyond the loop."

### What Flue gets wrong (for *our* purposes)

**The "core" is borrowed.** Flue's agent loop, reasoning, and provider layer are
`@earendil-works/pi-agent-core` + `@earendil-works/pi-ai`. The code is unambiguous: `Session`
constructs `new Agent({ ..., toolExecution: 'parallel' })` and a turn is just
`await this.harness.prompt(...)`. So Flue contributes *nothing new to the agent loop itself* — its
innovations are all packaging/deployment. For swe-term, which is *building* the core, Flue offers
framing but no engine lessons that pi-mono doesn't already provide more directly.

> **Implication:** Read Flue for the *framework* layer; read pi-mono for the *engine*. Don't conflate
> them. Flue's agent-loop critiques are pi-mono's critiques.

**Pre-1.0 instability.** Marked *Experimental* (`0.7.0`) with frequent breaking changes; APIs are
visibly in flux — the runtime moved out from under the `@flue/sdk` name to `@flue/runtime` +
`@flue/cli` (old `@flue/sdk/*` subpaths now throw migration errors), `defineConfig` moved to
`@flue/cli/config`, `getVirtualSandbox()` was removed, and the pi deps were renamed
`@mariozechner/*` → `@earendil-works/*`. It is not yet a stable foundation to design against.

---

## 2. Efficiency

### `just-bash` is emulation, not a sandbox

The default sandbox is an **in-process emulated bash with a virtual filesystem** — it runs in the
*same process* as the agent. It's fast and cheap, but it provides **no isolation boundary**. Calling
it a "sandbox" is generous; it's a convenience layer. Real isolation requires the remote/Cloudflare
container path.

> **Contrast with Harness/Codex:** Codex ships three *real* OS sandboxes (landlock/seccomp, seatbelt,
> Windows restricted token). swe-term should follow Codex here, not Flue. An "in-process virtual FS"
> is not a security control; treat it as a testing convenience at most.

### `local()` provides zero isolation by design

The Node `local()` sandbox is direct host access (`process.cwd()`, `child_process`), explicitly
"intended for environments where the runner itself provides the isolation boundary" (CI). That's a
reasonable stance *for a CI-invoked framework*, but it means Flue's isolation story is binary:
emulation (no isolation) or "trust the CI jail" (no isolation in-process) or remote containers (real,
but heavyweight and CF-flavored).

> **Lesson:** swe-term wants *in-process* safety for the common case (local dev on a developer's
> machine), which neither `just-bash` nor `local()` provides. The OS-sandbox-helper model (Codex) is
> the right target.

### Ephemeral sessions on Node

On the Node target, sessions are **in-memory by default and lost on restart** unless you supply a
custom `SessionStore`. For a framework that markets "continuity" and "durable execution," the default
non-Cloudflare experience is *non-durable*. Durability is effectively a Cloudflare feature.

> **Contrast with Harness:** swe-term's PLAN makes persistence a *core* concern (SQLite via
> `modernc.org/sqlite`, branchable tree, resume) on every platform — not a property of the deploy
> target. Durability should not depend on where you deploy.

---

## 3. Simplicity

### Durability is Cloudflare-coupled

Flue's strongest durable-execution story — Durable Objects + SQLite, `FlueRegistry` DO auto-injection,
cross-request resumption "for years" — is **tightly coupled to Cloudflare Workers**. The framework
aspires to be runtime-agnostic, but its best features light up on one vendor's platform. The Node
target is the lowest-common-denominator fallback.

> **Do not port:** Platform-coupled durability. swe-term is a local single binary; its durability is
> a local SQLite file, available identically everywhere. "Runtime-agnostic in principle,
> Cloudflare-best in practice" is the exact lock-in the Harness philosophy rejects (cf. the Claude
> Code critique's "avoid unnecessary environment lock-in").

### A build step for an agent

Flue's "agent = directory compiled into a server artifact" (esbuild on the Node target, `wrangler`
on the Cloudflare target) is elegant for a deploy-to-edge framework but is *overhead* for a local
CLI. swe-term's equivalent — compile Go to a single static binary — needs no esbuild, no `wrangler`,
no `flue.config.ts`, no artifact directory. The build *is* `go build`.

> **Contrast:** Flue's build system solves a problem (bundle TS for multiple JS runtimes) that a Go
> single-binary simply doesn't have. swe-term's Phase 7 recursive self-build is the relevant analog,
> and it's `go build ./cmd/...`, not a bundler pipeline.

### MCP is client-only and partial

Flue connects to remote streamable-HTTP MCP servers but does **not** spawn stdio servers, auto-detect
transports, or handle OAuth callbacks. Compared to Codex (full client *and* server, stdio + HTTP +
in-process, OAuth) or even Claude Code (multi-transport client), Flue's MCP is minimal.

> **Lesson:** A complete MCP story (client + server, multiple transports) is table stakes for a
> serious harness. Flue's partial implementation is a pre-1.0 gap, not a model.

---

## 4. Key Gaps (relative to Harness goals)

### No core to learn from (it's pi's)

swe-term's entire purpose is to *build* the agent loop, state machine, provider abstraction, and
session store. Flue *consumes* those from pi. So Flue offers no core-engineering lessons beyond what
pi-mono already provides — and pi-mono provides them more directly.

### No content-addressable cache, no analyzer

Same gaps as every other agent surveyed: no general `ContentKey(sha256) → value` store, no pre-LLM
context enrichment (repo maps, dependency graphs). These remain swe-term differentiators.

### Real isolation only via remote/Cloudflare

There is no in-process OS-level sandbox. For a local-first harness, that's the wrong default.

### Provider abstraction is inherited, not designed

Models are `provider/model-id` strings resolved by `pi-ai`. Flue adds Cloudflare Workers AI routing,
but the abstraction is pi's. swe-term must design its own `Provider` interface that bridges genuinely
different wire protocols (Anthropic Messages vs OpenAI Chat vs Responses) — work Flue never has to do.

---

## 5. Assumptions to NOT Port

### ❌ "An in-process virtual bash is a sandbox"

`just-bash` is a convenience, not a security boundary. Real isolation needs OS mechanisms (Codex's
model) or remote containers. Don't conflate emulation with isolation.

### ❌ "Durability is a property of the deploy target"

Cloudflare gets Durable Objects; Node gets ephemeral memory. swe-term's persistence must be uniform
(local SQLite) regardless of environment.

### ❌ "Agents need a bundler/build pipeline"

True for multi-runtime TypeScript; false for a Go single binary. swe-term's build is `go build`.

### ❌ "Runtime-agnostic" while best-on-one-vendor

The aspiration is right; the reality is Cloudflare-favored. A zero-lock-in harness should be equally
capable everywhere, which for swe-term means *one binary, one local data dir, no platform services*.

### ❌ "MCP client-only (HTTP, no stdio, no OAuth) is enough"

It's a pre-1.0 gap. A serious harness is both MCP client and server, multi-transport.

### ❌ "Borrow the engine, ship the framework"

Reasonable for Flue's goals; the opposite of swe-term's. swe-term's value *is* the engine. Borrowing
pi's loop wholesale would defeat the learning purpose and the "without its gaps" thesis.

---

## 6. What to Actually Port

| From Flue | To Harness | Why |
|-----------|------------|-----|
| Headless-first framing | `Frontend` interface design | Print/RPC frontends are first-class, not afterthoughts |
| "Agent as a deployable artifact" | Phase 7 recursive self-build | swe-term compiles to *itself* instead of a server |
| Primitive taxonomy (session/role/task/sandbox/skill/tool/MCP) | feature checklist | Confirms the surface area a harness needs beyond the loop |
| Roles as call-scoped persona overlays | system-prompt overlay per call | Swap persona without forking state |
| Tasks as detached child agents sharing a sandbox | `AgentTool` semantics | Subagent works in the same workspace, own history |
| Host-side tool execution, schema-only to model | tool secret hygiene | Credentials never enter the context |
| Typed results via schema (`prompt({ result })`) | structured-output tool | Agents as typed functions, not just chat |
| `dev` / `run` watch-and-reload DX | dev ergonomics | Fast iteration loop for tool/skill authoring |

---

## 7. Architectural Contrasts

```
Flue                                   Harness (swe-term)
─────────────────────────────────     ─────────────────────────────────
TypeScript, runtime-agnostic           Go single static binary
Framework over pi-agent-core+pi-ai     Builds its own core (ports pi)
Agent = dir → esbuild/wrangler → server  Agent = `go build` → one binary
Headless only (no TUI)                 Headless core + TUI/RPC/print frontends
just-bash (in-process emulation)       OS sandbox helper (seccomp/landlock)
local() = direct host (no isolation)   In-process safety as the default
Sessions in-memory on Node (ephemeral) SQLite persistence everywhere
Durable execution = Cloudflare DOs     Durable = local SQLite (uniform)
MCP client only (HTTP, no stdio/OAuth) MCP client + server, multi-transport
Provider layer inherited from pi-ai    Provider interface designed in-house
Experimental 0.7.0, breaking changes   Versioned, learning-oriented, stable target
Cloudflare-favored "agnostic"          Zero platform lock-in
```

---

## 8. Final Assessment

Flue is the right idea aimed at a different target than swe-term. Its contribution to the field is
the **reframing**: the agent harness as a deployable, headless framework with a build step and a
deploy story — "Next.js for agents." That framing is valuable and swe-term should absorb its spirit
in two places: the `Frontend` interface (headless core, frontends as first-class clients) and Phase 7
(agent as a compiled artifact — but compiled to *itself*, not to a Cloudflare Worker).

For everything else, Flue points the *opposite* way from the Harness philosophy. Its durability is
platform-coupled where swe-term wants it uniform and local. Its default "sandbox" is emulation where
swe-term needs real OS isolation. Its build pipeline solves a TypeScript-multi-runtime problem that a
Go single binary doesn't have. And its core is borrowed from pi-mono — so the deep engineering
lessons swe-term needs live in pi, not Flue.

The clean summary of the lineage that ties this whole doc set together:

- **pi-mono** is the **engine** swe-term ports (and the source of its 12-gap to-do list).
- **Flue** is a **framework** that wraps that engine for headless edge deployment — useful for its
  framing and its primitive taxonomy, not its core.
- **Codex** is the **protocol-and-safety reference** — steal the SQ/EQ seam and the layered exec
  sandboxing.
- **Claude Code** is the **anti-template** — proof you can ship a feature-complete monolith, and
  proof you shouldn't.

Take Flue's framing. Build the core from pi's lessons. Borrow Codex's seam and sandboxes. Avoid
Claude Code's monolith. That is the swe-term thesis, now triangulated against all four.
