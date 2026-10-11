# Critical Evaluation of DeepSeek Harness as an Agentic Harness

> Evaluated against the Harness (swe-term) philosophy:
> zero-dependency core, interface-driven plugins, immutable state,
> goroutine-native concurrency, content-addressable caching, single binary.
>
> **Validated against the local dsh source** at `deepseek-harness/`
> (`0.2.1-alpha.2`, HEAD `d743267388`), refreshed 2026-10-10. Previous
> validation: `c15f2f4d2` (2026-08-16). This is a pre-release composition
> platform, not a small coding loop.

## Verdict Changes (2026-10-10)

| Verdict | Before | Now | Evidence |
|---------|--------|-----|----------|
| Dispatch-time reconstruction invariant | dsh enforces it at runtime | dsh **removed** the runtime check; reconstruction is by construction (derive, freeze, brand) plus a test. Still the top steal, but swe-term keeps the runtime check dsh dropped. | `f028f25667`; `packages/core/agent-loop/src/agent.ts:688–700`; `packages/core/agent-loop/tests/request-reconstruction.spec.ts` |
| `JSON.stringify` equality footgun | Live concern | Moot: that code was deleted with the invariant | `f028f25667` |
| `SESSION_FORMAT_VERSION = 0` / "migrate or abandon" | Open risk | Resolved toward **migrate**: format 4, adjacent migration chain, V4 recorded as released | `packages/core/session/src/types.ts:70–89`; `docs/session-format-status.md` |
| Package explosion | 226 packages | **Worse**: 340 manifests, ~1.10M TS/TSX lines | `packages/*/*/package.json` |
| In-process `eval` self-modification | Research demo in `dsh-tool-cordis` | Narrowed: `cordis_define` moved to `dsh-cordis-host-runner` with human-approved client activation, web-app bundle only. "Do not port" stands. | `packages/extensions/cordis-host-runner/src/index.ts:1–5`; `packages/bundle/web-app/cordis.patch.yml` |
| Invariant-companion bureaucracy | Every package owns `./invariant` | Removed; other gates (per-file 100% coverage, doc budgets, bilingual docs) remain | `f028f25667`; `vitest.config.ts:377–387`; `AGENTS.md:173` |

Unchanged: seams, waterfall phases, turn/step vocabulary, fail-closed sandbox,
profiles-as-patches, `AgentFactory` dependency direction, "do not port Cordis".

---

## Verdict Summary

DeepSeek Harness is the **best written harness architecture** in this survey and
the **worst template to copy at face value**. The loop is 705 lines. The session
log is the only model history. Every loop request is derived from the log,
deep-frozen, and branded; a test proves reconstruction (the runtime invariant
was removed in `f028f25667`). Capability seams mean swapping the filesystem backend
moves Bash and LSP with it. Sandbox missing → fail closed. Profiles are ordered
patches. ACP is a thin adapter over `ctx.agents`.

It takes **340 packages**, a vendored actor/DI framework, per-file 100%
coverage, generated catalogs, bilingual doc budgets, and a Typert type-graph
gateway to hold that architecture in TypeScript. That is an org-scale machine
for a product that still advertises breaking changes, though it now migrates
session logs (`SESSION_FORMAT_VERSION = 4`).

**Import the log-centric loop contract and the seam taxonomy. Do not import
Cordis, the package explosion, or the gate factory. A Go core can be an order
of magnitude smaller while keeping the ideas that actually make agents
correct.**

---

## 1. Effectiveness

### What dsh does well

**The LLM sees only the log.** `deriveMessages()` at the step boundary plus a
frozen, `markAgentLoopRequest`-branded request
(`packages/core/agent-loop/src/agent.ts:688–700`) is the strongest
correctness property of any agent here. dsh dropped the runtime `llm/stream`
check (`f028f25667`) and now relies on construction and
`tests/request-reconstruction.spec.ts`. Prime's kernel state and oh-my-pi's
eval namespace are hidden context channels. dsh forbids them at the dispatch
boundary.

> **Lesson for Harness:** Before every provider call, reconstruct the message
> list from `SessionStore` and compare. In Go this is a function and a test,
> not a Cordis companion plugin. This is the single highest-leverage steal.
> Keep the runtime compare even though dsh removed theirs: in Go it costs one
> structured equality per step and catches desyncs that tests miss.

**Named extension phases beat god objects.** `agent/pre-step`, `agent/request`,
`tools/pre|execute|post`, `agent/turn-stopping` are the hook surface. Prime and
oh-my-pi keep growing `AgentSession` because they lack this cut. Deep Agents
has middleware; dsh has waterfalls with mandatory `next()`. Same idea, tighter
lifecycle.

> **Lesson for Harness:** `Hook` in `ARCHITECTURE.md` should be these phases,
> not a grab-bag. Waterfall semantics (must delegate) prevent silent
> short-circuits. Serial `turn-stopping` is the right shape for "should we
> continue?"

**Capability seams match Go interfaces.** Service definition / provider /
consumer is exactly `interface` + adapter + `Tool`. The "one execution world"
rule — fs and subprocess share a backend so tools don't fork — is how swe-term
should treat sandbox vs local vs remote.

> **Lesson for Harness:** Do not let `BashTool` know about Landlock. The tool
> calls `Subprocess`. `Subprocess` is wrapped by `Sandbox`. Pointing both at a
> remote world is a provider swap.

**Turn/step as durable vocabulary.** ACP stop reasons, telemetry, compaction,
and retries all speak the same units. pi's loop is implicit about this; dsh
makes it events.

> **Lesson for Harness:** Emit `turn/start`, `step/start`, `step/end`,
> `turn/end` as first-class protocol events. Compaction and resume attach to
> those boundaries.

**Fail closed, fail loud.** Missing Landlock does not "just run bash." Missing
config does not skip a plugin. That is the opposite of pi's URL-substring
provider sniffing.

> **Lesson for Harness:** Policy denial and missing backends are errors.
> Silent degradation is how agents exfiltrate.

**Frontends are bundles.** Headless vs web is a patch stack on `dsh-base`.
swe-term's TUI vs print vs RPC should be the same relationship: one core, three
clients.

**ACP as a consumer of `AgentFactory`.** The loop package is not imported by
the protocol adapter. That is the right dependency direction for
`ARCHITECTURE.md`'s wire-protocol-friendly frontend/core boundary.

### What dsh gets wrong (for *our* purposes)

**The platform is the product.** 340 packages (226 two months earlier) is not modularity; it is
granularity past the point of human navigation. "Add a tool" means a package,
a seam role, a README Model Experience section, an invariant companion, a
REAL-composition test, a snapshot, and a catalog regen. Correct for a lab with
a gate factory. Fatal for a small Go harness.

> **Implication:** Collapse to a handful of Go packages. Extensions are
> adapters, not npm workspaces.

**Cordis is a runtime, not a plugin API.** `ctx` proxies, fibers, isolate,
intercept, declaration-merged event maps — this is a framework the team
vendored and locally patched. swe-term's rule is interfaces in Go, not an
in-process actor kernel.

> **Implication:** Do not port Cordis. Port the *extension point names* as Go
> interfaces and a simple hook runner.

**Replaceable loop, frozen event language.** You can swap `AgentFactory`, but
replay, UI, ACP, and invariants require the same `SessionEventMap`. That is
fine — be honest that the log schema *is* the core. dsh sometimes writes as if
the loop were the only replaceable piece.

**JSON.stringify equality** was the old reconstruction check; it was deleted
with the runtime invariant (`f028f25667`). The lesson stands for Go: use
canonical encoding or structured compare, not string equality.

**Self-modification via in-process `eval`.** `cordis_define` accepts JS
function bodies. It now sits in `dsh-cordis-host-runner` behind human-approved
client activation and only in the web-app bundle; it still contradicts swe-term's "no dynamic in-process loading"
and the lethal-trifecta lessons in the Claude critique.

---

## 2. Efficiency

### Small loop, huge tree

The 705-line driver is efficient *to read*. The 340-package build, tsdown dual
host/client faces, and catalog generators are not efficient *to change*.
Contributor latency is the real cost.

### 100% per-file coverage

CI requires statements/branches/functions/lines at 100% on every
`packages/*/*/src` file. That produces tests. It also produces tests that exist
to satisfy the gate. swe-term should cover the loop, the invariant, sandbox
fail-closed, and snapshot transcripts — not every util at 100%.

### Dual host/client Cordis trees

The web profile mounts a second plugin universe in the browser. A terminal-first
harness does not need this. ACP + TUI over one event stream is enough.

---

## 3. Simplicity

dsh's *concepts* are simple (log, seam, waterfall, profile). The
*implementation* is a bureaucracy designed to keep 340 packages from lying.
Word budgets, bilingual pairing, export JSDoc verification — admirable, and
hostile to a three-person core. (Per-package invariant companions were
removed in `f028f25667`.)

`SESSION_FORMAT_VERSION` has left 0 (now 4), and dsh chose to pay migration
costs: an adjacent migration chain, a released-format record
(`docs/session-format-status.md`), and per-release upgrade guides
(`docs/upgrade-guide/`). That confirms the prediction and its price. swe-term
should pick a small schema and version it from day one (PLAN already says
SQLite).

---

## 4. Key Gaps (relative to Harness goals)

| Gap | dsh | Harness |
|-----|-----|---------|
| Single binary | Node + native addons + Python SDK | Go core |
| Plugin model | Cordis fibers + YAML | Go interfaces + optional sidecars |
| Package count | 340 | core + extensions |
| Session store | Log + SQLite backends (good) | SQLite (align) |
| Tokenization | LLM-reported + compaction plugin | Real tokenizer for triggers |
| Content-addressable cache | Not a design center | `core/cache` |
| Analyzer | LSP seam exists; not pre-LLM packing | `Analyzer` before the loop |
| Dynamic plugins | In-process JS | Starlark / subprocess, not eval |

dsh is **ahead** of pi on session-log honesty, sandbox fail-closed, and ACP.
It is **behind** swe-term's intended deployment posture (single binary,
cloud-agnostic, no framework kernel).

---

## 5. Assumptions to NOT Port

### ❌ "Everything is a plugin, including the loop"

The loop should be a function. Plugins attach at named hooks. Making the driver
itself a YAML row is cute and makes the event schema the *actual* core anyway.

### ❌ "A DI/actor framework is required for replaceability"

Go interfaces + a registry are enough. Cordis solves TypeScript's lack of a
boring module system.

### ❌ "100% coverage and generated catalogs keep architecture honest"

They keep *this repo* honest at lab scale. They will not keep swe-term small.

### ❌ "The agent should mount new plugins into its own process"

Inspect, yes. `eval` of model-authored JS, no.

### ❌ "Host and client are two Cordis trees"

One core, many frontends. The browser is a frontend, not a second runtime.

---

## 6. What to Actually Port

| From dsh | To Harness | Why |
|----------|------------|-----|
| `deriveMessages()` + frozen branded request (dsh dropped its runtime check) | `SessionStore.Messages()` checked before `Provider.Stream` | Ghost context becomes a failed test |
| Turn/step durable events | Protocol event types | Resume, ACP, telemetry share units |
| Waterfall hook phases | `Hook` with must-continue | Extension without loop forks |
| Seam triple (def/provider/consumer) | Go interfaces | Swap sandbox world, not tools |
| Profile = ordered patches | Headless vs TUI as config layers | One binary, multiple compositions |
| Fail-closed sandbox argv wrap | `Sandbox` adapter | Missing confinement is an error |
| `AgentFactory` vs protocol adapters | ACP talks to core, not loop internals | Dependency direction |
| Branded ids | Distinct Go types | SessionId ≠ string |
| `steer` / `followup` / `inject` | Keep from pi; dsh names them cleanly | Three inbox lanes are enough |
| SQLite schema version ≠ log version | Two version numbers | Physical layout vs event semantics |

---

## 7. Architectural Contrasts

```
DeepSeek Harness                         Harness (swe-term)
────────────────────────────────────     ─────────────────────────────────
Cordis plugin kernel                     Go interfaces, no DI framework
340 packages, generated catalogs         Small core + extensions
705-line loop + huge plugin tree         Small loop, same idea
Log-derived frozen request + test        Runtime invariant, SQLite store
YAML profiles / bundles                  Flags + config file layers
Landlock/bwrap/Seatbelt fail-closed      Same policy, adapter-driven
ACP + JSON-RPC + web + Python SDK        ACP + TUI + print; SDK later
Human-gated cordis_define eval (web)     Sidecar / Starlark, not eval
SESSION_FORMAT_VERSION = 4 + migrations  Version from v1, migrate or refuse
Per-file 100% coverage                   Critical-path + snapshots
```

---

## 8. Final Assessment

Read dsh as a **spec for harness correctness**, not as a codebase to resemble.
The spec says: the log is the truth; extension happens at named phases;
capabilities are seams; sandboxes fail closed; frontends are adapters;
protocol clients must not import the loop.

The codebase says: enforcing that spec in TypeScript at lab scale requires an
operating system of packages and gates.

swe-term should implement the spec in Go in a few thousand lines, then add
features as extensions. If the Go port ever needs 340 packages to stay honest,
the architecture has already failed.
