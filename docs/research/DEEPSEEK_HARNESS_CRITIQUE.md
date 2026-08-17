# Critical Evaluation of DeepSeek Harness as an Agentic Harness

> Evaluated against the Harness (swe-term) philosophy:
> zero-dependency core, interface-driven plugins, immutable state,
> goroutine-native concurrency, content-addressable caching, single binary.
>
> **Validated against the local dsh source** at `deepseek-harness/` (`0.1.0-rc.5`).
> This is a pre-release composition platform, not a small coding loop.

---

## Verdict Summary

DeepSeek Harness is the **best written harness architecture** in this survey and
the **worst template to copy at face value**. The loop is 496 lines. The session
log is the only model history. A runtime invariant refuses to send a request the
log cannot reconstruct. Capability seams mean swapping the filesystem backend
moves Bash and LSP with it. Sandbox missing → fail closed. Profiles are ordered
patches. ACP is a thin adapter over `ctx.agents`.

It takes **226 packages**, a vendored actor/DI framework, per-file 100%
coverage, generated catalogs, bilingual doc budgets, and a Typert type-graph
gateway to hold that architecture in TypeScript. That is an org-scale machine
for a product that still pins `SESSION_FORMAT_VERSION = 0` and advertises
breaking changes.

**Import the log-centric loop contract and the seam taxonomy. Do not import
Cordis, the package explosion, or the gate factory. A Go core can be an order
of magnitude smaller while keeping the ideas that actually make agents
correct.**

---

## 1. Effectiveness

### What dsh does well

**The LLM sees only the log.** `deriveMessages()` plus the `llm/stream`
invariant (`packages/core/agent-loop/src/invariant.ts`) is the strongest
correctness property of any agent here. Prime's kernel state and oh-my-pi's
eval namespace are hidden context channels. dsh forbids them at the dispatch
boundary.

> **Lesson for Harness:** Before every provider call, reconstruct the message
> list from `SessionStore` and compare. In Go this is a function and a test,
> not a Cordis companion plugin. This is the single highest-leverage steal.

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

**The platform is the product.** 226 packages is not modularity; it is
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

**JSON.stringify equality** for the reconstruction check is pragmatic in TS
and a footgun (key order, undefined). Go should use canonical encoding or
structured compare.

**Self-modification via in-process `eval`.** `cordis_define` accepts JS
function bodies. That contradicts swe-term's "no dynamic in-process loading"
and the lethal-trifecta lessons in the Claude critique.

---

## 2. Efficiency

### Small loop, huge tree

The 496-line driver is efficient *to read*. The 226-package build, tsdown dual
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
*implementation* is a bureaucracy designed to keep 226 packages from lying.
Word budgets, bilingual pairing, export JSDoc verification, package invariant
companions — admirable, and hostile to a three-person core.

Pre-release "foundation over blast radius" is the tell: they are still allowed
to rename everything. Once `SESSION_FORMAT_VERSION` leaves 0, this machine
either pays migration costs or abandons users. swe-term should pick a small
schema and version it from day one (PLAN already says SQLite).

---

## 4. Key Gaps (relative to Harness goals)

| Gap | dsh | Harness |
|-----|-----|---------|
| Single binary | Node + native addons + Python SDK | Go core |
| Plugin model | Cordis fibers + YAML | Go interfaces + optional sidecars |
| Package count | 226 | core + extensions |
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
| `deriveMessages()` + dispatch invariant | `SessionStore.Messages()` checked before `Provider.Stream` | Ghost context becomes a failed test |
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
226 packages, generated catalogs         Small core + extensions
496-line loop + huge plugin tree         Small loop, same idea
Session log + reconstruction invariant   Same invariant, SQLite store
YAML profiles / bundles                  Flags + config file layers
Landlock/bwrap/Seatbelt fail-closed      Same policy, adapter-driven
ACP + JSON-RPC + web + Python SDK        ACP + TUI + print; SDK later
In-process cordis_define eval            Sidecar / Starlark, not eval
SESSION_FORMAT_VERSION = 0               Version from v1, migrate or refuse
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
features as extensions. If the Go port ever needs 226 packages to stay honest,
the architecture has already failed.
