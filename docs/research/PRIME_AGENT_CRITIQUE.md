# Critical Evaluation of Prime Agent as an Agentic Harness

> Evaluated against the Harness (swe-term) philosophy:
> zero-dependency core, interface-driven plugins, immutable state,
> goroutine-native concurrency, content-addressable caching, single binary.
>
> **Validated against the local Prime Agent source** at `prime-agent/` (`0.7.2`).
> Prime Agent is a pi-mono fork. Loop critiques that apply to pi still apply;
> this document focuses on what Prime *added*.

---

## Verdict Summary

Prime Agent is the most serious **long-running-agent** design in this survey. It
answers a question pi-mono left open: *what happens when the human disconnects,
the task spans hours, and the agent needs children, memory, and a programming
environment that outlives a turn?* The answers — detachable workers, a
capability-gated local protocol, a rollbackable harness, typed host requests,
admission-not-await subagents — are real engineering, not slideware.

It is also a **warning about gravitational collapse**. `AgentSession` grew from
pi's already-large orchestrator into an **11,288-line** god object. Interactive
mode is **10,024** lines. The daemon worker handler is **7,064**. The product
replaced a multi-tool LLM surface with a Jupyter kernel that is explicitly **not
a sandbox**. Default RLM depth is 1, which is the authors telling you recursive
subagents are a loaded gun.

**Steal the execution/UI split, the protocol versioning, the harness
snapshot/rollback, and the host-request idea. Do not steal IPython-as-the-loop
or the 11k-line session class. swe-term should remain a Go loop over
interfaces; Prime is a product built *on* that loop, not a replacement for it.**

---

## 1. Effectiveness

### What Prime Agent does well

**The client does not own execution.** Interactive, print, JSON, and RPC all
talk to a supervisor over a versioned local protocol. That is the same payoff
as Codex's protocol seam and Flue's headless-first stance, implemented as
"reattach to a running worker." For a harness that wants TUI *and* scriptable
frontends, this is the right cut.

> **Lesson for Harness:** `Frontend` is a client of the core, never the owner of
> the run. Attach/detach/resume are protocol operations. Phase 4 of `PLAN.md`
> already says this; Prime shows the production shape, including recovery
> snapshots and generation-aware event cursors.

**Capability-gated daemon protocol.** `DAEMON_PROTOCOL_VERSION = 7` plus
`DAEMON_SCHEMA_REVISION = 16` plus explicit client/server capability sets
(`daemon-protocol.ts:52–103`) is how you evolve a local wire without
breaking old TUIs. Optional features degrade. Startup does not require a new
command unless gated.

> **Lesson for Harness:** ACP/JSON-RPC should version like this. Additive
> fields behind a capability; incompatible changes bump the protocol. Do not
> silently reinterpret payloads.

**Typed host-request bridge.** Python can *ask*; TypeScript *decides*. That
keeps credentials, transcripts, and child lifecycles out of the REPL while
still giving the model a programmatic surface. It is a better security story
than "the sandbox can call every tool" (oh-my-pi's eval bridge) because the
host enumerates request types.

> **Lesson for Harness:** If swe-term ever grows a code-exec sidecar, the
> sidecar speaks a closed host-RPC, not the full `Tool` registry. Go owns
> spawn, approvals, and persistence.

**Admission-not-await subagents.** `rlm()` returning a handle instead of a
blocking result is the correct concurrent primitive. Fan-out is just more
calls; results are messages. Combined with a spawn ledger, the parent can
reconstruct the family after compaction.

> **Lesson for Harness:** Subagent APIs should return job IDs. Blocking
> `run_subagent()` as the only interface serializes work the model wanted
> parallel. Pair it with a message bus and a depth cap (Prime's default of 1
> is conservative and correct for v1).

**Continual harness with rollback.** Separating the immutable base prompt from
reviewable supplemental state is the right memory model. Atomic apply + inverse
edits from `before` snapshots is operationally adult. This is closer to "the
agent can improve its operating manual" than to dumping thoughts into a
markdown file.

> **Lesson for Harness:** If swe-term grows durable memory, store it as
> versioned records with rollback, not as prompt concatenation. Keep the base
> system prompt a compile-time/config artifact the agent cannot rewrite.

**Catalog process.** Listing sessions without loading `AgentSession` is an
obvious idea that most agents skip until it hurts.

> **Lesson for Harness:** Session index (SQLite, already in PLAN) exists
> precisely so `st agents` does not deserialize 11k-line runtimes.

**Autonomous quality gates.** Continuation is bounded by turns/tokens/time
*and* optional shell gates. A passed gate checks only what that gate verifies
— the README is careful not to equate "hit the limit" with "task succeeded."

### What Prime Agent gets wrong (for *our* purposes)

**IPython as the only LLM tool is a product bet, not a harness primitive.**
It requires uv, ipykernel, ZMQ, dill snapshots, a Python wheel, and a
bootstrap path. Every file/shell/edit skill is now "Python the model wrote."
That is powerful for research agents and hostile to a Go single-binary with
thin sidecars. swe-term's tools should remain first-class `Tool` implementations;
code-exec can be *a* tool, not *the* tool.

> **Implication:** Do not port the RLM programming model as the default loop.
> Offer an optional eval sidecar later. Keep bash/read/edit as native tools.

**`AgentSession` is now worse than pi's.** pi's critique already called out a
3,060-line god object. Prime's is **11,288** lines and owns compaction, RLM,
goals, refine, messaging, kernel, and child lifecycles. `AgentSessionRuntime`
(802 lines) is a start at extraction and is not enough.

> **Harness fix:** Same as the pi critique, more urgently: the loop is a
> function over interfaces. Session, compaction, scheduler, subagent registry,
> and kernel (if any) are separate types.

**The daemon is a product, not a v1 requirement.** Three processes, recovery
journals, leases, idle eviction, catalog subprocess — justified for
"research evals that run overnight." Overkill for swe-term Phase 1–4. Build
the *protocol seam* first; add a resident supervisor when detach is a real
user need.

**Dual harness stores.** TypeScript `refinement.ts` and Python `harness.py`
both speak the state model. Two writers, one JSON file. That will desync.

---

## 2. Efficiency

### Kernel + host round-trips

Every privileged operation is a comm hop. Fine for spawn/goal/refine; painful
if the model uses IPython as a slow `cat`. Summarization and truncation live
in the TypeScript tool layer for ipython output, but the model can still dump
a dataframe into context.

> **Harness fix:** Native tools with hard output caps (oh-my-pi's summarized
> reads are the better token story). Code-exec results are artifacts, not
> prompt stuffing, unless the model asked for a slice.

### JSONL everywhere, no index

Same pi gap, now with more files: sessions, refinements, spawn ledger, logs.
Catalog process compensates for listing; it does not make branch navigation
O(1).

> **Harness fix:** SQLite for session metadata and topology. Keep JSONL as an
> export/debug format if you want human-readable transcripts.

### Compaction vs kernel state

Kernel state surviving compaction is a genuine win — the model can keep
variables while the chat is summarized. It is also a hidden context channel:
the REPL holds facts the transcript no longer does, so replay from JSONL
alone is incomplete.

> **Harness fix:** If a sidecar holds state, snapshot it into the session
> store on compact, or treat it as ephemeral and say so. "Model-visible means
> logged" (DeepSeek) is the invariant Prime violates by design.

---

## 3. Simplicity

Prime Agent is pi-mono plus a distributed runtime plus a Jupyter control
plane. The *ideas* are simple. The *files* are not.

| File | Lines | Smell |
|------|------:|-------|
| `agent-session.ts` | 11,288 | Every feature landed as a method |
| `interactive-mode.ts` | 10,024 | UI + slash commands + attach |
| `daemon-mode.ts` | 7,064 | Worker command switchboard |
| `daemon-supervisor.ts` | 5,236 | Fleet + recovery + eviction |

Extensions remain jiti `import()` — same pi gap, now running next to a
privileged kernel.

---

## 4. Key Gaps (relative to Harness goals)

| Gap | Prime | Harness |
|-----|-------|---------|
| Interfaces between layers | Concrete `AgentSession` | `Provider` / `Tool` / `SessionStore` / `Hook` |
| Single binary | Node + Python kernel + uv | Go core, optional sidecars |
| Sandbox | Example-only | Policy-gated, adapter-driven |
| Token accounting for compact | pi's heuristic heritage | Real tokenizer |
| Content-addressable cache | None | `core/cache` |
| Analyzer / pre-LLM enrichment | None | `Analyzer` interface |
| Session index | Catalog process + JSONL scan | SQLite |

---

## 5. Assumptions to NOT Port

### ❌ "The model should program the harness in Python"

A research-agent bet. swe-term's default surface is typed tools + optional
code-exec, not a REPL that *is* the product.

### ❌ "Lifecycle isolation is isolation"

Workers/kernels are crash domains. They are not sandboxes. Do not market
process splits as security.

### ❌ "One session object can absorb RLM, goals, refine, messaging, and the kernel"

It cannot. Prime is the existence proof.

### ❌ "Deep recursive subagents are the architecture"

Default depth 1. Treat multi-agent as a tree with a hard cap, not as unbounded
recursion.

### ❌ "Harness state can have two writers"

Pick one owner. In swe-term that owner is Go.

---

## 6. What to Actually Port

| From Prime Agent | To Harness | Why |
|------------------|------------|-----|
| Client ≠ runtime | `Frontend` over protocol | Detach, headless, TUI share one loop |
| Protocol version + schema revision + capabilities | ACP/JSON-RPC evolution | Additive features without silent breaks |
| Host-request closed RPC | Sidecar → core adapter | Privileged ops stay in Go |
| Subagent admission handles + depth cap | Job IDs, max depth 1 default | Parallelism without await-all |
| Spawn ledger separate from transcript | Subagent table in SQLite | Topology recoverable after compact |
| Harness snapshot + rollback | Versioned memory records | Agent-editable ops manual, reversible |
| Catalog/index without loading runtimes | Session index queries | `st agents` stays cheap |
| Autonomous bounds + quality gates | Optional continuation policy | Overnight work without pretending success |
| steer vs follow-up inbox | Already in pi loop; keep it | Interrupt vs queue is the right pair |

---

## 7. Architectural Contrasts

```
Prime Agent                              Harness (swe-term)
────────────────────────────────────     ─────────────────────────────────
pi-mono fork + IPython control plane     Go core port of pi layering
LLM tool: ipython only                   Native Tool set; eval optional
AgentSession 11,288 ln                   Split Session/Compaction/Registry
Daemon supervisor + workers              Protocol seam first; daemon later
JSONL + catalog process                  SQLite index + optional JSONL export
host_request from Python kernel          sidecar RPC, closed method set
Continual harness (TS + Python writers)  Single-owner versioned records
Not a sandbox (documented)               Policy + adapter sandbox
Default RLM depth 1                      Same cap; no recursive default
```

---

## 8. Final Assessment

Prime Agent is **pi-mono after it got a job**: long-running evals, detach,
multi-agent, self-refinement. The job pulled the architecture toward a
resident runtime and a REPL, and the session class ate the universe.

swe-term should take the **job description**, not the **org chart**. We want
detachable execution, versioned wire, recoverable subagent topology, and
reviewable memory. We do not want Jupyter in the hot path or an 11k-line
orchestrator. Keep the Go loop small; grow Prime's product features as
extensions behind the interfaces already in `ARCHITECTURE.md`.
