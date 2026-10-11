# Critical Evaluation of Prime Agent as an Agentic Harness

> Evaluated against the Harness (swe-term) philosophy:
> zero-dependency core, interface-driven plugins, immutable state,
> goroutine-native concurrency, content-addressable caching, single binary.
>
> **Validated against the local Prime Agent source** at `prime-agent/` version
> **0.10.0**, HEAD `2ee7e62` (2026-10-10). Previously validated at 0.7.2 (`2c34b82`,
> 2026-08-16). Since then Prime was ported from TypeScript to Rust (`39bc99a`,
> 2026-09-29) and its Jupyter kernel was replaced by a minimal CPython REPL
> (`61eb647`). Prime began as a pi-mono fork; the lineage now survives in layering
> and on-disk formats, not shared code. See
> [`PRIME_AGENT_DEEP_DIVE.md`](PRIME_AGENT_DEEP_DIVE.md) §Changes since 0.7.2.

## Verdict changes (2026-10-10)

| Verdict | 0.7.2 | 0.10.0 | Why |
|---------|-------|--------|-----|
| God-object session class | **Avoid** — 11,288-ln `AgentSession` | **Resolved upstream**; lesson kept | Rust `SessionEngine` is 971 ln in a 61-file module; largest non-generated file is 2,235 ln (`crates/pa-daemon/src/acp/daemon.rs`) |
| IPython-as-the-loop | Avoid — Jupyter/ZMQ stack | **Avoid (unchanged), cost lower** | Kernel is now `rlm.repl` over NDJSON; `ipython` is still the only LLM tool (`session_engine/engine.rs:432–443`) |
| Three-process daemon as v1 | Avoid | **Avoid (unchanged)**, now client + supervisor + worker | Catalog subprocess removed (`crates/pa-daemon/src/main.rs:15–43`) |
| Dual harness writers | Avoid — "will desync" | **Avoid (unchanged), risk reduced** | Writers now share a cross-process lock (`97b4a03`); two schema owners remain |
| Default depth 1 is "correct for v1" | Port depth 1 | **Port depth 1; note Prime moved to 2** | `9e49b73`; `DEFAULT_RLM_MAX_DEPTH = 2` |
| Admission-not-await | Port | **Port, strengthened** | `rlm.collect(timeout_ms)` adds a bounded typed join |
| Protocol versioning | Port | **Port, strengthened** | Schema 16 → 30 and a full language rewrite without a protocol bump |
| Extensions next to a privileged kernel | Gap | **Removed upstream** | `f54551d` deleted user extensions |

---

## Verdict Summary

Prime Agent is the most serious **long-running-agent** design in this survey. It
answers a question pi-mono left open: *what happens when the human disconnects,
the task spans hours, and the agent needs children, memory, and a programming
environment that outlives a turn?* The answers — detachable workers, a
capability-gated local protocol, a rollbackable harness, typed host requests,
admission-not-await subagents — are real engineering, not slideware.

It was also a **warning about gravitational collapse**. At 0.7.2 `AgentSession`
was an **11,288-line** god object, and it reached 15,101 lines before the Rust
port (`39bc99a^`). The port dissolved it: the largest non-generated Rust file is
now 2,235 lines. The lesson stands — Prime needed a full rewrite to escape the
class. The product still replaces a multi-tool LLM surface with a Python REPL that
is explicitly **not a sandbox** (`README.md:87`). Default RLM depth moved from 1 to
2 (`9e49b73`): the authors loosened the cap, but still cap it.

**Steal the execution/UI split, the protocol versioning, the harness
snapshot/rollback, and the host-request idea. Do not steal REPL-as-the-only-tool
or grow a god session class (Prime had to rewrite to escape one). swe-term should remain a Go loop over
interfaces; Prime is a product built *on* that loop, not a replacement for it.**

---

## 1. Effectiveness

### What Prime Agent does well

**The client does not own execution.** Interactive, print, JSON, RPC, and ACP all
talk to a supervisor over a versioned local protocol. That is the same payoff
as Codex's protocol seam and Flue's headless-first stance, implemented as
"reattach to a running worker." For a harness that wants TUI *and* scriptable
frontends, this is the right cut.

> **Lesson for Harness:** `Frontend` is a client of the core, never the owner of
> the run. Attach/detach/resume are protocol operations. Phase 4 of `PLAN.md`
> already says this; Prime shows the production shape, including recovery
> snapshots and generation-aware event cursors.

**Capability-gated daemon protocol.** `DAEMON_PROTOCOL_VERSION = 7` plus
`DAEMON_SCHEMA_REVISION = 30` plus client/server capability strings
(`crates/pa-types/src/daemon/mod.rs:12–25`) is how you evolve a local wire
without breaking old clients. Between 0.7.2 and 0.10.0 the schema moved 16 → 30
and the host was rewritten in Rust with byte parity, all without a protocol
bump. Optional features degrade.

> **Lesson for Harness:** ACP/JSON-RPC should version like this. Additive
> fields behind a capability; incompatible changes bump the protocol. Do not
> silently reinterpret payloads.

**Typed host-request bridge.** Python can *ask*; the Rust host *decides*. That
keeps credentials, transcripts, and child lifecycles out of the REPL while
still giving the model a programmatic surface. It is a better security story
than "the sandbox can call every tool" (oh-my-pi's eval bridge) because the
host enumerates request types: an unregistered type is rejected
(`crates/pa-core/src/kernel/manager/host_requests.rs:70–85`).

> **Lesson for Harness:** If swe-term ever grows a code-exec sidecar, the
> sidecar speaks a closed host-RPC, not the full `Tool` registry. Go owns
> spawn, approvals, and persistence.

**Admission-not-await subagents.** `rlm.spawn()` returning a handle instead of
a blocking result is the correct concurrent primitive. Fan-out is just more
calls. Since 0.7.2, `rlm.collect(targets, timeout_ms)` adds an explicit typed
join that blocks only the kernel call and never steers the parent
(`prime-agent-runtime/src/rlm/__init__.py:416–440`). Combined with a spawn
ledger, the parent can reconstruct the family after compaction.

> **Lesson for Harness:** Subagent APIs should return job IDs. Blocking
> `run_subagent()` as the only interface serializes work the model wanted
> parallel. Pair it with a bounded `collect`, a message bus, and a depth cap.
> Prime raised its default from 1 to 2 (`9e49b73`); 1 is still the conservative
> choice for swe-term v1.

**Continual harness with rollback.** Separating the immutable base prompt from
reviewable supplemental state is the right memory model. Atomic apply + inverse
edits from `before` snapshots is operationally adult. This is closer to "the
agent can improve its operating manual" than to dumping thoughts into a
markdown file.

> **Lesson for Harness:** If swe-term grows durable memory, store it as
> versioned records with rollback, not as prompt concatenation. Keep the base
> system prompt a compile-time/config artifact the agent cannot rewrite.

**Catalog without loading runtimes.** Listing sessions without loading a
session engine is an obvious idea that most agents skip until it hurts. Prime
folded the catalog subprocess into the supervisor
(`crates/pa-daemon/src/session_catalog.rs`; `pa-daemon` is now
`supervisor | worker`, `main.rs:15–43`). The property survived; the process did
not.

> **Lesson for Harness:** Session index (SQLite, already in PLAN) exists
> precisely so `st agents` does not deserialize session runtimes.

**Autonomous quality gates.** Continuation is bounded by turns/tokens/time
*and* optional shell gates. A passed gate checks only what that gate verifies
— the README is careful not to equate "hit the limit" with "task succeeded."

### What Prime Agent gets wrong (for *our* purposes)

**IPython as the only LLM tool is a product bet, not a harness primitive.**
At 0.7.2 it required uv, ipykernel, ZMQ, dill, a Python wheel, and a
bootstrap path. Since `61eb647` the kernel is a minimal CPython REPL over NDJSON
(`prime-agent-runtime/src/rlm/repl.md`), so ipykernel and ZMQ are gone. A Python
environment, provisioner (`crates/pa-core/src/kernel/provisioner.rs`, 1,784 ln),
and wheel remain. Every file/shell/edit skill is now "Python the model wrote."
That is powerful for research agents and hostile to a Go single-binary with
thin sidecars. swe-term's tools should remain first-class `Tool` implementations;
code-exec can be *a* tool, not *the* tool.

> **Implication:** Do not port the RLM programming model as the default loop.
> Offer an optional eval sidecar later. Keep bash/read/edit as native tools.

**`AgentSession` was worse than pi's — until a rewrite.** pi's critique called
out a 3,060-line god object. Prime's TS class was **11,288** lines at 0.7.2 and
15,101 just before the port. *Verdict changed:* the Rust port split it into
`SessionEngine` (`crates/pa-core/src/session_engine/engine.rs`, 971 ln) plus
60 sibling modules (compaction, goal driver, RLM host, refine, factory host) and
a daemon-side `agent_engine/`. The complexity did not shrink — `session_engine/`
alone is 28,338 lines — but it now has seams. It took a whole-product rewrite to
get them.

> **Harness fix:** Unchanged, and now with Prime's own evidence: the loop is a
> function over interfaces. Session, compaction, scheduler, subagent registry,
> and kernel (if any) are separate types from day one.

**The daemon is a product, not a v1 requirement.** Client, supervisor, and
worker processes, a worker recovery journal, launch budgets, roster, idle
eviction — justified for
"research evals that run overnight." Overkill for swe-term Phase 1–4. Build
the *protocol seam* first; add a resident supervisor when detach is a real
user need.

**Dual harness stores.** Rust `refinement/` and Python `harness.py` both write
the state model. Two writers, one JSON file. Since `97b4a03` (2026-10-07) they
share a cross-process lock directory (`crates/pa-core/src/platform/lock_dir.rs`),
so concurrent writes no longer race. The schema still has two implementations
in two languages, so drift remains the risk.

---

## 2. Efficiency

### Kernel + host round-trips

Every privileged operation is a comm hop. Fine for spawn/goal/refine; painful
if the model uses IPython as a slow `cat`. Summarization and truncation live
in the host tool layer for ipython output, but the model can still dump
a dataframe into context.

> **Harness fix:** Native tools with hard output caps (oh-my-pi's summarized
> reads are the better token story). Code-exec results are artifacts, not
> prompt stuffing, unless the model asked for a slice.

### JSONL everywhere, no index

Same pi gap, now with more files: sessions, refinements, spawn ledger, logs.
No SQLite in any crate. The in-supervisor catalog compensates for listing; it does not make branch navigation
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

Prime Agent is a pi-shaped harness plus a distributed runtime plus a Python
control plane. The *ideas* are simple. At 0.7.2 the *files* were not:

| File (0.7.2, TS) | Lines | Smell | 0.10.0 (Rust) |
|------|------:|-------|------|
| `agent-session.ts` | 11,288 | Every feature landed as a method | `session_engine/`: 61 files, 28,338 ln, max 1,502 |
| `interactive-mode.ts` | 10,024 | UI + slash commands + attach | `pa-tui/src/interactive/`: 5 files, 3,789 ln |
| `daemon-mode.ts` | 7,064 | Worker command switchboard | `pa-daemon/src/worker/`: 18 files, 9,973 ln |
| `daemon-supervisor.ts` | 5,236 | Fleet + recovery + eviction | `pa-daemon/src/supervisor/`: 14 files, 8,725 ln |

*Verdict changed:* the file-size smell is gone; total size is not. The Rust
workspace is 378,314 non-test lines (568,985 with tests) across 9 crates.

User extensions were deleted (`f54551d`), which removes the 0.7.2 concern about
jiti `import()` code running next to a privileged kernel.

---

## 4. Key Gaps (relative to Harness goals)

| Gap | Prime | Harness |
|-----|-------|---------|
| Interfaces between layers | Crates + traits (e.g. `HostRequestHandler`), concrete `SessionEngine` | `Provider` / `Tool` / `SessionStore` / `Hook` |
| Single binary | `pa-cli` + `pa-daemon` + Python REPL runtime | Go core, optional sidecars |
| Sandbox | None (example extension removed with extensions) | Policy-gated, adapter-driven |
| Token accounting for compact | pi's heuristic heritage | Real tokenizer |
| Content-addressable cache | None | `core/cache` |
| Analyzer / pre-LLM enrichment | None | `Analyzer` interface |
| Session index | In-supervisor catalog + JSONL | SQLite |

---

## 5. Assumptions to NOT Port

### ❌ "The model should program the harness in Python"

A research-agent bet. swe-term's default surface is typed tools + optional
code-exec, not a REPL that *is* the product.

### ❌ "Lifecycle isolation is isolation"

Workers/kernels are crash domains. They are not sandboxes. Do not market
process splits as security.

### ❌ "One session object can absorb RLM, goals, refine, messaging, and the kernel"

It cannot. Prime is the existence proof: the class reached 15,101 lines and was
only split by a full Rust rewrite.

### ❌ "Deep recursive subagents are the architecture"

Prime's default is now 2 (`9e49b73`), still capped. Treat multi-agent as a tree with a hard cap, not as unbounded
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
| Subagent admission handles + bounded collect + depth cap | Job IDs, `collect(timeout)`, max depth 1 default | Parallelism without await-all |
| Spawn ledger separate from transcript | Subagent table in SQLite | Topology recoverable after compact |
| Harness snapshot + rollback | Versioned memory records | Agent-editable ops manual, reversible |
| Catalog/index without loading runtimes | Session index queries | `st agents` stays cheap |
| Rewrite behind a frozen wire/disk contract | Golden wire/disk fixtures | Prime ported TS → Rust with byte parity and no protocol bump |
| Autonomous bounds + quality gates | Optional continuation policy | Overnight work without pretending success |
| steer vs follow-up inbox | Already in pi loop; keep it | Interrupt vs queue is the right pair |

---

## 7. Architectural Contrasts

```
Prime Agent                              Harness (swe-term)
────────────────────────────────────     ─────────────────────────────────
Rust rewrite of pi layering + Python REPL Go core port of pi layering
LLM tool: ipython only                   Native Tool set; eval optional
SessionEngine split into 61 modules      Split Session/Compaction/Registry
Daemon supervisor + workers              Protocol seam first; daemon later
JSONL + in-supervisor catalog            SQLite index + optional JSONL export
host_request from Python REPL            sidecar RPC, closed method set
Continual harness (Rust + Python, lock)  Single-owner versioned records
Not a sandbox (documented)               Policy + adapter sandbox
Default RLM depth 2                      Depth 1 cap; no recursive default
```

---

## 8. Final Assessment

Prime Agent is **pi-mono after it got a job**: long-running evals, detach,
multi-agent, self-refinement. The job pulled the architecture toward a
resident runtime and a REPL, and the session class ate the universe — until a
548-commit, TS → Rust rewrite broke it back into modules while holding the
wire and disk formats fixed.

swe-term should take the **job description**, not the **org chart**. We want
detachable execution, versioned wire, recoverable subagent topology, and
reviewable memory. We do not want a Python REPL in the hot path or an orchestrator
that needs a rewrite to split. Keep the Go loop small; grow Prime's product features as
extensions behind the interfaces already in `ARCHITECTURE.md`.
