# Critical Evaluation of Deep Agents as an Agentic Harness

> Evaluated against the Harness (swe-term) philosophy:
> zero-dependency core, interface-driven plugins, immutable state,
> goroutine-native concurrency, content-addressable caching, single binary.
>
> **Validated against the local Deep Agents source** at `/home/rohit/sandbox/swe-term/deepagents`
> (core `0.6.7`, `deepagents-acp 0.0.6`, `deepagents-evals` on `harbor>=0.6.4`). Per the user's
> focus, the **ACP integration** (`libs/acp`) and the **Harbor eval suite** (`libs/evals`) get
> dedicated scrutiny below. External references:
> [ACP](https://agentclientprotocol.com/protocol/overview), [Harbor](https://www.harborframework.com/docs).

---

## Verdict Summary

Deep Agents is the **best-engineered "harness as composition" design** in this survey, and the
single most useful one to learn *organizing principles* from — while being almost the **opposite** of
swe-term's deployment philosophy at the substrate level.

Its thesis is sharp and correct: **a harness is a curated bundle of middleware over a generic agent
loop.** It does not write an agent loop at all. `create_deep_agent` assembles ~11 `AgentMiddleware`
(todos, filesystem, sub-agents, summarization, skills, HITL, …) and hands them to LangChain's
`create_agent`, which compiles a LangGraph `CompiledStateGraph`. Capabilities are *layers*, not
forks. That is exactly the modularity swe-term wants — and Deep Agents demonstrates it works at the
118-eval, multi-provider, ACP-and-Harbor-integrated scale.

But as an *architectural template for a single Go binary*, Deep Agents is a cautionary tale of
**dependency depth**. The "core" is three frameworks deep (LangGraph → `create_agent` → deepagents)
before you reach a line of Deep Agents' own logic. Its state model, streaming, checkpointing,
persistence, and even its agent loop are LangGraph's. Its strongest features (delta-reduced
checkpoints, pluggable stores, interrupt-based HITL) are **LangGraph features Deep Agents configures**,
not invents. You cannot port Deep Agents; you can only port LangGraph's ideas, re-expressed.

**Steal the middleware-composition model, the pluggable-backend protocol, the two-tier eval scoring,
and the "agent-as-Harbor-BaseAgent / agent-as-acp.Agent" adapter pattern. Do not adopt the
three-framework dependency stack, the Python/asyncio concurrency model, or the LangSmith-coupled
observability as your core.**

---

## 1. Effectiveness

### What Deep Agents does well

**It correctly identifies that "harness" is a composition problem.** Where Claude Code, Codex, and
pi-mono each *own* a loop and bolt features onto it, Deep Agents declares the loop a commodity and
competes purely on what middleware you stack. `create_deep_agent` is ~600 lines of *assembly* — pick
model, resolve a per-model profile, order the middleware, call `create_agent`. This is the most
honest expression of "the harness is the configuration" of any agent here.

> **Lesson for Harness:** swe-term's plugin/interface design is the Go analogue. Deep Agents proves
> the model scales: filesystem, summarization, skills, sub-agents, and HITL are each independently
> togglable middleware. swe-term's equivalent — `Tool`, `Frontend`, `Sandbox`, `ContextManager`
> interfaces composed at startup — should aim for the same "add a capability = add a plugin, never
> fork the loop" property.

**The pluggable-backend protocol is the cleanest filesystem abstraction in the survey.** One
`BackendProtocol` (`ls`/`read`/`write`/`edit`/`glob`/`grep` + a `SandboxBackendProtocol.execute`
extension) is satisfied by an in-state virtual FS, real disk, a cross-thread store, a path-routed
composite, and remote sandboxes (Runloop/Modal/Daytona) alike. The agent sees only tools; the backend
decides where bytes live.

> **Lesson for Harness:** This is *exactly* swe-term's "pluggable filesystem" goal, validated.
> `CompositeBackend` routing `/memories/` → store and everything else → sandbox is a particularly
> good pattern to copy as a Go interface. Make `read_file`/`write_file` route through a
> `Filesystem` interface from day one.

**Context overflow spills to the agent's own filesystem.** Summarized history goes to
`/conversation_history/{thread_id}.md`; oversized tool results (>20K tokens) go to
`/large_tool_results/<id>` with a reference left in the transcript. The same backend that serves the
agent's files is its context-management substrate.

> **Lesson for Harness:** A genuinely good idea worth porting wholesale. swe-term's context manager
> should offload large tool outputs to the (content-addressable) filesystem and pass a handle, not
> the bytes. This is cheaper *and* more debuggable than in-memory truncation.

**Sub-agents fall out of state-scoping, and a compiled graph can *be* a sub-agent.** `task` spawns a
child with the parent's `messages`/`todos` stripped, merges back a single `ToolMessage`. A
hand-built LangGraph plugs in as a `CompiledSubAgent`. Delegation and composition are the same seam.

> **Lesson for Harness:** swe-term's sub-agent model should likewise be "spawn a child harness with
> a fresh, isolated context; return one summarized result to the parent." The "any compiled agent is
> a valid sub-agent" property is the right north star for a `SubAgent` interface.

### What Deep Agents gets wrong (for *our* purposes)

**There is no "core" to study — there are three.** The README's own FAQ admits it:
LangGraph is the runtime, `create_agent` is the loop, Deep Agents is middleware. Reading
`graph.py` teaches you *assembly order*, not how an agent works. The interesting mechanics
(checkpoint channels, the ReAct loop, interrupts, streaming) are in two upstream packages. For a
project whose value proposition is "study the great harnesses and build a clean Go core," Deep
Agents offers the *taxonomy* of features but not an *implementation* you can read end-to-end.

**"Trust the LLM" security pushes the entire safety burden to the tool/sandbox layer.** That's a
defensible stance for a hosted Python framework with cloud sandboxes — but the in-tree default
(`LocalShellBackend`) runs commands on the host with **no isolation**. Real isolation requires a
paid partner (Runloop/Modal/Daytona). swe-term's threat model (a layered, in-process execution
guard *before* you reach a sandbox) is stronger by default.

---

## 2. Efficiency

### Python/asyncio, not goroutines

The concurrency model is `asyncio`. Sub-agents, streaming, and the ACP server are all
single-event-loop coroutines. Parallel sub-agent fan-out exists but is bounded by the GIL for any
CPU-bound work and by asyncio's cooperative scheduling. This is fine for an I/O-bound LLM agent, but
it is the antithesis of swe-term's "goroutine-native concurrency with `errgroup`/`context.Context`"
goal. There is nothing to port here except the *shape* of the concurrency (isolated child contexts),
which Go does more cheaply.

### Approximate token counting

Like pi-mono's `chars/4`, context thresholds use `count_tokens_approximately` — a heuristic, not a
real tokenizer. Summarization triggers and the 20K-token eviction threshold are therefore
*estimates*. Acceptable, but the same caveat applies: budget decisions ride on an approximation, and
provider-specific tokenization drift is unmodeled.

### Checkpoint growth is solved — by LangGraph

The `DeltaChannel` reducer that keeps checkpoint growth O(N) instead of O(N²) is a genuinely good
optimization — and it is a **LangGraph channel feature** Deep Agents opts into, not something it
built. Worth knowing the pattern exists; not portable code.

### Cost and determinism of the eval suite

The behavioral suite runs real LLMs against ~118 evals across up to 29 models, plus an LLM-as-judge
(`claude-sonnet-4-6`) that adds its own cost and variance. The multi-trial workflow exists precisely
because results are non-deterministic; it measures stdev rather than asserting reproducibility.
Harbor/Terminal-Bench 2.0 at full concurrency (Daytona default 40) is expensive. This is honest, but
it means "did we regress?" is a *statistical* question with a real dollar cost — not a fast,
deterministic `go test`.

---

## 3. Simplicity

### The dependency stack is deep and wide

The core package pulls `langchain-core`, `langchain`, `langchain-anthropic`,
`langchain-google-genai`, `wcmatch`, and transitively the entire LangGraph stack. The full
experience (code TUI, MCP, sandboxes, evals, ACP) pulls Textual, `langchain-mcp-adapters`, partner
SDKs, `harbor`, `langsmith`, `openevals`, `modal`, and `agent-client-protocol`. This is the
**maximal-dependency** end of the spectrum — the exact opposite of swe-term's "zero-dependency core,
single static binary" tenet. Deploying Deep Agents means shipping a Python environment and a
dependency tree; deploying swe-term should mean copying one binary.

### Observability is LangSmith-shaped

Tracing, datasets, experiments, and eval feedback assume LangSmith. The behavioral eval suite
*requires* `LANGSMITH_TRACING=true` and exits otherwise. That's a great product story for LangChain
and a vendor coupling for everyone else. swe-term should keep observability behind an interface
(structured logs / OTel) rather than baking a SaaS into the test harness.

### MCP isn't in the core (and that's underadvertised)

The README lists "any MCP server" as a feature, but `libs/deepagents` has **no MCP dependency** —
MCP lives in `deepagents-code` and the CLI. So "Deep Agents supports MCP" is true at the *product*
layer, not the *SDK* layer. A reader evaluating the core SDK for MCP will be surprised.

---

## 4. The ACP Integration — Scrutiny

> Reference: the [ACP prompt-turn lifecycle](https://agentclientprotocol.com/protocol/prompt-turn)
> and [protocol overview](https://agentclientprotocol.com/protocol/overview).

**What's genuinely good.** `deepagents-acp` is the right *shape*: a thin adapter
(`AgentServerACP(acp.Agent)`, ~993 ln) that implements the protocol's agent side by translating
LangGraph stream events into `session/update` notifications, and maps Deep Agents' HITL interrupts
onto `session/request_permission`. It does not reimplement JSON-RPC or stdio — it leans on the
upstream `agent-client-protocol` SDK. The **model-switching-via-`session/set_config_option`** trick
(rebuild the `CompiledStateGraph` from a factory while reusing the same checkpointer + `thread_id`,
so history survives) is elegant and worth copying.

> **Lesson for Harness:** This is the protocol-seam dividend Codex also demonstrates. swe-term's
> `Frontend` interface should make an ACP frontend a first-class citizen — and the
> "stream events → protocol notifications" translator is a clean, testable unit. Copy the
> *adapter-implements-the-other-system's-interface* pattern.

**What's incomplete (and matters).** Measured against the [ACP spec](https://agentclientprotocol.com/protocol/overview),
the integration covers only the prompt-turn-with-tools core and skips a lot:

- **No client-side `fs/*` or `terminal/*` bridging.** The spec's whole point is that the editor
  *is* the environment — the agent should read/write through the client's filesystem and run
  commands through the client's terminal. Deep Agents instead uses its **own** local backends
  (`LocalShellBackend` + `CompositeBackend`), so it ignores the ACP host's file/terminal
  capabilities. The agent edits files in *its* view of disk, not Zed's. For an editor integration,
  that's a meaningful semantic gap (no shared diff/permission surface for file writes via the
  client).
- **No `session/load`, `authenticate`, `session/list`, fork/resume/close.** Session management is
  uuid-in-memory; resuming relies on the caller keeping a shared persistent checkpointer. The spec's
  optional-but-expected session lifecycle is absent.
- **No reasoning stream** (`agent_thought_chunk` unused), so thinking models show nothing during
  deliberation.
- **Tool status is `pending → completed` only** — no `in_progress`/`failed`, and `edit_file`
  completion updates are skipped. The spec explicitly models `in_progress` and failure; clients that
  render live tool progress get less than ACP allows.
- **Free-form `interrupt()` raises a `RequestError`**, and **audio prompts raise
  `NotImplementedError`.** These are honest failures, but they're failures.

> **Lesson for Harness:** When swe-term builds its ACP frontend, do the part Deep Agents skipped:
> route file and shell operations through the *client's* `fs`/`terminal` capabilities when present,
> and model the full tool-status lifecycle. The protocol's value is environment-sharing; an agent
> that brings its own filesystem under ACP is only using half the protocol.

---

## 5. The Harbor Eval Integration — Scrutiny

> Reference: [Harbor](https://www.harborframework.com/docs) — container-based agent evaluation,
> successor to Terminal-Bench.

**What's genuinely good — and the strongest thing to steal from this whole repo.** The eval design
is excellent and largely substrate-independent:

- **Two-tier scoring** (`.success()` hard-fails correctness; `.expect()` logs efficiency but never
  fails) is a *great* idea. It separates "did it get the right answer" from "did it take an
  efficient path," and refuses to let trajectory-shape churn break CI.
- **The Harbor adapter is a model of the right pattern.** `DeepAgentsWrapper(BaseAgent)` implements
  *Harbor's* interface, and `HarborSandbox(SandboxBackendProtocol)` implements *Deep Agents'*
  interface — the two systems meet at two thin adapters. The architectural decision to run **the
  agent on the orchestrator and only tool I/O in the container** (shell `exec` for read/grep/glob,
  native `upload`/`download` for write/edit to dodge `ARG_MAX`) is clean and correct.
- **Verification is delegated to Harbor.** The agent writes an ATIF `trajectory.json`; Harbor's
  in-container verifier produces the reward. Deep Agents doesn't grade itself.

> **Lesson for Harness:** Adopt the two-tier scoring split verbatim for swe-term's eval story
> (correctness gates merges; efficiency is observed, not gated). And adopt the adapter shape: a Go
> `BaseAgent`-equivalent that wraps the harness, and a sandbox backend that satisfies Harbor's
> environment contract, lets swe-term run Terminal-Bench 2.0 without entangling the core with the
> eval framework. This is the cleanest "eval as an external interface" example in the survey.

**What's limited.**

- **Coverage is sampled, not comprehensive.** External benchmarks are a *hard-set of 15 cases*
  (FRAMES/Nexus/BFCL v3); `tau2_airline` is 15 tasks; **SWE-bench is not wired**. Terminal-Bench 2.0
  is the only full sandboxed benchmark. The headline "we run benchmarks" is real but narrower than it
  sounds.
- **LangSmith coupling again.** Behavioral evals can't run without it; Harbor rewards are posted as
  LangSmith feedback. The eval suite is not portable to a team that doesn't use LangSmith.
- **HITL is untested.** `CONTRIBUTING.md` references a `test_hitl.py` that isn't in the tree, and the
  Harbor wrapper sets `auto_approve=True` — so the human-in-the-loop path, a headline feature, has no
  eval coverage.
- **CLI-agent mode in Harbor is disabled** (`# - cli # need to support properly first`), so the
  benchmark numbers reflect the SDK agent, not the shipped `deepagents-code` TUI agent.

---

## 6. Key Gaps (relative to Harness goals)

### No content-addressable cache, no analyzer
Same gap as every other agent surveyed. There is no general `ContentKey(sha256) → value` store and
no pre-LLM static analysis layer. Caching is prompt-caching (Anthropic) and LangGraph checkpointing,
not a content-addressable artifact store. swe-term's content-addressable-caching tenet is unaddressed
here.

### No single-artifact deployment
Deployment is "ship a Python env + LangGraph" or "deploy to LangGraph Platform." There is no
single-binary story. This is structural, not incidental — it's a Python framework.

### Concurrency is cooperative, not parallel
asyncio coroutines, GIL-bound. No goroutine-equivalent cheap parallelism for CPU work (search,
analysis, multi-sandbox fan-out).

### The core is borrowed (×2)
You can't learn "how an agent loop works" from Deep Agents — it delegates to `create_agent`, which
delegates to LangGraph. The novelty is *composition and offload strategy*, not mechanics.

---

## 7. Assumptions to NOT Port

### ❌ "The agent loop is someone else's problem"
True for a framework standing on LangGraph; false for swe-term, whose *reason to exist* is a clean,
readable, zero-dependency loop. Borrowing a loop means inheriting a dependency tree.

### ❌ "Trust the LLM; enforce only at the sandbox"
The default local backend has no isolation. swe-term wants a layered guard *before* the sandbox, on
by default.

### ❌ "Observability = LangSmith"
Don't bake a SaaS into the harness or the test suite. Keep it behind a logging/tracing interface.

### ❌ "Approximate tokens are good enough for budget decisions"
Acceptable for a heuristic ceiling; risky as the basis for eviction. Prefer a real tokenizer or at
least a provider-aware estimate.

### ❌ "An ACP agent can bring its own filesystem"
Under ACP, route file/terminal ops through the *client's* capabilities. Bringing your own backend
ignores the protocol's purpose.

### ❌ "MCP support" at the product layer counts as core support
Be honest about where MCP lives. If swe-term claims MCP, put it in the core (or clearly label it a
plugin).

---

## 8. What to Actually Port

The highest-value lessons in the survey for swe-term's *organization* (less for its substrate):

1. **Middleware composition over a generic loop** — capabilities are plugins, never forks. Deep
   Agents proves this scales to ~11 layers + profiles.
2. **One backend protocol, many implementations** — state / disk / store / composite / sandbox all
   behind `read_file`/`write_file`. Route by path prefix (`CompositeBackend`).
3. **Filesystem as context-overflow substrate** — offload summarized history and oversized tool
   results to (content-addressable) files; pass handles, not bytes.
4. **Isolated-context sub-agents; any agent is a valid sub-agent** — delegation and composition share
   one seam.
5. **Two-tier eval scoring** — correctness hard-fails, efficiency is logged. Copy this exactly.
6. **Adapter-implements-the-other-interface** — `acp.Agent` adapter and Harbor `BaseAgent`/
   `BaseEnvironment` adapters keep the protocol and eval frameworks *out* of the core. swe-term's
   ACP frontend and Harbor harness should be equally thin and external.
7. **Delta-reduced, channel-based state** — the O(N) checkpoint idea is worth re-implementing in Go's
   state model.

---

## 9. Architectural Contrasts

| Dimension | Deep Agents | Harness (swe-term target) |
|-----------|-------------|---------------------------|
| Language / runtime | Python / LangGraph | Go / single binary |
| Agent loop | **Borrowed** from `create_agent` (over LangGraph) | Owned, zero-dependency core |
| Extensibility | `AgentMiddleware` composition | Go interfaces / plugins |
| Concurrency | asyncio coroutines (GIL) | goroutines + `errgroup`/`context` |
| State | LangGraph delta channels + checkpointer | Immutable state, owned |
| Filesystem | `BackendProtocol` (state/disk/store/composite/sandbox) | `Filesystem` interface (same idea, native) |
| Context mgmt | summarize + offload to FS (approx tokens) | offload to content-addressable FS |
| Persistence | LangGraph checkpointer + `BaseStore` | owned, on-disk, portable |
| Security | "trust the LLM"; sandbox-layer | layered in-process guard, on by default |
| Sandboxing | partner SDKs (Runloop/Modal/Daytona) | native, pluggable |
| Caching | prompt-cache + checkpoints | content-addressable artifact store |
| Observability | LangSmith (required for evals) | logs/OTel behind an interface |
| Protocol | `deepagents-acp` (partial ACP) | first-class `Frontend` incl. full ACP |
| Evals | pytest + LangSmith + Harbor (two-tier) | port two-tier scoring + Harbor adapter |
| Deployment | Python env / LangGraph Platform | one static binary |
| Dependencies | deep & wide (LangChain++) | near-zero |

---

## 10. Final Assessment

Deep Agents is the **most instructive harness in the survey for *how to organize capabilities*,** and
the **least portable for *what to build them on*.** Its middleware-composition model, pluggable
backend protocol, filesystem-as-context-substrate, isolated sub-agents, two-tier eval scoring, and
thin protocol/eval adapters are all directly relevant to swe-term's design — they validate, with a
real and heavily-evaluated system, that the plugin/interface approach scales.

But everything underneath is a three-framework Python stack with cooperative concurrency, a deep
dependency tree, SaaS-coupled observability, and no single-binary deployment. Where pi-mono gives
swe-term a *core to port* and Codex gives it a *protocol seam to emulate*, Deep Agents gives it an
**organizational blueprint and an eval/protocol-integration pattern** — the *what* and the *how-to-
compose*, not the *substrate*.

> **One-line takeaway:** Deep Agents proves that "a harness is middleware over a generic loop plus a
> pluggable backend," and that protocol (ACP) and evals (Harbor) belong in thin external adapters —
> adopt all of that; just don't inherit the three frameworks it stands on.

---

*Companion to [`DEEPAGENTS_DEEP_DIVE.md`](DEEPAGENTS_DEEP_DIVE.md). Validated against the local Deep
Agents monorepo at `/home/rohit/sandbox/swe-term/deepagents` (core `0.6.7`, `deepagents-acp 0.0.6`,
`deepagents-evals` on `harbor>=0.6.4`). External references:
[ACP](https://agentclientprotocol.com/protocol/overview), [Harbor](https://www.harborframework.com/docs).*
