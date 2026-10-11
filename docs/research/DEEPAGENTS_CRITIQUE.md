# Critical Evaluation of Deep Agents as an Agentic Harness

> Evaluated against the Harness (swe-term) philosophy:
> zero-dependency core, interface-driven plugins, immutable state,
> goroutine-native concurrency, content-addressable caching, single binary.
>
> **Validated against the local Deep Agents source** at `/Users/rohit/sandbox/swe-term/deepagents`
> (commit `9f0e39d`, 2026-10-10; core `0.7.23`, `deepagents-acp 0.0.12`, `deepagents-code 0.1.83`,
> `deepagents-evals` on `harbor[langsmith]>=0.20.0`). Previously validated against core `0.6.7`
> (2026-06-06); see "Verdict changes since 0.6.7" below. Per the user's
> focus, the **ACP integration** (`libs/acp`) and the **Harbor eval suite** (`libs/evals`) get
> dedicated scrutiny below. External references:
> [ACP](https://agentclientprotocol.com/protocol/overview), [Harbor](https://www.harborframework.com/docs).

---

## Verdict changes since 0.6.7 (2026-06-06)

Evidence is in [`DEEPAGENTS_DEEP_DIVE.md`](DEEPAGENTS_DEEP_DIVE.md#changes-since-067-2026-06-06).
Only these verdicts moved; everything else stands as written.

| Verdict | Was | Now | Why |
|---|---|---|---|
| Harbor adapter pattern (§5, §8.6) | Steal `DeepAgentsWrapper(BaseAgent)` + `HarborSandbox`; agent on orchestrator, tool I/O in container | **Revised.** Steal "ship the agent as a standard graph artifact that the eval host runs in the container" | Both classes are gone; Harbor's built-in `langgraph` agent hosts `bare`/`dcode`/`tau3` graphs with `LocalShellBackend` in the container (`libs/evals/deepagents_harbor/langgraph_project/`, `.github/workflows/_harbor_run.yml:962-968`) |
| Benchmarks reflect the SDK, not the product (§5) | CLI-agent mode in Harbor disabled | **Withdrawn.** | `harbor.yml` defaults `agent_impl` to `dcode`; unified evals run `bare` and `dcode` (`.github/workflows/harbor.yml:117`, `libs/evals/UNIFIED_EVALS.md`) |
| Benchmark coverage "sampled" (§5) | Terminal-Bench 2.0 the only full sandboxed benchmark | **Softened.** | Unified battery adds harbor-index, τ³-bench, Context-Bench, DRBench with pass@K/avg@K. SWE-bench still not wired |
| ACP completeness (§4) | No session/load, no reasoning stream, status `pending → completed` only | **Softened.** Core gap (no client `fs/*`/`terminal/*`, MCP unused) unchanged | Opt-in `session/load`, `agent_thought_chunk`, and `failed` status landed in `deepagents-acp` 0.0.10–0.0.11 (`libs/acp/CHANGELOG.md`) |
| Observability coupling (§3, §7) | LangSmith-shaped evals | **Strengthened.** | `langsmith>=0.14.4` is now a *core* runtime dependency of `deepagents` (`libs/deepagents/pyproject.toml`), and runs carry `ls_integration` metadata (`graph.py:1004`) |
| Local-shell safety (§1, §7) | Default local backend has no isolation | **Softened, not reversed.** | `FilesystemBackend`/`LocalShellBackend` default to path-anchored `virtual_mode=True` and permissions can raise HITL interrupts (0.7.0, commit `540a0fa`; 0.6.8, `a090162`). Shell commands are still not process-isolated |

Unchanged after re-check: middleware-composition verdict, backend protocol, filesystem-as-overflow,
isolated sub-agents, two-tier scoring, approximate tokens, MCP-not-in-core, HITL untested, and the
dependency-depth verdict.

---

## Verdict Summary

Deep Agents is the **best-engineered "harness as composition" design** in this survey, and the
single most useful one to learn *organizing principles* from — while being almost the **opposite** of
swe-term's deployment philosophy at the substrate level.

Its thesis is sharp and correct: **a harness is a curated bundle of middleware over a generic agent
loop.** It does not write an agent loop at all. `create_deep_agent` assembles ~12 `AgentMiddleware`
(filesystem, sub-agents, summarization, skills, memory, HITL, …; `write_todos` became opt-in in 0.7.0) and hands them to LangChain's
`create_agent`, which compiles a LangGraph `CompiledStateGraph`. Capabilities are *layers*, not
forks. That is exactly the modularity swe-term wants — and Deep Agents demonstrates it works at the
~136-eval, multi-provider, ACP-and-Harbor-integrated scale.

But as an *architectural template for a single Go binary*, Deep Agents is a cautionary tale of
**dependency depth**. The "core" is three frameworks deep (LangGraph → `create_agent` → deepagents)
before you reach a line of Deep Agents' own logic. Its state model, streaming, checkpointing,
persistence, and even its agent loop are LangGraph's. Its strongest features (delta-reduced
checkpoints, pluggable stores, interrupt-based HITL) are **LangGraph features Deep Agents configures**,
not invents. You cannot port Deep Agents; you can only port LangGraph's ideas, re-expressed.

**Steal the middleware-composition model, the pluggable-backend protocol, the two-tier eval scoring,
the "agent-as-acp.Agent" adapter pattern, and the "agent as a standard artifact the eval host runs"
Harbor pattern.** Do not adopt the
three-framework dependency stack, the Python/asyncio concurrency model, or the LangSmith-coupled
observability as your core.**

---

## 1. Effectiveness

### What Deep Agents does well

**It correctly identifies that "harness" is a composition problem.** Where Claude Code, Codex, and
pi-mono each *own* a loop and bolt features onto it, Deep Agents declares the loop a commodity and
competes purely on what middleware you stack. `create_deep_agent` is ~700 lines of *assembly*
(`graph.py:277-1011`) — pick
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
(`LocalShellBackend`) runs commands on the host with **no process isolation**. Since 0.7.0 its file
paths are anchored under `root_dir` by default, and permission rules can raise HITL interrupts, but
shell commands are not confined. Real isolation requires `LangSmithSandbox` or a partner
(Runloop/Modal/Daytona/Vercel). swe-term's threat model (a layered, in-process execution
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

The behavioral suite runs real LLMs against ~136 evals across up to 25 models (`set0`), plus an LLM-as-judge
(`claude-sonnet-4-6`) that adds its own cost and variance. The multi-trial workflow exists precisely
because results are non-deterministic; it measures stdev rather than asserting reproducibility.
Harbor/Terminal-Bench 2.0 at full concurrency (Daytona default 40) is expensive. This is honest, but
it means "did we regress?" is a *statistical* question with a real dollar cost — not a fast,
deterministic `go test`.

---

## 3. Simplicity

### The dependency stack is deep and wide

The core package pulls `langchain-core`, `langchain`, `langchain-anthropic`,
`langchain-google-genai`, `langsmith` (new since 0.6.7), `wcmatch`, and transitively the entire
LangGraph stack. The full
experience (code TUI, MCP, sandboxes, evals, ACP) pulls Textual, `langchain-mcp-adapters`, partner
SDKs, `harbor`, `langsmith`, `openevals`, `modal`, and `agent-client-protocol`. This is the
**maximal-dependency** end of the spectrum — the exact opposite of swe-term's "zero-dependency core,
single static binary" tenet. Deploying Deep Agents means shipping a Python environment and a
dependency tree; deploying swe-term should mean copying one binary.

### Observability is LangSmith-shaped

Tracing, datasets, experiments, and eval feedback assume LangSmith. The behavioral eval suite
*requires* `LANGSMITH_TRACING=true` and skips otherwise (`libs/evals/tests/evals/conftest.py:34`).
Since 0.7.x, `langsmith` is also a hard dependency of the core SDK itself. That's a great product story for LangChain
and a vendor coupling for everyone else. swe-term should keep observability behind an interface
(structured logs / OTel) rather than baking a SaaS into the test harness.

### MCP isn't in the core (and that's underadvertised)

The README lists "any MCP server" as a feature, but `libs/deepagents` has **no MCP dependency** —
MCP lives in `deepagents-code` and `deepagents-talon` (`libs/cli` has been removed). So "Deep Agents supports MCP" is true at the *product*
layer, not the *SDK* layer. A reader evaluating the core SDK for MCP will be surprised.

---

## 4. The ACP Integration — Scrutiny

> Reference: the [ACP prompt-turn lifecycle](https://agentclientprotocol.com/protocol/prompt-turn)
> and [protocol overview](https://agentclientprotocol.com/protocol/overview).

**What's genuinely good.** `deepagents-acp` is the right *shape*: a thin adapter
(`AgentServerACP(acp.Agent)`, `server.py` 1,354 ln) that implements the protocol's agent side by translating
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
- **No `authenticate`, `session/list`, fork/resume/close.** `session/load` now exists but is opt-in
  (`load_sessions=True`) and needs a checkpointer that survives restarts (`server.py:229-252`).
  *(Softened since 0.6.7: load was absent.)*
- **`mcp_servers` are stored but never used** — written to `_session_mcp_servers` and never read
  (`server.py:273,375,419`), so editor-configured MCP servers do not reach the agent.
- ~~No reasoning stream~~ — **fixed in 0.0.11**: visible reasoning streams as `agent_thought_chunk`.
- **Tool status is `pending → completed | failed`** — `failed` was added (`server.py:915`), but
  `in_progress` is still never emitted for tool calls and `edit_file` completion updates are still
  skipped (`server.py:901`). Clients that render live tool progress still get less than ACP allows.
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
- **The Harbor seam got *thinner*, not thicker.** *(Verdict revised since 0.6.7.)* The old pair
  of adapters (`DeepAgentsWrapper(BaseAgent)` + `HarborSandbox(SandboxBackendProtocol)`, agent on
  the orchestrator, tool I/O proxied into the container) was deleted. Now the agent is packaged as a
  plain LangGraph project (`deepagents_harbor/langgraph_project/langgraph.json` → `bare`, `dcode`,
  `tau3` graphs) and Harbor's generic `langgraph` agent runs it *inside* the task container with an
  ordinary `LocalShellBackend` (`langgraph_agent.py:440-446`). Deep Agents no longer implements any
  Harbor interface; it ships a standard artifact. The cost is that the container is now the only
  isolation boundary, which the code states outright and mitigates with an env-var denylist for
  provider and LangSmith keys (`_SHELL_ENV_DENYLIST`).
- **Verification is delegated to Harbor.** Harbor's in-container verifier produces the reward, which
  is mirrored to LangSmith as `harbor_reward` (`deepagents_harbor/langsmith.py:603-622`). Deep Agents
  doesn't grade itself.
- **A cross-model scorecard now exists.** `unified_evals.yml` reports pass@K / avg@K across
  autonomous / conversation / context / research axes, with the harness (`bare` vs `dcode`) as an
  explicit input (`libs/evals/UNIFIED_EVALS.md`).

> **Lesson for Harness:** Adopt the two-tier scoring split verbatim for swe-term's eval story
> (correctness gates merges; efficiency is observed, not gated). For Harbor, Deep Agents' own
> evolution points to the simpler shape: make swe-term runnable as a self-contained agent *inside*
> the task container through a generic Harbor agent entry point, rather than maintaining a
> custom `BaseAgent` plus a remote-proxy sandbox backend. A single static binary fits that shape
> better than a Python graph does. Keep "eval as an external interface"; drop the bespoke proxy.

**What's limited.**

- **Coverage is broader but still not SWE-bench.** *(Softened since 0.6.7.)* The pytest side still
  samples FRAMES/Nexus/BFCL v3 and `tau2_airline`; the Harbor side now runs Terminal-Bench 2/2.1,
  harbor-index, τ³-bench, Context-Bench, and DRBench. **SWE-bench is still not wired.**
- **LangSmith coupling again.** Behavioral evals can't run without it; Harbor rewards are posted as
  LangSmith feedback. The eval suite is not portable to a team that doesn't use LangSmith.
- **HITL is still untested in evals.** `libs/evals/CONTRIBUTING.md:81` still lists a `test_hitl.py`
  that is not in `tests/evals/`, so the human-in-the-loop path, a headline feature, has no eval
  coverage.
- ~~CLI-agent mode in Harbor is disabled~~ — **withdrawn.** `harbor.yml` now defaults to the `dcode`
  product agent (`.github/workflows/harbor.yml:117`), so benchmark numbers can reflect the shipped
  agent as well as the bare SDK.

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
The default local shell backend has no process isolation (path anchoring arrived in 0.7.0, but
commands are unconfined). swe-term wants a layered guard *before* the sandbox, on by default.

### ❌ "Observability = LangSmith"
Don't bake a SaaS into the harness or the test suite. Keep it behind a logging/tracing interface.
Deep Agents moved the other way: `langsmith` became a core SDK dependency after 0.6.7.

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
   Agents proves this scales to ~12 layers + profiles, and 0.7.0 lets a same-named user middleware
   replace a default instead of forking the stack.
2. **One backend protocol, many implementations** — state / disk / store / composite / sandbox all
   behind `read_file`/`write_file`. Route by path prefix (`CompositeBackend`).
3. **Filesystem as context-overflow substrate** — offload summarized history and oversized tool
   results to (content-addressable) files; pass handles, not bytes.
4. **Isolated-context sub-agents; any agent is a valid sub-agent** — delegation and composition share
   one seam.
5. **Two-tier eval scoring** — correctness hard-fails, efficiency is logged. Copy this exactly.
6. **Keep protocol and eval frameworks out of the core** — the `acp.Agent` adapter still implements
   the other system's interface; the Harbor side now ships a standard graph artifact that Harbor's
   generic agent hosts. swe-term's ACP frontend and Harbor entry point should be equally thin and
   external.
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
| Evals | pytest + LangSmith + Harbor (two-tier; unified pass@K battery) | port two-tier scoring + in-container Harbor entry point |
| Deployment | Python env / LangGraph Platform | one static binary |
| Dependencies | deep & wide (LangChain++, `langsmith` in core) | near-zero |

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
Agents monorepo at `/Users/rohit/sandbox/swe-term/deepagents` on 2026-10-10 (commit `9f0e39d`; core
`0.7.23`, `deepagents-acp 0.0.12`, `deepagents-evals` on `harbor[langsmith]>=0.20.0`). External references:
[ACP](https://agentclientprotocol.com/protocol/overview), [Harbor](https://www.harborframework.com/docs).*
