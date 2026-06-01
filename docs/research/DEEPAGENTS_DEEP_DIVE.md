# Deep Agents Deep Dive — Architecture & Feature Analysis

> Analysis of [`langchain-ai/deepagents`](https://github.com/langchain-ai/deepagents) — LangChain's
> "batteries-included agent harness." An **opinionated agent that runs out of the box**, built as a
> thin opinionated layer of *middleware* on top of LangChain's `create_agent`, which itself sits on
> the **LangGraph** runtime.
> Language: Python. License: MIT. Distribution: PyPI (`deepagents`).
>
> **Validated against a local source checkout** at `/home/rohit/sandbox/swe-term/deepagents`
> (`deepagents` core `v0.6.7`, `deepagents-acp` `v0.0.6`, `deepagents-evals` using `harbor>=0.6.4`).
> Two areas get extra depth at the user's request: the **ACP (Agent Client Protocol) integration**
> (`libs/acp`) and the **Harbor-based eval suite** (`libs/evals`). All file:line references are from
> that checkout.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [The LangGraph → create_agent → Deep Agents Stack](#the-langgraph--create_agent--deep-agents-stack)
3. [Monorepo Layout](#monorepo-layout)
4. [`create_deep_agent`: The Entry Point](#create_deep_agent-the-entry-point)
5. [The Middleware Stack](#the-middleware-stack)
6. [State Model & Checkpointing](#state-model--checkpointing)
7. [Backends: The Pluggable Filesystem](#backends-the-pluggable-filesystem)
8. [Sub-agents](#sub-agents)
9. [Context Management (Summarization & Offload)](#context-management-summarization--offload)
10. [Skills, Tools & MCP](#skills-tools--mcp)
11. [Human-in-the-Loop](#human-in-the-loop)
12. [Shell & Sandbox Partners](#shell--sandbox-partners)
13. [The ACP Integration (`libs/acp`)](#the-acp-integration-libsacp)
14. [The Eval Suite & Harbor (`libs/evals`)](#the-eval-suite--harbor-libsevals)
15. [`libs/code` and `libs/cli`](#libscode-and-libscli)
16. [Key Design Patterns](#key-design-patterns)
17. [Summary Statistics](#summary-statistics)

---

## System Overview

Deep Agents is LangChain's answer to "what is the general-purpose part of Claude Code, and can we
ship it as a library?" The README is explicit about the lineage: *"Inspired by Claude Code: an
attempt to identify what makes it general-purpose, and push that further."* It is **not** a new
agent runtime — it is an *opinionated configuration* of LangChain's existing agent machinery.

The defining architectural choice: **Deep Agents writes no agent loop of its own.** It assembles a
list of `AgentMiddleware` and hands them to `langchain.agents.create_agent`, which compiles a
LangGraph `CompiledStateGraph` implementing the standard model↔tools ReAct loop. Everything Deep
Agents adds — planning, a virtual filesystem, sub-agents, summarization, skills — is a middleware.

**Tech stack:**

| Layer | Technology |
|-------|-----------|
| Language | Python (≥ 3.11) |
| Runtime | **LangGraph** (`CompiledStateGraph`, channels, checkpointers, streaming) |
| Harness base | **LangChain `create_agent`** (the ReAct model↔tool loop + middleware host) |
| Models | `init_chat_model` (provider:model strings); `langchain-anthropic`, `langchain-google-genai`, OpenAI, OpenRouter, … |
| State / persistence | LangGraph checkpointers (`MemorySaver`, durable) + `BaseStore` |
| Tool schema | LangChain `BaseTool` (Python-callable + JSON schema) |
| Glob/grep | `wcmatch` |
| Observability | LangSmith (tracing, datasets, experiments) |
| Editor integration | **ACP** via `deepagents-acp` (`agent-client-protocol` 0.9.0) |
| Evals | pytest + LangSmith + **Harbor** (`harbor>=0.6.4`) for Terminal-Bench 2.0 |

**Scale (core `libs/deepagents`):** ~48 Python source files, **13 middleware modules**, ~19–22K LOC.
The largest files are `middleware/filesystem.py` (~2,100 ln), `middleware/summarization.py`
(~1,666 ln), `profiles/harness/harness_profiles.py` (~1,317 ln), `middleware/skills.py` (~1,069 ln),
`middleware/async_subagents.py` (~955 ln), and the assembly file `graph.py` (830 ln).

---

## The LangGraph → create_agent → Deep Agents Stack

The single most important mental model is the three-layer stack (the README dedicates an FAQ to it):

```
LangGraph            the graph runtime: nodes, edges, channels, checkpointers, streaming
   │
create_agent         LangChain's minimal harness: the ReAct loop + an AgentMiddleware host
   │
Deep Agents          an opinionated bundle of middleware (todos, files, subagents, summarization,
                     skills, HITL) + pluggable backends, exposed via create_deep_agent()
```

The layers **compose downward**: any LangGraph `CompiledStateGraph` can be handed to a Deep Agent as
a sub-agent (`CompiledSubAgent`), so custom orchestration plugs in alongside the harness defaults.
This is fundamentally different from pi-mono/Codex/Claude Code, which each own their loop. Deep
Agents *borrows* the loop and competes on **middleware composition**.

---

## Monorepo Layout

A `uv`-managed Python monorepo (`libs/`):

| Package | PyPI name | Role |
|---------|-----------|------|
| **`libs/deepagents`** | `deepagents` (`0.6.7`) | Core SDK — `create_deep_agent`, middleware, backends, profiles |
| **`libs/acp`** | `deepagents-acp` (`0.0.6`) | Agent Client Protocol connector (run a Deep Agent inside Zed and other ACP editors) |
| **`libs/evals`** | `deepagents-evals` | Behavioral eval suite + **Harbor** integration (Terminal-Bench 2.0) |
| **`libs/code`** | `deepagents-code` | Pre-built coding agent with a **Textual TUI** (Claude-Code-like terminal app) |
| **`libs/cli`** | `deepagents-cli` | Deployment CLI (+ a TS `frontend/`) |
| **`libs/partners/*`** | `langchain-{runloop,modal,daytona,quickjs}` | Sandbox / code-execution provider integrations |
| **`examples/*`** | — | ~20 worked examples (deep research, text-to-SQL, RLM, swarm, deploy-* recipes) |

---

## `create_deep_agent`: The Entry Point

There is exactly one public constructor (no async variant), defined in
`libs/deepagents/deepagents/graph.py`:

```217:237:/home/rohit/sandbox/swe-term/deepagents/libs/deepagents/deepagents/graph.py
def create_deep_agent(  # noqa: C901, PLR0912, PLR0915  # Complex graph assembly logic with many conditional branches
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    permissions: list[FilesystemPermission] | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None = None,
    state_schema: type[DeepAgentState] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph[AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]]:
```

The body resolves the model and a **harness profile**, assembles the middleware list, and delegates:

```806:822:/home/rohit/sandbox/swe-term/deepagents/libs/deepagents/deepagents/graph.py
    return create_agent(
        model,
        system_prompt=final_system_prompt,
        tools=_tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
        state_schema=state_schema if state_schema is not None else DeepAgentState,
        transformers=[_subagent_factory],
    ).with_config(
        {"recursion_limit": 9_999, ...}
    )
```

Note the `recursion_limit: 9_999` — Deep Agents expects **long-horizon** runs (many loop turns), and
the LangGraph recursion guard is set high enough not to interfere. If `model=None`, it falls back to
a (deprecated) `ChatAnthropic("claude-sonnet-4-6")`; if `backend=None`, it uses a `StateBackend`
(in-state virtual filesystem).

---

## The Middleware Stack

This is where the "harness" actually lives. `graph.py` assembles middleware in a fixed order; each
contributes tools and/or wraps the model call. Default order:

| # | Middleware | Source | Contributes |
|---|-----------|--------|-------------|
| 1 | `TodoListMiddleware` | LangChain | `write_todos` planning tool |
| 2 | `SkillsMiddleware` *(if `skills=`)* | `middleware/skills.py` | Loads `SKILL.md` files into the system prompt on demand |
| 3 | `FilesystemMiddleware` **(required)** | `middleware/filesystem.py` | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`; permissions; large-result eviction |
| 4 | `SubAgentMiddleware` **(required)** | `middleware/subagents.py` | The `task` tool → spawn child agents |
| 5 | `SummarizationMiddleware` | `middleware/summarization.py` | Context compaction + history offload to disk |
| 6 | `PatchToolCallsMiddleware` | `middleware/patch_tool_calls.py` | Repairs dangling/malformed tool calls before a run |
| 7 | `AsyncSubAgentMiddleware` *(if remote specs)* | `middleware/async_subagents.py` | Background subagents on LangGraph Platform |
| — | *user middleware* (`middleware=`) | — | Inserted here |
| 8 | profile extras + `_ToolExclusionMiddleware` | `profiles/`, `middleware/_tool_exclusion.py` | Per-model tuning, tool filtering |
| 9 | `AnthropicPromptCachingMiddleware` | `langchain_anthropic` | Prompt caching (ignored on non-Anthropic) |
| 10 | `MemoryMiddleware` *(if `memory=`)* | `middleware/memory.py` | Injects `AGENTS.md`-style memory into the prompt |
| 11 | `HumanInTheLoopMiddleware` *(if `interrupt_on=`)* | LangChain | Tool-call approval/edit/reject |

`FilesystemMiddleware` and `SubAgentMiddleware` are declared `_REQUIRED_MIDDLEWARE` — core features
depend on them and they cannot be excluded by a profile:

```187:201:/home/rohit/sandbox/swe-term/deepagents/libs/deepagents/deepagents/graph.py
_REQUIRED_MIDDLEWARE: tuple[tuple[type[AgentMiddleware[Any, Any, Any]], tuple[str, ...]], ...] = (
    (FilesystemMiddleware, ()),
    (SubAgentMiddleware, ()),
)
```

The filesystem tools are created up front (`FilesystemMiddleware.tools`):

```717:725:/home/rohit/sandbox/swe-term/deepagents/libs/deepagents/deepagents/middleware/filesystem.py
        self.tools = [
            self._create_ls_tool(),
            self._create_read_file_tool(),
            self._create_write_file_tool(),
            self._create_edit_file_tool(),
            self._create_glob_tool(),
            self._create_grep_tool(),
            self._create_execute_tool(),
        ]
```

(A `RubricMiddleware` exists for grader sub-loops but is not in the default stack.)

---

## State Model & Checkpointing

State is a LangGraph state schema. `DeepAgentState` extends LangChain's `AgentState` and applies a
**delta reducer** to `messages` so checkpoint size grows O(N) rather than O(N²):

```63:66:/home/rohit/sandbox/swe-term/deepagents/libs/deepagents/deepagents/graph.py
class DeepAgentState(AgentState):
    """AgentState with DeltaChannel on messages to reduce checkpoint growth from O(N²) to O(N)."""

    messages: Required[Annotated[list[AnyMessage], DeltaChannel(_messages_delta_reducer, snapshot_frequency=50)]]
```

The virtual filesystem is a second delta-reduced channel added by the filesystem middleware:

```261:265:/home/rohit/sandbox/swe-term/deepagents/libs/deepagents/deepagents/middleware/filesystem.py
class FilesystemState(AgentState):
    """State for the filesystem middleware."""

    files: Annotated[NotRequired[dict[str, FileData]], DeltaChannel(_file_data_delta_reducer, snapshot_frequency=50)]
```

Persistence is whatever LangGraph checkpointer you pass (`MemorySaver` for ephemeral, a durable
checkpointer for production). A separate LangGraph `BaseStore` provides cross-thread (cross-session)
memory. This is the cleanest part of the design: **state is data in named channels, persistence is a
pluggable checkpointer, and cross-session memory is a separate store** — all inherited from LangGraph
rather than reinvented.

---

## Backends: The Pluggable Filesystem

Deep Agents abstracts "where do files live" behind a single `BackendProtocol`
(`backends/protocol.py`) with `ls`/`read`/`grep`/`glob`/`write`/`edit`/`upload_files`/`download_files`
(plus async `a*` variants). A `SandboxBackendProtocol` subtype adds `execute`/`aexecute` for shell.

Shipped implementations (`backends/__init__.py`):

| Backend | File | Semantics |
|---------|------|-----------|
| `StateBackend` *(default)* | `backends/state.py` | Virtual FS living in graph state; checkpointed per thread; not cross-thread |
| `FilesystemBackend` | `backends/filesystem.py` | Real disk under a `root_dir` |
| `StoreBackend` | `backends/store.py` | LangGraph `BaseStore` — persistent, cross-thread |
| `CompositeBackend` | `backends/composite.py` | Routes by path prefix (e.g. `/memories/` → store, rest → sandbox) |
| `LocalShellBackend` | `backends/local_shell.py` | `FilesystemBackend` + host shell `execute` (no isolation) |
| `BaseSandbox` | `backends/sandbox.py` | File ops implemented via a remote sandbox's `execute()` |
| `ContextHubBackend` | `backends/context_hub.py` | Files from a LangSmith Hub agent repo |
| `LangSmithSandbox` | `backends/langsmith.py` | LangSmith-managed sandbox |

The agent only ever sees the **tools** (`read_file`, `write_file`, …); the backend decides whether
those hit graph state, local disk, a LangGraph store, or a remote container. This is the same
"pluggable filesystem" idea swe-term wants — implemented as a Python ABC rather than a Go interface.

---

## Sub-agents

Sub-agents are the delegation primitive, exposed as a `task` tool by `SubAgentMiddleware`. Three
flavors:

- **`SubAgent`** (declarative dict): `name`, `description`, `system_prompt`, optional `tools`,
  `model`, `middleware`, `skills`, `permissions`, `response_format`. Compiled with `create_agent`.
- **`CompiledSubAgent`**: `name`, `description`, `runnable` — **any** `CompiledStateGraph` / Runnable.
  This is the composition seam: a hand-built LangGraph plugs in as a sub-agent.
- **`AsyncSubAgent`**: `graph_id` (+ optional `url`/`headers`) — a **remote** agent run on LangGraph
  Platform via the LangGraph SDK, in the background.

A default `general-purpose` sub-agent is provided unless overridden. Crucially, sub-agents run with
**isolated context** — the parent's `messages`/`todos` are stripped and the child gets only its task
description, then a single `ToolMessage` is merged back:

```534:540:/home/rohit/sandbox/swe-term/deepagents/libs/deepagents/deepagents/middleware/subagents.py
    def _validate_and_prepare_state(subagent_type: str, description: str, runtime: ToolRuntime) -> tuple[Runnable, dict]:
        ...
        subagent_state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}
        subagent_state["messages"] = [HumanMessage(content=description)]
        return subagent, subagent_state
```

This is the "isolated context window for delegated work" pattern Claude Code popularized — here it
falls out of LangGraph state-scoping.

---

## Context Management (Summarization & Offload)

Context management is two cooperating mechanisms, both backed by the filesystem:

1. **History summarization** — `create_summarization_middleware(model, backend)` wraps LangChain's
   summarization. When the thread approaches the model's input limit, older messages are summarized;
   the evicted history is **offloaded to disk** at `/conversation_history/{thread_id}.md` rather than
   discarded. Token counting uses `count_tokens_approximately` (a heuristic), with thresholds derived
   from `model.profile["max_input_tokens"]`.

2. **Tool-result offload** — when a single tool result exceeds `tool_token_limit_before_evict`
   (default **20,000 tokens**, estimated ≈ chars/4), the filesystem middleware writes it to
   `{artifacts_root}/large_tool_results/<tool_call_id>` and leaves a reference, so a giant grep/read
   doesn't blow the window (`_message_eviction.py`).

Both reuse the backend as scratch storage — the agent's own filesystem is also its context-overflow
spillover. (Note: this is an *approximate* token count, like pi-mono's `chars/4`, not a real
tokenizer.)

---

## Skills, Tools & MCP

- **Skills** (`middleware/skills.py`): Anthropic-style progressive disclosure. A skill is a directory
  with a `SKILL.md` (YAML frontmatter + Markdown body) plus optional helper files; the middleware
  surfaces skill metadata in the system prompt and the agent loads the full body on demand. Sources
  are passed via `skills=[...]` and resolved through the backend (so skills can live in state, on
  disk, or in a store).
- **Tools**: ordinary LangChain `BaseTool`s / callables passed to `create_deep_agent(tools=[...])`,
  merged with the middleware-provided tools.
- **MCP**: **not in the core package.** `libs/deepagents` has no `langchain-mcp-adapters` dependency.
  MCP support lives in `deepagents-code` (`mcp_tools.py`, via `langchain-mcp-adapters`) and the CLI.
  The README's "any MCP server" is a consumer-side integration, not a core feature.

---

## Human-in-the-Loop

HITL is LangChain's `HumanInTheLoopMiddleware`, appended only when `interrupt_on=` is set. It uses
LangGraph's `interrupt()`/checkpointer mechanism: when a guarded tool is about to run, the graph
**interrupts** and surfaces an interrupt payload with `action_requests` and `review_configs`
(allowed decisions: `approve`/`edit`/`reject`/`respond`). The caller resumes with
`Command(resume={"decisions": [...]})`. Sub-agents inherit the parent's `interrupt_on` unless
overridden. This same interrupt is what the ACP layer maps to `session/request_permission` (below).

---

## Shell & Sandbox Partners

Shell execution is a backend capability (`SandboxBackendProtocol.execute`). In-tree,
`LocalShellBackend` runs commands on the host with **no isolation**. Real isolation comes from the
partner packages:

| Partner | Class | What it is |
|---------|-------|-----------|
| `langchain-runloop` | `RunloopSandbox` | Runloop devbox cloud sandboxes |
| `langchain-modal` | `ModalSandbox` | Modal sandboxes |
| `langchain-daytona` | `DaytonaSandbox` | Daytona cloud workspaces |
| `langchain-quickjs` | `CodeInterpreterMiddleware` | In-process QuickJS REPL (a code-interpreter tool, not a shell) |

Each sandbox class implements the `execute()` half of the backend protocol, so the agent's file/shell
tools transparently run inside the remote environment.

---

## The ACP Integration (`libs/acp`)

> Background on the protocol: the [Agent Client Protocol (ACP)](https://agentclientprotocol.com/protocol/overview)
> is a **JSON-RPC 2.0** protocol (typically over stdio) that lets a *Client* (an editor like
> [Zed](https://zed.dev/)) drive an *Agent* subprocess. The flow is `initialize` → `session/new`
> (or `session/load`) → `session/prompt`, with the Agent streaming `session/update` notifications
> (`agent_message_chunk`, `tool_call`, `tool_call_update`, `plan`) and requesting approval via
> `session/request_permission`; the Client optionally exposes `fs/*` and `terminal/*` capabilities.
> See the [prompt-turn lifecycle](https://agentclientprotocol.com/protocol/prompt-turn).

`deepagents-acp` (`v0.0.6`) bridges a Deep Agent to ACP so it can run inside Zed. It is a **thin
adapter** (~1,400 LOC; `server.py` ~993 ln) over the PyPI `agent-client-protocol` package (imported
as `acp`, locked at **`0.9.0`**). The protocol library owns the stdio JSON-RPC transport and the
type/builder helpers; `deepagents_acp` owns only the translation to LangGraph.

### The adapter object

```92:93:/home/rohit/sandbox/swe-term/deepagents/libs/acp/deepagents_acp/server.py
class AgentServerACP(ACPAgent):
    """ACP agent server that bridges Deep Agents with the Agent Client Protocol."""
```

It accepts either a compiled Deep Agent or a **factory** `Callable[[AgentSessionContext], CompiledStateGraph]`
(needed for model switching), plus optional `modes` and `models`.

### Implemented ACP methods

| ACP method | `AgentServerACP` handler | Behavior |
|------------|--------------------------|----------|
| `initialize` | `initialize()` | Advertises **only** `promptCapabilities.image = true` |
| `session/new` | `new_session()` | `session_id = uuid4().hex`; used as the LangGraph `thread_id`; `mcp_servers` accepted but **ignored** |
| `session/prompt` | `prompt()` | The core turn (below) |
| `session/cancel` | `cancel()` | Sets a boolean flag polled in the stream loop |
| `session/set_mode` | `set_session_mode()` | Updates mode, rebuilds the agent |
| `session/set_config_option` | `set_config_option()` | `mode` or `model` switch → rebuild |
| `on_connect` | `on_connect()` | Stores the client conn for `session_update`/`request_permission` |

### The prompt turn → LangGraph translation

ACP content blocks are converted to LangChain multimodal content, then the graph is streamed and each
LangGraph event is translated into a `session/update`:

```644:656:/home/rohit/sandbox/swe-term/deepagents/libs/acp/deepagents_acp/server.py
        while current_state is None or current_state.interrupts:
            ...
            async for stream_chunk in agent.astream(
                Command(resume={"decisions": user_decisions})
                if user_decisions
                else {"messages": [{"role": "user", "content": content_blocks}]},
                config=config,
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
```

The mapping:

| LangGraph signal | ACP `session/update` |
|------------------|----------------------|
| AI text chunk (root graph only) | `agent_message_chunk` via `update_agent_message(text_block(...))` |
| `tool_call_chunks` on an `AIMessageChunk` | `tool_call` (`ToolCallStart`) — `edit_file` gets a diff via `start_edit_tool_call` |
| `ToolMessage` (non-`edit_file`) | `tool_call_update` with `status="completed"` and content |
| `write_todos` args / `updates["todos"]` | `plan` (`AgentPlanUpdate`) |
| LangGraph `__interrupt__` (HITL) | `session/request_permission` (a client RPC, not an update) |

Permission options surfaced are approve / reject / **allow_always** (with a command allowlist + a
dangerous-pattern guard); the user's choice becomes `Command(resume={"decisions": [...]})`. The turn
ends with `PromptResponse(stop_reason="end_turn")` or `"cancelled"`.

### Model switching via Session Config Options

The README's headline feature. `models=[{"value","name"}, …]` is advertised as a
`SessionConfigOptionSelect` (`id="model"`, `category="model"`). When the client picks a new model,
`set_config_option` calls `_reset_agent`, which rebuilds the `CompiledStateGraph` from the factory
with the new `AgentSessionContext.model` — **reusing the same checkpointer and `thread_id`**, so the
LangGraph conversation state survives the model swap:

```574:588:/home/rohit/sandbox/swe-term/deepagents/libs/acp/deepagents_acp/server.py
    def _reset_agent(self, session_id: str) -> None:
        """Reset the agent instance, re-creating it from the factory if applicable."""
        cwd = self._session_cwds.get(session_id)
        ...
            context = AgentSessionContext(cwd=self._cwd, mode=mode, model=model)
            self._agent = self._agent_factory(context)
```

### Transport

`run_agent(server)` (from the `acp` SDK) wires the agent into an `AgentSideConnection` over
`sys.stdin`/`sys.stdout` (50 MB buffer) and `await conn.listen()`. The adapter does not reimplement
JSON-RPC; Zed spawns the process (via `run_demo_agent.sh`) and speaks ACP on stdio.

### What it deliberately does **not** implement

This is an editor-facing convenience, not a full ACP server. Notably missing/partial:

- **No `session/load`, `authenticate`, `session/list`, fork/resume/close** — only the baseline turn.
- **No `set_session_model`** — model changes go through `set_config_option` only.
- **`mcp_servers` from `session/new` are ignored** — MCP isn't wired through ACP.
- **No client `fs/*` or `terminal/*` bridging** — the Deep Agent uses its **own** local backends
  (`LocalShellBackend` + `CompositeBackend` in the demo), *not* the editor's filesystem/terminal.
  So the agent does not honor the ACP host's file/terminal capabilities.
- **No reasoning/thought stream** (`agent_thought_chunk` unused).
- **Tool status is only `pending → completed`** (no `in_progress`/`failed`); `edit_file` completion
  updates are skipped.
- **Free-form LangGraph `interrupt()` is unsupported** — it raises a `RequestError` explaining the
  ACP limitation.
- **Audio prompts** raise `NotImplementedError`.

So the integration cleanly covers the *prompt-turn-with-tools-and-permissions* core of ACP and leaves
the resource-bridging and session-management surface unimplemented.

---

## The Eval Suite & Harbor (`libs/evals`)

> Background: [Harbor](https://www.harborframework.com/docs) (laude-institute) is "a framework for
> evaluating and optimizing agents and models in **container environments**" — the productionized
> successor to Terminal-Bench. It provides modular `environment`/`agent`/`task` interfaces, a
> **registry** of benchmarks/datasets, pre-integrated CLI agents, and integrations with cloud sandbox
> providers (Daytona, Modal, E2B, Runloop, Tensorlake) and optimizers (SkyRL, GEPA).

`deepagents-evals` has two distinct evaluation tracks.

### Track 1 — Behavioral evals (pytest + LangSmith)

~**118 evals** across **8 categories** (`file_operations`, `retrieval`, `tool_use`, `memory`,
`conversation`, `summarization`, `unit_test`, `langchain/middleware`). Each is a pytest test that
builds a real agent, runs it against a real LLM, captures the trajectory, and scores it. The scoring
contract is **two-tier**:

- **`.success(...)`** — correctness assertions that **hard-fail** the test (`final_text_contains`,
  `file_equals`, `llm_judge`, …).
- **`.expect(...)`** — efficiency expectations (step count, tool-call shape) that are **logged but
  never fail**.

```27:47:/home/rohit/sandbox/swe-term/deepagents/libs/evals/tests/evals/test_todos.py
def test_write_todos_sequential_updates_returns_text(model: BaseChatModel) -> None:
    """Creates a 5-item todo list and updates it 5 times, then responds with text."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        query=(...),
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=7,
            tool_call_requests=6,
            tool_calls=[tool_call(name="write_todos", step=i) for i in range(1, 7)],
        )
        .success(final_text_contains("DONE")),
    )
```

LLM-as-judge wraps **openevals** `create_llm_as_judge` (default judge `claude-sonnet-4-6`). Every
eval requires LangSmith tracing; results, correctness feedback, and step/tool metrics are logged to
LangSmith experiments. A separate **multi-trial** workflow (`scripts/run_trials.py`) runs the suite N
times for one model and aggregates **mean/median/stdev** — it measures *variance/stability*, not
pass@k. Model fan-out uses named **model groups** (`set0` = 29 models, `frontier`, `fast`, `open`, …).

A `radar.py` module renders per-category spider charts; `pytest_reporter.py` computes session
aggregates (correctness pass rate, step/tool ratios, solve-rate).

### Track 2 — Harbor / Terminal-Bench 2.0 (the focus)

The Harbor integration lets a Deep Agent run sandboxed container benchmarks. The key architectural
decision: **the agent runs on the orchestrator; only tool I/O runs in the container.** Harbor invokes
a Python agent class (not a CLI inside the container):

```372:380:/home/rohit/sandbox/swe-term/deepagents/.github/workflows/harbor.yml
          uv run harbor run \
            --agent-import-path deepagents_harbor:DeepAgentsWrapper \
            --dataset "$HARBOR_DATASET_NAME@$HARBOR_DATASET_VERSION" \
            -n "$HARBOR_CONCURRENCY" \
            ...
            --model "$HARBOR_MODEL" \
            --agent-kwarg use_cli_agent=...
```

`DeepAgentsWrapper` implements Harbor's `BaseAgent` (`name`/`setup`/`run`/`version`). Its `run()`
wraps the Harbor environment as a Deep Agents backend, builds an agent over it, and invokes it:

```260:294:/home/rohit/sandbox/swe-term/deepagents/libs/evals/deepagents_harbor/deepagents_wrapper.py
        backend = HarborSandbox(environment)
        ...
        if self._use_cli_agent:
            deep_agent, _ = create_cli_agent(
                model=self._model,
                assistant_id=environment.session_id,
                sandbox=backend,
                ...
                auto_approve=True,  # Skip HITL in Harbor
                enable_shell=False,  # Sandbox provides execution
            )
        else:
            deep_agent = create_deep_agent(
                model=self._model, backend=backend, system_prompt=system_prompt
            )
```

`HarborSandbox` implements `SandboxBackendProtocol`: `read`/`ls`/`grep`/`glob` run as **shell `exec`
inside the container**, while `write`/`edit` use Harbor's native `upload_file`/`download_file` (to
avoid `ARG_MAX` limits):

```45:51:/home/rohit/sandbox/swe-term/deepagents/libs/evals/deepagents_harbor/backend.py
class HarborSandbox(SandboxBackendProtocol):
    """A sandbox implementation using Harbor environments.

    Write and edit use Harbor's native file transfer (upload/download) for
    data content. Read, ls, grep, and glob execute shell commands in the
    environment.
    """
```

The lifecycle: Harbor pulls the `terminal-bench@2.0` dataset from its registry, provisions a sandbox
(`docker` / `daytona` / `modal` / `runloop`, or a custom `LangSmithEnvironment`), runs
`DeepAgentsWrapper`, the agent writes an **ATIF `trajectory.json`**, and **Harbor's verifier scripts**
(inside the container) produce the reward in `result.json`. A LangSmith bridge
(`deepagents_harbor/langsmith.py`) mirrors the Harbor dataset into a LangSmith dataset, creates an
experiment, traces each run, and posts the Harbor reward back as a `harbor_reward` feedback score.

### Track 3 — curated external benchmarks

`test_external_benchmarks.py` runs a **hard-set of 15 cases** drawn from three public benchmarks —
**FRAMES** (multi-hop retrieval), **Nexus** (nested function composition), **BFCL v3** (multi-turn
stateful tool calling, scored by **API state diff**). And `tau2_airline/` is a **simulated-user**
multi-turn eval (agent vs an LLM "customer" `UserSimulator`, scored on final DB state). These are
sampled, not full benchmark runners; **SWE-bench is not wired**.

### Eval CI

Three `workflow_dispatch` GitHub workflows: `evals.yml` (many models × 1 trial),
`evals_trials.yml` (1 model × N trials for variance), and `harbor.yml` (Terminal-Bench 2.0 via
Harbor). Unit tests run with `--disable-socket` (no live LLM).

---

## `libs/code` and `libs/cli`

- **`deepagents-code`** (`libs/code`) — a pre-built coding agent with a **Textual TUI** (the
  Claude-Code-like terminal app installed via `curl … | bash`). It was *forked from `deepagents-cli`*
  and calls `create_deep_agent` under the hood, adding CLI-specific middleware, **MCP** (via
  `langchain-mcp-adapters`), sandbox wiring, and local project context.
- **`deepagents-cli`** (`libs/cli`) — the deployment CLI (validate config, deploy to LangGraph
  Platform) plus a TypeScript `frontend/`.

So the "terminal coding agent" experience and MCP both live *above* the core SDK, mirroring
pi-mono's `coding-agent`-on-`agent-core` split.

---

## Key Design Patterns

### 1. Harness = middleware composition over a borrowed loop
Deep Agents owns no agent loop. It is a curated `AgentMiddleware` list handed to `create_agent`. New
capabilities are new middleware, not new graph code.

### 2. State as named LangGraph channels with delta reducers
`messages` and `files` are delta-reduced channels (O(N) checkpoints). Persistence is a pluggable
checkpointer; cross-session memory is a separate `BaseStore`.

### 3. The filesystem is the context-management substrate
The same backend that serves `read_file`/`write_file` is where summarized history and oversized tool
results are offloaded. Context overflow spills to the agent's own disk.

### 4. Pluggable backends behind one protocol
`StateBackend` / `FilesystemBackend` / `StoreBackend` / `CompositeBackend` / remote sandboxes all
satisfy `BackendProtocol`; the agent sees only the tools.

### 5. Sub-agents as isolated-context delegation
`task` spawns a child with the parent context stripped; a compiled LangGraph can *be* a sub-agent.

### 6. Protocol & eval as separate packages
ACP (`deepagents-acp`) and Harbor evals (`deepagents-evals`) are thin adapters in their own packages,
each implementing the *other* system's interface (`acp.Agent`, Harbor `BaseAgent`/`BaseEnvironment`).

### 7. "Trust the LLM" security
Per the README: the agent can do anything its tools allow; enforce boundaries at the **tool/sandbox**
layer, not by asking the model to self-police.

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Language / runtime | Python ≥ 3.11 / LangGraph |
| Core version | `deepagents` `0.6.7` (MIT) |
| Architecture | `create_deep_agent` → middleware list → `langchain.agents.create_agent` → `CompiledStateGraph` |
| Agent loop | **delegated to LangChain `create_agent`** (no custom graph in deepagents) |
| Core scale | ~48 src files, 13 middleware modules, ~19–22K LOC |
| Default middleware | todos, [skills], filesystem*, subagents*, summarization, patch-tool-calls, [async-subagents], profile, prompt-cache, [memory], [HITL] |
| State | `DeepAgentState` (delta-reduced `messages`); `FilesystemState.files`; LangGraph checkpointer + `BaseStore` |
| Backends | `StateBackend` (default), `FilesystemBackend`, `StoreBackend`, `CompositeBackend`, `LocalShellBackend`, remote sandboxes |
| Sub-agents | declarative `SubAgent`, `CompiledSubAgent` (any graph), `AsyncSubAgent` (LangGraph Platform) |
| Context mgmt | summarization + offload to `/conversation_history/`; tool-result eviction >20K tok to `/large_tool_results/`; `count_tokens_approximately` |
| Skills | `SKILL.md` (frontmatter + body), progressive disclosure |
| MCP | not in core; in `deepagents-code` + CLI |
| HITL | LangChain `HumanInTheLoopMiddleware` → LangGraph `interrupt()` |
| Sandboxes | partners: Runloop, Modal, Daytona; QuickJS in-process REPL |
| **ACP** | `deepagents-acp 0.0.6` over `agent-client-protocol 0.9.0`; `AgentServerACP(acp.Agent)`; implements initialize/new/prompt/cancel/set_mode/set_config_option; model switch via config options; **no** session/load, fs/terminal bridging, reasoning, or MCP |
| **Evals** | pytest + LangSmith, ~118 evals / 8 categories, two-tier scoring, openevals judge |
| **Harbor** | `harbor>=0.6.4`; `DeepAgentsWrapper(BaseAgent)` + `HarborSandbox(SandboxBackendProtocol)`; Terminal-Bench 2.0; agent on orchestrator, tool I/O in container; reward → LangSmith `harbor_reward` |
| Security model | "trust the LLM"; enforce at tool/sandbox layer |
| Downstream | `deepagents-code` (Textual TUI), `deepagents-cli`, `deepagents.js` (TS port) |

---

*Validated against the local Deep Agents monorepo at `/home/rohit/sandbox/swe-term/deepagents`
(core `0.6.7`, `deepagents-acp 0.0.6`, `deepagents-evals` on `harbor>=0.6.4`). External protocol/eval
references: [ACP](https://agentclientprotocol.com/protocol/overview) and
[Harbor](https://www.harborframework.com/docs). File and symbol references are from that checkout.*
