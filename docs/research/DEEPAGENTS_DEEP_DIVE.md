# Deep Agents Deep Dive — Architecture & Feature Analysis

> Analysis of [`langchain-ai/deepagents`](https://github.com/langchain-ai/deepagents) — LangChain's
> "batteries-included agent harness." An **opinionated agent that runs out of the box**, built as a
> thin opinionated layer of *middleware* on top of LangChain's `create_agent`, which itself sits on
> the **LangGraph** runtime.
> Language: Python. License: MIT. Distribution: PyPI (`deepagents`).
>
> **Validated against a local source checkout** at `/Users/rohit/sandbox/swe-term/deepagents`
> (commit `9f0e39d`, 2026-10-10; `deepagents` core `v0.7.23`, `deepagents-acp` `v0.0.12`,
> `deepagents-code` `v0.1.83`, `deepagents-talon` `v0.0.9`, `deepagents-evals` on
> `harbor[langsmith]>=0.20.0,<0.21.0`; versions from `.release-please-manifest.json` and
> `libs/evals/pyproject.toml`). Last refreshed **2026-10-10** (previous validation: core `0.6.7`,
> 2026-06-06). Two areas get extra depth at the user's request: the **ACP (Agent Client Protocol)
> integration** (`libs/acp`) and the **Harbor-based eval suite** (`libs/evals`). All file:line
> references are from that checkout.

---

## Changes since 0.6.7 (2026-06-06)

The checkout is shallow (history from 2026-09-28), so older changes are cited from
`libs/*/CHANGELOG.md` entries and their commits.

1. **`write_todos` left the default stack.** `TodoListMiddleware` is no longer added by
   `create_deep_agent`; only the OpenAI Codex harness profile re-adds it
   (`libs/deepagents/CHANGELOG.md` 0.7.0 BREAKING, commit `9340518`;
   `profiles/harness/_openai_codex.py`).
2. **Lean prompts.** The authored base prompt is empty and tool-usage prose constants
   (`FILESYSTEM_SYSTEM_PROMPT`, `TASK_SYSTEM_PROMPT`, …) were removed; `BASE_AGENT_PROMPT` is
   deprecated (0.7.0, commits `a8d1b32`, `d9f54fc`).
3. **Safer filesystem defaults, plus a destructive `delete` tool.** `FilesystemBackend` /
   `LocalShellBackend` default to `virtual_mode=True` (no `..` escape), `write_file` now overwrites,
   and a recursive `delete` tool is exposed when the backend supports it (0.7.0, commits `540a0fa`,
   `2506fcc`, `f2a21ec`). `FilesystemMiddleware(tools=[...])` takes an allowlist (`FsToolName`).
4. **Permissions became a HITL source.** `permissions=` rules with interrupt mode are merged into
   `interrupt_on` for the main agent and the general-purpose sub-agent
   (`graph.py` `_merge_fs_interrupt_on` / `_build_interrupt_on_from_permissions`; 0.6.8 commit
   `a090162`).
5. **Backend compatibility shims removed.** Backend factories (`BackendFactory`), `ls_info`/
   `grep_raw`-style methods, `files_update`, and `SummarizationMiddleware(history_path_prefix=)` are
   gone; `create_deep_agent(backend=)` takes a concrete `BackendProtocol` only (0.7.0, commit
   `540a0fa`).
6. **Sub-agent `mode="fork"`** (experimental) lets a child continue the parent's conversation;
   the default mode was renamed `handoff` → `isolated` (`middleware/subagents.py`;
   CHANGELOG #5714, #6030).
7. **Harbor integration re-architected.** `DeepAgentsWrapper` / `HarborSandbox` are gone. Harbor's
   built-in `langgraph` agent now runs a LangGraph project (`bare` / `dcode` / `tau3` graphs) inside
   the task container, with `LocalShellBackend` as the backend
   (`libs/evals/deepagents_harbor/langgraph_project/`, `.github/workflows/_harbor_run.yml`). A new
   `unified_evals.yml` reports pass@K / avg@K across capability axes (`libs/evals/UNIFIED_EVALS.md`).
8. **ACP grew session load, thought streaming, and failure status.** Opt-in durable `session/load`
   (`load_sessions=True`), `agent_thought_chunk` for visible reasoning, and `failed` tool status
   (`libs/acp/CHANGELOG.md` 0.0.10–0.0.12; `deepagents_acp/server.py`).
9. **`libs/cli` is gone; `libs/talon` is new.** `deepagents-talon` is an experimental local runtime
   host for long-running agents (Slack/Discord/Telegram/WhatsApp channels, cron, MCP, history
   stores) built on `deepagents-code` (`libs/talon/README.md`, `libs/talon/deepagents_talon/`).
   A `langchain-vercel-sandbox` partner was added (`libs/partners/vercel`).
10. **`langsmith` is now a core dependency** of `deepagents` (`libs/deepagents/pyproject.toml`).

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
15. [`libs/code` and `libs/talon`](#libscode-and-libstalon)
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
Agents adds — a virtual filesystem, sub-agents, summarization, skills, memory — is a middleware.
(Planning via `write_todos` is now opt-in; see the middleware stack below.)

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
| Observability | LangSmith (tracing, datasets, experiments); `langsmith` is a core dependency |
| Editor integration | **ACP** via `deepagents-acp` (`agent-client-protocol>=0.10.1`) |
| Evals | pytest + LangSmith + **Harbor** (`harbor[langsmith]>=0.20.0,<0.21.0`) |

**Scale (core `libs/deepagents/deepagents`):** 56 Python source files, **20 middleware modules**
(8 public, 12 private `_*.py` helpers), ~29.7K LOC. The largest files are `middleware/filesystem.py`
(3,739 ln), `middleware/summarization.py` (2,314 ln), `backends/sandbox.py` (2,116 ln),
`backends/filesystem.py` (1,576 ln), `middleware/rubric.py` (1,438 ln), `middleware/skills.py`
(1,432 ln), and the assembly file `graph.py` (1,011 ln).

---

## The LangGraph → create_agent → Deep Agents Stack

The single most important mental model is the three-layer stack (the README dedicates an FAQ to it):

```
LangGraph            the graph runtime: nodes, edges, channels, checkpointers, streaming
   │
create_agent         LangChain's minimal harness: the ReAct loop + an AgentMiddleware host
   │
Deep Agents          an opinionated bundle of middleware (files, subagents, summarization,
                     skills, memory, HITL) + pluggable backends, exposed via create_deep_agent()
```

The layers **compose downward**: any LangGraph `CompiledStateGraph` can be handed to a Deep Agent as
a sub-agent (`CompiledSubAgent`), so custom orchestration plugs in alongside the harness defaults.
This is fundamentally different from pi-mono/Codex/Claude Code, which each own their loop. Deep
Agents *borrows* the loop and competes on **middleware composition**. The repo now documents this
stack itself in `libs/ARCHITECTURE.md` ("The three layers").

---

## Monorepo Layout

A `uv`-managed Python monorepo (`libs/`):

| Package | PyPI name | Role |
|---------|-----------|------|
| **`libs/deepagents`** | `deepagents` (`0.7.23`) | Core SDK — `create_deep_agent`, middleware, backends, profiles |
| **`libs/acp`** | `deepagents-acp` (`0.0.12`) | Agent Client Protocol connector (run a Deep Agent inside Zed and other ACP editors) |
| **`libs/evals`** | `deepagents-evals` (`0.0.1`) | Behavioral eval suite + **Harbor** integration + unified cross-model evals |
| **`libs/code`** | `deepagents-code` (`0.1.83`) | Pre-built coding agent with a **Textual TUI**; pins `deepagents==0.7.23` and depends on `deepagents-acp` |
| **`libs/talon`** | `deepagents-talon` (`0.0.9`) | *Experimental* local runtime host: chat channels, cron, MCP, history stores (new) |
| **`libs/partners/*`** | `langchain-{runloop,modal,daytona,quickjs}`, `langchain-vercel-sandbox` | Sandbox / code-execution provider integrations |
| **`examples/*`** | — | 15 worked examples (deep research, text-to-SQL, `ralph_mode`, `rubric_middleware`, `talon`, deploy-* recipes) |

`libs/cli` (`deepagents-cli`) no longer exists in the tree. Maintainer docs now live alongside the
code: `libs/ARCHITECTURE.md`, `libs/deepagents/THREAT_MODEL.md`, and `libs/code/THREAT_MODEL.md`.

---

## `create_deep_agent`: The Entry Point

There is exactly one public constructor (no async variant), defined in
`libs/deepagents/deepagents/graph.py`:

```277:297:/Users/rohit/sandbox/swe-term/deepagents/libs/deepagents/deepagents/graph.py
def create_deep_agent(  # noqa: C901, PLR0912, PLR0915  # Complex graph assembly logic with many conditional branches
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    permissions: list[FilesystemPermission] | None = None,
    backend: BackendProtocol | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None = None,
    state_schema: type[DeepAgentState] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph[AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]]:
```

`backend=` no longer accepts a factory: 0.7.0 removed the backend compatibility shims
(`libs/deepagents/CHANGELOG.md`, commit `540a0fa`).

The body resolves the model and a **harness profile**, assembles the middleware list, and delegates:

```989:1010:/Users/rohit/sandbox/swe-term/deepagents/libs/deepagents/deepagents/graph.py
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
    ).with_config(
        {"recursion_limit": 9_999, "metadata": {"ls_integration": "deepagents", ...}}
    )
```

Note the `recursion_limit: 9_999` — Deep Agents expects **long-horizon** runs (many loop turns), and
the LangGraph recursion guard is set high enough not to interfere. The config also stamps LangSmith
integration metadata on every run. If `model=None`, it falls back to a deprecated
`ChatAnthropic(model_name="claude-sonnet-4-6")` (`_build_default_model`, slated for removal in
`1.0.0`); if `backend=None`, it uses a `StateBackend` (in-state virtual filesystem).

---

## The Middleware Stack

This is where the "harness" actually lives. `graph.py` assembles middleware in a fixed order; each
contributes tools and/or wraps the model call. Default order for the main agent
(`graph.py:888-970`):

| # | Middleware | Source | Contributes |
|---|-----------|--------|-------------|
| 1 | `FilesystemMiddleware` **(required)** | `middleware/filesystem.py` | `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`, `execute` (each only if the backend supports it); permissions; large-result eviction |
| 2 | `SubAgentMiddleware` **(required)** | `middleware/subagents.py` | The `task` tool → spawn child agents |
| 3 | `SummarizationMiddleware` | `middleware/summarization.py` | Context compaction + history offload to disk |
| 4 | `PatchToolCallsMiddleware` | `middleware/patch_tool_calls.py` | Repairs dangling/malformed tool calls before a run |
| 5 | `AsyncSubAgentMiddleware` *(if remote specs)* | `middleware/async_subagents.py` | Background subagents on LangSmith/LangGraph deployments |
| — | *user middleware* (`middleware=`) | — | Spliced in after the core; a user instance whose `.name` matches a default **replaces** it (0.7.0) |
| 6 | profile extras | `profiles/` | Per-model tuning |
| 7 | `SkillsMiddleware` *(if `skills=`)* | `middleware/skills.py` | Skill metadata in the prompt; body (and skill-scoped tools) loaded on read |
| 8 | prompt caching | `middleware/_prompt_caching.py` | Anthropic prompt caching (+ Fireworks session affinity when available) |
| 9 | `MemoryMiddleware` *(if `memory=`)* | `middleware/memory.py` | Injects `AGENTS.md`-style memory into the prompt |
| 10 | `HumanInTheLoopMiddleware` *(if `interrupt_on=` or interrupt-mode `permissions=`)* | LangChain | Tool-call approval/edit/reject |
| 11 | `UnsupportedContentMiddleware` | `middleware/unsupported_content.py` | Filters multimodal blocks the model cannot accept (new) |
| 12 | `_ToolExclusionMiddleware` *(if profile excludes tools)* | `middleware/_tool_exclusion.py` | Strips tools last, so custom middleware cannot restore them |

**`TodoListMiddleware` is no longer in the default stack** (0.7.0 breaking change, commit
`9340518`). Callers restore `write_todos` with `middleware=[TodoListMiddleware()]`; the built-in
OpenAI Codex harness profile does so itself (`profiles/harness/_openai_codex.py`). Skills moved from
near the top to just before prompt caching, so skill disclosure sees the compacted conversation.

`FilesystemMiddleware` and `SubAgentMiddleware` are declared `_REQUIRED_MIDDLEWARE` — core features
depend on them and a profile's `excluded_middleware` cannot strip them (it raises `ValueError`):

```247:250:/Users/rohit/sandbox/swe-term/deepagents/libs/deepagents/deepagents/graph.py
_REQUIRED_MIDDLEWARE: tuple[tuple[type[AgentMiddleware[Any, Any, Any]], tuple[str, ...]], ...] = (
    (FilesystemMiddleware, ()),
    (SubAgentMiddleware, ()),
)
```

The filesystem tools are now built from a factory list filtered by an optional allowlist
(`FilesystemMiddleware(tools=[...])`, typed `FsToolName`):

```1906:1906:/Users/rohit/sandbox/swe-term/deepagents/libs/deepagents/deepagents/middleware/filesystem.py
        self.tools = [factory() for name, factory in tool_factories if self._enabled_tools is None or name in self._enabled_tools]
```

Two tool semantics changed in 0.7.0: `write_file` now creates *or overwrites* (no file-exists
error), and the new recursive `delete` tool counts as a write for permission checks (commits
`2506fcc`, `f2a21ec`).

(`RubricMiddleware`, now 1,438 lines with grader integration hooks, exists for grader sub-loops but
is not in the default stack.)

---

## State Model & Checkpointing

State is a LangGraph state schema. `DeepAgentState` extends LangChain's `AgentState` and applies a
**delta reducer** to `messages` so checkpoint size grows O(N) rather than O(N²):

```74:77:/Users/rohit/sandbox/swe-term/deepagents/libs/deepagents/deepagents/graph.py
class DeepAgentState(AgentState):
    """AgentState with DeltaChannel on messages to reduce checkpoint growth from O(N²) to O(N)."""

    messages: Required[Annotated[list[AnyMessage], DeltaChannel(_messages_delta_reducer, snapshot_frequency=50)]]
```

The virtual filesystem is a second delta-reduced channel added by the filesystem middleware:

```1171:1174:/Users/rohit/sandbox/swe-term/deepagents/libs/deepagents/deepagents/middleware/filesystem.py
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
(`backends/protocol.py:404`) with `ls`/`read`/`grep`/`glob`/`write`/`edit`/`delete`/`upload_files`/
`download_files` (plus async `a*` variants). A `SandboxBackendProtocol` subtype (`protocol.py:886`)
adds `execute`/`aexecute` for shell. Since 0.7.0, `delete` is optional per backend (the tool is hidden
when unsupported), `GrepResult`/`GlobResult` carry a `truncated` flag, and the old `ls_info`/
`grep_raw`/`files_update` compatibility surface is gone (`libs/deepagents/CHANGELOG.md` 0.7.0).
`CompositeBackend(artifacts_root=...)` now owns where offloaded artifacts land.

Shipped implementations (`backends/__init__.py`):

| Backend | File | Semantics |
|---------|------|-----------|
| `StateBackend` *(default)* | `backends/state.py` | Virtual FS living in graph state; checkpointed per thread; not cross-thread |
| `FilesystemBackend` | `backends/filesystem.py` | Real disk under a `root_dir`; `virtual_mode=True` by default (paths anchored, `..` rejected) |
| `StoreBackend` | `backends/store.py` | LangGraph `BaseStore` — persistent, cross-thread; explicit `namespace` now required |
| `CompositeBackend` | `backends/composite.py` | Routes by path prefix (e.g. `/memories/` → store, rest → sandbox); `artifacts_root` for offloads |
| `LocalShellBackend` | `backends/local_shell.py` | `FilesystemBackend` + host shell `execute` (path-anchored by default, but no process isolation) |
| `BaseSandbox` | `backends/sandbox.py` | File ops implemented via a remote sandbox's `execute()` |
| `ContextHubBackend` | `backends/context_hub.py` | Files persisted in a LangSmith Hub agent repo; concurrent mutations batched (#5471) |
| `LangSmithSandbox` | `backends/langsmith.py` | LangSmith-managed sandbox; async client; captures oversized `execute` output in-sandbox |

The agent only ever sees the **tools** (`read_file`, `write_file`, …); the backend decides whether
those hit graph state, local disk, a LangGraph store, or a remote container. This is the same
"pluggable filesystem" idea swe-term wants — implemented as a Python ABC rather than a Go interface.

---

## Sub-agents

Sub-agents are the delegation primitive, exposed as a `task` tool by `SubAgentMiddleware`. Three
flavors:

- **`SubAgent`** (declarative dict): `name`, `description`, `system_prompt`, optional `tools`,
  `model`, `middleware`, `skills`, `permissions`, `response_format`, and `mode`
  (`"isolated"` default, or experimental `"fork"`). Compiled with `create_agent`.
- **`CompiledSubAgent`**: `name`, `description`, `runnable` — **any** `CompiledStateGraph` / Runnable.
  This is the composition seam: a hand-built LangGraph plugs in as a sub-agent.
- **`AsyncSubAgent`**: `graph_id` (+ optional `url`/`headers`) — a **remote** agent run on a
  LangSmith/LangGraph deployment via the LangGraph SDK, in the background.

A default `general-purpose` sub-agent is provided unless overridden; it inherits the parent's
permissions and permission-derived interrupts (`graph.py:870-884`). By default sub-agents run with
**isolated context** — excluded and private state keys are stripped and the child gets only its task
description, then a single `ToolMessage` is merged back:

```781:783:/Users/rohit/sandbox/swe-term/deepagents/libs/deepagents/deepagents/middleware/subagents.py
            subagent_state = {key: value for key, value in runtime.state.items() if key not in _EXCLUDED_STATE_KEYS | private_state_keys}
            subagent_state["messages"] = [HumanMessage(content=description)]
        return subagent_runnable, subagent_state
```

This is the "isolated context window for delegated work" pattern Claude Code popularized — here it
falls out of LangGraph state-scoping. Fields marked `PrivateStateAttr` (including caller
`state_schema` fields) never cross the boundary in either direction (0.7.0, commit `a4662c0`).

**New: `mode="fork"`** (experimental) instead passes the parent's message history (with the
summarization event folded in) plus a fork preamble, so the child continues the parent's
conversation rather than starting cold (`subagents.py:51-84`, `:363`; CHANGELOG #5714). Skills are
forbidden on forked sub-agents. Unknown `task` argument keys are now rejected (#6299).

---

## Context Management (Summarization & Offload)

Context management is two cooperating mechanisms, both backed by the filesystem:

1. **History summarization** — `create_summarization_middleware(model, backend)` wraps LangChain's
   summarization. When the thread approaches the model's input limit, older messages are summarized;
   the evicted history is **offloaded to disk** at `/conversation_history/{session_id}.md` (media under
   `<artifacts_root>/conversation_history/media/`) rather than discarded
   (`middleware/summarization.py:44-49`). Token counting defaults to `count_tokens_approximately` (a
   heuristic, `summarization.py:554`), with thresholds derived from `model.profile["max_input_tokens"]`.
   Compaction recovery is now bounded and input budgets validated (#6267).

2. **Tool-result offload** — when a single tool result exceeds `tool_token_limit_before_evict`
   (default **20,000 tokens**, estimated ≈ chars/4; `middleware/filesystem.py:1771`), the filesystem
   middleware writes it to `{artifacts_root}/large_tool_results/<tool_call_id>` and leaves a preview
   that states what was clipped (`_message_eviction.py`, `_overflow_clip.py`). Newer helpers offload
   binary blobs to the backend (`_blob_offload.py`, #6550), and opted-in sandboxes capture oversized
   `execute` output directly in the sandbox to save a round trip (0.7.0).

Both reuse the backend as scratch storage — the agent's own filesystem is also its context-overflow
spillover. (Note: this is an *approximate* token count, like pi-mono's `chars/4`, not a real
tokenizer.)

---

## Skills, Tools & MCP

- **Skills** (`middleware/skills.py`): Anthropic-style progressive disclosure. A skill is a directory
  with a `SKILL.md` (YAML frontmatter + Markdown body) plus optional helper files; the middleware
  surfaces skill metadata in the system prompt and the agent loads the full body on demand. Sources
  are passed via `skills=[...]` and resolved through the backend (so skills can live in state, on
  disk, or in a store). Recent additions: skill-scoped tools load only when the skill is read
  (`middleware/_skill_tools.py`, #6552), `pinned_skills` (#6810), and per-sub-agent
  `SkillsMiddleware` instances (#6820).
- **Tools**: ordinary LangChain `BaseTool`s / callables passed to `create_deep_agent(tools=[...])`,
  merged with the middleware-provided tools.
- **MCP**: **still not in the core package.** `libs/deepagents/deepagents` contains no MCP code and
  its `pyproject.toml` declares no MCP dependency. MCP support lives in `deepagents-code`
  (`mcp_tools.py`, now on `fastmcp` + `langchain.mcp.as_langchain_tool`) and in `deepagents-talon`
  (`mcp*.py`). The README's "any MCP server" is a consumer-side integration, not a core feature.

---

## Human-in-the-Loop

HITL is LangChain's `HumanInTheLoopMiddleware`, appended only when `interrupt_on=` is set or a
`permissions=` rule uses interrupt mode (the two are merged by `_merge_fs_interrupt_on` in
`graph.py:186`; interrupt-mode permissions landed in 0.6.8, commit `a090162`). It uses
LangGraph's `interrupt()`/checkpointer mechanism: when a guarded tool is about to run, the graph
**interrupts** and surfaces an interrupt payload with `action_requests` and `review_configs`
(allowed decisions: `approve`/`edit`/`reject`/`respond`). The caller resumes with
`Command(resume={"decisions": [...]})`. Sub-agents inherit the parent's `interrupt_on` unless
overridden. This same interrupt is what the ACP layer maps to `session/request_permission` (below).

---

## Shell & Sandbox Partners

Shell execution is a backend capability (`SandboxBackendProtocol.execute`). In-tree,
`LocalShellBackend` runs commands on the host with **no process isolation** (0.7.0 anchors its file
paths under `root_dir` by default, but shell commands are not confined). Real isolation comes from
`LangSmithSandbox` in core or the partner packages:

| Partner | Class | What it is |
|---------|-------|-----------|
| `langchain-runloop` | `RunloopSandbox` | Runloop devbox cloud sandboxes |
| `langchain-modal` | `ModalSandbox` | Modal sandboxes |
| `langchain-daytona` | `DaytonaSandbox` | Daytona cloud workspaces |
| `langchain-vercel-sandbox` *(new)* | see `libs/partners/vercel` | Vercel Sandbox |
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

`deepagents-acp` (`v0.0.12`) bridges a Deep Agent to ACP so it can run inside Zed. It is a **thin
adapter** (~1,750 LOC; `server.py` 1,354 ln) over the PyPI `agent-client-protocol` package (imported
as `acp`, now required at **`>=0.10.1`**, `libs/acp/pyproject.toml`). `deepagents-code` now depends on
it, so the shipped TUI agent also speaks ACP (`libs/code/deepagents_code/acp.py`). The protocol library owns the stdio JSON-RPC transport and the
type/builder helpers; `deepagents_acp` owns only the translation to LangGraph.

### The adapter object

```224:225:/Users/rohit/sandbox/swe-term/deepagents/libs/acp/deepagents_acp/server.py
class AgentServerACP(ACPAgent):
    """ACP agent server that bridges Deep Agents with the Agent Client Protocol."""
```

It accepts either a compiled Deep Agent or a **factory** `Callable[[AgentSessionContext], CompiledStateGraph]`
(needed for model switching), plus optional `modes` (deprecated in favor of config options),
`models`, and `load_sessions` (opt-in durable `session/load`; requires a checkpointer that survives
restarts — `server.py:229-252`).

### Implemented ACP methods

| ACP method | `AgentServerACP` handler | Behavior |
|------------|--------------------------|----------|
| `initialize` | `initialize()` | Advertises **only** `promptCapabilities.image = true` |
| `session/new` | `new_session()` | `session_id = uuid4().hex`; used as the LangGraph `thread_id`; `mcp_servers` stored per session but **never passed to the agent** |
| `session/load` *(opt-in, new)* | `load_session()` | Only when `load_sessions=True`; restores cwd/mode/model metadata, else `method_not_found` |
| `session/prompt` | `prompt()` | The core turn (below) |
| `session/cancel` | `cancel()` | Sets a per-session flag polled in the stream loop (scoped per session in 0.0.12) |
| `session/set_mode` | `set_session_mode()` | Updates mode, rebuilds the agent |
| `session/set_config_option` | `set_config_option()` | `mode` or `model` switch → rebuild |
| `on_connect` | `on_connect()` | Stores the client conn for `session_update`/`request_permission` |

### The prompt turn → LangGraph translation

ACP content blocks are converted to LangChain multimodal content, then the graph is streamed and each
LangGraph event is translated into a `session/update`:

```1007:1015:/Users/rohit/sandbox/swe-term/deepagents/libs/acp/deepagents_acp/server.py
            pending_interrupts: Sequence[Interrupt] = ()
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
| Visible reasoning chunk *(new, 0.0.11)* | `agent_thought_chunk` (`server.py:199`) |
| `tool_call_chunks` on an `AIMessageChunk` | `tool_call` (`ToolCallStart`) — `edit_file` gets a diff via `start_edit_tool_call` |
| `ToolMessage` (non-`edit_file`) | `tool_call_update` with `status="failed"` if the message is an error, else `"completed"` (`server.py:915`) |
| `write_todos` args / `updates["todos"]` | `plan` (`AgentPlanUpdate`) — only fires if the agent opted into `TodoListMiddleware` |
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

```928:941:/Users/rohit/sandbox/swe-term/deepagents/libs/acp/deepagents_acp/server.py
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

- **No `authenticate`, `session/list`, fork/resume/close.** `session/load` now exists but is
  opt-in (`load_sessions=True`) and off by default.
- **No `set_session_model`** — model changes go through `set_config_option` only.
- **`mcp_servers` from `session/new`/`session/load` are stored but unused** — `_session_mcp_servers`
  is written (`server.py:273,375,419`) and never read; `AgentSessionContext` carries only
  `cwd`/`mode`/`model` (`server.py:216-221`).
- **No client `fs/*` or `terminal/*` bridging** — the Deep Agent uses its **own** local backends
  (`LocalShellBackend` + `CompositeBackend` in the demo), *not* the editor's filesystem/terminal.
  So the agent does not honor the ACP host's file/terminal capabilities.
- ~~No reasoning stream~~ — **fixed**: visible reasoning streams as `agent_thought_chunk` (0.0.11).
- **Tool status is `pending → completed | failed`** — `failed` was added; `in_progress` is still never
  emitted for tool calls, and `edit_file` completion updates are still skipped (`server.py:901`).
- **Free-form LangGraph `interrupt()` is unsupported** — it raises a `RequestError` explaining the
  ACP limitation.
- **Audio prompts** raise `NotImplementedError`.

So the integration cleanly covers the *prompt-turn-with-tools-and-permissions* core of ACP and leaves
the resource-bridging surface (client `fs/*`, `terminal/*`, MCP pass-through) unimplemented.
Session management is partially covered since 0.0.10.

---

## The Eval Suite & Harbor (`libs/evals`)

> Background: [Harbor](https://www.harborframework.com/docs) (laude-institute) is "a framework for
> evaluating and optimizing agents and models in **container environments**" — the productionized
> successor to Terminal-Bench. It provides modular `environment`/`agent`/`task` interfaces, a
> **registry** of benchmarks/datasets, pre-integrated CLI agents, and integrations with cloud sandbox
> providers (Daytona, Modal, E2B, Runloop, Tensorlake) and optimizers (SkyRL, GEPA).

`deepagents-evals` has two distinct evaluation tracks, plus a newer unified cross-model battery.

### Track 1 — Behavioral evals (pytest + LangSmith)

~**136 test functions** in `tests/evals/` (counted from `def test_` definitions; catalogued in
`libs/evals/EVAL_CATALOG.md`, categories in `deepagents_evals/categories.json`). Each is a pytest test
that builds a real agent, runs it against a real LLM, captures the trajectory, and scores it. The
scoring contract is still **two-tier** (`TrajectoryScorer`, `tests/evals/utils.py:1198-1226`):

- **`.success(...)`** — correctness assertions that **hard-fail** the test (`final_text_contains`,
  `file_equals`, `llm_judge`, …).
- **`.expect(...)`** — efficiency expectations (step count, tool-call shape) that are **logged but
  never fail**.

```27:47:/Users/rohit/sandbox/swe-term/deepagents/libs/evals/tests/evals/test_todos.py
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

(This eval still builds `create_deep_agent(model=model)` with no `TodoListMiddleware`, although 0.7.0
removed `write_todos` from the default stack; the snippet is quoted as found.)

LLM-as-judge defaults to `claude-sonnet-4-6` (`tests/evals/llm_judge.py:25`). Every eval still
requires LangSmith tracing (`tests/evals/conftest.py:34`); results, correctness feedback, and
step/tool metrics are logged to LangSmith experiments. A separate **multi-trial** workflow
(`scripts/run_trials.py`) runs the suite N times for one model and aggregates
**mean/median/stdev** — it measures *variance/stability*. Model fan-out uses named **model groups**
(`set0` = 25 models, `frontier` = 5, …; `MODEL_GROUPS.md`). `deepagents_evals/radar.py` renders
per-category spider charts; `pytest_reporter.py` computes session aggregates.

### Track 2 — Harbor (re-architected)

The previous design — a `DeepAgentsWrapper(BaseAgent)` running on the orchestrator with a
`HarborSandbox(SandboxBackendProtocol)` proxying tool I/O into the container — **is gone**; neither
class exists in `libs/evals`. Harbor's **built-in `langgraph` agent** now loads a LangGraph project and
runs the graph next to the task:

```962:968:/Users/rohit/sandbox/swe-term/deepagents/.github/workflows/_harbor_run.yml
          uv run harbor run \
            --yes \
            --agent langgraph \
            --agent-kwarg project_path=deepagents_harbor/langgraph_project \
            --agent-kwarg config=langgraph.json \
            --agent-kwarg graph="$HARBOR_AGENT_GRAPH" \
            ...
```

`deepagents_harbor/langgraph_project/langgraph.json` registers three graphs:
`dcode` (`make_graph`, the `deepagents-code` product agent via `create_cli_agent`), `bare`
(`make_bare_graph`, plain `create_deep_agent`), and `tau3` (simulated-user conversation). The bare
graph uses an ordinary in-tree backend, not a Harbor-specific one:

```440:446:/Users/rohit/sandbox/swe-term/deepagents/libs/evals/deepagents_harbor/langgraph_project/langgraph_agent.py
    backend = LocalShellBackend(
        root_dir=_workdir(configurable), inherit_env=False, virtual_mode=False
    )
    ...
    return create_deep_agent(
        model=model,
        backend=backend,
```

Isolation is delegated to the task container ("isolation is the container's job", same file). A
shell-environment denylist strips provider and LangSmith keys from the agent's shell
(`_SHELL_ENV_DENYLIST`), and task-declared MCP servers forwarded by Harbor are connected through
`langchain.mcp` (`MCPAdapter` import; `_mcp_connections`). Dependencies are pinned to
`harbor[langsmith]>=0.20.0,<0.21.0` and `harbor-langsmith>=0.3.0,<0.4.0` (`libs/evals/pyproject.toml`).

`harbor.yml` defaults to `terminal-bench/terminal-bench-2` (2-1 selectable) and to the **`dcode`**
agent (`.github/workflows/harbor.yml:34,117`); sandboxes are `docker` or `langsmith` (default). The
LangSmith bridge (`deepagents_harbor/langsmith.py`) still posts the Harbor reward as a
`harbor_reward` feedback score (`langsmith.py:603-622`).

### Track 3 — unified evals and curated external benchmarks

`unified_evals.yml` (documented in `libs/evals/UNIFIED_EVALS.md`) runs a fixed battery per model
and reports **pass@K / avg@K** plus a radar chart across four capability axes: **autonomous**
(`harbor-index`), **conversation** (`tau3-subset`, τ³-bench), **context** (Context-Bench), and
**research** (DRBench, continuous scores). Each axis can use the `bare` or `dcode` harness. Harbor
adapters for these live in `harbor_adapters/` (`contextbench`, `drbench`). A `deepagents_clbench/`
package also exposes a Deep Agent to continual-learning-bench.

The pytest side still has `test_external_benchmarks.py` (sampled FRAMES / Nexus / BFCL v3 cases) and
`tau2_airline/` (simulated-user eval scored on final DB state). **SWE-bench is still not wired.**

### Eval CI

`workflow_dispatch` workflows: `evals.yml` (many models × 1 trial), `evals_trials.yml` (1 model ×
N trials), `harbor.yml` (Harbor datasets), `unified_evals.yml` (cross-model battery), and
`clbench.yml`; `_harbor_run.yml` / `_eval.yml` are the shared reusable jobs. A `test_hitl.py` is still
listed in `libs/evals/CONTRIBUTING.md:81` but is not in `tests/evals/`.

---

## `libs/code` and `libs/talon`

- **`deepagents-code`** (`libs/code`, `0.1.83`) — a pre-built coding agent with a **Textual TUI**. It
  pins `deepagents==0.7.23` and calls `create_deep_agent` under the hood, adding CLI-specific
  middleware, hooks (`HOOKS.md`), **MCP** (`mcp_tools.py` on `fastmcp`, plus MCP OAuth and a proxy),
  an **ACP** server mode (`acp.py`, depends on `deepagents-acp`), sandbox wiring, and local project
  context. It ships its own `THREAT_MODEL.md` and `ARCHITECTURE.md`.
- **`deepagents-talon`** (`libs/talon`, `0.0.9`, *experimental/alpha*) — "the local runtime host for
  long-running Deep Agents" (`libs/talon/README.md`): one event loop that owns chat-channel adapters
  (`channels/`: Slack, Discord, Telegram, WhatsApp), cron schedulers, MCP, tool approvals, and
  pluggable conversation-history stores (SQLite, Postgres, vector/embedding backends). It builds on
  `deepagents-code`.
- **`deepagents-cli`** (`libs/cli`) **has been removed** from the monorepo.

So the "terminal coding agent" experience, MCP, and now the always-on runtime all live *above* the
core SDK, mirroring pi-mono's `coding-agent`-on-`agent-core` split.

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
`task` spawns a child with the parent context stripped (or, experimentally, forked); a compiled
LangGraph can *be* a sub-agent.

### 6. Protocol & eval as separate packages
ACP (`deepagents-acp`) and Harbor evals (`deepagents-evals`) live in their own packages. The ACP side
still implements the *other* system's interface (`acp.Agent`). The Harbor side no longer does: instead
of a custom `BaseAgent` + sandbox backend, the agent is exposed as a plain LangGraph graph that
Harbor's generic `langgraph` agent hosts. The seam moved from "implement their interface" to "ship a
standard graph artifact".

### 7. "Trust the LLM" security
Per the README: the agent can do anything its tools allow; enforce boundaries at the **tool/sandbox**
layer, not by asking the model to self-police (`README.md:112`). 0.7.0 tightened the defaults that
model sits on: path-anchored local backends, permission rules that can raise HITL interrupts, and a
published `THREAT_MODEL.md` per package.

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Language / runtime | Python ≥ 3.11 / LangGraph |
| Core version | `deepagents` `0.7.23` (MIT), commit `9f0e39d` |
| Architecture | `create_deep_agent` → middleware list → `langchain.agents.create_agent` → `CompiledStateGraph` |
| Agent loop | **delegated to LangChain `create_agent`** (no custom graph in deepagents) |
| Core scale | 56 src files, 20 middleware modules (8 public), ~29.7K LOC |
| Default middleware | filesystem*, subagents*, summarization, patch-tool-calls, [async-subagents], *user*, profile, [skills], prompt-cache, [memory], [HITL], unsupported-content, [tool-exclusion]; todos now opt-in |
| State | `DeepAgentState` (delta-reduced `messages`); `FilesystemState.files`; LangGraph checkpointer + `BaseStore` |
| Backends | `StateBackend` (default), `FilesystemBackend`, `StoreBackend`, `CompositeBackend`, `LocalShellBackend`, `ContextHubBackend`, `LangSmithSandbox`, partner sandboxes; concrete instances only (no factories) |
| Sub-agents | declarative `SubAgent` (`mode` isolated/fork), `CompiledSubAgent` (any graph), `AsyncSubAgent` (remote deployment) |
| Context mgmt | summarization + offload to `/conversation_history/`; tool-result eviction >20K tok to `/large_tool_results/`; `count_tokens_approximately` |
| Skills | `SKILL.md` (frontmatter + body), progressive disclosure; skill tools loaded on read; pinned skills |
| MCP | not in core; in `deepagents-code` (`fastmcp`) and `deepagents-talon` |
| HITL | LangChain `HumanInTheLoopMiddleware` → LangGraph `interrupt()`; also driven by interrupt-mode `permissions` |
| Sandboxes | core `LangSmithSandbox`; partners: Runloop, Modal, Daytona, Vercel; QuickJS in-process REPL |
| **ACP** | `deepagents-acp 0.0.12` over `agent-client-protocol>=0.10.1`; `AgentServerACP(acp.Agent)`; initialize/new/prompt/cancel/set_mode/set_config_option + opt-in session/load; thought chunks; `failed` tool status; **no** fs/terminal bridging or MCP pass-through |
| **Evals** | pytest + LangSmith, ~136 eval tests, two-tier scoring; unified pass@K/avg@K battery (autonomous/conversation/context/research) |
| **Harbor** | `harbor[langsmith]>=0.20,<0.21`; Harbor's built-in `langgraph` agent hosts `bare`/`dcode`/`tau3` graphs; `LocalShellBackend` inside the task container; Terminal-Bench 2/2.1 default `dcode`; reward → LangSmith `harbor_reward` |
| Security model | "trust the LLM"; enforce at tool/sandbox layer |
| Downstream | `deepagents-code` (Textual TUI + ACP), `deepagents-talon` (experimental runtime host), `deepagents.js` (TS port, separate repo per `README.md:44`) |

---

*Validated on 2026-10-10 against the local Deep Agents monorepo at
`/Users/rohit/sandbox/swe-term/deepagents` (commit `9f0e39d`; core `0.7.23`, `deepagents-acp 0.0.12`,
`deepagents-code 0.1.83`, `deepagents-talon 0.0.9`, `deepagents-evals` on `harbor[langsmith]>=0.20.0`). External protocol/eval
references: [ACP](https://agentclientprotocol.com/protocol/overview) and
[Harbor](https://www.harborframework.com/docs). File and symbol references are from that checkout.*
