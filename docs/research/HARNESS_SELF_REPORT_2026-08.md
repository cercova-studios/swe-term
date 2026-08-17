# Harness Self-Report Round — 2026-08-16

> Raw evidence backing `docs/core/FRAMEWORKS.md` § "Agent Self-Report
> Synthesis (2026-08)" and the resulting deltas in `ARCHITECTURE.md`.

## Method

Three coding-agent CLIs installed on this machine were each given the same
self-contained prompt (`swe-term`'s current design snapshot — layers,
canonical interfaces, safety model, state/context, extensibility strategy,
port/avoid matrix highlights) and asked to answer as an agent that operates
*inside* a harness daily: where would you push back from lived operational
experience, what's missing, what should we steal from your own harness, what
anti-pattern should we avoid. Explicitly invited disagreement rather than
validation. Read-only / no-repo-write instruction; each ran non-interactively
(`-p`/`exec`), independently, with no visibility into the others' answers.

Backends: `prime-agent` and `omp` (oh-my-pi) are two of the exact harnesses
`docs/research/PRIME_AGENT_*` and `docs/research/OH_MY_PI_*` already studied
— asking them directly is asking the subject of the research about itself.
`codex` (OpenAI, `gpt-5.6-sol`, `--sandbox read-only`) is outside that
family, included as a cross-lab check; it read `GOLANG_TUI_PLAN.md` from the
actual repo during its run rather than working only from the prompt's
condensed summary, so its citations (`Tool.ReadOnly() bool`, `ApprovalRequest`,
`ChainApprover`) are grounded in the real planned interface, not invented —
verified by grep against `docs/core/GOLANG_TUI_PLAN.md` before use.

A fourth CLI, `agy`, hung indefinitely on this machine (both `-p` and the
`agents` subcommand timed out with no output) — likely needs interactive
login. Dropped rather than debugged further; not represented below.

Convergence table is in `FRAMEWORKS.md`. Full responses below, verbatim.

---

## prime-agent

<details>
<summary>Prime Agent's response</summary>

# Design Input: What Actually Costs Me Turns

## 1. Where I'd push back

### "Analyzer-like enrichment" cannot be invisible pre-LLM injection

I would push back on enrichment occurring "outside the loop core" unless it is still a first-class, replayable event in the session protocol.

Invisible context injection causes one of the most expensive failures I experience: I act on context without knowing its provenance, freshness, selection criteria, or whether it will exist on the next call. That produces irreproducible reasoning and "ghost context." When the result is wrong, neither I nor the operator can determine whether the model, retrieval, compaction, or analyzer failed.

Every injected artifact should be inspectable and identified by source, version/hash, timestamp, selection reason, and token cost. The exact message sent to the provider must be reconstructable from the event log. "Outside the core" should mean independently implemented, not semantically invisible.

### Deterministic compaction triggers are necessary but insufficient

A token threshold tells you *when* to compact, not *what must survive*. Harness summaries commonly preserve narrative while dropping operational state: commands already attempted, test results, dirty files, rejected hypotheses, user constraints, child-agent handles, and pending work.

That costs correctness more than raw context loss because I repeat work or unknowingly reverse an earlier decision. Compaction should be a typed checkpoint with protected fields:

- current objective and definition of done;
- constraints and approvals;
- changed files plus repository state;
- verified facts versus hypotheses;
- commands/tests run and their outcomes;
- pending tasks, subprocesses, and delegated work;
- artifact references for details omitted from the prompt.

Mechanical deduplication should happen first, but the durable work ledger matters more than the prose summary.

### "Single active task by default" should not become single-flight I/O

Serial reasoning is a good default. Serial waiting is not.

In my current harness, delegated work is asynchronous, but admission returns only a handle and results arrive later as messages. There is no clean in-turn wait/fan-in primitive. That means independent research can require extra conversational turns merely to let the scheduler advance. The architecture should distinguish one authoritative task state from multiple bounded background operations.

Permit explicit structured concurrency: spawn, observe, cancel, await-any, and await-all under a shared deadline and budget. Keep one writer/decision-maker if desired, but do not make the model burn turns while independent reads are in flight.

## 2. What's missing

### Result ergonomics are part of correctness

Tool output needs a universal contract for truncation, pagination, continuation, and artifact storage. Silent or arbitrary truncation makes me infer that missing output does not exist. Large results should return a bounded preview plus an explicit completeness flag, omitted-byte or omitted-record count, stable artifact reference, and continuation mechanism.

Exit status, stdout, stderr, elapsed time, timeout state, and cancellation cause should remain distinct. Flattening these into one text blob wastes diagnosis turns.

### A first-class worktree and mutation ledger

For long sessions, I need an immediate answer to: "What changed, by whom, against which base, and has it been verified?" Tool-call history is not enough.

Track repository root, base revision, dirty state before and after each mutation, affected paths, edit preconditions, rollback reference, and validation evidence. This becomes especially important once subagents exist. Parallel readers are cheap; parallel writers without ownership and conflict reporting are a correctness trap.

### Capability discovery must be lazy and task-scoped

Injecting every available skill, tool description, and policy into every turn creates substantial prompt tax and lowers attention on the actual task. Expose a compact capability index, then retrieve full contracts only when selected. Freeze the selected tool schema versions into the session snapshot so resume remains reproducible.

## 3. One thing to steal from my current harness

Steal the **persistent computational control plane**: I can keep named variables and intermediate results across tool calls, issue async operations with normal language-level `await`, and use that state to coordinate work without serializing every intermediate artifact back through natural language.

Do not make IPython mandatory, but preserve the mechanism: a typed, inspectable session workspace where handles, observations, parsed results, and temporary analysis survive turns and can be snapshotted or rolled back.

## 4. One anti-pattern to avoid

Avoid **eagerly injecting the entire harness manual into the agent prompt**. My current environment exposes extensive operational instructions and a very large capability catalog up front. It is discoverable, but the recurring context cost is high, precedence becomes harder to reason about, and relevant task instructions compete with framework documentation.

The cure is not fewer capabilities. It is layered disclosure: small immutable safety/core contract, task-relevant repository instructions, compact capability index, then on-demand details with explicit precedence and provenance.

</details>

---

## omp (oh-my-pi)

<details>
<summary>omp's response</summary>

# Design Input: Where swe-term Should Be More Opinionated

## 1. Where I'd Push Back

### A. Deterministic compaction triggers are necessary but insufficient

A token threshold is a poor proxy for context value. In long coding sessions, the least-recent material is often still load-bearing: the user's correction, a failed reproduction, an approval decision, a file-level invariant, or the exact error that disproved the previous hypothesis.

The failure mode is not merely "the model forgets." It confidently re-derives a rejected approach, repeats an already-failed command, or edits against a constraint that disappeared during compaction.

I would add a **pinned working set** to the state model. Pins should include:

- current user intent and latest corrections;
- active hypotheses and disproven hypotheses;
- approval/denial decisions;
- unresolved errors;
- files currently being edited;
- explicit invariants extracted from tool results.

Compaction can summarize ordinary history, but it must not evict pins without an explicit state transition. This fits the architecture's immutable snapshots and typed transitions better than treating compaction as a pure byte-budget operation.

### B. "Single active task by default" should not mean single active investigation

The default is sensible for mutation safety, but strictly serial discovery wastes turns. In practice, a coding task often has independent read-only branches: inspect the implementation, inspect tests, and inspect a reference pattern. Waiting for each branch serially increases latency and encourages shallow investigation.

I would distinguish:

- **one active mutation owner**;
- **multiple bounded read-only investigations**;
- explicit join points before edits or state transitions.

The admission handle/depth-cap idea from the snapshot is useful here, but the contract should also specify resource budgets, cancellation, and result merging. Otherwise "intentional parallelism" becomes either unused ceremony or uncontrolled context multiplication.

### C. Runtime/ad-hoc extensions need the same trust boundary as compiled extensions

The two-lane extensibility strategy is operationally attractive, but ad-hoc capabilities are exactly where policy drift appears. A runtime tool that bypasses the canonical `Tool` schema, timeout propagation, audit events, or read-only/mutating distinction can become a privileged escape hatch.

The failure mode is subtle: the core appears safe in review, while the development lane quietly gains arbitrary process, filesystem, or network access.

I would require every lane to emit the same capability manifest and policy-relevant events. The implementation may differ; the approval boundary, cancellation behavior, side-effect declaration, and audit record must not.

## 2. What's Missing

### A. Tool-call-level recovery semantics

Session persistence and resume are not enough after a process dies halfway through a side effect. The harness needs durable records for:

- tool invocation started;
- arguments and policy decision;
- side effect acknowledged or unknown;
- result committed to session state.

On resume, the core must distinguish "safe to retry" from "must inspect before retry." Without this, recovery can duplicate commits, file writes, deployments, or external API calls.

### B. Provenance and freshness for every context item

Context should carry source, snapshot/version, timestamp, truncation status, and confidence—not just text. A search result from a stale repository snapshot must not look equivalent to a freshly read file. Likewise, summarized output must loudly identify what was omitted and how to retrieve it.

This is especially important for analyzer-like enrichment: pre-LLM context is valuable only if the agent can tell whether it is authoritative, advisory, stale, or incomplete.

### C. A first-class correction path

"Steerable single active task" is not enough unless corrections are modeled explicitly. User instructions can invalidate a plan while old tool calls are still running. The harness needs cancellation and supersession semantics: which work is abandoned, which results are retained as evidence, and which mutations are still allowed to land.

## 3. One Mechanism to Steal From This Harness

Steal **snapshot-verified editing**. An edit names the exact file snapshot it was derived from; if the file changed, the edit is rejected instead of silently applying to shifted lines.

That mechanism is small but high leverage. It converts stale-context risk into a visible, recoverable failure. The same pattern should apply to more than edits: tool results and analyzer payloads should carry snapshot identifiers, and consumers should be able to reject stale inputs.

## 4. One Anti-Pattern to Avoid

Avoid **silent helpfulness**: truncating tool output, ignoring unsupported flags, retrying failed mutations, or normalizing malformed provider events without emitting a visible boundary event.

Agents operate through observed behavior, not implementation intent. A silent approximation creates a false world model; once that happens, later reasoning can be internally consistent and still completely wrong. Every lossy, stale, retried, or normalized operation needs an explicit event, provenance marker, and recovery path.

</details>

---

## codex (gpt-5.6-sol, `--sandbox read-only`)

<details>
<summary>codex's response</summary>

# Proposal: Operational Pushback from Inside a Harness

## 1. Where I'd push back

### 1.1 `Tool.ReadOnly() bool` is too weak—and conflates safety with scheduling

I would reject one boolean as both the approval signal and concurrency primitive. "Read-only" commands routinely have incidental effects: `git status` in this session attempted to create an `xcrun` cache file and hit the sandbox boundary. Package managers populate caches; compilers write metadata; language servers start processes; network reads disclose data.

This creates two failures:

- False-safe tools run concurrently or without sufficient scrutiny.
- Harmless operations receive repeated approvals because the harness cannot describe their bounded effects.

I want an invocation-level effect declaration: filesystem paths read/written, process spawning, network destinations, secret access, and host/sandbox scope. Scheduling should use resource conflicts; approval should use risk. If observed effects exceed the declaration, fail closed and emit an auditable mismatch event.

### 1.2 Compaction cannot be merely a pluggable `[]Message -> []Message` transformation

Deterministic triggering is good. Delegating the correctness-critical reduction policy entirely to a `Compactor` extension is not.

What costs me correctness after compaction is rarely losing prose; it is losing an exact constraint, command result, unresolved review finding, file ownership decision, or the reason an approach was rejected. Once that happens I repeat probes, reopen settled decisions, or produce a locally plausible but globally wrong change.

The core should enforce a loss contract: pinned user requirements, active task state, approvals, unresolved failures, workspace version, and artifact handles survive compaction mechanically. Summarization can remain pluggable, but it should summarize only the disposable narrative around that protected spine. Every compacted state should also be inspectable and replayable.

### 1.3 "Capability usually equals extension" needs a sharper exception rule

Extension-first is directionally right, but hook-heavy systems tend to push cross-cutting semantics outward until behavior depends on extension ordering. In my current environment, large capability and skill catalogs already create routing ambiguity and consume context before useful work begins.

Providers and analyzers are natural extensions. Cancellation, effect accounting, artifact identity, verification provenance, and context-retention rules are not: they are invariants every extension must obey. I would define a strict promotion test: something belongs in core when independent extensions must agree on it for safety, replay, or correctness—not merely after it becomes popular.

## 2. What's missing

### 2.1 Freshness-bound verification receipts

A harness should record exactly which build, test, lint, or runtime checks ran, their exit status, and the workspace snapshot they covered. If files change afterward, the receipt becomes stale automatically.

Without this, "tests passed" survives in conversation after the tested tree no longer exists. The agent must reconstruct freshness manually, which is error-prone and wastes turns.

### 2.2 First-class contested-workspace ownership

Long sessions happen in dirty trees shared with users and other agents. The harness needs file-level provenance, optimistic preconditions for edits, and a scoped change manifest. An edit should fail or require reconciliation when its input hash changed; staging should operate from the manifest rather than ambient working-tree state.

Snapshot/rollback alone is insufficient: rolling back another actor's legitimate work is worse than failing early.

### 2.3 Honest tool-output QoS

Silent truncation is poison. Every result should report whether it was truncated, original size, retained ranges, encoding, and a durable handle for retrieving the rest. Streaming tools need explicit completion, cancellation, and partial-result states. Otherwise I cannot distinguish "no match" from "the match was clipped," and I burn another turn rerunning the command differently.

## 3. One thing to steal from this harness

Steal handle-based, resumable tool execution. A long-running process can yield a session identifier; I can poll it, send input, or terminate it without restarting the process or stuffing its entire transcript into each turn. Preserve that mechanism across TUI, headless, and RPC frontends, with stable chunk identifiers and explicit truncation metadata.

## 4. One anti-pattern to avoid

Do not eagerly inject every tool schema, plugin, skill catalog, and duplicated policy document into every model call. My present harness often makes me reconcile substantial ambient instructions unrelated to the task, including overlapping capability definitions. That is a direct context tax and a source of instruction conflicts.

Expose a compact capability manifest, then lazily load the selected contract and its governing policy. The model should pay context for capabilities it uses, not everything installed.

</details>

---

## Points not carried into ARCHITECTURE.md (single-source, tracked here instead)

- **prime-agent**: replayable event log for every injected artifact (source,
  version/hash, timestamp, selection reason, token cost) — subsumed by the
  provenance/freshness bullet added to `ARCHITECTURE.md`, but the specific
  "reconstructable exact provider message from the event log" requirement is
  sharper than what was applied; worth revisiting when the event model is
  designed.
- **prime-agent / codex**: persistent computational control plane / IPython-
  style typed session workspace, and handle-based resumable tool execution.
  Both are "steal this mechanism" suggestions, not architecture-level
  deltas — relevant to a future long-running-tool-execution design, not to
  `ARCHITECTURE.md`'s canonical layer.
- **omp / codex**: tool-call-level recovery semantics (durable records
  distinguishing "safe to retry" from "must inspect before retry" after a
  crash mid-side-effect) and contested-workspace ownership (optimistic
  precondition on edits, scoped change manifest). Captured as a `BACKLOG.md`
  item (contested-workspace ownership) but recovery semantics specifically
  is not yet backlogged — should be, once `SessionStore` resume semantics
  are designed in detail.
- **codex**: `Tool.ReadOnly() bool` → effect declaration. Captured in
  `BACKLOG.md` as an interface rework, not yet applied to
  `GOLANG_TUI_PLAN.md`'s actual interface definition.
