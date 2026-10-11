# Critical Evaluation of opencode as an Agentic Harness

> Evaluated against the Harness (swe-term) philosophy:
> zero-dependency core, interface-driven plugins, immutable state,
> goroutine-native concurrency, content-addressable caching, single binary.
>
> **Validated against the local opencode source** at `opencode/` (`2.0.26`,
> HEAD `823ac87f58`, 2026-10-10). This critiques the **V2** tree
> (`packages/core`, `server`, `protocol`, `schema`, `ai`, `cli`). Architecture
> facts and line counts are in `OPENCODE_DEEP_DIVE.md`.

---

## Verdict Summary

opencode V2 is the **best-structured mainstream harness in this survey** and
the one whose *shape* is closest to swe-term's plan. It ships the protocol
seam swe-term wants (core behind a typed `HttpApi`, every UI a client), an
event-sourced session that the loop only reads from, permission as ordered
data with last-match semantics, and a loop split into files of a few hundred
lines. There is no 9k-line session object.

It is also a **product platform wearing a harness**. The app UI alone is
~154k lines. `@opencode/ai` carries 49 protocol files including video and
speech. Plugins are npm packages installed and run in-process. There is no OS
sandbox. Effect is load-bearing everywhere, so the architecture cannot be
read, or ported, without the framework.

**Steal the seams:** history-only `drain`, durable inbox with steer vs
next-turn, typed `once|always|reject|corrected` replies, `opaque`-not-guess
shell scanning, tool filtering ≠ authorization, explicit `truncated`/
`timeout` output, checkpoint compaction with verbatim recent tail, subagent
depth default 1. **Refuse the platform:** in-process npm plugins, a
framework-as-architecture, unbounded parallel tool fan-out, a second
provider SDK, and a sandbox story that is a permission prompt.

---

## 1. Effectiveness

### What opencode does well

**Loop reads history, never arguments.** `SessionRunner.drain` "runs one
local continuation from already-recorded Session history"
(`packages/core/src/session/runner/index.ts`). This is the DeepSeek
`deriveMessages` invariant swe-term already adopted (`docs/core/FRAMEWORKS.md`,
"Ghost-context prevention"), enforced by the type signature rather than
convention.

**Event-sourced session with a projector.** Durable events
(`packages/schema/src/session-event.ts`) are projected into SQLite tables
(`packages/core/src/session/projector.ts`, `session/sql.ts`). Execution
lifecycle is itself evented (`session.execution.started/succeeded/failed/
interrupted`), and crash restart uses claim attempts
(`database/migration/20260811161259_execution_claim_attempts.ts`,
`session/execution/restart.ts`). This is swe-term's SQLite-SSOT direction
with a concrete schema to compare against.

**Durable inbox for mid-run input.** `session/inbox.ts` (550 lines) and the
`Promotable` "steer" mode in `runner/index.ts` separate "change what the
active turn does" from "queue the next turn." Most harnesses conflate these.

**Permission as ordered data.** `evaluate` is `findLast` over flattened
rulesets, default `ask` (`core/src/permission.ts:87–97`). Missing agent
permissions default to `* * deny` (line 19). A decline with feedback is a
typed `CorrectedError` the model continues from (`core/src/tool/AGENTS.md`).
This is small, testable, and serializable.

**Shell scanning that admits ignorance.** `core/src/shell/scan.ts` returns
`{kind: "opaque", reason}` for command substitution, heredocs, dynamic
execution, etc. — "Opaque describes a parsing limitation, not a permission
decision." That is the correct contract for a prefix-matching permission
system: unparseable → escalate, never guess.

**Catalog visibility ≠ authorization.** "Tool filtering is catalog
visibility, not execution authorization" (`core/src/tool/AGENTS.md`). Leaves
own permission. This is a cleaner statement of the boundary than Claude Code's
or omp's.

**Explicit output states.** Shell returns `truncated` and `timeout` booleans
and a path to the full output (`tool/plugin/shell.ts:42–68`); generic bounding
runs after execution (`toolOutput.truncate` in `session/runner/step.ts`).
Matches swe-term's "never silently truncate" finding (`FRAMEWORKS.md`).

**Compaction as an updated checkpoint.** `session/compaction.ts` keeps a
verbatim recent tail beside the summary (line 45), supports provider-native
compaction (line 71), shrinks stepwise on overflow (line 228), and updates
one checkpoint with "newer history takes precedence" rules (lines 160–173).
The comment at line 161 records a real regression (#48058) where summaries
kept everything — evidence-driven prompt design.

**Subagents bounded by default.** Depth walk with
`experimental.subagent_depth ?? 1` (`tool/plugin/subagent.ts:118–132`);
background mode with completion notification. Same default as Prime.

**Bounded event fan-out that fails loudly.** SSE subscribers get a 4,096
queue and an explicit `SubscriberOverflowError`
(`packages/server/src/event-feed.ts`).

### What opencode gets wrong (for *our* purposes)

**No confinement.** No seatbelt/landlock/bwrap in shell, environment, or PTY
code (search over `core/src/shell*`, `environment/`, `tool/plugin/shell.ts`,
`persistent-pty.ts`). `environment/driver.ts` is a spawner seam only. The
2,409-line `shell/scan.ts` is effort spent making *prompts* precise, not
making execution safe. swe-term's rule is fail-closed argv sandbox
(`FRAMEWORKS.md`, "Sandbox").

**Unbounded parallel tool execution.** Every `tool-call` stream event forks
immediately (`session/runner/step.ts:126–137`). There is no read/write
partition; write ordering relies on a keyed mutex in `file-mutation.ts`.
That serializes same-file writes but not semantic conflicts (a shell `mv`
racing an `edit`). Claude Code's read-parallel/write-serial and swe-term's
effect-declaration backlog item (`FRAMEWORKS.md`, last table) are stricter.

**In-process npm plugins.** `plugin/module.ts` calls `npm.add(target)` and
loads the module into the host (lines 104–109). A plugin shares the server's
process, credentials, and filesystem. Same failure class as omp's N-API.

**Two provider stacks.** `@opencode/ai` (49 protocols) and the Vercel AI SDK
bridge (`core/src/aisdk.ts`, 1,078 lines; `@ai-sdk/*` pins in
`core/package.json`) coexist, plus a 1,636-line Copilot responses model
(`core/src/github-copilot/responses/`). Image, speech, video, and
transcription protocols are product, not harness.

**Framework as architecture.** Service identity, errors, concurrency,
cancellation, and lint rules are all Effect (`lint:effect-patterns` in root
`package.json`). The `uninterruptibleMask` + `forkScoped` choreography in
`step.ts` is correct but only legible to Effect readers. Go's
`context.Context` + goroutines + errgroup cover the same semantics natively.

**LSP is a TODO.** `core/src/file-mutation.ts:131`. V2 lost a V1 capability
during the rewrite; edits get no diagnostics feedback today.

**Instruction discovery reads `.claude/`.** `config/discovery.ts:41` scans
`.claude`, `.agents`, `.opencode`. Convenient, but it imports another
harness's config surface as an untrusted-input channel without saying so.

---

## 2. Efficiency

- **Startup:** a managed background server is elected and reused
  (`cli/src/server-process.ts`, `cli/src/services/service-registration.ts`),
  amortizing boot. Cost: a resident process and a version-replacement
  protocol (`AGENTS.md`, "Live V2 TUI Testing").
- **Snapshots:** captured concurrently with the provider request
  (`step.ts:80`), so they do not add latency. Storage is a git shadow repo
  (`core/src/snapshot.ts:76–98`) — content-addressed by git, close to
  swe-term's CAS goal.
- **Cache keys:** `model-request.ts:294` TODO notes forks do not yet reuse the
  root session's cache key — a known prompt-cache miss for subagents.
- **Edit fallback tiers** (exact → Unicode-normalized → trailing-whitespace,
  `tool/plugin/edit.ts:49–54,161–167`) cut retry turns cheaply, without
  omp-style hashline infrastructure.

---

## 3. Simplicity

Good: loop files are small (`runner/llm.ts` 375, `step.ts` 325), the session
is split into ~40 cohesive files, and dependency direction is a written rule
(`AGENTS.md`). Tool registry semantics are documented in one page
(`core/src/tool/AGENTS.md`), including a "Do not add a second executable
entry type" constraint.

Bad: 35 workspace packages, 139 endpoints still under audit
(`V2_HTTP_API_AUDIT.md`), 49 migrations in ~8 months including a
`reset_v2_session_state` migration, a 1,135-line V1 importer, and a 3,640-line
TUI session route (`packages/tui/src/routes/session/index.tsx`). The harness
is simple; the repository is not.

---

## 4. Key Gaps (relative to Harness goals)

| Goal | opencode | Gap |
|---|---|---|
| Zero-dependency core | 39 runtime deps in `core/package.json`, Effect, Drizzle, AI SDK | Large |
| Interface-driven plugins | Typed hook domains, but in-process npm | Isolation |
| Immutable state | Event log + projector; per-request tool snapshot | Small |
| Goroutine-native concurrency | Effect fibers; unpartitioned tool fan-out | Policy |
| Content-addressable caching | Git shadow snapshots; no CAS for tool output | Medium |
| Single binary | Bun compile + resident service + optional desktop/web | Medium |

---

## 5. Assumptions to NOT Port

### ❌ "Precise permission prompts are a sandbox"
`shell/scan.ts` is excellent prompt input. It is not confinement. swe-term
keeps fail-closed argv sandboxing; port the scanner's `opaque` contract only.

### ❌ "Every tool call should start the moment it streams"
`step.ts:126–137` forks per call. swe-term needs an effect declaration
(read/write/exec) and schedules writes serially.

### ❌ "Plugins are npm packages loaded in-process"
`plugin/module.ts:104–109`. swe-term plugins are interfaces in-process for
first-party code and sidecars over a protocol for third-party code.

### ❌ "Own the provider SDK and also keep the other one"
`packages/ai` + `core/src/aisdk.ts`. swe-term keeps one `Provider` port.

### ❌ "A framework can be the architecture"
Effect makes opencode's invariants enforceable *inside* Effect. swe-term
encodes the same invariants in Go types and `context.Context`.

---

## 6. What to Actually Port

| Pattern | opencode source | swe-term shape |
|---|---|---|
| History-only continuation | `session/runner/index.ts` `drain` | `Loop.Drain(sessionID)` reads `SessionStore`, takes no messages |
| Durable inbox, steer vs next-turn | `session/inbox.ts`, `Promotable` | Inbox table; `steer` amends active turn, else queue |
| Evented execution lifecycle + claims | `session-event.ts`, `execution/restart.ts` | `execution.{started,succeeded,failed,interrupted}` rows + claim attempt for crash recovery |
| Last-match ruleset, default ask, empty = deny | `permission.ts:19,87–97` | `[]Rule{Action,Resource,Effect}`; pure `Evaluate` func |
| Typed replies incl. corrected | `schema/permission.ts:42`, `CorrectedError` | `Reply{Once,Always,Reject,Corrected(msg)}` |
| Scanner returns `opaque` with reason | `shell/scan.ts` | Use `mvdan.cc/sh` parse; unparseable ⇒ ask |
| Visibility ≠ authorization | `tool/AGENTS.md` | Registry filters schemas; tools call `Permission.Assert` |
| Per-request tool snapshot | `tool/AGENTS.md` "Registration" | Immutable `ToolSet` value per request |
| Edit fallback tiers | `tool/plugin/edit.ts:49–54,161–167` | exact → Unicode-normalized → trailing-ws; report tier used |
| Checkpoint compaction + verbatim tail | `session/compaction.ts:45,160–173,228` | Summary checkpoint updated in place; protected recent tail |
| Bounded subscriber, loud overflow | `server/event-feed.ts` | Per-client buffered chan; overflow closes stream with error |
| Subagent depth default 1, background notify | `tool/plugin/subagent.ts:118–132` | Already planned; confirms default |

---

## 7. Architectural Contrasts

```
opencode V2                                 swe-term
Bun + Effect, TS                            Go stdlib core
Core behind HttpApi (139 endpoints)         Core behind versioned local protocol
SSE event feed                              Event stream on the same protocol
Event-sourced session → Drizzle/SQLite      SessionStore SQLite SSOT
drain(sessionID) from history               Loop reconstructs from store
Fork every tool call                        Effect-declared, write-serial
Last-match permission rules                 Same, plus fail-closed sandbox
No OS sandbox                               Argv sandbox adapter required
In-process npm plugins                      Interfaces in-core; sidecars for 3rd party
@opencode/ai + Vercel AI SDK                One Provider port
Git shadow snapshots                        CAS snapshots
LSP: TODO in V2                             Sidecar when needed
Resident managed service                    Single binary; daemon optional
```

---

## 8. Final Assessment

opencode V2 is the reference for **how the seams should look** and a
warning for **how much product can sit on top of them**. The event-sourced
session, history-only drain, durable inbox, and last-match permission data
are directly portable and validate swe-term's existing direction with
production evidence. The missing sandbox, in-process npm plugins, unpartitioned
tool fan-out, dual provider stacks, and Effect-as-architecture are the parts
swe-term should decline.

---

## Proposed FRAMEWORKS.md Additions

**Executive takeaway (one line):**

> **opencode** — protocol-first, event-sourced V2 core is the closest
> production match to swe-term's shape; port its session/inbox/permission
> seams, not its in-process npm plugins, unsandboxed shell, or Effect-shaped
> platform.

**Port/Avoid matrix row:**

| **opencode** | History-only `drain`, event-sourced session + projector, durable steer/next-turn inbox, last-match permission rules (empty=deny, typed corrected reply), `opaque` shell scan, visibility≠authorization, checkpoint compaction with verbatim tail, loud bounded SSE fan-out | In-process npm plugins, no OS sandbox, fork-every-tool-call fan-out, dual provider SDKs, Effect as architecture, product mass (app/console/media protocols) in-repo |
