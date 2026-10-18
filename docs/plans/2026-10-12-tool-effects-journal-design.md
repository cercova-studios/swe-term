# Design: Tool, EffectDeclaration, and ControlJournal SDK slice

Date: 2026-10-12
Status: Approved design; not yet implemented.
Consumer: obsidian-agent rebuild on swe-term SDK primitives
(Researcher → Summarizer → Zettelkasten → Persist), replacing the
LangGraph/LangChain tool path.

## Problem

ARCHITECTURE.md §5 names `Tool`, `EffectDeclaration`, `SessionStore`,
`ControlEvent` / `ControlJournal`, and `VerificationReceipt` as contracts.
Today only the reducers exist: `ApplyControlEvent`
(`internal/core/control_monitor.go`), the receipt gate, the V&V rung ladder,
and `ContextPacket` validation. There is no `Tool` type, effects are opaque
`[]string` tokens, and the control journal has no durable store — "control
events must not drop" (§6) is asserted but not implemented. PLAN.md Phase 1
already calls for `Tool` and `SessionStore`; obsidian-agent's cross-project
ledger (`../obsidian-agent/docs/cross-project/swe-term.md`, OA-001…003)
makes the need concrete. This slice builds the smallest durable subset.

## Decisions (settled 2026-10-12)

1. **Scope: contracts + journal, no loop.** `Tool`, typed
   `EffectDeclaration`, and an append-only `ControlJournal` feeding the
   existing monitor. No provider tool-calling loop, no SQLite
   `SessionStore`, no ACP wire protocol, no approvals UI.
2. **Effects: typed fields + observation level.** Not string tokens.
3. **Journal: interface + JSONL file implementation.** Not SQLite yet.

## Components (all in `internal/core`, no new dependencies)

### `tool.go` — Tool, Invocation, and ToolResult

```go
type Tool interface {
    Name() string
    Version() string
    Schema() json.RawMessage // JSON Schema for Invocation.Args
    Execute(ctx context.Context, inv Invocation) (ToolResult, error)
}

type Invocation struct {
    ID            string
    Args          json.RawMessage   // effective arguments passed to Execute
    RewrittenFrom json.RawMessage   // non-empty when a pre-tool hook/shim rewrote Args
    Declared      EffectDeclaration // post-rewrite declaration bound to the active lease
}

type ToolOutcome string

const (
    ToolOutcomeOK        ToolOutcome = "ok"
    ToolOutcomeTruncated ToolOutcome = "truncated"
    ToolOutcomeTimeout   ToolOutcome = "timeout"
    ToolOutcomeCancelled ToolOutcome = "cancelled"
    ToolOutcomeDenied    ToolOutcome = "denied"
    ToolOutcomeFailed    ToolOutcome = "failed"
)

type ArtifactHandle struct {
    ID     string // stable artifact identifier
    URI    string // durable local path or URI to full un-truncated payload
    SHA256 string // hex digest of the full payload
    Bytes  int64  // byte length of the full payload
}

type ToolResult struct {
    Outcome         ToolOutcome
    Output          string          // inline output (bounded when Outcome == ToolOutcomeTruncated)
    TruncatedFields []string        // field names truncated in compact views (e.g. ["output"])
    Artifact        *ArtifactHandle // required when Outcome == ToolOutcomeTruncated
    EffectiveArgs   json.RawMessage // disclosed to the model when Invocation.RewrittenFrom is set
}
```

- Deliberately **no `ReadOnly() bool`** (BACKLOG.md: a bool conflates
  approval risk with scheduling concurrency). Scheduling and policy read
  `EffectDeclaration` instead.
- **No tool-name policy enforcement** (Antigravity lesson
  `8c9fc83bb89cc6009bdd712c`): permissions and monitor checks evaluate
  `Invocation.Declared` (`EffectDeclaration`), never `Tool.Name()` strings,
  eliminating name-mismatch bypasses (`bash` vs. `run_command`).
- **Pre-tool argument rewrite disclosure** (`8c9fc83bb89cc6009bdd712c`):
  when a hook or shim rewrites tool arguments (`updatedInput`), the caller
  populates `Invocation.RewrittenFrom`, computes `Invocation.Declared` from
  the post-rewrite `Args`, and echoes `EffectiveArgs` in `ToolResult` so the
  model never reasons from stale pre-hook beliefs.
- `ToolResult` carries a closed outcome vocabulary — `ok`, `truncated`,
  `timeout`, `cancelled`, `denied`, `failed` — never flattened into
  successful text (§10 invariant 9). `ToolResult.Valid()` enforces that
  `ToolOutcomeTruncated` has a valid `ArtifactHandle` (`ID`, `URI`,
  `SHA256`, `Bytes > 0`) and non-empty `TruncatedFields`.
- Tool *argument* schemas are per-tool JSON Schema. Session-level typed
  final output (OA-002) is out of scope and must not be smuggled in here.

### `effect.go` — EffectDeclaration and ObservedEffect

```go
type EffectClass string

const (
    EffectClassRead    EffectClass = "read"
    EffectClassWrite   EffectClass = "write"
    EffectClassProcess EffectClass = "process"
    EffectClassNetwork EffectClass = "network"
    EffectClassSecret  EffectClass = "secret"
)

type ObservationLevel string

const (
    ObservationObserved     ObservationLevel = "observed"
    ObservationAttested     ObservationLevel = "attested"
    ObservationUnobservable ObservationLevel = "unobservable"
)

type PathEffect struct {
    Path string // normalized workspace-relative or canonical path prefix
    Glob bool   // true if Path is a directory/subtree prefix match
}

type EffectDeclaration struct {
    Reads       []PathEffect
    Writes      []PathEffect
    Processes   []string
    Network     []string         // destination hosts
    Secrets     []string         // names only, never values (§10 invariant 14)
    Observation ObservationLevel // default/ceiling observation level ("observed" | "attested" | "unobservable")
}

type ObservedEffect struct {
    Class       EffectClass
    Target      string
    Observation ObservationLevel
}
```

- `Validate()` fails closed:
  - Rejects empty declarations (at least one effect across the five slices
    must be declared, and total entries cannot exceed `maxDeclaredEffects`).
  - Rejects invalid or empty `ObservationLevel` (must be `observed`,
    `attested`, or `unobservable`).
  - Rejects duplicate entries within any slice, empty targets/paths, and
    path traversal (`..` segments).
  - Rejects secret *values* (`Secrets` entries containing `=`, `:`,
    whitespace, or high-entropy bearer prefixes such as `sk-`, `ghp_`,
    `Bearer `; only upper/lower identifier names like `OPENAI_API_KEY` are
    valid).
  - **OA-003 strictness rule:** `HasMutation()` is true when `Writes` or
    `Processes` is non-empty. A mutating declaration with
    `Observation == ObservationUnobservable` is rejected by `Validate()`
    (and therefore by `ControlEffectsDeclared` with
    `control.effect.declaration_invalid`). Provider-hosted tools whose
    effects no harness can observe (`web_search`) may declare read/network
    effects as `unobservable`, or be denied entirely by a strict profile,
    but may never claim governed workspace mutation under `unobservable`.
- `Covers(observed ObservedEffect) bool` is the structural membership test
  the monitor uses:
  - Validates `observed.Valid()` (valid `Class`, non-empty `Target`, valid
    `ObservationLevel`).
  - Checks observation-level compatibility: an `unobservable` runtime event
    is only covered if the declaration itself declared `unobservable`; an
    `attested` event is covered by `attested` or `unobservable`; an
    `observed` event is covered by any valid declared level.
  - Checks target membership in the matching class (`Reads`, `Writes`,
    `Processes`, `Network`, `Secrets`), supporting exact path match or
    clean directory boundary prefix match when `PathEffect.Glob` is true.
  - An observed effect outside the declaration produces
    `control.effect.undeclared`.

### `journal.go` — ControlJournal

```go
type ControlJournal interface {
    Append(event ControlEvent) (ControlEvent, error) // store assigns Sequence = last + 1
    Replay(visit func(ControlEvent) error) error
}
```

- `FileControlJournal` (JSONL file implementation): one compact JSON object
  per line, append under a `sync.Mutex` (single writer), monotonic
  `Sequence` (`1..N`) assigned by the store (if caller passes a non-zero
  `Sequence` that does not match `last + 1`, `Append` rejects it without
  writing), flushed and `Sync()`ed before returning.
- `Replay` streams from line 1 and folds `ApplyControlEvent` from genesis
  (via a `ReplayControlMonitor(journal)` helper or visitor callback);
  replayed `ControlMonitorState` must equal live state.
- A truncated, unterminated (missing trailing newline), or corrupt JSON
  line — or a sequence gap — is a loud error naming the line number, never
  a silently skipped record.
- The monitor stays a pure reducer; the journal owns history, ordering,
  and durability on disk.

### Monitor migration

`ControlEvent.Effects []string` becomes `EffectDeclaration`,
`ControlEvent.Effect string` becomes `ObservedEffect`, and
`ControlMonitorState.DeclaredEffects []string` becomes `EffectDeclaration`.
Existing rule IDs (`control.approval.required`,
`control.effect.declaration_invalid`, `control.effect.undeclared`, …) are
the external contract and do not change. When `ControlEffectObserved` is
accepted, any mutating observed effect (`write`, `process`, or
`unobservable`) invalidates `CurrentReceipt` so a fresh verification
receipt is required before `ControlLifecycleClaimed`; pure declared `read`
effects within the active lease do not clobber an already-recorded receipt
unless the target identity itself changes. The table-driven trace tests in
`control_monitor_test.go` are updated to the typed shape, not weakened.

## Data flow (one obsidian-agent step)

```
host resolves tool + (optional) pre-tool hook rewrites Args
  → approval_granted → lease_acquired
  → effects_declared (bound to post-rewrite Invocation.Args)
  → Tool.Execute
  → effect_observed × n (each tagged with Class, Target, ObservationLevel)
  → receipt_recorded → lifecycle_claimed("done") → lease_released
```

Researcher (web/MCP/shell reads, network egress declared), Summarizer and
Zettelkasten (model steps bounded by tool schemas), and Persist (vault
writes under `Writes`, receipt bound to the written state) all traverse
the same path; no step gets a private side channel.

## Testing

- Table-driven traces (existing style): happy path plus each rejection —
  unapproved mutation, lease conflict, undeclared effect, unobservable
  mutation declaration, stale receipt, claim after cancel.
- Journal: append/replay state equivalence; corrupt-tail and missing-newline
  rejection; sequence mismatch rejection; idempotent replay of the last
  event in the reducer.
- Effects: `Covers` truth table across all five classes (`read`, `write`,
  `process`, `network`, `secret`) and three observation levels (`observed`,
  `attested`, `unobservable`); `Glob` path prefix boundary checks (`dir`
  covers `dir/file.go` but never `dir-other/file.go`); mutating
  `unobservable` declaration rejected at `Validate`; secret values rejected
  at `Validate`.
- Tool: closed `ToolOutcome` validation, `ToolOutcomeTruncated` requiring
  `ArtifactHandle` + `TruncatedFields`, and `RewrittenFrom` /
  `EffectiveArgs` disclosure check.
- Durable tests live beside the code (`tool_test.go`, `effect_test.go`,
  `journal_test.go`); no new test framework.

## Documentation updates (with the implementation change)

- `docs/core/PLAN.md`: follow-on entry under Phase 1/2 (this file).
- `docs/core/MECHANISM_LEDGER.md`: Effect declarations and Temporal
  control-journal monitor rows move toward Implemented when code lands.
- `ARCHITECTURE.md` §5: statuses change only with source + tests + doc
  together (§14); reducer-only caveats stay explicit until then.
- New `docs/cross-project/obsidian-agent.md` in this repo answering
  OA-001 (deferred: host callbacks need the wire protocol), OA-002
  (deferred: session-level typed output), OA-003 (accepted: observation
  levels in `EffectDeclaration`).

## Non-goals

Agent loop and steering; `SessionStore`/SQLite; ACP framing and
host-hosted tool callbacks (OA-001); typed session output (OA-002);
approvals UI; Fleet CPG adapter wiring. Each is a later slice gated on
this one.
