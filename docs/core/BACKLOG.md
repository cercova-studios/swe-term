# Backlog

Consolidated candidate list for integrations and standalone tools.

## Integrations to Explore

- Zoekt for code search (MCP integration path)
- ast-grep for structural search (MCP integration path)
- Semgrep for policy/security scanning (MCP integration path)
- **Do not rewrite Joern.** Rebirth CPG-lite as `extensions/swe_graph` (tree-sitter + SCIP ingest + SQLite). Optional later: wrap `joern-parse` with canned queries only. Evidence: `docs/research/JOERN_*`.

## Lightweight CLI Tools to Build

- X search (tweets and personal bookmarks via X API)
- Webpage to markdown converter
- PDF parser for extraction workflows
- Codebase research and architecture-insight generator
  - architecture-as-code outputs
  - intent formalization
  - formal verification helpers
  - chaos/performance analysis workflows

## Harness ergonomics (from 2026-08 self-report round)

Evidence: `docs/core/FRAMEWORKS.md` "Agent Self-Report Synthesis" — three
external agent CLIs (prime-agent, omp, codex) independently critiqued the
architecture snapshot; four findings converged and are already applied to
`ARCHITECTURE.md`. One is unresolved:

- Rework `Tool.ReadOnly() bool` (currently planned in `GOLANG_TUI_PLAN.md`
  §Tool interface) into a per-invocation effect declaration (paths
  read/written, processes spawned, network destinations, secret access).
  Approval reasons about risk from the declaration; scheduling reasons about
  concurrency from resource conflicts. A single bool conflates the two and
  produces both false-safe concurrent execution and unnecessary re-approval
  of harmless ops. Codex's concrete example: `git status` under
  `--sandbox read-only` still attempted an `xcrun` cache write — read-only
  classification did not predict the actual effect surface.
- Contested-workspace ownership for edits: file-level provenance + optimistic
  precondition (edit rejected if the input snapshot hash changed) +
  scoped change manifest, rather than relying on snapshot/rollback alone.
  Resonates with the existing `<parallel-agents>` contested-tree doctrine —
  this generalizes it from "check before you edit" to a structural
  precondition the core enforces.

## Harness capabilities to steal (from 2026-08 research)

Evidence: `docs/research/{PRIME_AGENT,DEEPSEEK_HARNESS,OH_MY_PI}_*` and `docs/core/FRAMEWORKS.md`.

- Session-log reconstruction check before every provider call (dsh invariant)
- Hashline / snapshot-verified apply as an edit sidecar, with a per-model format table (omp)
- Summarized default `Read`/`Grep` with hard caps (omp)
- Fail-closed sandbox argv wrapper (dsh Landlock/bwrap/Seatbelt chain)
- Closed host-RPC for any code-exec sidecar — enumerated methods, not full tool re-entry (Prime, not omp)
- Capability-gated protocol version + schema revision (Prime daemon protocol)
- Mechanical compaction (elide superseded reads / artifact-offload) before LLM summary (omp shake)
- Versioned harness/memory records with rollback; base system prompt immutable (Prime `/refine`)

## Control / OODA / V&V (from 2026-08 Codex design critique)

Evidence: independent `codex exec --sandbox read-only` pass over
`ARCHITECTURE.md`, `JOERN_*`, `HARNESS_SELF_REPORT`, and the swe_graph /
OODA canvases. Amp/Grok delegates did not produce usable output this round.
Verified against `GOLANG_TUI_PLAN.md` (lossy bus at Event Bus §2; `ReadOnly()`
still executable; `AgentState` is messages/UI; persistence in Phase 6).

**Adopt (core contracts — several already applied to `ARCHITECTURE.md`):**

- Durable ordered control-event journal + tiny runtime monitor: approval
  before mutation, one active mutation owner, observed effects within
  declaration, receipt envelopes still current. Telemetry may drop; control
  events must not. Checkpoint monitor state in `SessionStore`. Hand-authored
  closed rule IDs and table-driven trace tests — no agent-authored policy
  language. (`GOLANG_TUI_PLAN.md` Event Bus / Phase 1 must stop treating the
  bus as both audit substrate and drop-oldest channel.)
- Verification **envelopes**, not file-only receipts (verifier digest, args,
  rule/config, sandbox/runtime, lockfiles).
- Three-valued graph reachability (`found` / `not_found_in_complete_scope` /
  `unknown`). `wiring` must not treat “no path on a fuzzy graph” as fail-closed safety.
- Graph-bounded **mutation trust region**: predicted semantic blast vs observed
  `swe_graph.impact`; escape stales receipts and freezes the inner loop; outer
  loop may widen. Derived-file / formatter exceptions required.
- Counterexample-guided graph refinement: `may` (name match) / `must-within-index`
  (SCIP) / `unknown`; refine only the slice implicated by a failed check;
  bounded budget. Do not promote a runtime trace to a proof.
- Controllability preflight: if no authorized actuator or adequate sensor
  exists for an obligation, mark `uncontrollable` before the first retry.
  Privilege may only expand by explicit approval (monotonic confinement).
- Diagnostic observability matrix on obligations (`sensor_set`, `blind_spots`,
  `unobservable` ≠ discharged). Qualitative scopes, not invented probabilities.
- Mechanical **minimum V&V rung** from obligation kind/risk; LLM may escalate
  only.
- Pull `Tool.ReadOnly() bool` rework (already listed above) **before** encoding
  inner-loop control — otherwise the actuator model is wrong.
- `AgentState` in the plan must grow a mutation lease, workspace snapshot,
  obligation ledger, and receipt index — not remain messages/UI flags — if
  those are core invariants. Journal/receipt contracts belong in Phase 1 even
  if production backends wait.

**Adopt later (after receipts + journal exist):**

- Budgeted active-sensing scheduler (value-of-information among checks;
  hard minima override optimization).
- Mutation-adequacy sampling in an isolated snapshot, never the live worktree.
- Assume-guarantee overlays only from committed machine-readable contracts
  (API/schema/protocol), not LLM-drafted prose.

**Must not:** a second “V&V agent” loop; model-authored temporal/SMT/Scala;
scalar “correctness score”; treating coverage or a missing fuzzy edge as
absence-proof; AgentSpec/Progent-style LLM-generated policies at the safety
boundary (steal deterministic enforcement only).

## Tool output ergonomics (from axi.md, reviewed 2026-08-22)

Evidence: <https://axi.md> — “Agent eXperience Interface”, 10 principles plus
browser/GitHub benchmarks where a principled CLI beat both raw CLI and MCP on
success/cost/turns (MCP conditions used 2.3× the input tokens, dominated by
upfront schema loading). Caveats: author-run benchmarks, 14 tasks, one domain
per study; code-mode's cost ranking flipped between the two studies. Direction
agrees with Anthropic's code-execution-with-MCP findings; treat magnitudes as
indicative, not settled.

Scope: the CLI surface of swe-term-native tools (`swe_distiller`, future
`swe_graph`, backlog CLIs above). AXI says nothing about approval, effects,
mutation ownership, or receipts — it is read-path ergonomics only. Core
contracts in `ARCHITECTURE.md` govern what tools may do; this governs what
their output feels like. Where an AXI principle has a stronger swe-term
counterpart, the swe-term version wins.

**Adopt for swe-term-native tools:**

- Minimal default schemas (3–4 fields per list item) with flag-gated detail.
- Loud truncation with size hints and a `--full` escape hatch — subsumed by
  the stronger ARCHITECTURE.md rule (truncation/timeout/cancellation are
  distinct explicit states with a durable handle to the omitted remainder);
  AXI is independent validation, not the spec.
- Pre-computed aggregates in list output (counts, status rollups) to
  eliminate follow-up round trips. For `swe_graph`, design query commands
  to return rollups, and fuse common multi-step sequences into single
  combined commands — trajectory analysis attributes most turn savings to
  combined operations, more than to output format.
- Definitive empty states — upgraded to the three-valued form already
  adopted (`found` / `not_found_in_complete_scope` / `unknown`); never a
  bare “0 results” when scope completeness is in question.
- Strict exit-code contract: 0 success, 1 error, unknown flag → exit 2,
  never silently ignored; no interactive prompts; idempotent mutations.
  (Same doctrine as the interposed-coreutils exit-code contract.)
- Consistent concise `--help` per subcommand.
- Contextual next-step suggestions in output trailers, with a provenance
  constraint AXI lacks: suggestion text is tool-authored context injection.
  Permitted only when generated from tool-owned state, never templated from
  wrapped untrusted data (issue titles, page content, chat messages) — that
  is a prompt-injection surface. The ARCHITECTURE.md provenance rule
  (source/snapshot/freshness on injected items) applies to trailers too.

**Reject:**

- TOON as default output format. The ~40% token saving is real for uniform
  tabular data, but models have deep JSON/TSV priors and no TOON priors;
  escaping edge cases (delimiters inside fields) are exactly the silent-
  corruption class the shim-fidelity doctrine exists to prevent. Compact
  output yes; TOON only as an experiment behind a flag.
- Structured errors on stdout. Defensible for bespoke agent tools, but it
  inverts the Unix contract and would break the ≥99.5% fidelity target for
  anything interposed over real coreutils. swe-term-native tools keep
  errors on stderr with structured machine-readable bodies; interposed
  shims follow GNU reference behavior, no exceptions.
