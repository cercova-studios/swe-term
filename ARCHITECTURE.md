# swe-term Architecture

This is the authoritative architecture contract for the repository. It describes
both the implemented system and the accepted target design; every contract is
marked accordingly so plans are not mistaken for shipped behavior.

Plans, comparative evidence, design history and detailed validation records live
in the [designated development archive](docs/README.md#designated-development-archive).
Those records retain their dates, status and limitations.

## 1. Document Contract

- Read this file before architectural work, feature design, or non-trivial code
  changes.
- `Implemented` means the contract is present in current source and can be cited
  as runtime behavior.
- `Target` means the contract is an accepted design constraint but is not yet a
  runtime claim.
- The status of a contract changes only when source, tests, and this document
  change together.
- `docs/core/ARCHITECTURE.md` is a compatibility pointer, not a second source of
  truth.

## 2. Design Goals

- A general-purpose, minimal agent harness that can extend and improve itself
  under the harness's authority (see the
  [agent-editable harness direction](docs/plans/2026-09-05-agent-editable-harness.md)).
- One runtime behind three surfaces: the TUI, headless mode, and language SDKs
  (Python and TypeScript for Node/Bun) that embed the agent in another
  program, in the manner of the Claude Agent SDK and pi's
  `@earendil-works/pi-coding-agent`. obsidian-agent is the first SDK
  consumer and replaces LangGraph with it.
- Minimal, stable Go core with explicit interface boundaries.
- Single-binary, local-first operation with optional sidecars and services.
- ACP-aligned protocol model at the frontend/core boundary.
- Deterministic safety and clear operational behavior.
- Cloud-agnostic capability ports without vendor lock-in.
- Cheap operation in constrained environments: stream instead of buffer, keep
  resident state bounded, and load capabilities lazily.

## 3. Core Principles

- Keep the core small; push specialized logic to extensions.
- Treat architecture as contracts: interfaces over framework magic.
- Prefer immutable snapshots and explicit state transitions.
- Separate agent infrastructure from product surface concerns.
- Optimize for debuggability, replay, and predictable rollback.
- Spend complexity only on correctness or a measured operational constraint.
- The harness owns authority, state, and ground truth; the model owns
  cognition. Boundaries (approval, effect declarations, confinement, the
  control journal, receipts, provenance, budgets) are durable core contracts.
  Cognitive aids (planning scaffolds, prompt structure, reflection loops,
  per-model edit formats, summary-based compaction) are optional, configured
  per model, and expire when an ablation shows no lift. The core never imposes
  a fixed plan/act/reflect structure on the loop. The register is
  [`docs/core/MECHANISM_LEDGER.md`](docs/core/MECHANISM_LEDGER.md).

## 4. System Shape and Boundaries

### Layers

1. **Frontends** — TUI, headless/pipe mode, and SDK clients (target), all
   over the same protocol.
2. **Core (Go)** — provider event contracts today; target agent loop, state,
   approval, policy, journal, and verification control plane.
3. **Extensions** — providers, tools, analyzers, and sidecar adapters.
4. **External services (optional)** — retrieval, indexing, and sandboxes behind
   narrow adapters.

### Boundaries

- Frontend/core communication remains wire-protocol friendly.
- Core/extension interaction is interface-driven.
- Extension internals never leak into the core state model.
- Frontends render state; they do not own provider pricing, policy, verification,
  or persistence semantics.
- Heavy or language-specific engines run out of process behind bounded adapters.

### Current runtime slice

`main.runOnce` sends one `StreamRequest` to a `Provider`, collects the terminal
stream into a `Response`, and renders it. The interactive TUI consumes the same
provider event contract. There is not yet a multi-step tool-using agent loop.

Because there is no loop, a trace of this slice is a single span per turn.
`internal/telemetry` is in place and exports real `gen_ai.*` spans over OTLP,
but its cross-turn value arrives with in-flight refactor 1; until then it
earns its keep as per-turn cost and token attribution, not as a call tree.

## 5. Domain Types

| Contract | Status | Role | Source or intended home |
|---|---|---|---|
| `Message`, `StreamRequest`, `StreamEvent`, `Model` | Implemented | Provider-neutral request and streaming event vocabulary | [`internal/core/provider.go`](internal/core/provider.go) |
| `Provider` | Implemented | Streams completions and exposes model capabilities | [`internal/core/provider.go`](internal/core/provider.go) |
| `Usage`, `UsageTotals`, `Response` | Implemented | Provider-reported completion and session accounting. `Usage.InputTokens` is **uncached** input; `gen_ai.usage.input_tokens` is defined cache-inclusive, so exporters must reconstruct the total rather than copy the field | [`internal/core/usage.go`](internal/core/usage.go) |
| `Tool` | Implemented | Declares a versioned schema and executes a bounded invocation | [`internal/core/tool.go`](internal/core/tool.go) |
| `EffectDeclaration` | Implemented | Per-invocation paths, processes, network destinations, and secret names, with an observation level | [`internal/core/effect.go`](internal/core/effect.go) |
| `SessionStore` | Target | Persists sessions, journal checkpoints, task snapshots, and receipt indexes | [`internal/core/`](internal/core/) |
| `TaskSnapshot` | Target | Externalized, typed task state updated only from attributable evidence | [`internal/core/`](internal/core/) |
| `ControlEvent` / `ControlJournal` | Implemented | Durable ordered lifecycle record used by the safety monitor; JSONL file journal assigns sequence and fails loudly on corrupt or gapped records | [`internal/core/journal.go`](internal/core/journal.go), [`internal/core/control_monitor.go`](internal/core/control_monitor.go) |
| `VerificationReceipt` | Target | Fresh, source-bound evidence that an obligation was discharged | [`internal/core/`](internal/core/) |
| `Obligation` | Target | Required check with kind, risk policy, status, and minimum V&V rung | [`internal/core/`](internal/core/) |
| `MutationLease` | Target | Enforces a single active mutation owner | [`internal/core/`](internal/core/) |
| `ContextPacket` / `Provenance` | Target | Pre-model context item with source, snapshot version, freshness, three-valued scope completeness, and resolution tier | [`internal/core/`](internal/core/) |

Target type names describe contracts, not frozen Go identifiers. A feature plan
may refine their representation without weakening Sections 6 or 10.

## 6. Control and Safety Model

- Risky or mutating actions pass through policy-gated approval.
- Read-only scheduling and safety authorization are separate decisions.
- A tool invocation declares bounded effects before execution. Observed effects
  outside that declaration fail closed and produce an auditable mismatch event.
- Approval, mutation lease, observed effect, snapshot, and verification events
  are written to a durable ordered control journal. Lossy telemetry is separate.
- A small deterministic monitor enforces ordering and closed, hand-authored rule
  IDs. The model may explain rules; it cannot create or weaken them.
- Tool truncation, timeout, cancellation, denial, and execution failure are
  distinct states. Omitted output has a durable artifact handle.
- Sandbox and escalation are adapters; missing required confinement is an error.
- Secret names may enter declarations and receipts. Secret values never do.

## 7. State and Context

- Prefer immutable snapshots and typed transitions over mutable global state.
- A token threshold decides when to compact, not what survives compaction.
- Compaction preserves a protected spine: active constraints and approvals,
  unresolved errors, disproven hypotheses, dirty-file state, obligations, and
  artifact handles.
- Every injected context item carries its source, snapshot/version, and
  freshness. Staleness is checked rather than assumed away.
- A verification receipt binds source state to verifier identity: binary digest,
  arguments, rule/config digest, sandbox/runtime, and relevant lockfiles.
- Graph reachability is three-valued: `found`,
  `not_found_in_complete_scope`, or `unknown`.
- Recoverability requires persisted sessions, resume semantics, and replayable
  control events.
- Multiple bounded read-only investigations may run concurrently under a shared
  budget, but they join before mutation.

## 8. Portability and Deployment

- Provider/model, storage, sandbox/compute, and retrieval/indexing are capability
  ports with swappable backends.
- The reference deployment is a local modular monolith and single Go binary.
- Sidecars are optional, independently restartable processes with versioned,
  bounded protocols.
- **Implemented:** [`extensions/distiller`](extensions/distiller/README.md) is
  a standalone Rust URL/HTML-to-Markdown/JSON/EPUB CLI and library. Install locally with
  `cargo install --path extensions/distiller --locked`; the binary is `distiller`.
  Its deterministic path preserves code text/whitespace and numeric content,
  uses TeX annotations or MathML for equations, and retains complex tables as
  sanitized HTML. Prose cleanup excludes technical content; DOM sanitization
  and URL resolution operate on attributes. Renderer support remains external.
  Its `--stdin --base-url URL` extraction accepts bounded UTF-8 HTML without source fetching
  or invoking a model; EPUB cover production may separately call Muse. JSON stdout uses `article-extraction/v1` with the exact
  caller-supplied URL, snapshot SHA-256, extractor provenance, and Markdown,
  allowing independent consumers to validate retained evidence through pipes.
  Versioned arXiv EPUB/KFX inputs prefer the complete HTML article and check its
  matching PDF; a versioned local PDF stamp selects the same route. Other HTML
  can pair with a local `--pdf-source`. `html_first.rs` compares visible prose
  witnesses, restores uniquely associated missing artwork and isolated text, and
  rejects ambiguous gaps. A sidecar receipt binds both sources and output hashes
  while recording unchecked short spans and native-region equivalence. Semantic
  completeness and reading-order correctness still require source inspection.
  Its deterministic EPUB mode packages cleaned structured HTML through sandboxed
  Pandoc, with typography, metadata and heading navigation. Raster images are
  linked to retained caption descriptions; explicit table-header roles and narrow
  LaTeXML visual-header inference improve accessible structure. Empty layout tables
  without media are removed while preserving anchors; equation tables use separate typography.
  Visual header inference and caption fallback do not prove semantic accessibility.
  Raster images are
  decoded under limits; local files stay beneath the HTML resource directory,
  and URL/file modes fetch bounded HTTP(S) assets. Stdin never fetches. WebP
  becomes PNG. Output publication never overwrites an existing EPUB.
  Every successful EPUB has a cover: a caller-supplied PNG/JPEG (`--cover`), a
  verified cache entry, or Muse artwork with exact locally rasterized typography.
  `MODEL_API_KEY` is injected at runtime; one request per cache miss, no automatic
  paid retries. Generation failure prevents publication. Cache reservations
  retain ambiguous billing after interruption. `cover.rs` owns the bounded Muse
  request, licensed bundled font, grayscale profile and PNG provenance.
  Born-digital PDF conversion uses a bounded Rust/PDFium worker to extract
  source spans, font/position evidence, bookmarks and raster figures. A separate
  `pdf_reconstruct.rs` stage uses OpenRouter `deepseek/deepseek-v4.1-flash` with
  strict JSON Schema to assign source IDs
  to semantic blocks. Rust rejects missing/duplicate IDs, invented prose,
  invalid joins and unsafe furniture removals before EPUB publication. Each page
  pair requires a draft and a separately validated source review before caching.
  Oversized source packets split into single pages and shrink optional neighboring
  context without truncating owned spans; the 256 KiB cap covers source JSON.
  Exact source spellings can preserve compound separators. Related spellings
  select a bounded join-only model review, with evidence-bound decision caches.
  Unresolved decision-cache misses require credentials; invalid replies prevent publication.
  Bounded heading history, the book outline and intact-word witnesses aid review;
  captions remain separate from prose continuing across floating tables.
  Validated geometry repairs baseline numeral spacing, same-line run-ins and
  uniquely located linked callouts on fresh and cached plans. A bounded typography
  heuristic closes em dashes within aligned adjacent lines; explicit spacing and
  ambiguous page/column boundaries remain unchanged. The receipt hashes the effective
  normalized plan and rendered HTML for comparison. Draft and source-review passes
  start with low reasoning and treat candidates as user data.
  Malformed replies are not replayed.
  Draft retries can retain a typed source-ID candidate for focused correction.
  Every rejected-response retry disables reasoning within the existing attempt cap.
  Native regions are rendered before model calls. Opaque source anchors protect
  glyph/image/graphic membership; the model chooses semantic roles and reading order.
  Review policy participates in cache identity and receipts; diagnostics stay
  outside the reading flow. Scans,
  graphic-only pages and rotated/skewed/stretched Forms fail explicitly. Native
  bold/italic ranges survive in prose. Equations, mathematical/complex tables and
  vector figures use source-bound color crops with complete intersecting-object
  ownership and pixel receipts from a second bounded native worker. Native anchors
  can remain within paragraph/note flow; hidden numeric members remain link candidates.
  Ordinary text tables remain semantic. A narrowly aligned bold native header can
  become real column-header cells above a stable semantic body; math, spanning
  labels and referenced callouts retain native rendering. Inseparable inline formulas retain a whole
  source line as an image. General textbook semantics, accessible mathematical
  notation and reading-device acceptance remain targets.
  CLI-only KFX output reuses that EPUB path and invokes independent `epub-to-kfx`
  (Amazon Previewer/KFX Output). `scripts/save2kindle.py` composes URL/local-file
  conversion and independent `kindle-send` delivery, retaining KFX on transfer
  failure. Delivery uses the configured Kindle SSH alias and SHA-256 verification
  without overwriting existing books. Its Go tool adapter remains a target,
  as described in Section 9.
- Moving to services requires a measured isolation, scaling, or ownership pain;
  it is not the default evolution path.
- State formats and journal schemas outlive implementations and require explicit
  versioning and migration paths.

## 9. Extension Points

- **Provider adapters — implemented.** Add model backends without changing core
  stream semantics.
- **Observability decorators — implemented.** Wrap any `core.Provider` to emit
  telemetry without changing stream semantics.
  [`internal/telemetry`](internal/telemetry) is the first such decorator: it
  traces each `Stream` call with OpenTelemetry `gen_ai.*` attributes and is a
  strict pass-through, forwarding every `StreamEvent` unchanged, in order, and
  exactly once. A decorator here may observe but never repair, reorder,
  filter, or synthesise an event, because doing so would mask a Section 10
  invariant violation rather than report it; enforced by
  `internal/telemetry/provider_test.go`. No OpenTelemetry type enters
  `internal/core` (invariant 13). Tracing is off unless an OTLP endpoint is
  configured, so a default local run opens no connection. Design evidence:
  [`docs/plans/2026-10-06-openinference-openllmetry-integration-research.md`](docs/plans/2026-10-06-openinference-openllmetry-integration-research.md).
- **Tool adapters — contract implemented, adapters target.** Add bounded actions with the same manifest, effect,
  journal, and receipt contracts in every implementation language
  ([`internal/core/tool.go`](internal/core/tool.go),
  [`internal/core/effect.go`](internal/core/effect.go),
  [`internal/core/journal.go`](internal/core/journal.go)). The
  manifest declares a replay policy: the journal records a call's intent
  before its effect, and on resume a call cut off mid-run reruns only if it
  is declared replay-safe; otherwise it ends with an explicit `interrupted`
  result, never a silent rerun (invariant 9).
- **Analyzer/enrichment adapters — target.** Add pre-model context with explicit
  source, version, scope completeness, and freshness. The Fleet CPG engine is
  the first such adapter (target; not yet implemented): a code property graph
  engine producing a materialized context packet per PR, out-of-process
  behind `internal/analyzer/cpg`. Evidence for its design is preregistered
  `benchmark` experiments curated under `experiments/evidence/fleet-cpg-*`;
  the implementation plan is
  [`docs/plans/2026-09-05-fleet-cpg-engine-implementation.md`](docs/plans/2026-09-05-fleet-cpg-engine-implementation.md).
  An adapter of this kind must refuse to merge a diff overlay against a base
  snapshot whose revision is not the diff's parent revision, and report
  staleness explicitly instead of merging — a base one unrelated commit away
  from the true parent was measured to silently corrupt facts for files the
  diff never touched.
- **Session stores — target.** Swap persistence without changing state-machine
  semantics.
- **Policy and lifecycle hooks — target.** Intercept declared phases without
  rewriting the loop; hooks cannot bypass deterministic invariants.
- **Frontend protocol — target beyond the TUI.** TUI, headless, and ACP clients
  consume the same versioned event/state model.
- **SDK clients — target.** Thin Python and TypeScript libraries that start the
  `st` binary and speak the frontend protocol; they hold no loop, policy, or
  persistence of their own. A client may register tools implemented in its
  own language: the core calls them back over the protocol and applies the
  same effect, approval, journal, and receipt contracts as any tool adapter;
  it never loads client code in process. A client may supply a JSON Schema for
  the session result; the core validates it, providers declare support, and a
  violation is a failure. Submissions carry a caller request ID, and a
  repeated ID returns the existing submission. Go programs may import the
  core packages directly.
  The client-over-protocol shape keeps one implementation of the core across
  languages. obsidian-agent's requests OA-001 to OA-003
  ([ledger](docs/cross-project/obsidian-agent.md)) are its first contract.

Capabilities have two lanes: runtime/ad-hoc for iteration and compiled for
durable first-class support. Both emit the same versioned manifest and
policy-relevant events. Discovery is lazy and task-scoped; selected schema
versions are frozen into the session snapshot.

A capability moves into core only when independent extensions must agree on it
for safety, replay, or correctness. Popularity alone is not sufficient.

## 10. Invariants

These are non-negotiable and require mechanical enforcement or tests:

1. Exactly one terminal completion exists for a successful provider stream.
2. Approval precedes every mutating action that requires approval.
3. At most one mutation lease is active for a workspace.
4. Observed effects remain within the approved declaration.
5. Read-only investigations join and refresh contested state before mutation.
6. A lifecycle claim such as `verified`, `done`, or `ready_to_merge` requires a
   current receipt bound to the current source and verifier state.
7. Obligation kind and risk determine a minimum V&V rung; a model may escalate
   but never downgrade it.
8. Control events required for replay and safety do not drop.
9. Truncation, timeout, cancellation, denial, and failure are never flattened
   into successful text output.
10. Compaction cannot silently evict the protected spine.
11. Incomplete graph absence is `unknown`, not proof of non-existence.
12. Unknown pricing, provenance, freshness, or confinement remains explicit; it
    is never converted to a reassuring zero or success.
13. Extension internals and vendor-specific types do not enter core state.
14. Secret values are absent from prompts, journals, receipts, and artifacts.

## 11. Verification and Operations

- Table-driven trace tests cover fail-closed ordering and journal monitor rules.
- User-journey tests use mock providers and real local tool wiring; they assert
  observable behavior rather than implementation helpers.
- Receipt verification is independent of the model that requested the check.
- Evaluations separate tool invocation, task outcome, side effects, evidence
  quality, and reproducibility. Aggregate scores do not replace component data.
- Harness research follows the VCS-neutral specification, preregistration,
  isolation, and evidence-promotion contract in
  [`experiments/README.md`](experiments/README.md). Experiment infrastructure is
  tooling around the core, not a second agent loop or state model.
- Two experiment kinds share that contract: `mechanism-hypothesis` (paper-backed
  causal A/B claims about harness mechanisms) and `benchmark` (descriptive
  measurement of an external tool or artifact, gated by a design-document
  reference instead of a paper). A `benchmark` experiment is how a `Target`
  extension point in Section 9 — e.g. an analyzer/enrichment adapter — earns
  evidence before it has code. The Fleet CPG engine's nine curated bundles
  under [`experiments/evidence/fleet-cpg-*`](experiments/evidence/) are the
  first `benchmark` evidence promoted under this contract.
- Paper-backed hypotheses enter that contract through the mechanism-first
  discovery, source fallback, signal/noise triage, and selection workflow in
  [`docs/research/papers/README.md`](docs/research/papers/README.md). Discovery
  may propose an experiment; it cannot promote a claim into architecture.
- Architectural drift is sensed continuously by [`cmd/drift`](cmd/drift)
  against [`docs/reports/drift-baseline.json`](docs/reports/drift-baseline.json).
  It is a **ratchet, not a threshold**: it blocks new or worsened values rather
  than enforcing an absolute target. Re-baselining is therefore a deliberate,
  justified act, never a way to clear a red signal. Every baseline move must
  name the change that caused it and why the new level is accepted; the record
  lives in
  [`docs/reports/research-and-experiments-summary.md`](docs/reports/research-and-experiments-summary.md).
- Every harness mechanism has a row in
  [`docs/core/MECHANISM_LEDGER.md`](docs/core/MECHANISM_LEDGER.md). A cognitive
  aid needs a `mechanism-hypothesis` ablation before it ships by default and is
  re-ablated when a supported model changes; a boundary is measured for the
  friction it costs, not for whether it should exist.
- Operational failures must retain enough state to resume, explain the failed
  rule, and identify whether retry is safe.
- Rollback restores versioned session/harness records; it does not erase the
  audit trail.

## 12. In-Flight Refactors

These are accepted direction, not completed runtime claims:

1. **One-shot stream to agent loop.** Introduce tool dispatch and steering while
   retaining the current provider completion invariant.
2. **Tool safety contract.** Replace the planned `ReadOnly() bool` design with
   per-invocation effect declarations before landing the first mutating tool.
   Contract landed (`internal/core/effect.go`, `tool.go`, `journal.go`); the
   first mutating tool is still pending.
3. **State expansion.** Define session snapshot, obligation ledger, mutation
   lease, receipt index, and durable control journal before production
   persistence.
4. **Frontend protocol seam.** Keep TUI behavior on core events while defining a
   versioned protocol usable by headless, ACP, and SDK clients, including
   client-hosted tool callbacks and schema-validated session results.
5. **Architecture path migration.** Root `ARCHITECTURE.md` is now authoritative;
   the old `docs/core/ARCHITECTURE.md` path remains a temporary compatibility
   pointer.
6. **Analyzer adapter: Fleet CPG engine.** An out-of-process Rust sidecar
   (`cpgd`, sibling Cargo workspace `engine/`) behind `internal/analyzer/cpg`;
   slices defined in
   [`docs/plans/2026-09-05-fleet-cpg-engine-implementation.md`](docs/plans/2026-09-05-fleet-cpg-engine-implementation.md).

Plans that touch these items must state whether they advance, defer, or conflict
with the refactor. Conflicts require an explicit architecture decision.

## 13. Source Hierarchy

The root `ARCHITECTURE.md` owns architectural constraints and their
`Implemented` / `Target` status. Current source and tests decide whether a
constraint is implemented. If code contradicts an implemented claim here, stop
and resolve the drift before building on either version.

Scoped operating contracts govern their workflows unless they conflict with this file:

- [CONTRIBUTING.md](CONTRIBUTING.md) governs change delivery and documentation placement.
- [Mechanism ledger](docs/core/MECHANISM_LEDGER.md) governs mechanism classification and shipping rules.
- [Experiments](experiments/README.md) governs execution and evidence promotion.
- [Paper discovery](docs/research/papers/README.md) governs research selection.

The [development archive](docs/README.md#designated-development-archive) preserves
roadmaps, rationale, comparisons and historical validation. Those records explain
decisions; a plan or dated result does not override current source or establish
current verification. Accepted targets remain explicit in this architecture contract.

## 14. How to Keep This Document Honest

Update this file in the same change when work alters:

- a layer or dependency direction;
- a domain contract in Section 5;
- an extension point or promotion boundary;
- an invariant, safety rule, or minimum verification requirement;
- a state, journal, receipt, persistence, or protocol schema;
- deployment shape, sidecar boundary, or source hierarchy;
- the status of an in-flight refactor.

For every update:

1. Label contracts `Implemented` or `Target` from current source evidence.
2. Link the owning source, test, plan, or evidence document.
3. Search for superseded names and paths across code, tests, prompts, and docs.
4. Remove stale compatibility surfaces unless their owner and removal condition
   are documented.
5. Run reference checks and the smallest relevant verification suite.
