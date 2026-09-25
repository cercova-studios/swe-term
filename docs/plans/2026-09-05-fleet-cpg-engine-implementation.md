# Fleet CPG Engine — tech stack and implementation plan

Status: proposed. Evidence gathered (9 preregistered `benchmark` experiments,
curated under `experiments/evidence/fleet-cpg-*`); promotion into
`ARCHITECTURE.md` awaiting explicit human acceptance per
`experiments/README.md` § Evidence and promotion.

Design reference: the Fleet CPG Engine design document
(<https://claude.ai/code/artifact/0c277fff-2382-485d-8316-31af8f4179e7>).
This plan is the bridge between that document's build order — every phase of
which now has empirical evidence — and code in this repository.

## Problem

`ARCHITECTURE.md` §9 names **Analyzer/enrichment adapters** as a `Target`
extension point: "add pre-model context with explicit source, version, scope
completeness, and freshness." No adapter exists. The Fleet CPG Engine is the
first one: a code property graph engine that turns a PR into a materialized
context packet (touched definitions, blast radius, provenance) for the review
agents, at O(diff) cost per PR rather than O(repo).

The design document was written before integration with `swe-term` was
decided. This plan reconciles it with the repository's actual constraints:
Go core, single local binary, sidecars behind bounded protocols (§4, §8),
and the invariants in §10.

## What the experiments settled

Every row below is a measured fact from `experiments/evidence/fleet-cpg-*`
and the design consequence it forces. Nothing here is inferred from the
design document alone.

| Measured | Where | Consequence for this plan |
|---|---|---|
| Joern's cost is per-invocation JVM/Ammonite startup (7.2s at 21k LOC → ~224s at 3.03M LOC); in-graph query latency is sub-ms once resident | baseline-{fastapi,zod,excalidraw,home-assistant-core} | The engine is a **resident process** with a warm store. Process-per-query is the one shape ruled out by data. |
| Joern's `jssrc2cpg` is not deterministic at ~173k LOC (typeDecl count wobbled by 1) | baseline-excalidraw | "Incremental ≡ from-scratch" is a **CI-enforced differential test**, not an assumption; the experiments' evaluators become that test. |
| Cost scaling is sublinear with LOC for Joern on home-assistant's shape | baseline-home-assistant-core | Do not extrapolate a per-line budget from small repos; slice 1 measures the Rust engine on all four pinned repos before any budget is written down. |
| Whole-file overlay extraction is 18–44× below full rebuild on real PRs; cost tracks touched-file size, not diff line count | phase1-overlay-zod | **Whole-file re-extraction in v1.** Hunk-scoped extraction is a named upgrade with a measured trigger (below). |
| COW merge is set-exact vs. ground truth (9/9); overlay storage 5–12.5% of base | phase1-commit-zod | Layered store with query-time newest-wins merge; the base is never rewritten by an overlay. |
| An overlay merged against a base one unrelated commit away from the diff's parent silently corrupts untouched files | phase1-commit-zod (pilot) | **Freshness precondition, enforced in code**: base snapshot revision must equal the diff's parent revision, or the adapter refuses and reports `stale`. This is the mechanism behind §7 "staleness is checked rather than assumed away." |
| DuckDB per-connection overhead (150–500ms) dominates every commit-path measurement | phase1-commit, phase2, phase3 | No per-request database handle. In a resident Go process this cost disappears by construction; DuckDB itself is not carried into v1 (see stack). |
| End-to-end PR → packet composes with zero manual steps; packets round-trip as plain files | phase2-blast-radius-zod | The packet **file tree is the adapter's output contract**; the core receives a path plus provenance, never engine types (§10 inv. 13). |
| Name-only callee resolution produces concrete false positives (`Array.prototype.map` resolved to an unrelated touched function named `map`) | phase2-blast-radius-zod | Every fact and every impact edge carries a **resolution tier**; blast-radius scope is reported three-valued (§10 inv. 11). Compiler-integrated resolution is a named upgrade. |
| Re-flattening the base every push is O(repo) (~8.5s/step); a layered store keeps per-step cost tied to that step's diff | phase3-differential-zod (pilot + official) | LSM-style layers with **compaction as a separate, background operation**, never on the push path. |
| Merge-query cost is flat to 6 layers; nothing is known beyond that | phase3-differential-zod | Compaction threshold is **unknown**; it is a configurable knob with instrumentation and a follow-up experiment, not a constant. |
| A second language (Python) required zero bytes changed in the shared core, verified by hash | phase4-second-language | The L0/L1 ↔ engine boundary becomes a **dependency-direction test in CI** (engine packages may not import language packages), which is stronger than a file hash. |

## Tech stack

The design document's reference architecture is the unconstrained target.
Each row states the reference, the v1 choice, and — where v1 is a downgrade —
the measured condition that turns the dial back up. Downgrades are labeled;
none is silent.

| Component | Reference (design doc) | v1 in `swe-term` | Why / dial |
|---|---|---|---|
| Engine language | Rust | **Rust**, matching the reference — a sibling Cargo workspace (`engine/`), built as an out-of-process sidecar. Confirmed as the explicit choice over the Go-sidecar downgrade this plan originally proposed. | Not a downgrade — the reference choice. This also fits `ARCHITECTURE.md` §4 more cleanly than a Go engine would have: "heavy or language-specific engines run out of process behind bounded adapters" is close to forced once the engine is a different language from the core, whereas an in-process Go engine could have tempted a boundary blur. Cost: two toolchains, two test runners, `cargo` + `go` both in CI. |
| Parsing | tree-sitter | `tree-sitter` crate v0.27.0 + `tree-sitter-typescript` v0.23.2, `tree-sitter-python` v0.25.0 (verified resolvable on crates.io 2026-09-05 via `cargo add --dry-run`; pin exact versions per snapshot — a grammar bump is a re-index event) | Not a downgrade. Grammar pins are recorded in the snapshot manifest. |
| Fact storage | Owned mmap columnar segments, Arrow-compatible buffers | **Arrow IPC files** via the `arrow`/`arrow-ipc` crates (v59.3.0, verified resolvable) plus `memmap2` (v0.9.11) for zero-copy reads, inside an owned, versioned snapshot/layer manifest (JSON) | Partial downgrade, labeled, independent of the language decision above. Rust removes the *implementation-cost* argument against building the owned segment format directly (native control over memory layout, no cgo/FFI tax to fight) — but the experiments only validated correctness/timing at small scale on DuckDB/Parquet scaffolding, and nothing yet says Arrow IPC is insufficient. Dial up: owned segment format when a needed index (symbol trie, reverse-call adjacency) cannot be expressed as an Arrow column, or when compaction throughput is measured to be IPC-bound — cheaper to pull forward now than it would have been from Go. |
| Rejected for v1 | — | DuckDB (`duckdb-rs`: the measured per-connection tax that dominated every Phase 1–3 commit-path number is a DuckDB architectural property, not a host-language one); Parquet as the primary store (encoded, not zero-copy — fine as an export format) | — |
| Metadata | Postgres (fleet) | JSON manifests on disk, content-addressed by commit SHA, `schema_version`ed | Deployment dial, not a design dial: `swe-term`'s reference deployment is a local single binary (§8). Postgres returns when a fleet needs the readiness join across hosts. |
| Event backbone | NATS JetStream | None. The adapter is invoked on demand by the harness; a filesystem/VCS watch hook is a later addition | Deployment dial. |
| Core ↔ engine protocol | gRPC readers | Long-lived sidecar over a unix socket, newline-delimited JSON messages with an explicit protocol version; independently restartable; adapter starts it lazily | Matches §8 "sidecars are optional, independently restartable processes with versioned, bounded protocols." A single Go consumer doesn't yet justify gRPC's schema/codegen weight; that's the dial when a second consumer language appears — orthogonal to the engine-language decision. |
| Agent-facing interface | Materialized file tree + shim-routed symbol queries | Materialized file tree (packet). Shims are out of scope for this plan | Same as reference for the half that exists. |
| Observability | OTel, shared span taxonomy | OTel Go SDK on the core side; the Rust sidecar emits structured JSON logs correlated by `profile_version`/`pr_id` (an OTel Rust SDK is a later addition, not required for v1's single-sidecar topology) | Same as reference in intent; Rust-side OTel is deferred as a dial, not a downgrade of the taxonomy itself — correlation keys are the same either way. |
| Isolation | gVisor/Firecracker for extraction | Per-file parse size cap and bounded parse time; store mounted read-only to readers; no network in the sidecar | Deployment dial. Verified against the actual pinned 0.27.0 API (there is no timeout setter): `Parser::parse_with_options` takes a `ParseOptions` whose `progress_callback` runs periodically during parsing and returns `ControlFlow::Break(())` to cancel — bound wall time by returning `Break` once a deadline passes inside that callback, not by calling a timeout method. |

## Package layout

```
engine/                          Cargo workspace, sibling to the Go module
  cpgd/                           sidecar binary crate (the process; owns the protocol loop)
  cpg-schema/                     L1 fact types (Def, Call), schema_version
  cpg-store/                      snapshots, layers, manifest, newest-wins merge, compaction
  cpg-derive/                     L2 blast radius, three-valued scope
  cpg-packet/                     packet materialization
  cpg-engine/                     ingest base, overlay, differential step
  cpg-lowering-typescript/
  cpg-lowering-python/
internal/core/                    ContextPacket, Provenance — the only types core sees
internal/analyzer/cpg/            core-side adapter: sidecar client (spawn, unix-socket
                                  protocol, restart), freshness precondition, degrade-to-unknown
```

Boundary enforcement, now structural rather than merely tested: `cpg-schema`,
`cpg-store`, `cpg-derive`, `cpg-packet`, and `cpg-engine`'s `Cargo.toml` files
declare no dependency on `cpg-lowering-*` or on `tree-sitter*` — Cargo cannot
compile an illegal edge that was never declared, which is stronger than the
Phase 4 hash check (a mechanical audit of `Cargo.toml`s in CI catches an edge
added by mistake before it would even fail to build in a way that reveals the
violation). The Go side keeps its own, smaller version of the Phase 4 check:
`internal/analyzer/cpg` and `internal/core` may not import any Rust FFI or
CGo — trivially true once the engine is a separate process with no cgo bridge.

## Core contract (§5 addition, proposed)

```go
// ContextPacket is what an analyzer adapter hands the core. Engine types
// never cross this boundary (ARCHITECTURE.md §10 inv. 13).
type ContextPacket struct {
    Path       string      // materialized file tree root
    Provenance Provenance
}

type Provenance struct {
    Source           string // e.g. "fleet-cpg"
    BaseRevision     string // snapshot the overlay was merged against
    HeadRevision     string // the diff's head
    Freshness        Freshness // Fresh | Stale{BaseRevision != parent(Head)} | Unknown
    Scope            Scope     // Complete | Partial{missing} | Unknown
    ResolutionTier   string    // "syntactic-heuristic" | "compiler"
    ProfileVersion   string    // snapshot manifest digest; joins to telemetry
}
```

Three-valued reachability (§7, §10 inv. 11) is carried per impact edge inside
the packet and summarized in `Scope`. With the v1 heuristic tier, an absent
edge is `unknown`, never `not_found_in_complete_scope`.

## Invariants touched (§10)

- **11** — blast radius reports `found` / `not_found_in_complete_scope` /
  `unknown`; the heuristic tier can only produce `found` and `unknown`.
- **12** — a missing or unreachable sidecar yields no packet and a context
  item marked `unknown`; never an empty "success." Freshness is always
  explicit.
- **13** — only `ContextPacket`/`Provenance` enter core state.
- **14** — packets contain symbol names and line numbers, not source text or
  literals; no secret values can transit. Still scanned in the packet test.
- New precondition (proposed home: §9 adapter contract, not a new invariant):
  an overlay may only be merged against a base whose revision equals the
  diff's parent; otherwise the adapter reports `Stale` and does not merge.

## In-flight refactors (§12): advance, defer, or conflict

1. Agent loop — **independent.** The adapter is callable before a loop
   exists (a headless `swe-term context <base>..<head>` command in slice 2).
   No conflict.
2. Tool safety contract — **defer alignment.** The analyzer is read-only on
   the repo and writes only its own store directory; when
   `EffectDeclaration` lands, the adapter declares exactly that. No conflict.
3. State expansion — **depends on.** Where `Provenance` is recorded in a
   session snapshot is decided by that refactor; v1 returns packets without
   persisting them into session state. No conflict.
4. Frontend protocol — not touched.
5. Architecture path migration — not touched.

## Implementation slices

Each slice has an entry gate, a definition of done, and the check that proves
it. No slice starts on faith that the previous one would have passed.

**Slice 0 — Contracts. Done 2026-09-20** — `internal/core/context_packet.go`,
27 table-driven cases green.

Correction to this slice as originally written: it said §5 "flips to
`Implemented` in the same change the types land." That was wrong and has not
been done. §1 defines `Implemented` as a contract that "can be cited as
runtime behavior," and nothing *produces* a packet — there is no adapter yet.
`VerificationReceipt` and the V&V gate are in exactly this position (reducers
with no loop around them) and are correctly still `Target`. `ContextPacket`
stays `Target` until an adapter emits one. `ARCHITECTURE.md` is untouched.

What landed, and the reasoning behind the parts that are not obvious:

- `DetermineFreshness(base, diffParent)` — equal is `fresh`, different is
  `stale`, **either missing is `unknown`, never an optimistic default.**
- `Provenance.ConcludeAbsence()` implements invariant 11's three values.
  Absence proves non-existence only under **complete scope *and* compiler
  tier**. A complete scan with heuristic resolution still cannot support the
  claim, because a missed edge there is a resolution failure rather than a
  coverage gap — the `map` false positive from phase 2 is the worked example.
- `Validate()` refuses a `fresh` claim that lacks the two revisions justifying
  it. Making a claim carry its own evidence is the same move as the V&V
  ladder making downgrade unreachable: the assertion cannot be made
  unsupported.
- `Unavailable(source)` returns a **well-formed packet carrying honest
  unknowns**, not an error and not an empty success — invariant 12.
- `UsableAsAuthoritative()` returns a reason, not a bool, and the stale reason
  names both revisions so a caller can act on it (M6 typed feedback). Callers
  are expected to degrade and label, not drop.

**Slice 1 — Engine core + TypeScript (Rust).** `cargo new --lib` the
`cpg-schema`, `cpg-store` (Arrow IPC segments, manifests, newest-wins merge),
`cpg-lowering-typescript`, and `cpg-engine` crates in `engine/`. Port the
experiments' evaluators into a golden differential test suite over the pinned
zod commits used in phase1–3 (incremental == from-scratch at every step,
`cargo test` in CI). Done: the suite passes, and the engine reproduces the
recorded counts on the pinned commits (`defs=1388`, `calls=43888` at zod
`5ff9566`) or documents every divergence with a reason. Also: run the full
base build on all four Phase 0 repos and record wall time and RSS — the first
Rust-engine numbers against the Joern baseline. Preregister as
`fleet-cpg-engine-v1-baseline`.

**Slice 2 — Value path.** `cpg-derive` (blast radius, three-valued scope),
`cpg-packet`, and a minimal `cpgd` binary that can be invoked once per call
(no persistent socket loop yet) plus a Go-side `internal/analyzer/cpg` that
shells out to it. A headless `swe-term context` command. User-journey test
with the mock provider: PR in, packet path plus provenance out, no manual
steps. Done: the journey test and a packet round-trip test pass.

**Slice 3 — Resident sidecar + operations.** `cpgd` grows a persistent
unix-socket protocol loop with version negotiation; `internal/analyzer/cpg`
gains lazy start, restart, and the freshness precondition (refuse to merge,
report `Stale`, rather than crash or silently proceed); OTel spans on the Go
side keyed by `profile_version`; compaction as a background job in
`cpg-store` with a configurable layer threshold. Preregister
`fleet-cpg-compaction-depth`: merge cost vs. layer depth on
home-assistant-core scale — the open question Phase 3 could not answer at six
layers. Done: sidecar kill/restart test (the Go adapter observes and recovers
from a killed `cpgd` process); compaction differential test (compacted ==
uncompacted view); the experiment's report.

**Slice 4 — Second language.** `cpg-lowering-python`; the `Cargo.toml`
dependency-direction audit in CI. Done: Python passes the same golden
differential suite on the pinned fastapi commit; the audit passes.

**Slice 5 — Named upgrades, each with a trigger.**
- Hunk-scoped extraction — trigger: overlay p95 exceeds the per-PR budget on
  the Phase 0 repo set and profiling attributes it to touched-file size.
- Compiler-integrated resolution (`tsc` API, pyright) — trigger: false-positive
  rate on a golden impact-set fixture exceeds a written threshold; the
  `map` collision is the first fixture.
- Owned segment format — trigger: an index need Arrow columns cannot express,
  or measured IPC-bound compaction. Cheaper to pull forward now than it would
  have been from a Go host, since there is no FFI boundary to design around.

## Operability

- **Deploy:** a second binary (`cpgd`), built from the sibling Cargo
  workspace and shipped alongside the Go binary; the core works without it —
  a missing or unreachable `cpgd` degrades to `Provenance.Freshness = Unknown`,
  never a crash (§10 inv. 12).
- **Observe:** structured logs and (later) OTel spans keyed by
  `profile_version`; the store manifest is human-readable JSON; `cpgd inspect
  <store>` prints snapshots, layers, and grammar pins.
- **Roll back:** the store is `schema_version`ed and content-addressed;
  rolling back either binary independently leaves old snapshots readable or
  rejected loudly by version, never misread. The Go core and the Rust engine
  version independently — the wire protocol version is the compatibility
  contract between them, not the two binaries' release versions.
- **Debug at 3am:** packets are plain files; every impact edge names its
  resolution tier; every merge names the base revision it used; a `cpgd`
  crash shows up as a socket error the adapter turns into `Unknown`, not a
  hang.

## Constrained-runtime budget

Segments are mmap'd (`memmap2`) rather than heap-materialized; extraction
streams one file at a time with a size cap; no whole-repo AST is ever
resident. Rust removes the GC-pause and memory-layout uncertainty a Go engine
would have carried into this budget, but the crate boundary (`cpg-store` never
copies a segment it can slice a view into) still has to be verified, not
assumed, once written. The only number worth writing down today is the one
slice 1 measures: full base build wall time and peak RSS on
home-assistant-core (3.03M LOC) for the Rust engine versus Joern's
50–61s / 13–16GB. That measurement, not a target picked in advance, sets the
budget.

## Open questions carried forward

| Question | Origin | Where it is answered |
|---|---|---|
| Overlay granularity (file vs. hunk) | phase1-overlay | Slice 5 trigger |
| Freshness enforcement in code | phase1-commit pilot | Slice 3 (adapter test) |
| Compaction cost at real layer depth | phase3 | Slice 3 experiment |
| Heuristic resolution false positives | phase2 | Slice 5 trigger + golden fixture |
| L1 schema under a third language | phase4 | Not scheduled; first candidate should be a language with a construct `defs/calls` cannot represent |
| Base build cost of the Rust engine at 3M LOC | none yet | Slice 1 experiment |

## Proposed `ARCHITECTURE.md` changes (pending acceptance)

1. §5 — add row: `ContextPacket` / `Provenance` — Target — "Pre-model context
   item with source, snapshot version, freshness, three-valued scope
   completeness, and resolution tier" — `internal/core/`.
2. §9 — under Analyzer/enrichment adapters, append: the Fleet CPG engine is
   the first adapter (target); evidence in `experiments/evidence/fleet-cpg-*`;
   plan in this document; an adapter must refuse to merge an overlay against
   a base snapshot not keyed to the diff's parent revision and report
   staleness instead.
3. §11 — link the curated evidence directories as the first promoted
   `benchmark` evidence.
4. §12 — add item 6: "Analyzer adapter: Fleet CPG engine as an out-of-process
   Rust sidecar (`cpgd`, sibling Cargo workspace `engine/`) behind
   `internal/analyzer/cpg`; slices in this plan."

No invariant is added or changed. The freshness precondition is proposed as
adapter-contract text in §9; promoting it to a §10 invariant is a separate
human decision.
