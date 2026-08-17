# Critical Evaluation of Joern as a swe-term Extension (and of a Rust Rewrite)

> Evaluated against the Harness (swe-term) philosophy:
> small Go core, interface-driven plugins, content-addressable caching,
> single binary, heavy engines as **sidecar binaries** (not FFI),
> Analyzers enrich context *before* the LLM.
>
> Pair with [`JOERN_DEEP_DIVE.md`](JOERN_DEEP_DIVE.md). This document answers
> one question: **should we rewrite Joern in Rust and ship it as a swe-term
> extension?**

---

## Verdict Summary

**Do not rewrite Joern in Rust.** A faithful port is a multi-year compiler-team
project whose hardest pieces are *not even Joern’s* (Eclipse CDT, Soot, Ghidra,
Roslyn, Graal). `cpg-rs` already shows the cheap part — serde types for the
schema — and stops there for a reason. AppThreat’s `chen` forked Joern and is
**still on the JVM**. Joern 4 already rewrote the graph store (OverflowDB →
flatgraph) *in Scala* for 40% RAM; that is the rewrite that was worth it, and
they did it.

**Do rebirth the CPG *idea* as a swe-term Analyzer**, in Rust, at 5–10% of
Joern’s power and ~90% of agent value: tree-sitter tags + import/call graph +
SCIP ingest + SQLite, behind the existing `swe_distiller` sidecar pattern.
Optionally **wrap** stock `joern-parse` later for security-review taint — with
a **closed query pack**, never the unsandboxed Scala HTTP interpreter.

| Option | Effectiveness for swe-term | Effort |
|--------|----------------------------|--------|
| Full Joern → Rust | **Negative ROI** | Multi-year, team |
| Wrap Joern JVM as always-on sidecar | **Poor** (startup, RAM, Scala API, no incrementality) | Weeks, operational pain forever |
| Wrap Joern for opt-in security mode, canned queries | **Fair**, later | Days–weeks |
| Rebirth: `extensions/swe_graph` (CPG-lite) | **High** | Matches Phase 5 Analyzer plan |
| Point LLM at `joern --server` | **Forbidden** (RCE) | — |

---

## 1. Effectiveness

### What Joern does that agents actually want

**Blast radius and reachability are real questions.** “Who calls this?”, “can
this request reach `exec`?”, “what files move if I change X?” are the
out-of-diff bugs the Context Gathering Engine OKRs care about. Grep cannot
answer them. A graph can.

**One IR across languages is the right *interface*.** swe-term’s `Analyzer`
should not have a Java-shaped API and a Go-shaped API. CPG’s node/edge
vocabulary (`METHOD`, `CALL`, `REACHING_DEF`) is a decent target *schema* even
if we never run Joern.

**Fuzzy parse without a build** matches agent reality (dirty worktrees, missing
deps). SCIP indexers often need a successful compile. Tree-sitter and Joern
both degrade more gracefully.

**Taint + library semantics** is the slice nothing else in PLAN covers.
Semgrep is pattern/SAST. SCIP is names. Joern is dataflow. That is the only
reason to ever shell out to Joern.

### What Joern gets wrong for a harness

**The query surface is a programming language.** Agents need JSON in / JSON
out. Joern’s server is:

```
POST { "query": "<arbitrary Scala>" }
```

Docs demonstrate `println("remote execution vector")` and state the server
**is not a security boundary**. Feeding model-authored strings to that endpoint
is RCE with extra steps. A swe-term Tool that “lets the agent write CPGQL”
fails the Approver model.

**Batch indexer, not a daemon for a TUI.** Full rebuild on the order of
**minutes** (issue #5865: ~10 min Java project). `importCode` starts a
**second JVM**. Kernel-scale heaps are **tens of GB**. That cannot sit on the
hot path before every prompt. Analyzers in `GOLANG_TUI_PLAN.md` are supposed
to run on *changed files* with content-addressable reuse.

**Language maturity is lopsided.** C/Java “Very High.” Go — swe-term’s own
implementation language — is **Medium**. Ruby/C# Medium-Low. An agent working
in a TS/Go/Python monorepo does not get Joern’s best frontends.

**Correctness cliffs.** `ReachingDefPass` gives up at 4000 defs/method. Bundled
JS and generated code — common in the repos agents touch — silently lose
dataflow.

**JVM tax vs single-binary local-first.** JDK 21, fat zips, `-Xmx` rituals,
optional gcc for headers. The distiller pattern exists so we *don’t* pull this
into the Go process. Wrapping it is allowed; pretending a Rust rewrite removes
the tax while keeping CDT/Soot is fantasy.

---

## 2. Why a Rust Rewrite Fails a Cost Model

Decompose Joern. Rewrite each layer independently:

| Layer | What you’d rewrite | Rust win? | Agent need? |
|-------|-------------------|-----------|-------------|
| Frontends | Re-wrap CDT/Soot/Ghidra/Roslyn/Graal **or** lose fidelity | No — those aren’t Rust | Partial (TS/Go/Py) |
| CPG schema | Already specified; `cpg-rs` exists | Trivial | Schema yes, full spec no |
| Overlay passes (CFG, types, call linker) | The actual analysis | Maybe faster | **Yes, subset** |
| flatgraph | Already rewritten in Scala for RAM | Diminishing | No at kernel scale |
| Scala DSL | Replace entirely | N/A — don’t port | **JSON algebra** |
| dataflowengineoss | Research-grade taint | Hard, high skill | Security mode only |
| querydb | Vuln pack | Don’t port; use Semgrep/CodeQL | Low for SWE loop |
| REPL / --server | Don’t port | — | Forbidden as-is |

The layers with a Rust win (graph store, some passes) are not the layers that
make Joern *Joern*. The layers that make Joern Joern (frontends + taint +
decade of querydb) are the layers you cannot honestly finish as a sidecar
hobby.

**Historical evidence:** OverflowDB → flatgraph was the internal performance
rewrite. They did not switch to Rust. chen/atom forked and stayed on Scala 3.
weggli, the successful Joern-*inspired* Rust tool, **refused the CPG** and
did AST pattern matching for C/C++ only. That is the Pareto frontier.

**FFI/N-API lesson from oh-my-pi:** even if you wrote a Rust CPG engine,
swe-term’s rule is sidecar spawn, not in-process addon. A rewrite does not
change the integration shape; it only changes who maintains 12 frontends.

---

## 3. What “Rebirth” Should Mean

Not “Joern in Rust.” **CPG-lite as the Analyzer the plan already named.**

```
extensions/swe_graph/          # Rust sidecar, swe_distiller template
  swe_graph index <root>       # tree-sitter + import/call edges → SQLite
  swe_graph query callers --of Foo
  swe_graph query path --from A --to B
  swe_graph query blast --path src/x.go --depth 2
  swe_graph ingest-scip index.scip

cli/graph/                     # Go: Analyzer + Tool
  Analyze() → repomap / important symbols (pre-LLM)
  Query()  → structured JSON (agent tool)
```

Contracts:

- **JSON only.** Closed operations (`callers`, `callees`, `path`, `blast`,
  `symbol`). No eval.
- **Content-addressable.** File hash → subtree; Merkle skip unchanged files
  (`GOLANG_TUI_PLAN` cache story).
- **Incremental by construction.** Joern’s missing feature is our default.
- **SCIP when present, tree-sitter when not.** Precise refs are an ingest, not
  a rewrite of rust-analyzer.
- **Deterministic.** Zero LLM in this path (CGE doctrine).

This is 5–10% of Joern:

| Joern | swe_graph v1 |
|-------|----------------|
| Full AST-as-graph | File + symbol nodes, not every expression |
| CFG / PDG | Optional later; not v1 |
| Interprocedural taint | Out of scope v1 |
| 12 compiler frontends | tree-sitter pack + SCIP |
| Scala DSL | 6–8 JSON ops |
| 20 GB kernel graphs | Repo-scale SQLite, eviction |

It is ~90% of what a SWE agent uses: navigate, impact-set, “don’t miss the
other file.”

---

## 4. If We Ever Wrap Joern Anyway

Allowed as **opt-in security-review profile**, same as Semgrep:

1. Ship/require `joern-cli` (or Docker) — do not vendor the Scala tree.
2. `joern-parse` on a snapshot worktree; cache `cpg.bin` by commit hash.
3. Run **named queries from disk** (`querydb` or our YAML pack). Map each name
   to a Tool schema. The model picks `query_id`, never writes Scala.
4. Hard timeouts, `-Xmx` cap, fail loud if JDK missing.
5. Never enable `joern --server` for the agent. If we need a long-lived
   process, we still only send allowlisted scripts we authored.

This is a **Semgrep-shaped** integration, not a “Joern REPL tool.”

---

## 5. Assumptions to NOT Port

### ❌ “Rewrite it in Rust and we get Joern without the JVM”

You get weggli-plus-graph, or you wrap the same native parsers. Pick one and
name it honestly.

### ❌ “The agent can write CPGQL/Scala”

Unsandboxed interpreter. Closed ops only.

### ❌ “We’ll run full CPG before every turn”

Minutes + tens of GB. Analyzer must be incremental and cheap, or async
precompute (the AST_SERVICE doc’s Cloud Run story) — not a blocking sidecar
on the TUI path.

### ❌ “Joern replaces tree-sitter, SCIP, ast-grep, and Semgrep”

Those four already partition the space (syntax, precise refs, structural
pattern, SAST). Joern overlaps all of them badly for agent UX and wins only
at taint.

### ❌ “atom/chen is the shortcut”

Still JVM, still a Joern-shaped fork. Don’t inherit a second analysis OS.

---

## 6. What to Actually Take

| From Joern | To swe-term | How |
|------------|-------------|-----|
| CPG as *vocabulary* (METHOD/CALL/REACHING_DEF) | Analyzer result types | JSON schema, not Scala classes |
| Overlays / passes | Pipeline stages in `swe_graph` | AST tags → edges → optional SCIP |
| Fuzzy parse without build | tree-sitter default | SCIP optional upgrade |
| Library semantics as data | Later taint pack | YAML, not hardcoded |
| Packaged queries | Semgrep + optional Joern query ids | Model never authors the engine language |
| Export IR for later | SQLite + optional GraphSON | Content-addressed artifacts |
| `joern-parse` CLI split from REPL | Sidecar has no REPL | `swe_graph` is parse+query, not a shell |

---

## 7. Architectural Contrasts

```
Joern                                      swe-term Analyzer path
────────────────────────────────────       ─────────────────────────────────
Scala 3 / JDK 21 / fat zip                 Go core + Rust sidecar binary
12 compiler-wrapped frontends              tree-sitter + SCIP ingest
flatgraph, tens of GB in RAM               SQLite, per-file hashes
Scala DSL + unsandboxed --server           Closed JSON ops + Approver
Batch importCode (minutes)                 Incremental Merkle index
Taint + querydb (vuln workbench)           Optional later / wrap Joern
REPL as the product                        Headless CLI as the product
```

---

## 8. Placement in the existing plan

`GOLANG_TUI_PLAN.md` already lists `analyzers/treesitter`, `zoekt`, `astgrep`,
`semgrep`. Joern is **not a missing fifth of the same kind**. It is either:

- the **unifying graph store** behind those (that is `swe_graph`), or
- a **heavy optional SAST backend** next to Semgrep.

Do not add `analyzers/joern` as a Phase 5 peer of tree-sitter. Build
`swe_graph` first. Revisit a Joern wrapper only if security-review dogfood
shows Semgrep + SCIP cannot answer reachability.

---

## 9. Final Assessment

Joern is an excellent **research and vuln-hunting workbench** and a poor
**always-on coding-agent brain**. The CPG idea is the part that belongs in
swe-term. The JVM platform, Scala query language, and compiler-frontend zoo
are the part that does not.

A Rust “rebirth” is effective **if and only if** we shrink the problem to
what agents query every hour (symbols, edges, blast radius) and implement
that as a sidecar. Calling that “rewriting Joern” would be a lie that sets
the wrong bar and the wrong timeline.

**Ship `swe_graph`. Do not ship Joern-in-Rust. Maybe wrap Joern later, canned
queries only.**
