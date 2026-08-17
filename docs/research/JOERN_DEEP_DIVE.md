# Joern Deep Dive — Architecture & Feature Analysis

> Analysis of [`joernio/joern`](https://github.com/joernio/joern) — “The Bug Hunter's
> Workbench.” A Scala/JVM platform that turns source, bytecode, and binaries into a
> **Code Property Graph (CPG)** and mines it with a Scala DSL. License: Apache-2.0.
> ~3.4k GitHub stars. Spec: [cpg.joern.io](https://cpg.joern.io). Docs:
> [docs.joern.io](https://docs.joern.io). Commercial lineage: ShiftLeft → Qwiet AI
> (Ocular).
>
> **Validated against a local checkout:** `swe-term/joern`, commit `84ac957ce`
> (2026-08-13), tag **v4.0.604**, Scala **3.8.3**, `codepropertygraph` **1.7.70**
> (flatgraph arrives transitively through it — neither `flatgraph` nor the CPG
> domain classes live in this repo). Remaining external claims are triangulated
> from official docs, the README, DeepWiki, GitHub issue #5865, and the 4.0.0
> changelog. Docs.joern.io still shows JDK 19 / Joern 2.0.x in places; **the
> clone is treated as current** where they conflict.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [What a CPG Actually Is](#what-a-cpg-actually-is)
3. [Frontends: Wrappers, Not a Unified Parser](#frontends-wrappers-not-a-unified-parser)
4. [Building a CPG](#building-a-cpg)
5. [Graph Store: OverflowDB → FlatGraph](#graph-store-overflowdb--flatgraph)
6. [Query Language and Layers](#query-language-and-layers)
7. [Dataflow / Taint](#dataflow--taint)
8. [CLIs and Server](#clis-and-server)
9. [Incremental Analysis (Mostly Missing)](#incremental-analysis-mostly-missing)
10. [Scale and Cost](#scale-and-cost)
11. [Adjacent Projects](#adjacent-projects)
12. [Key Design Patterns](#key-design-patterns)
13. [Why It Matters for swe-term](#why-it-matters-for-swe-term)
14. [Rust Rewrite Feasibility as an swe-term Extension](#rust-rewrite-feasibility-as-an-swe-term-extension)
15. [Summary Statistics](#summary-statistics)

---

## System Overview

Joern is **not** a coding-agent tool. It is a static-analysis *workbench* whose
unit of work is “load a program into a graph, then write Scala to find patterns.”
The original research target was vulnerability discovery in C (Linux kernel). The
product later became a **language-agnostic IR** so the same queries can run on
Java, JS, Python, Go, binaries, etc.

The stack, top to bottom:

```
joern REPL / joern-scan / joern-slice / HTTP --server
        │
semanticcpg  (typed traversal DSL)
querydb      (packaged vuln queries)
dataflowengineoss  (reaching-defs + taint)
        │
overlay passes (CFG, PDG, call graph, types, …)
        │
flatgraph store  (columnar in-memory; cpg.bin / cpg.fg)
        │
x2cpg + language frontends  (c2cpg, javasrc2cpg, …)
        │
foreign parsers (Eclipse CDT, JavaParser, Soot, Ghidra, Roslyn, …)
```

That last layer is the load-bearing fact for any rewrite discussion: Joern’s
language coverage is **borrowed compiler/parser stacks**, not a Joern-owned
grammar family.

---

## What a CPG Actually Is

Yamaguchi et al., *Modeling and Discovering Vulnerabilities with Code Property
Graphs*, IEEE S&P 2014: merge three classic IRs into one attributed multigraph.

| Layer | Answers |
|-------|---------|
| **AST** | What is the syntax? methods, locals, calls, literals |
| **CFG** | What can execute after what? |
| **PDG** (CDG + DDG) | What does this value depend on? what controls this statement? |

A CPG is a **directed, edge-labeled, attributed multigraph**. Nodes have types
(`METHOD`, `LOCAL`, `CALL`, `CONTROL_STRUCTURE`, …) and key-value properties.
Edges have labels (`AST`, `CFG`, `CONTAINS`, `REACHING_DEF`, `ARGUMENT`, …).
Multiple edges may exist between the same pair.

Later ShiftLeft/Joern extensions:

- **Overlays** — extra layers of abstraction (HTTP endpoints, findings) queried
  with the same DSL.
- **Interprocedural** analysis and library **semantics** (how `foo(a,b)` taints).
- A **generic statement/expression container** so many frontends emit one schema
  ([codepropertygraph](https://github.com/ShiftLeftSecurity/codepropertygraph)).

This is a different object from tree-sitter (syntax only) and from SCIP (precise
symbol/index). A CPG is **syntax + control + data dependence in one store**.
That is Joern’s actual invention. Everything else is packaging.

---

## Frontends: Wrappers, Not a Unified Parser

The clone at `joern-cli/frontends/` holds **14 language frontends plus the
shared `x2cpg` kit** (main-source Scala LOC from the checkout):

| Frontend | Language | Built with | Maturity (docs) | Main LOC |
|----------|----------|------------|-----------------|----------|
| `c2cpg` | C/C++ | Eclipse CDT | Very High | 6.5k |
| `javasrc2cpg` | Java (source) | JavaParser | Very High | 9.8k |
| `jssrc2cpg` | JavaScript/TS | astgen (JSON AST) | High | 6.3k |
| `pysrc2cpg` | Python | JavaCC | High | 7.9k |
| `ghidra2cpg` | x86/x64 | Ghidra | High | 2.4k |
| `jimple2cpg` | JVM bytecode | Soot | Medium | 2.1k |
| `kotlin2cpg` | Kotlin | IntelliJ PSI | Medium | 6.5k |
| `php2cpg` | PHP | PHP-Parser | Medium | 6.2k |
| `gosrc2cpg` | Go | go.parser | Medium | 3.2k |
| `swiftsrc2cpg` | Swift | SwiftSyntax / SwiftAstGen | Medium | 16.3k |
| `rubysrc2cpg` | Ruby | ANTLR | Medium-Low | 8.2k |
| `csharpsrc2cpg` | C# | Roslyn | Medium-Low | 4.4k |
| `rust2cpg` | Rust | prebuilt `rust_ast_gen` native binary | new | 2.7k |
| `abap2cpg` | ABAP | JSON AST ingestion | new | 1.8k |
| `x2cpg` | (shared kit) | — | — | 15.5k |

`rust2cpg` and `abap2cpg` are newer than the docs.joern.io table. `x2cpg` is
the shared frontend kit; it also contains `FrontendHTTPServer`, letting a
frontend stay **resident as an HTTP service** instead of paying JVM startup per
parse (exercised by the frontends' tests). Each `*2cpg` is otherwise a
**standalone executable** invoked by `joern-parse` / `importCode`.

Parsers are **not tree-sitter** — a grep across the clone's frontends finds no
tree-sitter usage at all. The newer frontends (`jssrc2cpg`, `swiftsrc2cpg`,
`rust2cpg`) instead use the **astgen pattern**: an external native binary emits
a JSON AST, and the Scala side only maps JSON → CPG. Fuzzy parsing is a
first-class goal: import code **without a working build** and with missing
headers.

Implication: a Rust rewrite that “just uses tree-sitter” is **not Joern**. It is
a different, weaker IR. A rewrite that keeps fidelity must keep CDT/Soot/Ghidra
or reimplement them.

---

## Building a CPG

Typical paths:

1. **REPL:** `importCode("/path")` guesses language, creates a workspace
   project, spawns the frontend **in a second JVM**, writes `cpg.bin`, opens it,
   runs default overlays.
2. **CLI:** `joern-parse /src --language JAVASRC -o cpg.bin` then
   `joern-export --repr pdg` / `joern-scan`.
3. **Manual frontend:** `./c2cpg.sh -J-Xmx30G -o linux.odb /path/to/linux` then
   `importCpg(...)`. Docs recommend this for large trees because `importCode`
   doubles memory.

Overlays (CFG, types, call linker, dataflow) run **after** the AST CPG exists.
`run.ossdataflow` is required before PDG-ish dumps.

Serialization: domain classes from `codepropertygraph` (historically protobuf
schema + generated accessors). On-disk `cpg.bin` / flatgraph `cpg.fg`. Export
also to DOT, GraphML, GraphSON, Neo4j CSV.

---

## Graph Store: OverflowDB → FlatGraph

Older Joern used general-purpose graph DBs + Gremlin, then **OverflowDB**
(off-heap, overflow-to-disk). As of **Joern 4.0.x**, OverflowDB is replaced by
[`joernio/flatgraph`](https://github.com/joernio/flatgraph) (columnar arrays).

Documented reasons (changelog `4.0.0-flatgraph.md`):

- ~40% less memory, faster traversals (~40% on default passes / import).
- Overflow-to-disk **was not reimplemented** — too slow to be useful.
- Edges: **at most one property** (schema never needed more).
- Linux 4.1.16 (workstation, rough): import 18 min → 11 min; heap after import
  33 GB → 20 GB; min `-Xmx` 80 GB → 30 GB.

Linux 4.1.16 CPG size from install docs (older Joern): **Graph [47,542,978 nodes]**.
That is the scale Joern considers “large but in-scope.” It is not an agent
sidecar’s default working set.

---

## Query Language and Layers

Queries are **Scala**, not Cypher/Gremlin (Gremlin was removed; OverflowDB
Traversal is a Scala collection with graph steps).

```scala
def source = cpg.call("source")
def sink   = cpg.call("sink")
sink.reachableBy(source)
sink.reachableByFlows(source)
```

Modules:

| Module | Role | Scala LOC (clone, incl. tests) |
|--------|------|-------------------------------|
| `semanticcpg` | High-level typed traversal API (`cpg.method`, `cpg.call`, …) | ~10.0k |
| `dataflowengineoss` | Reaching definitions, taint `Engine`, library semantics | ~5.2k |
| `querydb` | Packaged `@q` vuln queries; `joern-scan --list-query-names` | ~6.5k |
| `console` | REPL, `importCode`, workspace | ~4.7k |
| `macros` | DSL construction | ~0.3k |

The striking split: the **analysis core is small** (~27k lines across these
modules) while `joern-cli` — dominated by the frontends — is ~243k. And the two
hardest parts, the **CPG schema/domain classes** (`codepropertygraph`) and the
**flatgraph store**, live in *separate repositories* entirely.

The agent-relevant problem: the public remote API is this same interpreter
(see [CLIs and Server](#clis-and-server)). There is no first-class JSON query
algebra. You send Scala source.

---

## Dataflow / Taint

`ReachingDefPass` solves reaching-definitions **per method**, emits
`REACHING_DEF` edges (optional `Variable` property). The taint `Engine` walks
DDG **backwards** from sinks, in parallel (`ExecutorService`), applying
`Semantics` for external calls.

Semantics language (rudimentary): method full name + argument index pairs
(`1 -> -1` means arg1 taints return; `0` is receiver). Regex method names
supported. `PASSTHROUGH` for “don’t kill, don’t cross-taint.” Missing mappings
are assumed killed.

Hard limit: if a method exceeds `maxNumberOfDefinitions` (default **4000** —
confirmed in the clone at
`dataflowengineoss/.../passes/reachingdef/ReachingDefPass.scala:14` and
`.../layers/dataflows/OssDataFlow.scala:15`), the pass **bails** with a warning
and leaves that method without reaching defs. That is a
correctness cliff on generated/bundled code (the exact case atom/chen tried
to fix with a different engine).

This is Joern’s unique value versus tree-sitter/SCIP: **“can untrusted X reach
dangerous Y?”** as a graph walk, not a guess.

---

## CLIs and Server

| Binary | Job |
|--------|-----|
| `joern` | Interactive Scala REPL |
| `joern-parse` | Frontend → `cpg.bin` |
| `joern-export` | AST/CFG/PDG/CPG14/ALL → DOT/GraphML/… |
| `joern-scan` | Run `querydb` |
| `joern-slice` | Program slice / backwards dataflow subset |
| `joern --server` | HTTP interpreter on :8080 |

Server routes ([docs.joern.io/server](https://docs.joern.io/server/)):

- `POST /query` `{query: scala}` → `{uuid}`
- `GET /result/$uuid` → stdout/stderr
- `POST /query-sync` (used in the curl helper)
- `ws://host:8080/connect` completion notifications

The docs’ own example is:

```
joern-remote 'println("remote execution vector - this prints on the server")'
```

And the warning:

> the server exclusively implements remote access to an interpreter, **it does
> not implement sandboxing**.

For a coding agent, that API is a **remote code execution primitive**, not a
query protocol.

Docker: `ghcr.io/joernio/joern:master`. Releases are fat JVM zips per
GOOS/GOARCH. Daily automated releases.

---

## Incremental Analysis (Mostly Missing)

Issue [#5865](https://github.com/joernio/joern/issues/5865) (javasrc2cpg):
re-index after a file change currently requires full `importCode`. Reporter:
~**10 minutes** on a large Java project. Prototype `IncrementalCpgUpdater`:
delete file subtree, re-parse those files, re-run post-passes — **~350 ms/file**
(~1700×). Not documented as shipped product behavior.

swe-term’s Analyzer story (content-addressable per-file hash, Merkle diffs)
assumes incrementality. Joern as it ships is a **batch indexer**.

---

## Scale and Cost

| Cost | Reality |
|------|---------|
| Runtime | JDK 21 + optional gcc/g++ for C headers |
| Memory | Tens of GB for kernel-scale; `importCode` **spawns a second JVM** |
| Startup | Fat CLI; REPL + frontend process |
| Query UX | Expert Scala; not an LLM tool schema |
| Language quality | Very High only for C/Java; Go/Ruby/C# medium or worse |
| Contributor history | Decade-scale (fabsx00, ShiftLeft/Joern org, 10+ regulars) |

Joern is cheap **if you already live in a JVM analysis lab**. It is expensive
**as a always-on agent sidecar**.

---

## Adjacent Projects

| Project | Relation | Lesson |
|---------|----------|--------|
| [codepropertygraph](https://github.com/ShiftLeftSecurity/codepropertygraph) | Schema + generated bindings | The IR is specified; you can *speak* CPG without *being* Joern |
| [flatgraph](https://github.com/joernio/flatgraph) | Store rewrite already happened (in Scala) | They already paid the “faster graph in same language” tax |
| [AppThreat/chen](https://github.com/appthreat/chen) + atom | Joern fork, still Scala/JVM, CPG 1.0 | Forking Joern does not escape the JVM |
| [cpg-rs](https://github.com/gbrigandi/cpg-rs) | Rust serde types for CPG JSON | Schema port ≠ engine port |
| [weggli](https://github.com/weggli-rs/weggli) | Rust tree-sitter pattern search, C/C++ | The *right-sized* Joern-inspired Rust tool: AST patterns, no CPG |
| [Fraunhofer CPG](https://github.com/Fraunhofer-AISEC/cpg) | Independent Java CPG + LLVM-IR | Second JVM CPG; still not Rust |
| SCIP / tree-sitter / ast-grep / Semgrep | Agent-native stack already in PLAN | Cover most SWE questions cheaper |

---

## Key Design Patterns

1. **One IR, many languages** — query once, frontend per language.
2. **Fuzzy parse** — analysis without a build is a feature.
3. **Overlays** — add CFG/PDG/types as passes, not a second database.
4. **Semantics files** — library taint as data, not hardcoded per API.
5. **Packaged queries** (`querydb`) — vuln knowledge as a plugin, not the core.
6. **Export the IR** — Joern is also an extraction tool for ML papers (Devign, …).

---

## Why It Matters for swe-term

swe-term already wants an `Analyzer` that injects structure *before* the LLM
(`ARCHITECTURE.md`, `GOLANG_TUI_PLAN.md` Phase 5: tree-sitter, zoekt, ast-grep,
semgrep). Joern is the **upper bound** of that idea: full CPG + taint.

The question is not “is CPG a good idea?” (yes). It is “is *Joern* the
implementation swe-term should own in Rust?” That is the critique.

---

## Rust Rewrite Feasibility as an swe-term Extension

Grounded in the local clone, with swe-term’s extension model as the target
(per `docs/core/ARCHITECTURE.md`: Go core; extensions are interface-driven
adapters — `swe_distiller` is the precedent: a standalone Rust CLI binary the
core shells out to, JSON/markdown over stdio, no linkage into the core).

**Integration is the easy part.** A Rust analyzer binary in `extensions/`
speaking JSON over stdio slots into the “Analyzer-like enrichment interfaces”
contract exactly like `swe_distiller` does. Nothing in swe-term’s architecture
blocks it; the compiled-extension lane exists for precisely this.

**The rewrite itself is the hard part.** What the clone shows you would have
to port for a *faithful* rewrite:

| Layer | Size / location | Rust reality |
|-------|-----------------|--------------|
| Analysis core (`semanticcpg` + `dataflowengineoss` + `console` + `macros`) | ~27k Scala LOC, this repo | Portable — small, well-factored; reaching-defs + taint engine is a few kLOC |
| CPG schema + domain classes | `codepropertygraph` (separate repo, codegen-heavy) | [`cpg-rs`](https://github.com/gbrigandi/cpg-rs) covers serde types only; no traversal layer |
| Graph store | `flatgraph` (separate repo) | Columnar arrays port naturally to Rust; but this is a rewrite of a rewrite they just finished |
| Frontends | ~97k main-LOC Scala across 14 languages, **plus** the borrowed parsers underneath (CDT, JavaParser, Soot, Ghidra, Roslyn, IntelliJ PSI, GraalVM…) | **The killer.** These parser stacks are JVM/CLR-native with no Rust equivalents at comparable fidelity. This layer is why the maturity column exists |
| Scala query DSL | The entire UX | Would not be ported — an agent extension wants a JSON tool schema, which Joern itself lacks |

**Verdict: a full-fidelity Rust rewrite is infeasible** at extension scale —
multi-engineer-year, and most of the value (frontend maturity) lives in
borrowed JVM parser stacks that cannot be ported, only re-earned. This matches
`JOERN_CRITIQUE.md` §2/§5.

**Two feasible shapes exist**, both compatible with the extension lane:

1. **Wrap, don’t rewrite.** Run stock Joern as a sidecar (its own process/
   container), drive it via `joern-parse`/`joern-export`/`joern-slice` CLIs —
   never the unsandboxed `--server` — and put a thin Rust/Go adapter in
   `extensions/` that turns exported DOT/GraphML slices into agent-consumable
   JSON. Cost: JVM + memory footprint, batch-only indexing. Use only for
   on-demand taint questions, not per-turn enrichment.
2. **CPG-lite in Rust.** Own a deliberately smaller IR: tree-sitter AST + CFG +
   intraprocedural def-use for the 2–3 languages swe-term actually targets,
   emitting a `weggli`-class capability with a JSON query surface. This is
   *inspired by* Joern (the CPG idea, overlays-as-passes, semantics-as-data),
   not a port of it. Notably, Joern’s own newest frontends (`rust2cpg`,
   `swiftsrc2cpg`, `jssrc2cpg`) already externalize parsing to native astgen
   binaries emitting JSON AST — evidence that the “native parser process →
   JSON → graph builder” seam is the right cut line, and the one an swe-term
   extension should exploit.

Option 2 is the recommendation for the roadmap; option 1 is the escape hatch
when a real interprocedural taint question shows up.

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Language / runtime | Scala 3.8.3 / JDK 21 |
| Version validated | v4.0.604 (`84ac957ce`, 2026-08-13, local clone) |
| Stars (2026-08) | ~3,414 |
| License | Apache-2.0 |
| Frontends | 14 in-repo (`rust2cpg`, `abap2cpg` newest) + `x2cpg` kit |
| Analysis core size | ~27k Scala LOC; frontends ~97k main LOC (~243k incl. tests) |
| Schema & store | `codepropertygraph` 1.7.70 + flatgraph — **separate repos** |
| Graph store | flatgraph (v4+) |
| Linux 4.1.16 CPG | ~47.5M nodes; ~20 GB heap post-import (v4) |
| Query language | Scala DSL |
| Remote API | Unsandboxed Scala interpreter |
| Incremental index | Not a supported product feature |
| Agent-native JSON query algebra | None |
