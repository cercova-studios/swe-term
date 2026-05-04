# Design: Standalone `books` Fuzzy Keyword Search Engine for ~/polymathy

**Date**: 2026-05-03  
**Status**: Approved (Sections 1–3)  
**Owner**: User + AI architect (following swe-term AGENTS.md principles)  
**Related**: swe-term/docs/CLI_TOOLS.md, GOLANG_TUI_PLAN.md, PLAN.md (lightweight Go CLI tools pattern)

## 1. Problem Restatement + Assumptions

Build a simple, standalone terminal command `books` (invokable directly in PATH) that provides fast fuzzy + keyword search over the personal book library stored in `~/polymathy` (symlink to external drive with topic-organized directories containing EPUB, PDF, MOBI, AZW, and other formats).

**Must-have (v1)**: Keyword search on filenames and full paths, with fuzzy matching for typos/partial words. Ranked results.  
**Nice-to-have (later)**: Semantic search via embeddings.

**Key assumptions**:
- Standalone binary, completely separate from the swe-term agent binary.
- Start simple (YAGNI): filename/path-based only. No metadata extraction, no full-text indexing, no persistent index in v1.
- Library size: personal scale (hundreds to low thousands of files) — full scan on every invocation is acceptable (<500ms target).
- Go language (aligns with swe-term orchestration/CLI preference).
- Minimal dependencies (prefer stdlib; at most one small MIT fuzzy lib if needed).
- Linux primary (xdg-open), with macOS `open` fallback.
- Output to stdout for piping/scripting; optional explicit open action.
- Idempotent, observable, safe (read-only on library).

**Success criteria**: User can type `books "deep learning"` and immediately see highly relevant books ranked at the top, with the ability to open the best match via flag or alias.

## 2. Tradeoff Analysis & Chosen Approach

**Chosen**: Approach 1 — Pure-Go minimal scanner (scan on demand + stdlib + optional small fuzzy scorer).

**Why rejected**:
- Approach 2 (fzf wrapper): Introduces external runtime dependency and makes semantic evolution harder.
- Approach 3 (indexed from day 1): Violates "start simple", adds unnecessary complexity/storage/edge cases for a personal tool.

**Key tradeoffs accepted**:
- Rescan every run vs. index: simplicity and zero state wins for v1.
- Explicit `--open` flag vs. default open: safety and predictability > one-keystroke convenience.
- Filename-only vs. metadata/full-text: massive reduction in scope and failure modes; easy to layer later.
- Stdlib `flag` + `tabwriter` vs. heavier CLI/table libs: zero deps, fast compile, readable.

**Blast radius**: Extremely low. New binary, read-only access to `~/polymathy`, no writes except optional temp files. No impact on swe-term or other tools.

**Operational simplicity**: Single binary, `go build` or `go install`, works at 3am with no config or services.

## 3. Architecture & Components (v1)

Single small Go program (one package or flat `main.go` for ultimate simplicity).

**Components**:
- **CLI layer** (`main.go`): flag parsing, query normalization, dispatch.
- **Scanner** (`scan.go`): `filepath.WalkDir` over root, collect regular files, compute scores.
- **Scorer** (`score.go`): `score(path, query) float64` — keyword `strings.Contains` + word overlap + optional Levenshtein/fuzzy.
- **Renderer** (`render.go`): `text/tabwriter` for table, or JSON/TSV marshaling.
- **Opener** (`open.go`): best-effort `exec.Command` for `xdg-open`, `open`, `ebook-viewer`.

**Data flow**:
```
books "query" [flags]
  → parse + validate
  → expand ~ + WalkDir(root)
  → for each file: score() → filter(score > 0)
  → sort by score desc + depth/recency bonus
  → take top N
  → if --open: exec opener on top result
  → else: render table/TSV/JSON to stdout
```

All errors are visible (stderr + non-zero on hard failures), graceful degradation on permission skips.

## 4. CLI Surface & Flags (Locked)

See Section 2 for full flag table and examples.  
Default behavior: ranked list.  
`-o` / `--open`: open top result (best-effort).  
Recommended user alias: `alias book='books -o'`.

**Output formats**:
- `table` (default, colored if tty)
- `tsv`
- `json` (array of objects with score, title, path)

## 5. Scoring Algorithm (v1)

Composite score (0–1 range):
- Keyword exact substring match on basename or full path: high base.
- Multi-word query: bonus for covering more terms.
- Fuzzy component (if enabled): edit-distance or `sahilm/fuzzy` match quality (0.6–0.95).
- Path depth bonus: slightly prefer shallower (more "top-level") matches.
- Optional recency: small bonus for recently modified files.

Pure stdlib implementation first; add `github.com/sahilm/fuzzy` only if real-world testing shows weak ranking.

## 6. Error Handling, Robustness & Observability

- Permission errors on subdirs: log to stderr (or silent with `--quiet`), continue.
- No matches: "No matches for 'query'" + exit 0 (scripting friendly).
- Invalid root or flags: clear usage + exit 2.
- Opener failure: fallback message + still print the path.
- Large output: limit enforced; no memory bloat (stream-friendly).
- UTF-8 filenames: handled correctly via Go paths.
- Timeouts: none in v1 (scan is fast); can add `context` later if needed.

Failures are visible and cheap to debug (just run with a bad query or permission issue).

## 7. Implementation Sketch & Package Layout

See Section 3 sketch.  
Target: <150 LOC core, <5MB binary, <1s build.

**Recommended layout** (flat for v1 simplicity):
```
books/
├── go.mod
├── main.go     # entry + flags + main flow
├── scan.go
├── score.go
├── render.go
├── open.go
└── README.md
```

Can start as a single `main.go` and split only when it grows.

**Build/Install**:
```bash
go build -o books .
# or
go install .@latest
```

## 8. Verification & Testing

- Unit tests for scorer (table-driven: exact, fuzzy, multi-word, path cases).
- Integration: temp dir tree + `go test` that execs the binary.
- Manual: run against real `~/polymathy`, verify ranking feels intuitive.
- Performance: `time books "common query"` < 500ms on target hardware.
- Cross-platform: Linux + macOS opener paths.
- Piping: `books "q" --format json | jq '.[0].path'`

## 9. Evolution Path (Start Simple → Semantic)

- **v1 (this design)**: Filename/path keyword + fuzzy, explicit open.
- **v2**: Optional persistent SQLite index (on mtime change detection) for speed.
- **v3**: Semantic search — embed file titles/paths using the Xenova/all-MiniLM-L6-v2 model already cached in the workspace (`.fastembed_cache`), store vectors (LanceDB or in-memory), add `--semantic` flag. Reuses patterns from osgrep.
- **v4**: Metadata (title/author from PDF/EPUB) + optional full-text snippets.
- **v5+**: Interactive TUI picker (Bubble Tea), `books open` subcommand, config file, fzf integration.

This follows the project's "rule of three" and evolution model exactly.

## 10. Why This Design Wins

It delivers immediate value with minimal complexity, stays aligned with swe-term's engineering principles (simplicity, observability, portability, no premature abstraction), and has a clear, low-risk path to the semantic capability the user mentioned.

**Full design approved by user on 2026-05-03.**

---

Next step (per brainstorming process): Write this doc (done), then invoke writing-plans skill for the detailed implementation plan.