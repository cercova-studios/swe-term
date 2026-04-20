# swe_distiller Architecture

## Purpose

`swe_distiller` is a Rust-first extraction pipeline and CLI that turns a URL into clean article content, typically Markdown.

Primary contract:

```bash
swe_distiller <url> -o webpage.md
```

## Design Principles

- Keep the default path deterministic and fast.
- Isolate site-specific behavior in extractors.
- Keep acquisition concerns separate from extraction concerns.
- Keep output conversion centralized and testable.
- Prefer simple module seams over heavy abstractions.

## High-Level Flow

```text
URL
  -> fetch layer (HTTP acquisition, retries, charset decode, optional browser fallback feature)
  -> parse + extractor selection
  -> content selection + cleanup/removal stages
  -> metadata extraction
  -> markdown conversion (AST-based)
  -> optional LLM extraction override (--llm) with safety checks and heuristic fallback
  -> output file (markdown/json)
```

## Module Layout

```text
src/
├── main.rs                 CLI
├── lib.rs                  Pipeline orchestration
├── types.rs                Options/response types
├── fetch.rs                URL acquisition and fallbacks
├── extractors/             Site-specific extractors + registry
├── find_content.rs         Main content candidate discovery
├── removal/                Cleanup/removal passes
├── standardize/            Structural normalization passes
├── metadata.rs             Title/author/date/site/meta extraction
├── markdown_ast.rs         DOM-walking markdown conversion
├── markdown.rs             Markdown post-processing helpers
├── extraction/
│   ├── llm.rs              --llm provider orchestration
│   └── checks.rs           LLM safety/quality gates
└── observability.rs        Structured debug logging
```

## Pipeline Contracts

### Acquisition Contract

- `fetch::fetch_page(...) -> Result<String>`
- Returns decoded HTML only.
- LLM provider calls are not part of fetch.
- Browser fallback exists only when built with `--features browser`.

### Extractor Contract

- `fn extract(&Html, &str) -> Option<ExtractorResult>`
- Returns `content_html` plus optional metadata overrides.
- Registry owns URL-pattern dispatch.

### Output Contract

- `DistillerResponse` contains content, markdown, metadata, parse timings, and meta tags.
- `ParseMode` controls markdown/json output behavior.

## Runtime Modes

### Default Mode

- Heuristic extraction pipeline only.
- Fast and deterministic.

### `--llm` Mode

- Tries configured providers in order.
- Applies safety checks before accepting provider output.
- Falls back to heuristic output automatically.

## Observability

With `--debug`, JSON logs are emitted to stderr.

Current tracked events include:
- pipeline start/done
- fetch attempts and outcomes
- extractor presence
- per-removal step deltas (word/char counts)
- LLM pipeline outcome

## Extensibility Notes

The current architecture is prepared for multi-content-type expansion:

- fetch/acquisition is isolated from extraction logic.
- extractor registry is modular.
- markdown conversion is centralized.
- LLM provider orchestration is isolated behind one module.

Natural next step is introducing an explicit `InputKind`/`RawDocument` preprocess stage for HTML/PDF/image routing without changing CLI behavior.
