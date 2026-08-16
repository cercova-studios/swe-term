# swe_distiller Architecture

## Purpose

`swe_distiller` is a Rust-first extraction pipeline and CLI that turns a URL into clean article content, typically Markdown.

Primary contract:

```bash
swe_distiller <url> -o webpage.md
```

## Design Principles

- Keep the default path deterministic and fast.
- Represent HTML as a DOM for structural cleanup; avoid nested regex over tags.
- One markdown converter (`markdown_ast`) plus markdown post-processing helpers.
- Isolate site-specific behavior in extractors.
- Keep acquisition concerns separate from extraction concerns.
- Prefer simple module seams over heavy abstractions.

## High-Level Flow

```text
URL
  -> fetch (HTTP, size-capped stream decode, optional Medium RSS / browser fallback)
  -> Html::parse_document once
  -> site extractor OR readability/content find
  -> DOM cleanup (selectors, hidden, images, scoring, chrome patterns)
  -> metadata extraction
  -> markdown_ast conversion + chrome postprocess
  -> optional LLM override (--llm), gated against heuristic markdown
  -> output file (markdown/json)
```

## Module Layout

```text
src/
├── main.rs                 CLI
├── lib.rs                  Pipeline orchestration
├── types.rs                Options/response types
├── fetch.rs                URL acquisition and fallbacks
├── dom_ops.rs              Fragment parse/serialize + node detach helpers
├── extractors/             Site-specific extractors + registry
├── find_content.rs         Main content candidate discovery
├── removal/                DOM-based cleanup passes
├── standardize/            Footnotes/callouts/content normalization
├── metadata.rs             Title/author/date/site/meta extraction
├── markdown_ast.rs         DOM-walking markdown conversion
├── markdown.rs             Markdown post-processing helpers only
├── extraction/
│   ├── llm.rs              --llm provider orchestration
│   └── checks.rs           LLM safety/quality gates
└── observability.rs        Structured debug logging
```

## Pipeline Contracts

### Acquisition Contract

- `fetch::fetch_page(...) -> Result<String>`
- Returns decoded HTML only.
- Bodies are streamed with a hard byte cap (`MAX_SIZE`).
- Medium `Referer` is applied only for Medium hosts.
- LLM provider calls are not part of fetch.
- Browser fallback exists only when built with `--features browser`.

### Extractor Contract

- `fn extract(&Html, &str) -> Option<ExtractorResult>`
- Returns `content_html` plus optional metadata overrides.
- Registry owns URL-pattern dispatch.

### Cleanup Contract

- Removal stages operate on `scraper` DOM fragments via `dom_ops`.
- Serialize once per stage (or after detach).
- Nested HTML-eating regex is intentionally avoided for structural pruning.

### Output Contract

- `DistillerResponse` contains content, markdown, metadata, parse timings, and meta tags.
- `ParseMode` controls markdown/json output behavior.

## Runtime Modes

### Default Mode

- Heuristic extraction pipeline only.
- Fast and deterministic.

### `--llm` Mode

- Runs after the heuristic pass.
- Safety checks compare candidates to heuristic markdown (not raw page markdown).
- Falls back to heuristic output automatically.

## Recovery

- At most one relaxed retry when word count is below threshold.
- Retry disables partial selectors, content-pattern pruning, and low-score pruning together.

## Observability

With `--debug`, JSON logs are emitted to stderr.

Current tracked events include:
- pipeline start/done
- fetch attempts and outcomes
- per-removal step deltas (word/char counts)
- LLM pipeline outcome
