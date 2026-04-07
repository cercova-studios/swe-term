# TODO: swe_distiller Roadmap

## 1) Multi-Content Pipeline (Next)

- [ ] Add explicit acquisition contracts:
  - `InputKind` (`Html`, `Pdf`, `Image`, `Other`)
  - `RawDocument` (bytes + content-type + URL)
  - `PreprocessedDocument` (normalized HTML + hints)
- [ ] Split fetch into:
  - `fetch_bytes(...)`
  - `classify_input_kind(...)`
- [ ] Route by kind before parse:
  - HTML -> current extraction path
  - PDF/Image -> preprocessing adapter -> shared extraction path

## 2) Browser Fallback Hardening

- [ ] Add integration coverage for browser fallback with `--features browser`.
- [ ] Add better challenge detection markers and retries in browser mode.
- [ ] Add doc note for expected local browser binary prerequisites.

## 3) LLM Pipeline Maturity

- [ ] Add provider-level metrics in debug logs (latency, rejection reason).
- [ ] Add explicit provider error categories (timeout, bad response, safety rejection).
- [ ] Add test coverage for provider ordering and fallback behavior.

## 4) Extractor Coverage Expansion

- [ ] Improve extractor accuracy for:
  - Reddit thread/comment prioritization
  - X/Twitter thread stitching
  - YouTube transcript/description handling
  - Hacker News comment tree shaping
  - Chat transcript turn grouping
- [ ] Add fixture tests for each extractor target family.

## 5) Markdown Conversion Quality

- [ ] Improve nested list indentation in AST converter.
- [ ] Add stronger code language inference from class/attrs.
- [ ] Add richer table handling (alignment, uneven row widths).
- [ ] Add regression tests for malformed HTML edge cases.

## 6) Observability + Hygiene

- [ ] Add a stable event schema for all debug logs.
- [ ] Add optional summary diagnostics output for automated agents.
- [ ] Add `make lint`/`make test` style helper script for local developer ergonomics.

## 7) Packaging and UX

- [ ] Add release profile guidance and binary size checks.
- [ ] Add shell completion generation for CLI.
- [ ] Add explicit exit code mapping for common failure classes.

