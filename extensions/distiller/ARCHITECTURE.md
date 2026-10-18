# distiller Architecture

## Purpose

`distiller` turns URLs, local HTML files or stdin snapshots into clean article
content, with Markdown, JSON, EPUB3 and optional Kindle KFX outputs.
Born-digital local PDFs select matching versioned arXiv HTML when identified,
with a separate Rust/PDFium cloud reflow path for other PDFs.

Primary contract:

```bash
distiller <url> -o webpage.md
```

Plans and historical validation belong to the
[development archive](../../docs/README.md#designated-development-archive).
The [roadmap](../../docs/plans/distiller-roadmap.md) records development work;
this document owns current pipeline and operating contracts.

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
  -> output file (markdown/json) OR sandboxed Pandoc EPUB3 packaging
```

## Module Layout

```text
src/
├── main.rs                 CLI
├── kfx.rs                  CLI-only adapter to independent local EPUB-to-KFX encoder
├── lib.rs                  Pipeline orchestration
├── types.rs                Options/response types
├── epub.rs                 Bounded image embedding and atomic Pandoc EPUB3 export
├── cover.rs                Mandatory Muse artwork, local lettering and verified cover cache
├── pdf.rs                  Isolated native PDF worker, page reflow and figure extraction
├── pdf_reconstruct.rs      Source-linked OpenRouter reconstruction, validation and rendering
├── html_first.rs           Versioned arXiv/explicit HTML-PDF pairing and source-bound recovery
├── fetch.rs                URL acquisition and fallbacks
├── dom_ops.rs              Fragment parse/serialize + node detach helpers
├── extractors/             Site-specific extractors + registry
├── find_content.rs         Main content candidate discovery
├── removal/                DOM-based cleanup passes
├── standardize/            Footnotes/callouts
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
- Heuristic pruning protects code, math, tables, superscripts and subscripts.
  Exact chrome/security removal still runs.
- Sanitization and URL resolution edit DOM attributes, never literal code text.
  This is extraction hygiene, not a complete HTML rendering security boundary.

### Output Contract

- For EPUB/KFX, versioned arXiv HTML/abstract/PDF URLs acquire the matching HTML
  and bounded PDF. Local PDFs with a recognized versioned arXiv margin stamp
  select the same path; other PDFs retain cloud reconstruction. `--pdf-source`
  pairs other URL/local HTML with a local PDF, with caller-declared identity.
  The paired path requires one article and a heading corroborated by the PDF's
  first page. Versioned acquisition also checks the PDF stamp and HTML PDF link.
  The document title precedes pre-title notices to avoid a synthetic Pandoc title
  chapter; LaTeXML abstract headings become level two. LaTeXML expandable notes
  become native linked EPUB footnotes, preserving short bodies and collapsing
  marker-only tooltips. Note numbers follow Pandoc's reading order. Note-list IDs
  remain on their list items until the HTML reader resolves them.
  It retains the full article rather than heuristic prose pruning. NFKC-normalized
  visible prose witnesses (24 alphanumeric characters minimum, including text
  inside native regions) are compared without hidden TeX annotations or phantom
  math. Missing numbered artwork requires one caption destination and one
  source-bound graphics region above it with no intervening prose. Unsupported
  mixed panels are replaced together. Captions remain live HTML once. Isolated
  absent prose lines require a unique aligned same-column predecessor ending its HTML
  block; shared insertion boundaries, larger or interior gaps fail. Isolated
  small bottom-margin text is retained as an end note. Short spans and semantic visual/math
  equivalence remain explicitly unchecked in the hash-bound sidecar receipt.
  EPUB and receipt are staged before no-clobber publication. A receipt race can
  leave a retained EPUB with an explicit error. KFX copies its receipt beside the
  final output and records both hashes. Stdin never auto-fetches a PDF.

- `scripts/save2kindle.py` is standalone shell composition, not a library or Go
  adapter: HTTP(S)/HTML/PDF uses `distiller --mode kfx`, EPUB uses `epub-to-kfx`,
  and KFX goes directly to `kindle-send`. PDF URL paths ending in `.pdf` use a
  temporary bounded download; other URLs retain Distiller acquisition. Existing
  outputs are refused; failed delivery retains the local KFX for explicit retry.
  Conversion alone receives the cover credential, through the configured Doppler
  project when absent. Delivery success remains the independent sender's receipt.
- CLI `--mode kfx` selects the existing EPUB extraction/cover path internally,
  then invokes the independent `epub-to-kfx` executable on a temporary EPUB.
  Library ParseMode and extraction semantics stay unchanged. The adapter checks
  encoder prerequisites and destination availability before extraction or cover
  spending. KFX export is file-only, supports HTML/stdin/PDF through their existing
  EPUB constraints, and has a 370-second conversion deadline. The independent
  encoder uses Amazon Previewer plus KFX Output, publishes with a no-clobber hard
  link, and owns its bounded process deadlines and cleanup. Amazon may resize and
  recompress images. Distiller does not invoke bookmark discovery or SSH delivery.
- `DistillerResponse` contains content, markdown, metadata, parse timings, and meta tags.
- `ParseMode` controls Markdown/JSON/EPUB output behavior. EPUB preserves the
  structured DOM rather than replacing it with a longer plain-text schema body;
  it retains footnote backrefs and defers URL resolution until images are embedded.
- Local files share the bounded strict UTF-8 input path with stdin. `--base-url`
  is optional for files and required for stdin; URL positional input rejects it.
- EPUB consumes cleaned HTML directly, includes trusted typography in page headers,
  and repairs output structure before packaging. Truly empty layout tables are
  removed only without meaningful media, preserving their source anchors. Equation tables retain natural
  typography without table or cell borders, with presentational table roles that retain MathML.
  Authored roles remain. Head-group headers retain scope;
  body headers use explicit ARIA row/column roles because Pandoc may emit them as data cells.
  Existing header references remain. Group associations require separate verification;
  unmarked LaTeXML tables may infer an initial header band only from ruled, styled
  labels with consistent bold or bounded spanning cells. Numeric matrices and
  ambiguous bands retain source semantics. This inference is heuristic, not a
  semantic proof. Caption text retains a live ID for image/table descriptions;
  authored alternatives survive, placeholder alternatives gain panel labels.
  LaTeXML auxiliary headings use their enclosing section's rank. Caption fallback
  does not replace a full diagram description or establish accessibility conformance.
  The writer delegates EPUB3 packaging, metadata, heading navigation and chapter splitting
  to Pandoc on PATH (tested 3.12.1). Pandoc runs with `--sandbox`, warning failures
  and uses explicit extracted title metadata before a first-heading fallback.
  It has a 30-second timeout. Missing Pandoc fails without publishing output.
  Every EPUB requires a cover. `--cover` overrides generation with a decoded PNG/JPEG (4 MiB maximum,
  included in the 64 MiB total) and works through the same packager for PDFs.
  Pandoc 3.12.1 needs the owned cover copy supplied as both cover-image and
  reference-doc to preload that file in its sandbox; no directory access is granted.
  The EPUB writer ignores reference-doc otherwise. Distiller owns this workaround;
  remove it when Pandoc's cover option itself preloads correctly.
  Wrappers containing headings are flattened while retaining their IDs as inline
  anchors, because Pandoc excludes headings nested in Div blocks from its TOC.
  Heading wrappers with language/direction attributes fail explicitly until
  flattening can preserve their inherited semantics.
- EPUB embeds decoded PNG/JPEG/GIF bytes as data URLs (4 MiB each, 64 MiB total);
  local regular files must canonicalize beneath the input HTML's parent directory.
  HTTP(S) assets are streamed with a 10-second request timeout, five redirects and
  the same byte caps in URL/file modes. Stdin permits only inline data images.
  Decoding caps dimensions at 12000 per axis and allocations at 128 MiB; WebP
  becomes lossless PNG. Unsupported media fail explicitly. Dates are normalized
  to ISO dates; unparseable dates are omitted. Paragraph/list-item/figure/table IDs become inline anchors because Pandoc drops
  those container IDs. Atomic no-clobber publication protects existing books.
  Empty mailto targets lose their href while retaining their displayed text.
- HTML EPUB extraction/packaging is deterministic and file-only; PDF reconstruction and cover artwork use models.
  Muse (`muse-image-1.0`, fixed https://api.meta.ai/v1/responses) generates one
  text-free illustration on a cache miss, with a 90-second deadline, no redirects,
  an 8 MiB response cap and 4 MiB PNG cap. Only title, author and six bounded
  section labels are sent. A bundled SIL OFL Noto Sans Bold font typesets exact
  title/author on opaque panels in a 1024×1536 grayscale cover. Unsupported
  glyphs/shaping and unreadable overflow fail before dispatch; precomposed Latin,
  Greek and Cyrillic are supported, combining marks and complex scripts are not.
  The source content fingerprint includes the PDF's existing source SHA-256 ledger.
  Cache identity binds source, metadata, font, prompt/profile and Muse settings.
  PNG iTXt provenance travels inside the EPUB; a receipt hashes the final image.
  Cache image/receipt files and directory publication are synchronized before
  deleting the exclusive reservation; this requires filesystem directory-sync support.
  Completed cache survives packaging failure; no expiry or background eviction.
  Pending requests wait at most five seconds, then fail without retry. Recovery
  requires checking provider history before explicitly removing the named `.pending`
  file. Cache corruption is an error, never a hidden paid regeneration.
  `DISTILLER_COVER_CACHE` defaults to `~/.cache/distiller/covers`; runtime
  `MODEL_API_KEY` is needed only for a cache miss. No generation enable/disable flag.
  EPUB rejects extraction LLM/stdout modes. Multi-article
  books remain unsupported; EPUBCheck verifies packaging, not
  source fidelity or reading-system display.
- PDF uses PDFium 7881 from ~/.local/share/distiller/lib or the system library
  path, in a child Rust process with a 180-second extraction timeout. Input limits:
  200 MiB, 400 pages, 100000 characters/objects per page, Form depth 16 and
  32 million pixels per image. No Python production runtime or OCR is used.
  The worker retains all non-whitespace glyphs in stable text spans, including
  PDFium's line-end hyphen flag, font size, x/y position, right edge of retained
  glyphs and source-page identity. It also retains native glyph rectangles, bold/italic
  ranges, math-font evidence and connected graphic bounds. Font metadata is cached
  per document. Bookmark depth starts at one.
  It emits a JSON layout rather than guessing body/note regions or flattening
  columns. `pdf_reconstruct.rs` submits owned page pairs plus adjacent-page
  context and 1200×1800-bounded grayscale page previews (1 MiB JPEG each) to
  OpenRouter `deepseek/deepseek-v4.1-flash`, requesting strict JSON
  Schema and endpoints requiring parameter support. PDF text and page previews are sent to
  this model; covers still receive only bounded metadata. No local model download.
  Uncached source packets exceeding 256 KiB split into single owned pages.
  Neighboring context can shrink to fit; owned spans remain complete. A page
  that still exceeds the packet cap fails explicitly. The cap excludes prompt,
  schema, candidate and image data. Existing validated pair caches remain reusable.
  Output assigns span IDs and bounded join operations to paragraphs, headings,
  quotations, list items, notes, tables, captions and figures; it cannot replace prose.
  Tables and explicit captions preserve the prior body continuation target;
  headings and preformatted blocks remain barriers.
  Every owned span/image/graphic must occur exactly once. Arbitrary fields, unknown IDs,
  invalid hyphen removals, broken note targets, and unsupported furniture
  classifications fail validation. A digit-bearing segment immediately before
  a marked hyphen cannot request its removal; later alphabetic compound segments
  may still be dehyphenated. Furniture eligibility uses
  page-edge and repeated-position/page-number evidence, or a native-rotated arXiv
  identifier stamp wholly in the outer side margin; other rotated margin text remains
  protected. Semantic classification remains heuristic.
  A trailing hyphen can be repaired when PDFium marks it or the ordered spans
  cross a physical line/page; unmarked within-line literals remain protected.
  Source SHA-256 remains in a nonvisible HTML comment for cover identity.
  Notes move to a top-level end section, with backlinks when a matching numeric callout
  is explicitly associated. Page anchors are invisible; no generated page labels,
  raster captions, blank-page prose or conversion warnings enter the reading flow.
  Source bookmarks guide actual heading placement; unmatched bookmarks remain
  in the evidence layout and are not fabricated as body headings.
  Each page pair requires a draft followed by source review using the same
  original text/visual evidence. Review checks joins, superscripts, paragraph
  boundaries, continuations, contents and heading hierarchy. Up to 32 previous
  headings with source font sizes and the full bookmark outline provide context.
  Candidate hyphen joins receive up to three intact source-span witnesses per
  spelling. After reconstruction, an exact intact compound witness preserves its
  separator when the joined spelling is unattested. Related compound families
  only select questions for a focused, structured spelling review. That review
  may change existing hyphen joins, including continuations, but cannot change
  source IDs or text. Both-spelling ambiguity retains the reconstruction choice.
  Evidence-bound decision caches allow offline replay; a new unresolved review
  needs OpenRouter credentials even when page plans are cached. Invalid or
  incomplete decisions stop publication. Receipts retain applied decisions.
  Each focused packet contains at most 64 boundaries, with a 256 KiB evidence
  and response cap, 8192 output tokens and one request; failed packets are not
  retried automatically. Existing complete replies remain available for inspection.
  Both passes enforce identical ownership and rendering checks before acceptance.
  After validation, measured geometry restores spaces between baseline words and
  numerals, joins same-line run-in headings to prose unless bookmarked, and places
  explicitly linked raised callouts after a unique physical predecessor. Ambiguous
  placement remains unchanged. These repairs run identically on fresh and cached plans.
  An omitted numeric callout is restored only when an explicit note reference,
  its separate printed label and a unique physical body predecessor agree.
  Raw model replies remain unchanged in the evidence cache; repairs are logged.
  A misplaced empty horizontal rule from an earlier page returns between two
  horizontally overlapping source blocks that bracket it vertically. Same-page,
  mixed-page and ambiguous placements retain their continuation barriers.
  Measured blank gaps also clear an explicit continuation between adjacent prose
  blocks after a completed sentence. Floats, native passages and page boundaries
  remain outside that override.
  A typography heuristic closes em dashes between aligned adjacent source lines
  within a run, preserving explicit whitespace, preformatted content and ambiguous
  page/column boundaries. It assumes symmetric dash spacing and requires source inspection.
  Review receives the candidate as user data; malformed replies are never replayed.
  Draft retries retain only deserialized source-ID candidates for focused correction.
  An explicit rerun can resume a bounded, complete draft envelope as advisory data.
  The matching model and completion status are checked; typed candidates are
  revalidated against the current source and prior blocks before correction.
  A resumed draft still requires a model correction pass and source review.
  Draft and separate source-review passes start with low reasoning.
  Every rejected-response retry disables reasoning, with the same validation and
  attempt bounds. Native regions are prepared and rendered before model requests.
  Each opaque anchor owns its hidden glyph/image/graphic membership; the model
  selects role and reading order, not crop boundaries. Author/affiliation grids and parallel prose columns are text
  blocks, rather than data tables.
  The renderer can promote an adjacent native header into semantic column cells
  only for ordinary fully bold labels, horizontal graphics, a single wrapped
  label per column and stable body-column positions. Page, gap and intervening
  content checks prevent distant association. Math, spanning labels, captions
  and referenced callouts retain their source crop. This is conservative inference.
  Each attempt records its requested effort. Requests have a 32768-token output budget, a 300-second
  deadline and 2 MiB response cap, with at most three
  validation attempts per pass and no model fallback. Review adds at least one
  request per uncached pair. OpenRouter selects
  the serving endpoint by throughput; the provider is recorded with usage.
  Truncated responses consume a validation attempt and never publish output.
  Reviewed pairs are atomically cached under `DISTILLER_PDF_CACHE`
  (default `~/.cache/distiller/pdf`) with source/model/prompts/schema/endpoint identity;
  old unreviewed caches are not reused. Cache hits are revalidated. Draft/review
  responses and usage have separate names. The effective normalized plan is saved
  as `reconstructed.json`; the receipt hashes that plan and the rendered HTML and records
  `review: source-review/v12`. These artifacts support
  inspection and resume. Corrupt cache fails explicitly. `OPENROUTER_API_KEY` is
  injected only at runtime; test endpoints permit HTTP only on loopback.
  Nested Forms are traversed; rotated/skewed/stretched Forms, rotated/mirrored/skewed
  images, scans and graphic-only
  pages fail. JPEG bytes survive only if PDFium raw/processed pixels match;
  other rasters use lossless PNG with adaptive compression and EPUB byte caps.
  Images render at integer source-pixel scale, with the original matrix restored.
  Tight glyph bounds drive crop membership; original advances drive spacing.
  A short bottom-page rule joins a unique native footnote immediately below it
  when a smaller raised numeric label, aligned source ink and a clear gap agree.
  Short displayed equalities retain a uniquely aligned parenthesized number when
  no intervening source ink separates it from the equation.
  Native traversal propagates conservative linear clip envelopes in page coordinates.
  Wholly excluded glyphs remain in the source ledger as `clipped` furniture; spans
  split at visibility changes. Partial glyph bounds intersect the effective clip.
  Partial raster clips require native source regions; wholly clipped rasters fail
  explicitly. Curved clip paths retain inherited constraints without claiming exact visibility.
  Native bold/italic ranges survive rendering. Measured blank gaps split merged
  prose; normal leading reconnects only unfinished aligned passages. Isolated
  heading styles and contiguous same-page numbered lists provide narrow role repairs.
  Native segmentation seeds mathematical source lines, connected graphic groups,
  aligned numeric/math table rules and fraction bars. Transitive intersection
  closes ownership; shared source ink and nearby formula groups merge before
  rendering. Table seeds require numeric/math evidence and aligned cells; short
  bars require nearby math or short symbolic numerator/denominator evidence. Detection uses common
  mathematical fonts, operators and geometry. Plain-font plus suffixes stay prose.
  Script-aware line closure preserves parentheses; intersected numbered Figure/Table
  captions extend through matching aligned lines at normal leading;
  aligned same-page hyphen wraps remain together inside native passages, including chains;
  ambiguous layouts and undetected notation remain source-inspection risks.
  Each region has a source anchor (an empty synthetic anchor only for graphic-only
  regions). Hidden members are not model-selectable. Selecting an anchor consumes
  all its source spans/images/graphics exactly once. Native anchors render as
  pictures inside paragraph/note parts; floating figures separate native pictures
  from ordinary caption parts. Isolated `visual` blocks contain one native anchor
  and remain continuation barriers. Hidden numeric members are reference-only
  candidates; note validation checks ownership through selected anchors and rejects
  references to the note's own members. Ordinary text tables retain semantic rows.
  Figures support multiple raster panels per caption.
  A second 180-second native worker pins the source bytes by SHA-256, renders
  color at 3x scale (216 dpi), and crops floor/ceil pixel rectangles. It enforces
  12000-pixel axes, 32 million page pixels, 4 MiB per PNG and 64 MiB asset totals.
  `regions.json` binds source rectangles; `region-receipt.json` records source,
  pixel rectangles, dimensions and encoded/decoded pixel hashes. Crops and receipts
  stay in the reconstruction cache. Native rendering finishes before EPUB publication.
  Visual math/table cells are images, not searchable semantic notation; inseparable
  inline formulas retain a whole source line. Model-selected ordering, joins,
  crop grouping and note associations still require source comparison; ID coverage
  does not prove fidelity. Original PDF link annotations remain unsupported.
- `--stdin --base-url URL` reads bounded (5 MiB), strict UTF-8 HTML and calls
  `parse_html` without source fetching or extraction LLM use. EPUB's mandatory
  cover can make a separate outbound Muse request. Conflicting acquisition/LLM flags fail
  before input is consumed. Stdin defaults to stdout; explicit output paths remain.
- Stdin JSON is `article-extraction/v1`: exact caller-supplied source URL, SHA256
  of input bytes, HTML/UTF-8 media identity, extractor version/method/options,
  and title/Markdown. URL JSON retains its existing `DistillerResponse` shape.
  This is a data contract, not a downstream tool dependency or review receipt.
- Fenced code keeps source text, indentation and blank lines; a terminal newline
  is added only when required to close a fence. Delimiters exceed source backticks.
- Inline text follows DOM boundaries; numeric content and inline code spaces survive.
- Math uses single-line TeX annotations when available; multiline annotations remain MathML, otherwise sanitized MathML.
  Superscripts/subscripts retain semantic HTML.
- Rectangular tables with a header row use GFM with empty cells intact. Spanning,
  mixed-header or nested/block tables retain sanitized HTML instead.
- Prose cleanup skips code and semantic HTML/TeX. Whole-HTML whitespace and entity
  rewriting is absent. HTML/math display depends on the consumer's renderer.

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
