# distiller

`distiller` extracts readable article content from URLs or UTF-8 HTML files and writes Markdown, JSON, EPUB3 or Kindle KFX.

```sh
distiller 'https://example.org/article' -o article.md
distiller article.html --mode epub --cover cover.png -o article.epub
```

Local PDFs use matching versioned arXiv HTML when available through their source
stamp; other PDFs use cloud reconstruction. HTML extraction is deterministic by
default. EPUB/KFX output requires Pandoc and a supplied, cached, or generated cover.

## Quickstart

### Read in the stock Kindle reader

`save2kindle` composes conversion and verified delivery in one shell command:

```sh
save2kindle 'https://www.anthropic.com/engineering/building-effective-agents'
save2kindle ~/books/situationalawareness.pdf
save2kindle existing.epub
save2kindle existing.kfx
# Choose a new destination, or reuse a finished cover:
save2kindle article.html -o article.kfx --cover cover.png
```

Install the wrapper from the repository root (Python 3.8+):

```sh
install -m 755 extensions/distiller/scripts/save2kindle.py ~/.local/bin/save2kindle
python3 extensions/distiller/tests/save2kindle_journey.py
```

It requires the installed `distiller`, `epub-to-kfx`, and `kindle-send` tools,
plus the existing Amazon Previewer/KFX Output setup and `kindle` SSH alias.
HTML/PDF conversion generates a mandatory Muse cover. PDF reconstruction also
needs `OPENROUTER_API_KEY`. Without `--cover` or an
injected `MODEL_API_KEY`, the wrapper uses Doppler project `llm-experiments`,
config `dev_personal`, injecting the missing cover/reconstruction keys into conversion. A cache miss
makes one paid artwork request; EPUB/KFX inputs retain their existing cover.
No model credential is passed to the delivery process.

Supported inputs are HTTP(S) article links and local HTML, born-digital PDF,
EPUB or KFX files. PDF links ending in `.pdf` are downloaded temporarily with
a 200 MiB cap and a 60-second read budget (each blocking read may add up to
15 seconds); extensionless
PDF links must first be downloaded locally. PDF reflow retains the limitations below.
Other local formats fail explicitly. URL outputs use a readable slug plus URL
hash in the current directory; local outputs sit beside the source. `-o` chooses
a new KFX path. Existing outputs and conflicting Kindle books are never overwritten.
If delivery fails, the completed KFX stays local; retry with `save2kindle file.kfx`.
Successful stdout is the sender's `kindle-transfer/v1` hash-verification receipt.

The individual tools remain independently usable:

```sh
distiller 'https://example.org/article' --mode kfx -o article.kfx
kindle-send article.kfx
# An existing EPUB can use the converter independently:
epub-to-kfx article.epub -o article.kfx
```

KFX uses the same mandatory-cover EPUB writer, then invokes the independent
`epub-to-kfx` command on PATH. That command uses the installed local Amazon
Previewer engine and KFX Output packager. Missing prerequisites or an existing
destination fail before extraction and cover generation. HTML, stdin and local
PDF input retain their EPUB constraints; KFX is file-only and rejects `--llm`.
Amazon can resize/recompress diagrams. EPUB remains available without this
converter. `kindle-send` uses the separately configured `kindle` SSH alias and
verifies file hashes without overwriting books. Distiller itself never sends files.

### Make a readable EPUB

```sh
distiller article.html --mode epub -o article.epub
distiller article.html --base-url 'https://example.org/article' --mode epub -o article.epub
distiller 'https://example.org/article' --mode epub -o article.epub
distiller article.html --mode epub --cover cover.png -o article.epub
```

EPUB export requires Pandoc on `PATH`. It uses cleaned HTML
directly, retains structured DOM instead of plain-text schema replacements, and
adds a title page, heading navigation, chapter breaks, metadata and reader-friendly
typography. It preserves code tabs, MathML, spanning tables and footnote links.
Equation layout tables use borderless typography and presentational roles that retain MathML.
Empty layout tables without
media are removed while their anchors remain. Caption descriptions stay linked to images
and tables. Head-group headers retain scope; body headers retain row/column roles.
Complex header associations still need source review. For unmarked LaTeXML
tables, a narrow visual-header heuristic recognizes bold or spanning label bands;
ambiguous tables retain their source markup. These repairs improve structure but
do not establish screen-reader conformance or full diagram descriptions.
Pandoc can normalize terminal code newlines; equation display depends on the reader.

Versioned arXiv links use the full HTML article first and check the corresponding
PDF. Other HTML sources can be paired with a local PDF explicitly:

```sh
distiller 'https://arxiv.org/html/1706.03762v7' --mode epub -o paper.epub
distiller paper.html --pdf-source paper.pdf --base-url 'https://example.org/paper' --mode epub -o paper.epub
```

Local PDFs carrying a versioned arXiv margin stamp also use that HTML-first path.
Other PDFs retain cloud reconstruction. The paired path preserves MathML and
semantic tables, restores uniquely associated missing artwork from the PDF, and
recovers isolated missing text only at a verified paragraph boundary. Ambiguous
gaps fail without publishing a book. Isolated small bottom-margin text becomes
an end note. It needs no reconstruction model credential.
The mandatory cover still requires a supplied image, cache entry, or Muse.

`paper.completeness.json` records input/output hashes, matched source spans,
recovered content, and unchecked short spans and native regions. This checks
prose witnesses and numbered artwork presence; it does not prove equation,
image or reading-order equivalence. KFX retains the receipt beside its final file.
Stdin never fetches a companion PDF. Unversioned links and arbitrary PDF discovery
remain on their existing paths. EPUB and receipt are staged before publication;
if a concurrent receipt publication fails, the error names the retained EPUB.

Every successful EPUB has a cover. On a cache miss, Muse generates text-free
artwork and Rust composes exact title/author text using a bundled licensed font.
Inject the Meta key with Doppler:

```sh
doppler run --project llm-experiments --config dev_personal --only-secrets MODEL_API_KEY --no-fallback -- distiller article.html --mode epub -o article.epub
```

`--cover PATH` overrides generation with a supplied PNG/JPEG for reader libraries,
including Books. The regular file must decode within the image limits and be at
most 4 MiB; its bytes count toward the 64 MiB total. It works with HTML, stdin
and PDF EPUBs/KFX files and is rejected for Markdown/JSON. Supplied and verified cached
covers need no credential or image request. No generation toggle is required.
Missing credentials, invalid responses, timeouts and unsupported typography fail
without publishing an EPUB. Markdown/JSON never generate covers.

Generated covers are 1024×1536 grayscale PNGs with large local title/author text,
opaque panels and a geometric illustration. Precomposed Latin, Greek and Cyrillic
are supported; complex scripts/combining marks require a supplied cover. Muse is
asked for artwork without text, but unintended model-drawn symbols in the artwork
are not detected. Physical e-ink display quality remains a reader acceptance check.

The cache defaults to `~/.cache/distiller/covers`; `DISTILLER_COVER_CACHE` overrides
the directory. Entries contain `cover.png` and a hash-checked `receipt.json`;
PNG provenance also travels inside the EPUB. There is no automatic eviction.
One request is attempted per miss, with a 90-second deadline and no redirects.
Distiller does not report actual image-provider billing. Failed/interrupted requests retain a `.pending` reservation because
billing may be uncertain. Check provider history before explicitly removing that
named reservation. Corrupt entries require explicit restoration/removal, not a
silent paid retry. Atomic cache publication requires directory synchronization.

Each export creates one article per book. It embeds local, base64 or HTTP(S)
PNG, JPEG and GIF images, with 4 MiB per image and 64 MiB total. Images must decode
within dimension/allocation caps; WebP is converted losslessly to PNG. Local images must stay
beneath the HTML file's directory, including symlinks, and must be regular files.
Remote requests have a 10-second timeout and five-redirect limit. Unsupported
media, SVG and MathML image glyphs cause an error. Pandoc runs in its sandbox
with a 30-second timeout. Existing EPUBs
are never overwritten; failures leave no partial destination.

EPUB accepts stdin with `--stdin --base-url URL --mode epub -o article.epub`.
It rejects `--llm`, `--stdout` and `-o -`. Local HTML files accept at most 5 MiB
of strict UTF-8 and reject `--llm` and `--proxy`.

Run the public CLI journey after building:

```sh
python3 tests/epub_journey.py --binary target/debug/distiller
# Optional independent conformance check:
python3 tests/epub_journey.py --binary target/debug/distiller --epubcheck /path/to/epubcheck.jar
```

### Reflow a born-digital PDF

```sh
doppler run --project llm-experiments --config dev_personal --only-secrets OPENROUTER_API_KEY,MODEL_API_KEY --no-fallback -- distiller ~/books/situationalawareness.pdf --mode epub -o situationalawareness.epub
python3 tests/pdf_journey.py --binary target/debug/distiller --epubcheck /path/to/epubcheck.jar
```

The converter is Rust, using `pdfium-render` and PDFium 7881. Install the matching
platform library from [PDFium's binary release](https://github.com/bblanchon/pdfium-binaries/releases/tag/chromium%2F7881)
in `~/.local/share/distiller/lib` (or the system library path), preserving its license.
Pandoc packages the final EPUB. Python and ReportLab/Pillow are test dependencies only.

The worker accepts up to 200 MiB and 400 pages and stops after 180 seconds. It
extracts text spans with stable IDs, font/position/right-edge evidence, bookmarks and raster
figures. Original JPEG bytes are
kept when their decoded pixels match PDFium's processed output; other figures use
lossless PNG. Image and page-object limits fail explicitly instead of dropping data.
Scans, graphic-only pages, rotated/skewed/stretched Forms and rotated/mirrored/skewed
images are rejected. Figures render at integer source-pixel scale to avoid resampling.

PDFs then use `deepseek/deepseek-v4.1-flash` through OpenRouter with strict JSON
Schema. The model sees source text
and geometry plus bounded grayscale page images and returns source-linked blocks,
tables, note associations and joins;
it cannot rewrite the prose. This sends the PDF text to a cloud provider, without
downloading a local model. Supply `OPENROUTER_API_KEY` through Doppler at runtime.
The ordinary HTML extraction path remains deterministic.

Rust requires every source span/image/graphic exactly once and rejects invalid references,
joins and furniture removal. Hyphen repairs require a trailing hyphen and either
a PDFium marker or a physical line/page boundary; unmarked within-line literals
remain protected. A digit-bearing segment immediately before a hyphen retains that
separator; a later alphabetic segment can still be dehyphenated. Rare wrapped
alphanumeric words may retain a line hyphen.
Native bold and italic ranges survive in the text. Measured blank gaps can split
merged paragraphs; normal leading reconnects only unfinished, aligned prose.
Isolated heading styles and contiguous numbered lists provide narrow role repairs.
The model selects printed heading positions; notes move to a separate
end section with links when associated. Source-page anchors stay invisible; conversion diagnostics are retained outside
the reading flow.

Source packets have a 256 KiB JSON cap. Oversized page pairs split into single
pages, with optional neighboring context reduced as needed. Owned source text
is never truncated. A single-page packet that still exceeds the cap fails.
Prompt, schema, candidate and image data are outside that cap.

Each owned-page batch requires a draft followed by a separate source review, using
the same model and original page evidence. Review checks word/compound joins,
superscript placement, paragraph splits and continuations, contents entries,
heading hierarchy, and floating tables/captions. The model receives up to 32
previous headings with their source font sizes, the full bookmark outline, and
intact source-word witnesses for candidate hyphen joins. Witnesses are hints:
both `re-sign` and `resign` can be intentional. An exact intact source spelling
can preserve a compound hyphen when the joined spelling is unattested. Related
compound families only select boundaries for a focused structured spelling review;
they never decide the spelling automatically. The review changes joins only.
Its evidence-bound cache supports offline replay. A new unresolved boundary can
require OpenRouter access even when the page plans are cached; failed review
prevents publication. No dictionary rewrites source prose.
Explicit captions stay separate from body paragraphs interrupted by a table.
After validation, measured geometry restores baseline word/numeral spacing,
retains same-line run-ins as prose unless bookmarked, and relocates explicitly
linked raised callouts when there is one physical predecessor. Ambiguous geometry
does not trigger these repairs. Fresh and cached plans use the same normalization.
A typography heuristic also closes em dashes between aligned adjacent source lines
within a run. It preserves explicit whitespace and preformatted content, and leaves
page/column ambiguity to review. It assumes symmetric dash spacing; inspect the result.

Draft and separate source-review passes start with low reasoning.
Every rejected-response retry disables reasoning while keeping the same schema and
validation. Native code protects detected mathematical passages, tables and vector
figures before either model pass. Each fixed region has one opaque source anchor;
the model chooses its role and position without selecting crop geometry or hidden glyphs.
Author/affiliation grids and parallel prose columns remain ordinary text blocks,
rather than being forced into data-table cells.
Each attempt records the requested effort. Requests use a 32768-token output budget,
a 300-second timeout and a 2 MiB response cap. Each pass allows at most
three validation attempts, including incomplete responses. Both passes enforce
the same source ownership and rendering checks; a valid draft alone is never
published. Candidates are supplied as user data; draft retries retain only typed
source-ID candidates, never malformed prose or truncated replies. The review adds
at least one model request per uncached page pair,
increasing cost and latency. Exhausted attempts fail
without publishing output. OpenRouter prioritizes endpoint throughput; endpoint
pricing can differ, and recorded usage includes the serving provider.
Reviewed page pairs are cached and revalidated on resume under `~/.cache/distiller/pdf`
(`DISTILLER_PDF_CACHE` overrides it). Source layout, model responses, provider usage,
reconstructed HTML, source hashes and warnings remain there for inspection. Cache
corruption is an error. Draft and review responses/usage have separate artifact names;
the review prompt and schema participate in cache identity, so old unreviewed
entries are not reused. The receipt records `review: source-review/v12`.
It hashes `reconstructed.json`, the effective normalized plan, and the rendered HTML.
The `save2kindle` wrapper allows up to four hours for book
conversion and keeps delivery bounded separately.

Model-selected reading order and semantic associations still need source inspection.
Equations, mathematical/complex tables and vector diagrams use bounded native
color crops at 216 dpi. Native geometry closes each crop over intersected glyphs,
rasters and graphics; connected figures, aligned numeric/math table rules and nearby
formula ink retain their spatial relationships. Numeric/math evidence guards table
enclosures against ordinary decorative rules. A crop's anchor can appear within its paragraph or note, or in a
figure with its caption. Hidden numeric callouts remain reference-only link candidates.
Ordinary text tables can remain HTML tables; raster panels can share one caption.
An isolated bold header crop can become real column headers when its labels
uniquely align with stable body columns on the same page. Wrapped labels stay
within their column. Math, spanning labels, uncertain geometry and note callouts
retain native rendering; header inference remains heuristic.
The separate render worker verifies the input SHA-256 and records pixel rectangles
and hashes before any reconstruction request.
Crops preserve appearance rather than semantic math or selectable table cells.
An inseparable inline formula retains its complete source line as an image.
Measured same-page word wraps stay within that image. Native clip evidence keeps
fully hidden text out of the reading flow while retaining its source IDs.
Partial image clips use source-rendered regions; wholly clipped raster objects fail
explicitly. Curved clipping and ambiguous caption boundaries still require inspection.
Detection uses common mathematical fonts, operators and geometry. Dense or ambiguous
layouts and undetected notation still require source inspection.
Original PDF link annotations remain unsupported. Language defaults to English.
Packaging validity and source-ID coverage do not establish semantic fidelity or
correct display on every reading device.

### Pipe a retained HTML snapshot

```sh
distiller --stdin --base-url 'https://example.org/article' --mode json < source.html
```

`--stdin` accepts at most 5 MiB of UTF-8 HTML and writes stdout by default.
Extraction performs no source fetching or LLM calls; EPUB cover generation can
separately call Muse. Stdin conflicts with the URL argument,
`--llm`, `--proxy`, and `--lang`. `--base-url` preserves the exact original URL
and resolves relative links. Markdown remains the default output mode.

JSON stdin output uses this producer-neutral record contract:

```json
{
  "schema": "article-extraction/v1",
  "source": {"url": "https://example.org/article", "sha256": "<64 hex digits>", "media_type": "text/html", "encoding": "utf-8"},
  "extractor": {"name": "distiller", "version": "0.1.0", "method": "deterministic-html", "options": {}},
  "document": {"title": "Article title", "markdown": "Extracted Markdown"}
}
```

The input hash covers exact stdin bytes, not normalized HTML. Empty options mean
the version's deterministic parser defaults. Source URL labels are supplied by
the caller; a hash does not authenticate authorship or establish source fidelity.
URL-mode JSON keeps its existing response format. Neither mode invokes a consumer,
stores knowledge, or reviews claims. Use shell `pipefail` when composing producers.

### Install the CLI

From the repository root:

```bash
cargo install --path extensions/distiller --locked
distiller "https://example.com/article" --stdout
```

Cargo installs `distiller` in `~/.cargo/bin` by default. Ensure that directory is on `PATH`.
To include optional browser fallback, add `--features browser` to the install command.

### 1. Build

```bash
cargo build
```

### 2. Run (default markdown output)

```bash
cargo run -- "https://example.com/article" -o webpage.md
```

### 3. Choose output mode

```bash
# JSON
cargo run -- "https://example.com/article" --mode json -o webpage.json
```

### 4. Enable LLM extraction pipeline

```bash
cargo run -- "https://example.com/article" --llm -o webpage.md
```

The default LLM provider is Ollama's [`gemma4:31b-cloud`](https://ollama.com/library/gemma4:31b-cloud), with no secondary
provider fallback. If it fails or its output fails the quality checks, extraction
keeps the heuristic output. `DISTILLER_LLM_PROVIDERS` overrides this default.

Optional environment variables:

```bash
export DISTILLER_LLM_PROVIDERS="gemma4:31b-cloud"
export DISTILLER_LLM_TIMEOUT_MS=12000
export DISTILLER_LLM_MAX_INPUT_CHARS=60000
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export JINA_API_KEY="..."
```

### 5. Enable browser fallback support

Browser fallback is behind a feature flag:

```bash
cargo run --features browser -- "https://example.com/article" -o webpage.md
```

### 6. Run tests

The cover regression journey requires Pandoc and Python 3 on `PATH`; it uses a
local mock Muse server and makes no paid requests. PDF journeys additionally
need PDFium and the test-only ReportLab/Pillow dependencies.

```bash
cargo test
cargo build
python3 tests/pdf_journey.py --binary target/debug/distiller
python3 tests/pdf_fidelity_journey.py --binary target/debug/distiller
python3 tests/pdf_repair_journey.py --binary target/debug/distiller
```

## Attribution

This project is inspired by the extraction approach and architecture patterns from [`defuddle`](https://github.com/nicholasgasior/defuddle).  
`distiller` is an independent Rust implementation with its own pipeline, CLI, and module layout.

## Deterministic fidelity

The default path preserves highlighted code values, indentation and blank lines.
Inline code spaces and numeric paragraphs survive cleanup. MathML and sup/sub
retain semantic HTML; single-line TeX annotations become math delimiters; multiline annotations remain MathML. Simple
tables retain empty cells and links; complex tables remain HTML. A compatible
renderer is required for math and raw HTML. Sanitization edits real DOM attributes
and does not rewrite attribute-like strings inside code.

Source-derived regression fixtures remain in [tests/fixtures/](tests/fixtures/),
beside the public CLI journeys. Historical comparisons and validation records
belong in the [development archive](../../docs/README.md#designated-development-archive);
they do not establish current general fidelity.
