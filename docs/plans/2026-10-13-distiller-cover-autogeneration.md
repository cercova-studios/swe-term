# Mandatory EPUB cover generation

Status: implemented C1–C5, 2026-10-13; C6 reader/device acceptance is partial.
Muse artwork plus local licensed Noto Sans typography is implemented in `cover.rs`.
User requirement: cover generation belongs to EPUB creation,
without an enable flag. Covers must be minimal, aesthetic and clear on Kindle
e-ink displays and in Books library thumbnails.

## Current contract and intended change

`extensions/distiller/src/main.rs` currently accepts an optional `--cover`.
HTML/stdin call `epub::write_epub`; `pdf::write_epub` extracts an HTML/metadata
intermediate and calls that same packager. `epub.rs` validates PNG/JPEG covers
with a 4 MiB byte limit, dimension/allocation limits, and the 64 MiB aggregate
image limit. It packages through sandboxed Pandoc and publishes without overwrite.
These contracts remain. Root ARCHITECTURE.md Sections 5, 9 and 10 keep specialized
image work in the Rust extension; the Go tool adapter remains Target. Section 12's
agent-loop and tool-contract refactors are unaffected.

Target behavior for the existing command:

```sh
distiller article.html --mode epub -o article.epub
distiller book.pdf --mode epub -o book.epub
```

Every successful EPUB has one validated cover. Resolve it in this order:

1. Use the caller's `--cover` override when supplied.
2. Reuse an identical previously generated cover when its receipt and image hash
   match the current cover specification and generator identity.
3. Otherwise generate artwork, compose the cover locally, validate it, then embed it.

There is no `--generate-cover` or `--no-cover` flag. A missing generator, missing
credentials, timeout, corrupt response or failed composition prevents publication.
Report the failure clearly and keep reusable completed cover work. Ordinary Markdown
and JSON extraction stay deterministic and do not generate covers.

This changes offline behavior: a first conversion requires a configured generator
or supplied cover. Subsequent identical conversions may reuse a verified cache.
HTML stdin remains acquisition-free; mandatory generation would introduce a separate,
explicitly documented outbound image request when no supplied/cached cover exists.
Update stdin help and architecture's no-model promise explicitly: extraction still
never invokes a model, but EPUB cover production may call the configured generator.
`allow_remote` continues to control source assets only; it is not a cover-generation
permission. `--llm` remains an extraction option and is still rejected for EPUB;
describe deterministic extraction/packaging separately from stochastic artwork.

## Cover pipeline

```text
extracted title + author + bounded topic hints
  → versioned e-ink cover specification
  → supplied cover / verified cache / generated text-free artwork
  → local title and author typesetting
  → validated PNG + generation receipt
  → existing EPUB packager
```

Resolve the same metadata used by EPUB packaging first: the current writer prefers
the first nonempty H1 over `article.title`. Do not generate from a different title.
Use title/author plus at most six source section labels, bounded to 160 characters
each, as topic hints; no extra summarization model or new topics response field.
PDF hints can come from its existing outline. Check content/images, destination
directory and existing output before making a paid request.

Generate the illustration only. Render the exact title and author in Rust, so
spelling and line wrapping do not depend on the image model. A supplied complete
cover is embedded unchanged after validation. Do not invent an author when source
metadata is missing. Long titles wrap or reduce size within a defined readable
range; fail visibly instead of clipping text. Reject unsupported glyphs rather
than substitute incorrect characters.

Use one built-in e-ink design profile initially: 1024 × 1536 pixels, large title,
generous margins, a small number of bold geometric forms and strong grayscale
contrast. Artwork must leave a reserved text area. Compose text on an opaque
high-contrast panel so artwork cannot obscure it. Avoid tiny detail, gradients,
textures and dependence on color. Choose light/dark composition reproducibly from
the cover specification; retain creative subject variation in the illustration.
Use a 64-pixel safe margin, reserve the upper 40% for title text and lower 12% for
author text, and render lettering against a solid panel. These are initial layout
bounds to validate with long-title fixtures, not proof of reader legibility.

Start with one concrete generator adapter and a narrow request/result boundary
inside the extension. Keep provider/model identity explicit in configuration and
receipts; add a second adapter only when requested. Select and verify the first
provider's official API, authentication, output limits and cost controls before
implementation. Codex's conversation-only image tool is not a callable CLI backend.
No resident service, provider framework, queue or Go-core change is required.

Send only title, author and bounded topic hints by default, not the entire source
document. Source text is input data, not generator instructions. Credentials enter
the process through the repository's Doppler runtime workflow and never enter
prompts, files or diagnostics. Use one generation attempt per cache miss; do not
retry ambiguous paid requests automatically. Report known usage/cost and preserve
unknown cost as unknown. Require a configured request/cost policy supported by the
chosen provider; do not invent a universal price or budget guarantee.
Give generation a 90-second deadline and bounded response/image decoding. The
existing 120-second HTML deadline remains the outer limit; enforce the shared
generation deadline for PDF too, whose current 180-second limit covers extraction
only. Cancellation may leave billing unknown, but never publishes partial output.

The cache key covers a document fingerprint, the exact specification, prompt/profile version, font/layout
identity, provider/model and generation parameters. Record image digest, dimensions,
prompt, generator identity and available usage/request identity. Validate bytes and
receipt on every hit. Use the original PDF digest already emitted by the worker;
for HTML, hash retained structured content plus resolved metadata. The fingerprint
is cache identity, not proof of source authenticity. A malformed cache entry is an
explicit error, not a hidden paid regeneration. Interrupted cache writes are not
valid entries; publish a staged image/receipt directory atomically as one entry.
Concurrent invocations use a per-key exclusive reservation and bounded wait. A
crashed/ambiguous reservation requires explicit recovery; do not issue a second
paid request under the claim that the first never happened. Keep a completed cover
after a later Pandoc failure so retrying the
conversion does not incur another generation request. An interrupted request may
still have been billed; report that ambiguity. Define the bounded cache location
and retention behavior during implementation; no background eviction service.
Retain origin (supplied/generated), actual reported response model, output digest
and usage state. Put generated-cover provenance in an embedded PNG text chunk so
it travels with the EPUB, not just its private cache. Test that packaging preserves
the chunk; avoid a new EPUB XML vocabulary or a second-file publication transaction.
The external receipt hashes the final composed PNG; embedded provenance cannot
contain its own final-byte digest. Supplied images remain byte-identical.

## Delivery tasks

| ID | Depends on | Responsibility and affected files | Done when |
|---|---|---|---|
| C1 | — | Define metadata inputs, one versioned e-ink profile and layout bounds. Proposed `src/cover.rs`; inspect `types.rs` and PDF worker metadata without changing JSON extraction schemas. | HTML, stdin and PDF resolve the same title/author rules; long and missing metadata cases have explicit outcomes. |
| C2 | C1; provider selection | Implement one bounded generator adapter, receipt and atomic verified cache in `cover.rs`. Add only required dependencies to Cargo.toml/lock. | A fake generator exercises success, timeout, invalid output, cache reuse/tampering and ambiguous failures without live keys or hidden retries. |
| C3 | C1 | Compose exact title/author over text-free art in Rust. Proposed licensed font asset and minimal font rasterizer, selected after checking existing dependencies and license coverage. | Long titles, punctuation and supported glyphs render without clipping; thumbnail and grayscale previews remain legible; unsupported glyphs fail explicitly. |
| C4 | C2, C3 | Resolve the mandatory cover once at `epub::write_epub`, using resolved metadata after deterministic validation; forward the PDF digest/outline from `pdf.rs`. Preserve `--cover` as override and update stdin/EPUB help in `main.rs`. | Default HTML/stdin/PDF EPUB journeys embed a cover; supplied/cache cases make zero generation requests; cover and EPUB titles agree; any cover failure produces no destination and preserves existing files. |
| C5 | C4 | Extend `tests/epub_journey.py` and `tests/pdf_journey.py`, with a fake generator/provider. Update README, extension/root architecture and TODO. | Tests assert one EPUB cover-image entry, exact composed metadata, valid internal references, caps, cache behavior, mode conflicts, cancellation/failure and no-overwrite publication. EPUBCheck passes. |
| C6 | C5 | Inspect both acceptance books in Books and Calibre, plus grayscale/thumbnail previews; check physical Kindle when available. Retain reader evidence and cover provenance. | Both titles are recognizable in library grids; covers, chapter navigation and body content render correctly. Untested hardware is labeled unverified. |

Implement C1–C4 as one runnable vertical slice before adding more providers or
design profiles. Reuse the existing bounded image validation and Pandoc cover
embedding, including its documented sandbox preload workaround. Existing CLI
journeys need offline supplied covers or the fake generator; no live API keys in CI.

## Acceptance and limits

- Every successful EPUB contains a cover; no silent bare-book fallback.
- Default generation preserves exact metadata and keeps title text legible at a
  representative small library thumbnail. Human inspection complements geometric
  checks; contrast or image scores alone do not establish aesthetic quality.
- Cache hits and supplied covers cause no network request. One miss causes at
  most one deliberate generation attempt, with reported failure/usage states.
  Concurrency and cancellation journeys test duplicate-request prevention; recovery
  after an ambiguous paid request is explicit rather than an automatic retry.
- Source article/PDF content and navigation remain unchanged by cover composition.
  Repeat the existing two-source fidelity benchmark as a regression check, with
  generated artwork measured separately from retained source figures.
- Physical e-ink acceptance requires actual hardware. Books/Calibre screenshots
  and grayscale previews are useful checks, not substitutes for that evidence.

Out of scope: cover galleries, multiple generated candidates, automatic aesthetic
judging, OCR for model-drawn titles, background generation and automatic provider
fallback. Rollback removes the mandatory resolver and restores the existing
supplied-cover path; generated assets/receipts remain recoverable.
