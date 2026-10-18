# EPUB fidelity: two-source case study

Completed 2026-10-12 America/Detroit using the final installed distiller 0.1.0, PDFium 7881, xberg 1.3.8, Pandoc 3.12.1 and EPUBCheck 5.4.0. All conversion paths were deterministic; no model calls or local model downloads. The PDF converter is Rust; Python is test/evaluation tooling only.

Distiller retains more of the measured article components and more PDF outline headings in this case study. Both PDF pipelines retain every raster figure. Reader inspection confirms working navigation and fitting diagrams but exposes PDF prose-formatting limits, so these component results do not establish overall readability superiority.

## Anthropic: Building effective agents

| Component | Distiller | xberg + Pandoc | Direct Pandoc |
|---|---:|---:|---:|
| Source headings retained | 18/18 | 18/18 | 18/18 |
| Source section titles in navigation | 18/18 | 18/18 | 0/18 |
| Exact normalized source text blocks | 92/92 | 90/92 | 92/92 |
| Article diagrams included | 8/8 | 4/8 | 8/8 |
| Diagrams with exact decoded source pixels | 8/8 | 0/8 | 8/8 |
| EPUBCheck errors / warnings | 0 / 0 | 0 / 0 | 3 / 0 |

The two unmatched xberg blocks are parent list items with nested Sectioning/Voting examples. This is a block/grouping difference, not proof that two entire paragraphs disappeared. Its four included images differ in decoded pixels; cause/default image policies were not tuned. Direct Pandoc includes ten image elements (eight diagrams plus two additional images). Its validation errors are two duplicate title IDs and one SVG namespace error. Additional source cleanup could improve that baseline.

## Situational Awareness: 165 pages

| Component | Distiller | xberg + Pandoc | Direct Pandoc |
|---|---:|---:|---:|
| Source raster figures with exact decoded pixels | 40/40 | 40/40 | Unsupported input |
| Total image elements | 40 | 68 | — |
| Source-page anchors | 165/165 | 0/165 | — |
| Outline titles retained as headings and navigation | 40/41 | 35/41 | — |
| Original bookmark links with exact source-page destination | 41/41 | 0/41 | — |
| EPUBCheck errors / warnings | 0 / 0 | 0 / 0 | — |

Direct Pandoc exits 21: `Unknown input format 'pdf'`. No third-party PDF preprocessor was supplied to that baseline. The 28 additional xberg image elements are reported separately rather than scored as retained source figures. Seven previously missing headings are recovered by matching quote variants and nearby alphabetic hyphenated line boundaries. The one unmatched title is Introduction: the source bookmark targets the cover and has no printed heading there. Distiller retains that original destination in its labeled source outline and does not invent a printed heading. All 41 source bookmarks remain accessible.

Five selected PDF text probes survive distiller and two survive xberg. Three probes include printed line-end hyphens; xberg may dehyphenate those words. These are presence probes, not reading-order proof or a score of prose quality. Distiller still exposes printed split words, page boundaries and separated margin notes.

## Fixes and regression evidence

- Preserve PDFium hidden hyphen markers before filtering controls; traverse nested Forms; reject scans, graphic-only pages and unsupported transforms explicitly.
- Render raster images at integer source-pixel scale, restore placement matrices, preserve matching raw JPEGs, and compress other images losslessly without increasing byte caps.
- Validate outline labels against complete visible text. Put PDF chapter headings outside page containers; keep note labels out of the TOC.
- Flatten HTML heading wrappers in source order while retaining their anchor IDs. Reject language/direction wrappers that cannot yet be flattened safely rather than erase inherited semantics.
- Preserve WebP pixels through PNG conversion, reject corrupt images, normalize dates, preserve footnote links, and keep atomic no-overwrite publication.

Final checks passed: 54 Rust tests, formatting, Clippy with warnings denied, and both installed CLI EPUB/PDF journeys with EPUBCheck. The navigation regression failed against the previous installed binary before the fix. A scratch mutant disabling PDFium hyphen handling failed the PDF journey as intended. The new journey also checks navigation order and language/direction rejection. No production Python converter remains.

## Method and limits

Preregistration passed before each recorded run. All pipelines received the same frozen HTML DOM and shared cached public images; links were resolved, WebP was converted losslessly, and srcset was removed without selecting article content. The PDF was copied unchanged into isolated workspaces. Article oracle: source article h2/h3 headings, 92 paragraph/list-item blocks, eight diagrams. EPUB inspection follows OPF spine order and actual navigation links, not ZIP entry order. Four footer markers are checked; their absence is not proof of all chrome removal.

The PDF image oracle uses original DCT streams for JPEGs because pypdf's exported JPEG `image.data` re-encodes bytes. Earlier apparent 35/40 results were oracle artifacts; the corrected final comparison is 40/40 for both pipelines. Historical incomplete/development runs are preserved but excluded from these conclusions.

Full reading-device acceptance, a random/held-out corpus, OCR, multi-column order, semantic equations, vector graphics, clipping, PDF links/emphasis, note associations and cross-page paragraph joins remain unverified or unresolved. One repetition per pipeline; no speed/RSS superiority claim. Language/direction heading wrappers currently fail explicitly.

[Reviewed component receipt](../../evidence/distiller-reading-fidelity/receipt.json). Raw final run: `experiments/runs/distiller-reading-fidelity/20261013T034735.061342Z` contains commands, terminal records, source/tool/artifact digests, shared assets and EPUBCheck output. Original PDF SHA256: `68735f1b5a3d30fb276e42493389d016d4afb46fef922b44ed7ebf7ad55189d3`; original input remains unchanged. No human architecture-promotion decision is implied.

## Reader inspection and replacement covers

Calibre E-book viewer 9.16.0 was operated through its native UI. Article navigation worked, serif prose and hierarchy were legible, and the augmented-LLM and prompt-chaining diagrams fitted the page without clipping. PDF checks followed the recovered Intelligence Explosion chapter, the source-outline Introduction link to source page 1, and source page10 with its reconstructed drop cap and raster figure. Page47/48 showed the disclosed interruption by margin notes. Small text inside PDF diagrams requires zoom. Repeated heading text, printed hyphen splits and note/page interruptions remain visible; the PDF is not yet a polished semantic reflow.

Both books now carry original generated covers, replaced at the user's request with minimal, high-contrast e-ink-oriented designs. Their exact PNG bytes and EPUB cover-image metadata were checked, both replacements were viewed in Calibre, and both covered deliverables passed EPUBCheck with zero errors/warnings. macOS Books import/library display and physical Kindle hardware were not tested. [Generated-cover prompts and edit provenance](/Users/rohit/.codex/visualizations/2026/10/12/01a12e31-44b0-78c2-9ebc-d2dc42b645d3/distiller/cover-prompts.json). Artwork uses the built-in image generator; CLI generation remains Target. `--cover PATH` embeds a supplied PNG/JPEG for HTML/stdin/PDF EPUBs and applies the existing validation and atomic publication rules.

Pandoc 3.12.1's cover metadata path was not preloaded by its sandbox. Supplying the same validated owned image as reference-doc explicitly preloads only that file; independent probes confirmed an unrelated file remains blocked. No unsandboxed writer, Lua filter or ZIP editing layer was added. The workaround's owner and removal condition are documented in the extension architecture. Source: [Pandoc sandbox preload contract](https://raw.githubusercontent.com/jgm/pandoc/3.12.1/src/Text/Pandoc/App/OutputSettings.hs).

New public regressions cover curly/TeX quote mismatches, wrapped heading hyphens, unchanged body punctuation/internal hyphens, preserved unmatched bookmark links, valid cover metadata/bytes, corrupt/missing/nonregular/unsupported cover rejection, output-mode conflicts and duplicate XHTML IDs. The heading regression failed against the prior installed binary. The duplicate-ID sensor also fails on the preserved invalid development EPUB and passes on the corrected book. Final 54 Rust tests, formatting, Clippy with warnings denied, and both installed CLI journeys passed. Development runs with the temporary ID collision are retained and excluded from final conclusions. Source-fidelity pipelines run without generated covers; covers are validated separately as added artwork.
