# Distiller, Indra and Marker PDF-to-EPUB benchmark

Status: preregistered descriptive benchmark. No architecture promotion.

## Question and frozen inputs

Does Distiller retain more source content in a readable EPUB than local Indra or Marker extraction plus Pandoc? Root `ARCHITECTURE.md` section 8 and the extension architecture define source-linked PDF reconstruction and EPUB packaging.

The independent variable is the conversion pipeline. The null is that a required component is lost, conversion fails, or reader acceptance is missing, preventing a superiority claim. No summed quality score decides a winner.

`fixtures/source.json` binds the fresh 18-page MRTA paper by SHA-256. The original stays in `~/papers`; the runner verifies and copies it into each run. The source reviewer inspected the PDF before seeing conversion outputs and froze 32 witnesses in `fixtures/oracle.json`. This paper has no standalone figures, so it cannot measure figure fidelity. All variants use a supplied neutral cover.

## Procedure and budgets

Run `just experiment-ready distiller-indra-marker-epub`, then inject only `OPENROUTER_API_KEY` through Doppler to `python3 run.py` in this directory. The runner creates an append-only directory under `experiments/runs/` and records identities, commands, terminal states, wall time and output hashes.

Three repetitions run in fixed order: Distiller, Indra, Marker. Distiller starts with an empty reconstruction cache each time. Local model files may remain warm; first-run downloads/setup are visible in logs and reported separately. Indra 0.3.2 uses automatic extraction, math enabled and four cores. Marker 2.0.0 uses its fast mode for CPU/MPS, with no optional remote LLM. Their native Markdown is retained, then packaged with the same Pandoc, title, language and cover.

Each command has a 1,200-second process-group timeout. Execution is sequential. Memory/CPU limits are observations, not enforced guarantees. All three repetitions and failures must be reported. Unknown model provenance, detached-server memory, cloud cost or incomplete terminals remain unknown. Dependencies live in a separate temporary virtual environment.

Competitor processes run from fresh output directories with an allowlisted environment and temporary HOME. They receive no credentials. Network is allowed for official weight downloads. Only Distiller receives the OpenRouter key and sends this public PDF through its existing configured reconstruction path.

## Evaluation and stop conditions

Inspect each frozen witness against source and output. Do not reward excerpts buried in noisy, out-of-order text. Equation structure, table relationships, source coverage, notes, headings, EPUBCheck and sampled reader behavior are separate dimensions. Original pixels preserve visual content but do not prove semantic accessibility. Record dispersion and every discordant case.

Stop each command at its declared deadline and retain failures. Do not modify Distiller, fixtures or scoring after benchmark results; further fixes require a new run identity. Raw results do not become curated evidence or an architectural claim without human review and promotion.

## Results and limitations

Completed raw runs: `20261018T203447Z` (initial setup) and `20261018T203858Z`
(warm dependencies), under the manifest's raw-run directory. Distiller failed
all three conversions at its 256 KiB model-input guard. Indra and Marker each
published three EPUBs; their native content was identical across repetitions.
Source review found mathematical and table errors in both. Indra EPUBs also
failed EPUBCheck with duplicate title IDs; Marker EPUBs passed EPUBCheck.

The completed warm run predates subsequent EPUB styling and mixed-header repairs.
Automatic approval review rejected an additional full rerun because it could send
the local PDF to OpenRouter. Keep the recorded binary identities; do not treat
these results as measurements of that later build. No evidence was promoted.

A single selected fresh paper cannot establish general parity or superiority.
Human interpretation is required before evidence promotion.
