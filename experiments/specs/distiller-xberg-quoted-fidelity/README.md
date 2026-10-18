# distiller versus xberg fidelity

Status: preregistered benchmark. See manifest.json for the frozen hypothesis,
budgets, variants, evaluator rubric and directory digest.

The measured seam is deterministic URL extraction in root ARCHITECTURE.md §9.
The sole independent variable is the installed CLI implementation. No LLM runs.

## Frozen inputs and procedure

article.html is the original HTML fetched on 2026-10-12 from
https://purplesyringa.moe/blog/vectorized-clz-and-ctz/ (PurpleSyringa).
technical.html is an authored regression fixture, not a held-out evaluation.
Source provenance is also recorded in fixtures/README.md.

Run `just experiment-digest distiller-xberg-quoted-fidelity`, then validate and ready.
After ready passes, `python3 experiments/specs/distiller-xberg-quoted-fidelity/run.py`
runs the two installed CLIs against a loopback server in a fresh temporary
directory with scrubbed PATH/HOME/TMPDIR. One warmup and five measured repetitions
per fixture and variant alternate execution order. No cache or OCR in xberg;
no user configuration discovery. CLI flags are frozen in the manifest.

The runner records binary, runner, manifest, source and artifact digests,
timestamps, arguments, terminal states, wall times and macOS time -l RSS.
Raw runs remain ignored under experiments/runs/distiller-xberg-quoted-fidelity.
Temporary fixture copies and HOME are deleted after execution; failed raw runs
remain available for diagnosis. Timeout is enforced; output size is checked after
termination. RSS and CPU are observations without hard resource limits.
Network confinement is procedural: only the loopback URLs are supplied; this
process adapter does not enforce a network sandbox.

## Evaluator and acceptance

Independent Python HTMLParser gathers source pre text, MathML tag/attribute/text trees
and non-fragment link targets. Markdown fences must retain exact code content,
except for one required terminal newline. Math comparison measures structure
retention, not equivalence of possible alternate LaTeX representations.
Table checks apply to the authored fixture only. No opaque combined score.

A fidelity winner must dominate every measured fidelity component with at least
one improvement. Otherwise report a tie or tradeoff. Timing cannot compensate
for content loss. Abort on nonzero exit, timeout, excess output or inconsistent
fidelity across repetitions. Evaluator self-check injects indentation/tree losses.

## Limits and promotion

This is a two-fixture comparison on one machine, with warm filesystem caches.
Loopback timings exclude WAN and page acquisition variability. Original source
links are resolved against the local server for both variants. Math renderer
support and alternate math equivalence require separate review.
Human acceptance is required before any evidence or architecture promotion.
No result is promoted automatically. Results belong in each raw run's summary.json.

## Revised input adapter

The original loopback-only specification aborted during the first xberg warmup:
xberg 1.3.7 rejects loopback URLs under its SSRF policy. No successful xberg output
was inspected before this revision. This new preregistration supplies xberg the
identical frozen HTML as a local file using its documented URI argument.
Distiller still fetches the loopback fixture. Timing therefore includes a small
HTTP acquisition cost for distiller; it is a CLI comparison, not pure parser time.
Relative links from xberg file mode may stay relative; the oracle resolves them
against the same synthetic source URL before comparison.

## Explicit HTML input

The preceding local-file run aborted in xberg's XML parser because automatic
MIME detection treated this HTML/MathML source as XML. The source is unchanged.
This preregistration sets the documented --mime-type text/html flag for xberg,
before inspecting any successful xberg output. Failed attempts remain retained.

## Evaluator correction and prior inspection

The preceding paired run exposed a false negative in the code oracle: three
xberg code blocks were intact inside blockquotes, but the oracle recognized only
top-level fences. The corrected oracle removes the enclosing quote prefix and
preserves literal code characters. Fixtures and variants are unchanged.
This repeat follows output inspection, not held-out validation.
Earlier raw results and evaluator versions remain untouched.

## Completed comparison

Final installed-build run: 20261012T043411.486652Z. All paired repetitions
completed. [RESULTS.md](RESULTS.md) records component results, failures,
evaluator correction and limitations. Distiller wins technical structure
retention on these fixtures; code preservation ties. Raw evidence remains
ignored and unpromoted.
