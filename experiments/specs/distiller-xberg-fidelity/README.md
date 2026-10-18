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

Run `just experiment-digest distiller-xberg-fidelity`, then validate and ready.
After ready passes, `python3 experiments/specs/distiller-xberg-fidelity/run.py`
runs the two installed CLIs against a loopback server in a fresh temporary
directory with scrubbed PATH/HOME/TMPDIR. One warmup and five measured repetitions
per fixture and variant alternate execution order. No cache or OCR in xberg;
no user configuration discovery. CLI flags are frozen in the manifest.

The runner records binary, runner, manifest, source and artifact digests,
timestamps, arguments, terminal states, wall times and macOS time -l RSS.
Raw runs remain ignored under experiments/runs/distiller-xberg-fidelity.
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

## Aborted run

20261012T042716.741020Z aborted at article-xberg-0 (exit 1): crawlberg
ssrf_policy_violation: loopback. The raw directory retains terminal states,
outputs, manifest and runner. There is no paired verdict. The revised independent
preregistration is ../distiller-xberg-fidelity-file/README.md.
