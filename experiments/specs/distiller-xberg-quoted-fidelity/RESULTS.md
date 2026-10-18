# Deterministic extraction comparison

Distiller preserves more technical structure on these fixtures than xberg-cli
1.3.7. Both retain all seven article code blocks exactly. Xberg flattens MathML
and superscripts/subscripts into plain text; distiller retains semantic markup.
This is a fixture-scoped result, not a general extractor ranking.

## Setup

Final installed distiller build, no --llm; xberg 1.3.7 with config discovery,
OCR and cache disabled, explicit text/html and Markdown output.
Frozen PurpleSyringa article plus one authored technical fixture, one warmup
and five measurements each, alternating order, scrubbed process environment.
Distiller uses loopback HTTP; xberg reads identical bytes as a local file.
The actual article URL was separately retried successfully without an LLM.

## Component results

| Article component | distiller | xberg |
|---|---:|---:|
| Exact source code blocks | 7/7 | 7/7 |
| MathML structures retained | 48/48 | 0/48 |
| Whole-page link destinations retained | 3/7 | 3/7 |
| Observer wall median | 20.56 ms | 40.60 ms |
| Observer wall range | 20.33–20.75 ms | 39.80–40.84 ms |
| Maximum process RSS | 15.34 MiB | 24.69 MiB |

Both retained the three article-body destinations. The four omitted destinations
are page navigation/profile links, so 3/7 does not mean body links were lost.
There are no tables in the article; table metrics are inapplicable there.

On the authored fixture both tools preserve its code block, both links, and
empty table cells. Distiller preserves its MathML and colspan/rowspan attributes;
xberg flattens both. Xberg's plain `x2` rendering removes the superscript
relationship; the article similarly renders powers such as `232` as plain text.
These are observed representation losses, not proof that every equation is wrong.

## Verification and retained evidence

52 Rust tests passed (39 unit, 13 integration). Four added public parse_html
journeys cover highlighted numeric code, exact whitespace and dynamic fences,
inline boundaries and code spaces, single/multiline annotations and MathML,
sanitizer behavior on real attributes versus literal code, empty/spanning/mixed
tables, and the frozen article's complete code/MathML. All four initial new
journeys failed against the old implementation.

Evaluator self-check detects indentation/tree/attribute loss, non-link URL
false positives and quoted fences without stripping literal code greater-than
signs. The process timeout kill path was exercised with a sleeping process.
Each benchmark run has binary/manifest/source/runner identities, timestamps,
commands, exit status, stderr, stdout and artifact hashes.

Final raw run (ignored, not promoted):
../../runs/distiller-xberg-quoted-fidelity/20261012T043411.486652Z/summary.json
and terminal.jsonl. The installed binary SHA-256 is in identity.json.

Live output: /tmp/distiller-vectorized-clz-and-ctz-fidelity.md.

## Corrections and limits

- First run aborted because xberg denies loopback URLs under its SSRF policy.
- File-input retry aborted when xberg auto-detected HTML containing MathML as
  XML. Explicit HTML MIME fixed acquisition; the source bytes stayed unchanged.
- First paired oracle undercounted xberg code as 4/7 by missing quoted fences.
  Manual inspection found intact code. This separate preregistration corrected
  that evaluator; the final count is 7/7. Earlier raw runs remain retained.
- These are regression fixtures inspected during development, not held-out data.
- Popen timeout polling adds observer timing overhead. /usr/bin/time reports
  roughly 0.01 s versus 0.02 s on the article, with coarse resolution. Do not infer
  a precise parser speed ratio from observer medians.
- RSS is macOS process maximum, not a hard memory limit or aggregate tree budget.
  Timeout is enforced; output bytes are checked after process completion.
- Math oracle compares tags/attributes/text with whitespace removed. It cannot
  prove semantic equivalence, especially whitespace-sensitive mtext or alternate
  math encodings. The Rust article journey additionally checks exact original
  serialized MathML expressions.
- Link oracle handles simple inline/raw-HTML destinations, not all Markdown.
- HTML/math rendering and complete XSS protection were not evaluated. Consumers
  need compatible rendering and their own rendering security boundary.
- Human review/acceptance is required for evidence promotion. No architecture
  promotion or commit/push was performed.
