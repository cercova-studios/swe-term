# Hugging Face Papers discovery

This is the upstream research contract for paper-backed harness experiments. It
turns a broad topic into a small, auditable set of mechanisms worth testing. It
does not automate taste by ranking papers; it makes the judgments that constitute
taste visible enough to review, reproduce, and improve.

Use this workflow before adding a paper-derived hypothesis to
[`experiments/`](../../../experiments/). The output is a discovery packet, not an
experiment result or an architectural decision.

## Boundary

This first version covers discovery through Hugging Face Papers and the original
paper when needed. It does not crawl arbitrary blogs, social media, citation
graphs, or private corpora. Those sources may suggest a query, but they do not
replace the paper as the evidence source.

The workflow optimizes for three outcomes:

1. find a mechanism that addresses an observed local failure;
2. understand what evidence does and does not support it; and
3. formulate the cheapest experiment capable of falsifying its transfer to
   swe-term.

## Discovery packet

Create one directory per research question:

```text
docs/research/papers/<topic>/
  search-plan.md
  notes/
    <arxiv-id>.md
  synthesis.md
```

Start from:

- [`templates/SEARCH_PLAN.md`](templates/SEARCH_PLAN.md)
- [`templates/PAPER_NOTE.md`](templates/PAPER_NOTE.md)
- [`templates/SYNTHESIS.md`](templates/SYNTHESIS.md)

Keep rejected and deferred candidates in the packet. The negative trail is part
of the evidence: it shows which attractive ideas failed the transfer test and
prevents a later agent from repeating the same search without learning from it.

## The funnel

### 1. Anchor the question in repository reality

Read root `ARCHITECTURE.md` and the source or traces behind the suspected
failure. State:

- the observed failure or unresolved design decision;
- the affected domain type, invariant, or extension point;
- what is implemented versus only targeted;
- the runtime, cost, access, and safety constraints; and
- the smallest decision the research needs to inform.

Do not start with a field label such as “multi-agent systems” or “agent memory.”
Start with a failure such as “a stale verification result can still authorize a
completion claim” or “compaction can silently discard an active constraint.”

### 2. Build a mechanism inventory

Translate the failure into mechanisms and synonyms before searching. Search
across the parts of the system, not just the product vocabulary.

| Local concern | Example mechanism phrases |
|---|---|
| lifecycle claims | evidence gate, proof-carrying state, verification receipt, stale evidence |
| unsafe sequences | temporal monitor, runtime enforcement, event ordering, policy automaton |
| task continuity | external task state, verified memory, checkpointed state, provenance |
| context loss | memory consolidation, protected facts, faithful compression, state preservation |
| tool usability | agent-computer interface, structured feedback, action schema, diagnostics |
| evaluation | evaluator validity, judge disagreement, process benchmark, side-effect metric |

Each mechanism becomes a query family. Add failure terms, evaluation terms, and
known synonyms. A query family is more useful than many cosmetic rewrites of one
phrase.

### 3. Search Hugging Face Papers in layers

Use the configured Hugging Face Papers capability when available. The official
fallback path is:

1. search with
   `GET https://huggingface.co/api/papers/search?q=<query>&limit=<n>`;
2. fetch a candidate as machine-readable Markdown at
   `https://huggingface.co/papers/<arxiv-id>.md`;
3. fetch `https://huggingface.co/api/papers/<arxiv-id>` when structured authors,
   linked code, datasets, models, Spaces, or project metadata matter; and
4. fall back to the arXiv abstract, HTML, or PDF when the Hugging Face page is
   missing or insufficient.

Record the query, date, source path, result count when available, and any
fallback. If an advertised connector returns an unavailable-tool or not-found
error, record that operational fact and continue through the official endpoint.
Do not silently switch to a third-party summary.

Hugging Face recency, trending position, and upvotes are discovery priors only.
They are not evidence quality signals.

### 4. Screen title and abstract for mechanism fit

Advance a candidate to deep reading only if the abstract exposes a plausible
causal mechanism or a useful evaluation warning. Reject or defer candidates that
are merely adjacent by vocabulary.

Fast-screen questions:

- What concrete harness behavior changes?
- Where would enforcement or state live?
- Can the mechanism be separated from a larger model, dataset, or topology?
- Does the paper evaluate the claimed mechanism, or only a complete system?
- Is there a locally observable failure it could improve?
- Could a constrained experiment distinguish the mechanism from its null?

Typical noise:

- a headline benchmark gain with no isolatable mechanism;
- “multi-agent” as a topology rather than an explanation;
- a new framework whose relevant idea is ordinary orchestration;
- a memory paper evaluated only by answer similarity when state fidelity matters;
- a safety claim judged solely by the model being constrained;
- code availability presented as independent validation; or
- a paper selected because its terminology already matches our design.

### 5. Deep-read for discriminating evidence

Read the method, evaluation design, ablations, failure analysis, and limitations.
The abstract is an index, not the evidence. Extract:

1. **Claim** — the narrowest result the evidence supports.
2. **Mechanism** — what changes between treatment and control.
3. **Enforcement point** — prompt, tool interface, state transition, runtime
   monitor, evaluator, or another boundary.
4. **Evidence** — comparator, fixtures or benchmark, repetitions, metrics,
   ablations, and discordant cases.
5. **Evaluator trust** — whether measurement is deterministic, human, model
   judged, or circularly produced by the system under test.
6. **Boundary conditions** — assumptions, scale, model dependence, and known
   failure modes.
7. **Transfer** — the swe-term failure, invariant, type, or extension point the
   mechanism might affect.
8. **What not to copy** — topology, infrastructure, benchmark-specific machinery,
   or unsupported policy language that is not required by the mechanism.
9. **Cheapest falsification** — the smallest paired or deterministic experiment
   that could show the transfer does not help.

Prefer papers that make their mechanism legible over papers that merely make
their result impressive. Negative results, ablations, and precise failure
taxonomies often contain more transferable information than the top-line score.

### 6. Apply admission gates before qualitative ratings

A paper may become an experiment candidate only when all four gates pass:

- **Local fit:** it maps to an observed failure or an explicit architecture
  decision, not a generic aspiration.
- **Mechanism isolation:** the proposed causal change is separable from the
  paper's full system and model capability.
- **Falsifiability:** a bounded local experiment and null hypothesis can be
  stated before implementation.
- **Evidence legibility:** the supporting evaluation and its limitations can be
  described without relying on the authors' headline.

After the gates, record `high`, `medium`, `low`, or `unknown` for each dimension:

| Dimension | Question |
|---|---|
| Transfer relevance | How directly does this mechanism address the local failure? |
| Evidence strength | Do controls, ablations, and cases discriminate the claimed cause? |
| Evaluator validity | Is the measurement independent and appropriate to the claim? |
| Reproduction feasibility | Can the mechanism be tested with available models, fixtures, and compute? |
| Operational fit | Can it respect local confinement, cost, memory, and observability constraints? |
| Independent support | Do unrelated papers or established methods converge on the mechanism? |

Do not add the ratings into a total. A high average can conceal a fatal evaluator
or falsifiability problem. Unknown remains unknown until investigated.

### 7. Triangulate mechanisms, not citations

Group candidates by mechanism and compare:

- a foundational or established anchor, when one exists;
- the newest direct proposal;
- an evaluation or critique that could falsify the measurement; and
- an independent implementation or result, when available.

Check whether apparently independent papers share authors, datasets, model
families, judges, or benchmark assumptions. Several papers can be one evidence
line wearing different titles.

Treat a recent preprint as a hypothesis source. Linked code improves
inspectability, not truth. Convergence across independent mechanisms and
measurements is stronger than repeated citation or engagement.

### 8. Make an explicit selection decision

Every deep-read paper receives one final disposition:

- `experiment-candidate` — passes all gates and supplies a bounded test;
- `reference-only` — useful mental model or evaluation warning, but not an
  experiment driver;
- `defer` — promising, but blocked by unavailable compute, artifacts, access, or
  an unresolved prerequisite; or
- `reject` — mechanism mismatch, inseparable treatment, invalid evaluator, or no
  decision-relevant test.

Write the reason in one or two sentences. “Interesting” is not a disposition.

### 9. Hand the mechanism to the experiment contract

Discovery does not authorize a run. For an `experiment-candidate`:

1. link the discovery packet from the experiment specification;
2. copy the exact paper claim into `manifest.json` rather than its headline;
3. state the local hypothesis, null, and one independent variable;
4. preserve the paper note's boundary conditions and “what not to copy” section;
5. freeze fixtures, identities, metrics, evaluator, and budgets; and
6. pass `just experiment-ready <id>` before producing evidence.

If the experiment cannot preserve the isolated mechanism, return the paper to
`defer` rather than quietly testing a different claim.

## Stop conditions

Stop searching when all are true:

- every mechanism family has at least one screened candidate or a documented
  no-result;
- the leading candidate has an evaluation challenge or independent comparison;
- new queries produce duplicates or vocabulary-adjacent papers rather than new
  mechanisms; and
- one or more bounded experiments can now change a concrete decision.

Do not keep searching to make a bibliography look comprehensive. Reopen search
when an experiment produces a surprise, an evaluator proves invalid, a required
artifact appears, or the local architecture question changes.

## Worked example: the first harness portfolio

The initial swe-term paper pass started from six local failure classes, searched
for mechanisms, and then reduced the results to experiments. It did not attempt
to reproduce paper leaderboards.

| Candidate | Extracted mechanism | Disposition and rationale |
|---|---|---|
| [SWE-agent](https://huggingface.co/papers/2405.15793) | agent-computer interface design | `reference-only` anchor for treating tool interfaces as an experimental variable |
| [Structured Feedback Improves Repair in an LLM Agent Loop](https://huggingface.co/papers/2607.14167) | typed verifier feedback | `experiment-candidate`; separable paired treatment with a local repair corpus |
| [Proof-or-Stop](https://huggingface.co/papers/2607.14890) | evidence-gated lifecycle claims | `experiment-candidate`; reducible to deterministic source-bound receipt traces |
| [Enforcing Temporal Constraints for LLM Agents](https://huggingface.co/papers/2512.23738) | deterministic temporal monitoring | `experiment-candidate`; testable as a closed event reducer without an agent loop |
| [LongHorizon-Harness](https://huggingface.co/papers/2608.01964) | externally verified task state | `experiment-candidate`; topology can be excluded while state transitions are isolated |
| [TRUSTMEM](https://huggingface.co/papers/2606.25161) | verified memory consolidation | `experiment-candidate`; translates into protected-state preservation fixtures |
| [Benchmarking the Benchmarks](https://huggingface.co/papers/2607.02577) | evaluator disagreement and validity | `experiment-candidate`; calibrates measurements used by the other experiments |
| [AutoSaddler](https://huggingface.co/papers/2608.23041) | automatic harness optimization | `defer`; self-modification is premature before deterministic gates and trustworthy evaluators exist |

The resulting portfolio is tracked in
[`docs/plans/2026-08-27-harness-hypothesis-experiments.md`](../../plans/2026-08-27-harness-hypothesis-experiments.md).
The important artifact is the chain from local failure to mechanism to evidence
to falsifiable test—not the number of papers collected.

## Automation boundary

Phase 0 is intentionally structured Markdown. Once at least two discovery
packets expose stable repeated fields, a later tool may automate query logging,
metadata retrieval, duplicate detection, packet creation, and stale-source
checks. It must not automatically:

- convert engagement or recency into evidence quality;
- accept a paper into the experiment portfolio;
- summarize away limitations or discordant evidence;
- invent a local transfer mechanism; or
- promote an experimental result into architecture.

Those are judgment points. Automation should surface their inputs and preserve
their audit trail, not impersonate research taste.
