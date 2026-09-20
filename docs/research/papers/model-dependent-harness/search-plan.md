# Paper search plan: model-dependent harness mechanisms

Status: complete

## Decision to inform

Turn the remaining model-dependent harness hypotheses into bounded local
experiments without confusing a paper's full system result with a local causal
claim.

## Local anchor

The root architecture's target state, compaction, and evaluation contracts need
evidence before implementation. The portfolio in
`docs/plans/2026-08-27-harness-hypothesis-experiments.md` identifies four
separable questions: verifier feedback, externally verified task state,
protected compaction, and evaluator validity.

## Mechanism inventory

| Mechanism family | Synonyms and failure terms | Why it could affect the local failure |
|---|---|---|
| verifier feedback design | structured diagnostics, agent-computer interface, repair feedback | the shape of a failed check's output may change how often the next edit is invalid |
| externally verified task state | task snapshot, state externalization, attributable facts | stale actions after context pressure may come from state living only in the transcript |
| protected compaction | memory consolidation, preservation, faithfulness, superseded facts | free-form summaries may drop active constraints, approvals, and obligations |
| evaluator validity | judge disagreement, judge variance, process-versus-outcome scoring | a single aggregate score may hide safety-relevant disagreement |

## Query log

| Date | Query or candidate | Source or endpoint | Result count | Fallback or error | Notes |
|---|---|---|---:|---|---|
| 2026-08-31 | Structured Feedback Improves Repair in an LLM Agent Loop (`2607.14167`) | Hugging Face paper page | 1 direct candidate | none | verifier feedback family |
| 2026-08-31 | LongHorizon-Harness (`2608.01964`) | Hugging Face paper page | 1 direct candidate | none | external task state family |
| 2026-08-31 | TRUSTMEM (`2606.25161`) | Hugging Face paper page | 1 direct candidate | none | protected compaction family |
| 2026-08-31 | Benchmarking the Benchmarks (`2607.02577`) | Hugging Face paper page | 1 direct candidate | none | evaluator validity family |
| 2026-08-31 | AgentRewardBench (`2504.08942`) | Hugging Face paper page | 1 direct candidate | none | evaluator validity family; comparison point, not a mechanism |

## Candidate ledger

Ratings are abstract-level reads recorded in `notes/`; `unknown` means the
deep read did not establish the dimension, not that it is weak.

| Paper | Mechanism | Local fit | Evidence | Evaluator | Reproduction | Operational fit | Independent support | Screen decision |
|---|---|---|---|---|---|---|---|---|
| Structured Feedback Improves Repair (`2607.14167`) | information design of verifier feedback | high | medium | medium | medium | high | unknown | deep-read → [note](notes/2607.14167.md) |
| LongHorizon-Harness (`2608.01964`) | state updated only from attributable facts | high | unknown | unknown | medium | medium | unknown | deep-read → [note](notes/2608.01964.md) |
| TRUSTMEM (`2606.25161`) | memory-transition verifier (coverage, preservation, faithfulness) | high | unknown | unknown | medium | medium | unknown | deep-read → [note](notes/2606.25161.md) |
| Benchmarking the Benchmarks (`2607.02577`) | decomposed evaluator signals and disagreement audit | high | medium | medium | medium | high | low | deep-read → [note](notes/2607.02577.md) |
| AgentRewardBench (`2504.08942`) | judge-versus-human agreement benchmark | medium | unknown | unknown | low | low | unknown | watch (reference-only evaluator comparison; no note) |

## Admission boundary

Each candidate passes local fit, mechanism isolation, falsifiability, and
evidence-legibility only as a draft. None is preregistered: a named model,
version, parameters, prompt/tool-schema digests, frozen fixtures, and an
evaluator identity still need explicit selection. Selecting those is a material
experimental choice, not infrastructure work an agent should invent.

## Exclusions

Do not import a manager/executor/auditor topology, train a memory model,
automatically improve a harness, or use an LLM judge as its own sole authority.

## Search stop conditions

- [x] Each mechanism family has a screened candidate or documented no-result.
- [ ] The leading mechanism has an evaluation challenge or independent
      comparison. Only the evaluator-validity family has one (AgentRewardBench
      as a comparison point); the other three rest on a single preprint each.
- [ ] New queries mostly produce duplicates or vocabulary-adjacent results.
      Not established: one query per family was run and no follow-up queries
      were recorded, so saturation is unknown.
- [x] At least one bounded experiment could now change the stated decision.

The search stopped because each family has a candidate reducible to a single
isolated variable, not because the literature is exhausted. Reopen it for a
family if its experiment cannot be preregistered without a model choice the
candidate does not constrain, or if a discordant result points at a mechanism
the candidate does not name.

## Open uncertainties

- Independent support for the three non-evaluator families is unknown; no
  replication or comparison paper was screened.
- Every candidate's evidence is model- and benchmark-bound; transfer to Go
  repair tasks and swe-term traces is an inference until run locally.
