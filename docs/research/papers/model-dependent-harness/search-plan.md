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

## Query log

| Date | Candidate | Source | Decision |
|---|---|---|---|
| 2026-08-31 | Structured Feedback Improves Repair in an LLM Agent Loop (`2607.14167`) | Hugging Face paper page | experiment-candidate |
| 2026-08-31 | LongHorizon-Harness (`2608.01964`) | Hugging Face paper page | experiment-candidate |
| 2026-08-31 | TRUSTMEM (`2606.25161`) | Hugging Face paper page | experiment-candidate |
| 2026-08-31 | Benchmarking the Benchmarks (`2607.02577`) | Hugging Face paper page | experiment-candidate |
| 2026-08-31 | AgentRewardBench (`2504.08942`) | Hugging Face paper page | reference-only evaluator comparison |

## Admission boundary

Each candidate passes local fit, mechanism isolation, falsifiability, and
evidence-legibility only as a draft. None is preregistered: a named model,
version, parameters, prompt/tool-schema digests, frozen fixtures, and an
evaluator identity still need explicit selection. Selecting those is a material
experimental choice, not infrastructure work an agent should invent.

## Exclusions

Do not import a manager/executor/auditor topology, train a memory model,
automatically improve a harness, or use an LLM judge as its own sole authority.
