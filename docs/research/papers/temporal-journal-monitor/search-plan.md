# Paper search plan: temporal control-journal monitor

Status: complete

## Decision to inform

Determine whether swe-term can enforce its first safety-ordering rules with a
small closed-event monitor before introducing a durable journal, a tool loop, or
a policy language.

## Local anchor

- Observed failure or unresolved question: a future tool loop could mutate
  before approval, hold competing mutation leases, observe undeclared effects,
  or make a lifecycle claim from stale evidence.
- Relevant source, trace, or evidence: root `ARCHITECTURE.md` §5 names target
  `ControlEvent`, `ControlJournal`, and `MutationLease`; §10 invariants 2–4 and
  6 require mechanical enforcement; §11 requires table-driven trace tests.
- Architecture seam: target control and safety model, plus state expansion in
  §12. This is an experimental reducer only; it does not change the stated
  implementation status of those contracts.
- Implemented versus target: provider streaming is implemented. Tool safety,
  session persistence, control journal, and lifecycle hooks are target work.
- Constraints: local Go tests, synthetic events, no model, network, database,
  container, or workspace mutation.

## Mechanism inventory

| Mechanism family | Synonyms and failure terms | Why it could affect the local failure |
|---|---|---|
| temporal runtime enforcement | temporal constraint, sequence monitor, safety automaton, event ordering | makes illegal action sequences reject at the state-transition boundary |
| least privilege | capability, approval, mutation lease, declared effects | keeps an action inside an explicit authorization and effect envelope |
| formal policy languages | DSL, SMT, constrained generation | useful comparison, but excessive for four fixed repository invariants |

## Query log

| Date | Query or candidate | Source or endpoint | Result count | Fallback or error | Notes |
|---|---|---:|---:|---|---|
| 2026-08-31 | Enforcing Temporal Constraints for LLM Agents (`2512.23738`) | Hugging Face paper page | 1 direct candidate | none | direct source for temporal ordering enforcement |
| 2026-08-31 | Progent (`2504.11703`) | Hugging Face paper page | 1 direct candidate | none | comparison for deterministic privilege enforcement |
| 2026-08-31 | AgentSpec (`2503.18666`) | Hugging Face paper page | 1 direct candidate | none | comparison for runtime constraint specifications |

## Candidate ledger

| Paper | Mechanism | Local fit | Evidence | Evaluator | Reproduction | Operational fit | Independent support | Screen decision |
|---|---|---|---|---|---|---|---|---|
| Enforcing Temporal Constraints for LLM Agents | formal temporal enforcement during generation | high | medium | medium | medium | medium | low | deep-read |
| Progent | deterministic least-privilege tool policy | high | medium | medium | medium | medium | medium | deep-read |
| AgentSpec | customizable runtime constraint enforcement | high | medium | medium | medium | medium | medium | deep-read |

## Exclusions

- Do not import a general-purpose DSL, SMT solver, constrained decoding, or
  model-authored policy generator to test four fixed rules.
- Do not treat paper-reported safety numbers as a guarantee for swe-term.
- Do not execute tools or persist events in this experiment.

## Search closure

The sources converge on runtime enforcement, but their policy-language and
generation-time machinery is not needed to falsify the local claim. Reopen only
if a closed reducer misses a required rule, cannot replay deterministically, or
needs unbounded state.
