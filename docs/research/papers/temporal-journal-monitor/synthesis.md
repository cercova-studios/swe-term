# Paper synthesis: temporal control-journal monitor

Status: complete

## Decision and local failure

Before any production tool loop, test whether a small deterministic reducer can
mechanically protect swe-term’s target ordering invariants. This advances the
experimental evidence for state expansion; it does not claim that a durable
control journal or lifecycle hook is now implemented.

## Evidence map

| Mechanism | Foundational anchor | Direct proposal | Evaluation challenge | Independent support | Remaining unknown |
|---|---|---|---|---|---|
| temporal event enforcement | none established in this packet | [Agent-C](notes/2512.23738.md) | claims depend on formal policy and application models | [Progent](notes/2504.11703.md), [AgentSpec](notes/2503.18666.md) support runtime enforcement generally | whether four fixed rules are sufficient and replay-safe locally |

## Convergence and disagreement

All three sources put enforcement outside the model. They differ in scope:
Agent-C uses formal temporal constraints at generation time; Progent and
AgentSpec expose programmable policy languages. The local experiment adopts
neither language nor generator because those surfaces would confound the four
repository invariants with policy-authoring quality.

## Mechanisms selected

| Mechanism | Paper note | Local seam | Why now | Cheapest falsification | Final disposition |
|---|---|---|---|---|---|
| closed event reducer | [Agent-C](notes/2512.23738.md) | target control journal and mutation lease | deterministic prerequisite for tool-loop safety | table-driven legal/illegal/replay traces | experiment-candidate |

## Rejected and deferred paths

| Candidate or mechanism | Disposition | Reason | Reopen condition |
|---|---|---|---|
| policy DSL and SMT solver | defer | unnecessary machinery for four fixed rules | a future invariant cannot be represented by the closed vocabulary |
| model-authored policies | reject | makes enforcement quality depend on the model | none before independently verified policy review exists |
| durable journal | defer | persistence does not change reducer falsification | reducer semantics and replay traces stabilize |

## Experiment handoff

[`experiments/specs/temporal-journal-monitor/`](../../../../experiments/specs/temporal-journal-monitor/)
freezes synthetic traces for approval, lease, effect declaration, receipt, and
lifecycle ordering. It has no model, network, tool execution, or persistence.
The safety-critical counter is any illegal trace accepted by treatment; it must
remain zero. Raw output is required before any architecture promotion.

## Research-taste update

The highest-signal commonality is not a product topology; it is an enforcement
boundary independent of model text. A good transplant removes machinery until
the failure remains observable. A broad policy framework is not evidence that a
new policy language is the next local need.

## Search closure

Search is closed until a trace reveals an omitted rule, an invalid evaluator, or
an unbounded-state requirement. Those outcomes would name the next query rather
than restart a generic search for agent safety papers.
