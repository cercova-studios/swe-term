# Paper synthesis: evidence-gated lifecycle claims

Status: complete

## Decision and local failure

Before adding an agent loop, determine whether swe-term can fail closed when a
lifecycle claim relies on a receipt whose relevant verification inputs have
changed. The decision affects target `VerificationReceipt` and `Obligation`
contracts plus lifecycle invariant 6 in root `ARCHITECTURE.md`.

## Evidence map

| Mechanism | Foundational anchor | Direct proposal | Evaluation challenge | Independent support | Remaining unknown |
|---|---|---|---|---|---|
| source-bound receipt gate | software verification and provenance practice | [Proof-or-Stop](notes/2607.14890.md) | paper is a v1 preprint with model/corpus limits | none established in this packet | which fields fully capture swe-term's verification scope |

## Convergence and disagreement

The mechanism converges with the repository's existing requirement that a
`verified`, `done`, or `ready_to_merge` claim needs a current receipt. The paper
offers a concrete transfer hypothesis, not independent confirmation. Its
self-hosted and one-model-family evaluation is deliberately not used as a
general effectiveness claim.

## Mechanisms selected

| Mechanism | Paper note | Local seam | Why now | Cheapest falsification | Final disposition |
|---|---|---|---|---|---|
| sealed receipt identity compared to current obligation identity | [Proof-or-Stop](notes/2607.14890.md) | target receipt and obligation state | deterministic and upstream of any agent loop | replayed trace suite with changed identity fields | experiment-candidate |

## Rejected and deferred paths

| Candidate or mechanism | Disposition | Reason | Reopen condition |
|---|---|---|---|
| unattended autonomous loop | reject | confounds receipt gating with model, review, and orchestration quality | receipt reducer passes and a later agent-loop experiment isolates a remaining gap |
| persistent receipt store | defer | persistence is not needed to falsify identity invalidation | trace semantics are stable and replay durability becomes the next bottleneck |
| temporal multi-rule monitor | defer | broader closed-event monitor belongs to Experiment 3 | receipt traces establish the minimum event vocabulary |

## Experiment handoff

The proposed experiment is
[`experiments/specs/evidence-gated-lifecycle/`](../../../../experiments/specs/evidence-gated-lifecycle/).

- exact claim: lifecycle transitions require fresh, tracked-source-state-bound,
  mechanically verifiable evidence;
- boundary conditions: the paper's evaluation is one model family, 24 ablation
  tasks, and a self-hosted corpus; the transfer is an inference that holds only
  if every relevant verification input is represented in the identity and
  refreshed, and it cannot establish semantic correctness or scoped test
  selection;
- local hypothesis: identity binding eliminates false promotion for the frozen
  trace corpus;
- null: exact identity binding either fails to distinguish stale evidence from
  current evidence or rejects an unchanged verification scope unpredictably;
- independent variable: result-only control versus identity-bound receipt gate;
- fixture shape: deterministic receipt and identity transitions;
- evaluator: closed Go trace assertions with explicit rule IDs;
- safety counter: false lifecycle promotions;
- resources: local Go process, network disabled, one repetition; and
- excluded machinery: model, reviewer, background loop, persistence, database,
  and network.

## Research-taste update

High-signal research exposes an enforceable transition boundary, a failure
taxonomy, and a way to falsify the mechanism locally. Low-signal research would
ask us to recreate the paper's whole loop merely to repeat its headline metric.

## Search closure

Search is closed until the deterministic traces reveal a missing identity field,
an ambiguity in scope, or an evaluator failure. Any of those outcomes should
reopen discovery around that concrete gap.
