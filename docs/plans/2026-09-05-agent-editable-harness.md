# Agent-editable harness

Status: accepted product direction; target behavior, not implemented. This
document does not authorize autonomous promotion, implementation, or experiment
execution. Root [ARCHITECTURE.md](../../ARCHITECTURE.md) remains authoritative.

## Positioning

swe-term is an agent-editable engineering environment where improvements are
demonstrated, versioned, and reversible. Its durable value is execution within
granted authority, continuity of work, and trustworthy evidence. Better models
should improve outcomes directly and allow obsolete scaffolding to be deleted.

The model may identify friction in its environment and propose changes to it.
An agent's explanation of its own limitations is a hypothesis: preserve a
reproducible case before attributing failure to the harness rather than missing
information, model error, or an impossible task.

## Stable contracts, replaceable strategies

| Stable responsibility | Replaceable strategy |
|---|---|
| Execute within granted authority | Planning and tool composition |
| Persist work and recover from interruption | Summarization and retrieval |
| Record source state and verification evidence | Proposed checks and feedback presentation |
| Enforce budgets and cancellation | Delegation and continuation tactics |
| Expose accurate tools and artifacts | Model-specific schemas and context helpers |

Do not prescribe planner/executor/reviewer pipelines or resident runtimes without
evidence of need. Specialized computation can be an optional sidecar; native
tools remain available. An agent may replace many calls with one program, but
the program inherits the same authority, effect, budget, and evidence contracts.

## Improvement lifecycle

```text
observe friction and preserve a reproducer
  -> propose a bounded candidate change
  -> test in an isolated copy under a frozen evaluation contract
  -> compare outcomes, failures, and costs
  -> accept, reject, or revise through the authorized promotion policy
  -> preserve evidence, version identity, and rollback
  -> re-evaluate usefulness when models or workloads change
```

Editing source and activating a candidate are separate operations. The active
harness and evaluation authority remain outside the candidate's write scope.
No candidate can grant itself permissions, increase its budget, change the
evaluator judging it, or relax acceptance criteria. Proposed changes to those
contracts require separate review; they cannot validate themselves.

Recursive improvement uses an explicit shared budget and bounded depth. A child
trial cannot reset those limits. Exhaustion, incomplete evidence, and failure
are recorded outcomes, not evidence of improvement.

## Scope and authority

| Scope | Target authority |
|---|---|
| Working scripts, tools, context helpers, optional computation | Agents may create and test within existing permissions, effects, and budgets. |
| Persistent skills, memory, compaction, retrieval, orchestration | Agents prepare versioned candidates and evidence; activation requires an explicit promotion decision. |
| Core loop, permissions, evaluator, evidence and promotion rules | Agents may propose patches in isolation; cannot authorize their own activation or weaken invariants. |

Initially, persistent promotion is human-approved. A future policy permitting
automatic promotion for a narrow class of extensions is a separate architecture
decision, not implied by acceptance of this direction. The existing
[experiment contract](../../experiments/README.md) continues to govern research
preregistration, confinement, evidence review, and human promotion. Do not add
a second experiment runner or bypass its gates. If a proposed trial does not
fit that contract, resolve the lifecycle gap explicitly before running it.

## Evidence and retirement

Every proposed intervention records the observed failure, source and model
identities, applicable tasks, expected benefit, added dependencies and costs,
independent disable path, owner, and removal condition. These are requirements
for a future change record, not a new manifest format or command surface.

Compare the smallest working baseline with the candidate using comparable
information, tools, authority, and budgets. Preserve failures and inconclusive
results. Candidate-authored tests may reproduce a bug but cannot be the sole
acceptance authority. Evaluation includes independent regressions and cases
outside the motivating example; repeated model runs report variability.

Assess correctness, unintended effects, latency, tokens, memory, and maintenance
separately. Do not trade away a safety-critical failure through an aggregate
score. A model release triggers re-evaluation of reasoning aids, not relaxation
of permissions or evidence requirements. Remove interventions that no longer
earn their cost; clean up configuration, prompts, dependencies, tests, and docs
rather than retain compatibility by default.

Memory stores attributable facts and scoped, tested procedures with freshness
and version identity. One successful trajectory does not establish a permanent
instruction. Durable assets include reproducers, verifiers, adapters, and
reviewed evidence, rather than ever-growing behavioral prompts.

## First delivery slice

Start with an agent-authored tool extension after the core tool loop, execution
boundary, session persistence, and extension manifest contracts exist. Advance
the root architecture's loop, tool safety, state, and protocol refactors;
do not replace them with a self-editing control plane.

Example: repeated truncation motivates a paginated search adapter. Demonstrate
that it exposes the needed evidence within output and resource bounds. A later
candidate may delete a summarizer that no longer helps a stronger model.

Acceptance requires observable journeys:

- Create a candidate extension in isolation without changing the active one.
- Compare it against a frozen baseline and retain attributable results.
- Reject attempts to change the evaluator, authority, or shared budget.
- Activate an approved version at a controlled boundary and record its identity.
- Recover from interruption with the previous active version intact or the new
  version completely activated; no ambiguous partial installation.
- Roll back without erasing evidence; resume sessions with their selected
  versions or explicitly refuse an incompatible version.
- Disable and remove an obsolete intervention without changing core semantics.

Use existing session, artifact, and journal contracts to expose candidate,
evaluation, decision, activation, and rollback state. Do not invent a separate
product CLI. Keep trials lazy, bounded, and out of process where needed; no
resident Python kernel, daemon fleet, or unbounded in-memory history is required
by this direction.

## Limits and tradeoffs

Even unlimited engineering cannot supply a complete verifier for arbitrary
harness changes. Model judgment is not proof, and reversible edits can still
cause irreversible external effects. Confinement and independently controlled
promotion remain necessary. Test coverage, trial breadth, and convenience of
automation are adjustable quality investments; authority and evidence integrity
are correctness constraints, not budget compromises.

The reference target retains a small Go core and cheap default runtime. Start
with extensions because their effects and rollback are easier to bound, then
expand scope only when evidence supports it. The success criterion is better
engineering outcomes with improving models and less unnecessary machinery.

## Related decisions

- [Delivery roadmap](../core/PLAN.md): core foundations, then extension pilot.
- [Framework synthesis](../core/FRAMEWORKS.md): DSH capability seams and Prime
  versioned harness state inform this direction; their runtime packaging is
  not a requirement.
- [Prime critique](../research/PRIME_AGENT_CRITIQUE.md): closed sidecar RPC,
  single-owner state, and rollback rather than IPython as the default loop.
- [DSH critique](../research/DEEPSEEK_HARNESS_CRITIQUE.md): log-derived context
  and named extension phases rather than in-process model-authored eval.
