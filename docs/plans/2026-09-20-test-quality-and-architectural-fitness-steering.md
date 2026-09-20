# Steering agents away from brittle tests and compounding debt

Status: research + design proposal. Nothing here is promoted into
`ARCHITECTURE.md`; per [`experiments/README.md`](../../experiments/README.md)
these claims need preregistered evidence and explicit human acceptance first.
Section 8 proposes the experiments that would earn that. Section 5 answers the
build-vs-adopt question for Hegel, Bombadil, and semi-formal methods.

## 1. The two failure modes, stated precisely

**A. Brittle tests in brownfield.** An agent extends existing code and writes
tests that couple to *how* the code works (mocks, patched internals,
assertions on call counts and call order) rather than *what it does*
observably. These pass, add coverage, catch nothing, and break on every
refactor — so they simultaneously fail to protect the codebase and actively
tax future change.

**B. Compounding-debt blindness.** An agent produces a locally correct
solution with no visibility into what it costs to extend. The patch is
defensible in isolation and corrosive in aggregate: it widens a public
surface, deepens a dependency cycle, adds the third copy of a shape that
should have been factored, or couples two modules that then always change
together.

These look like separate problems. They are the same problem: **the harness
rewards a local, immediately-verifiable signal, and both "will this test
still be meaningful in six months" and "what will this cost to extend" are
non-local signals that nothing in the loop measures.** An agent optimizing
"make the checkmark green" writes the cheapest test that goes green — which
is the mock-heavy one — and stops thinking at the boundary of the current
task, because nothing asks it not to.

## 2. What the evidence actually says

Both failure modes are empirically documented, and — importantly — so is the
failure of the obvious fix.

- **Over-mocking is real and measured.** Hora and Robbes, *Are Coding Agents
  Generating Over-Mocked Tests? An Empirical Study*
  ([arXiv:2602.00409](https://arxiv.org/abs/2602.00409), MSR 2026), is the
  first study of mocks in agent-generated tests of real-world systems. Their
  own recommendation is to "include guidance on mocking practices in agent
  configuration files" — i.e. prompting. See the next bullet for why that is
  probably insufficient on its own.

- **Prompting agents about tests does not change outcomes.** *Rethinking the
  Value of Agent-Generated Tests for LLM-Based Software Engineering Agents*
  ([arXiv:2602.07900](https://arxiv.org/abs/2602.07900)) analyzed six strong
  LLMs on SWE-bench Verified and ran a prompt-intervention study revising
  prompts to increase or reduce test writing. Result: *"prompt-induced changes
  in the volume of agent-written tests do not significantly change final
  outcomes."* They also found resolved and unresolved tasks show similar
  test-writing frequencies, and that when agents do write tests, the tests
  *"mainly serve as observational feedback channels, with value-revealing
  print statements appearing much more often than assertion-based checks."*

  Two consequences for harness design. First, an instruction in `AGENTS.md`
  is the weakest available mechanism and should not be the plan. Second — and
  this is the more useful insight — **agents are frequently using "tests" as
  probes, not as specifications.** Those are different activities and the
  harness currently conflates them, so throwaway instrumentation gets
  committed wearing the costume of a test suite.

- **Coverage is the wrong gate; mutation score is better but is itself
  contested.** Reports of LLM suites reaching high line/branch coverage with
  very low mutation scores are common. But the honest counter-evidence:
  *Do Coverage and Mutation Scores of LLM-Generated Test Suites Correlate with
  Their Effectiveness? (Replicability Study)*
  ([arXiv:2607.22880](https://arxiv.org/abs/2607.22880)) finds that
  correlations among coverage, mutation score, and real-bug detection can
  **largely vanish once test-suite size is controlled**. Mutation score is a
  sharper proxy than coverage; it is not ground truth, and a harness that
  gates on it is choosing a proxy with known validity limits. This repo
  already has an open `evaluator-validity-audit` spec for exactly this class
  of concern, and that spec should cover any metric we gate on.

- **Industrial precedent exists.** *Mutation-Guided LLM-based Test Generation
  at Meta* ([arXiv:2501.12862](https://arxiv.org/pdf/2501.12862)) uses
  mutants to drive generation rather than to grade it after the fact.

### What Antithesis contributes

Antithesis' framing is the clearest articulation of *why* example-based tests
under-deliver, and two of their ideas transfer directly even though their
product does not.

- **The state-space framing.** From their docs: *"A piece of stateful software
  is essentially a giant state machine... software testing is about exploring
  a very large state space. An integration test threads a single path through
  this immense space."*
  ([How Antithesis works](https://antithesis.com/docs/properties_assertions/properties/))
  A brittle mock-heavy unit test is worse than a single path — it is a path
  through a *model of* the system the agent just wrote, not the system.

- **Sometimes assertions, and why coverage is structurally wrong.** Antithesis
  distinguishes *always* assertions (invariants) from *sometimes* assertions
  (this state is reachable at least once). Their argument against coverage is
  precise: coverage tools are *"equivalent to adding `assertSometimes(true)`
  to every line"*, and are both **not sensitive enough** (*"code coverage only
  covers locations, while sometimes assertions cover situations"*) and **too
  sensitive** (most lines don't matter, so percentages hide whether vital
  locations were reached).
  ([Sometimes Assertions](https://antithesis.com/docs/best_practices/sometimes_assertions/))

  For an agent harness this is the missing check on the most common failure:
  *a test that runs but never exercises the new behavior.* A sometimes-style
  reachability assertion makes "did your test actually reach the thing you
  changed" a machine-checkable question.

- **What does *not* transfer — stated plainly.** Antithesis' own docs say the
  FoundationDB approach (designing all nondeterministic components to be
  pluggable) is *"generally impractical for systems already in production"*,
  and the alternative is running under their deterministic hypervisor — a
  commercial product.
  ([DST docs](https://antithesis.com/docs/resources/deterministic_simulation_testing/))
  They also scope DST's sweet spot to distributed databases, transaction
  engines, consensus protocols, and microservice/async systems. **Recommending
  full DST as the answer to "my agent writes brittle tests in a brownfield
  repo" would be cargo-culting.** What transfers cheaply is the *discipline*:
  properties over examples, reachability over coverage, seeded and replayable
  randomness, and faults as first-class inputs.

### The harness-engineering frame (Thoughtworks)

Birgitta Böckeler's [*Harness engineering for coding agent users*](https://martinfowler.com/articles/harness-engineering.html)
(02 Apr 2026, on martinfowler.com) supplies better vocabulary than this
document originally used, and it is converging into industry-standard
terminology. `Agent = Model + Harness`; the outer harness is what *we* build.

Two axes:

|  | **Computational** (deterministic, CPU, ms–s, reliable) | **Inferential** (LLM/judge, slower, non-deterministic) |
|---|---|---|
| **Guides** (feedforward — steer *before* the agent acts) | code mods, LSP/code intelligence, structural context | `AGENTS.md`, skills, conventions |
| **Sensors** (feedback — observe *after*, enable self-correction) | tests, linters, type checkers, ArchUnit, mutation testing | AI code review, LLM-as-judge |

Her diagnostic is the sharpest thing in the article, and it indicts a very
common setup: *"you get either an agent that keeps repeating the same
mistakes (feedback-only) or an agent that encodes rules but never finds out
whether they worked (feed-forward-only)."*

She also names three **regulation categories**, which map almost exactly onto
this document's two failure modes:

- **Maintainability harness** — easiest; lots of existing tooling.
- **Architecture fitness harness** — *"Basically: Fitness Functions."* This is
  failure mode B.
- **Behaviour harness** — *"the elephant in the room."* This is failure mode A,
  and she is explicitly pessimistic: the common approach (trust the
  AI-generated suite, check coverage, *maybe* mutation testing) *"puts a lot
  of faith into the AI-generated tests, that's not good enough yet."*

**That is a direct challenge to §4's M3** and should be read as one: mutation
gating is the current state of the art for this and a practitioner with more
field exposure than this document has still calls it insufficient on its own.

Three further ideas worth importing:

- **Harnessability / ambient affordances** (Ned Letcher's term): *"structural
  properties of the environment itself that make it legible, navigable, and
  tractable to agents."* Strong typing, clear module boundaries, and
  frameworks all make a codebase more governable. And the line that states
  our brownfield problem exactly: **"the harness is most needed where it is
  hardest to build."**
- **Ashby's Law of Requisite Variety** — *"a regulator must have at least as
  much variety as the system it governs, and it can only regulate what it has
  a model of."* This reframes the Fleet CPG engine: it is not a
  context-retrieval optimisation, it is **the regulator's model of the
  system's structure.** Without it, no architecture-fitness harness is
  possible, because there is nothing to regulate *against*.
- **Keep quality left, and separate the two sensor timings.** Fast sensors
  pre-commit (linters, fast tests); expensive ones post-integration (mutation
  testing, broad review). And distinctly: **continuous drift sensors that run
  outside the change lifecycle entirely** (dead code, test-quality analysis,
  dependency scanning). This document originally treated everything as a
  per-change gate; that was wrong.

Her [follow-up on sensors](https://www.thoughtworks.com/en-de/insights/blog/generative-ai/harness-engineering-agent-feedback-exploring-ai-coding-sensors)
reports a TypeScript dashboard experiment running an agent with and without a
sensor suite (ESLint, Semgrep, Dependency Cruiser for module boundaries,
coverage + mutation testing): with sensors, *"it was able to improve quality
over time."* Her framing of the human's role is worth keeping: harness
engineering *"isn't about total automation; it's really about situational
awareness for the developer... humans should sit on top of a higher-abstraction
steering loop."*

Industrial corroboration she cites: an OpenAI team enforcing layered
architecture with custom linters and structural tests plus recurring "garbage
collection" scans for drift — their stated conclusion, *"our most difficult
challenges now center on designing environments, feedback loops, and control
systems"* — and Stripe's "minions" using pre-push hooks that select linters
heuristically, emphasising shift-feedback-left.

### Evolutionary architecture

Ford, Parsons, and Kua's *Building Evolutionary Architectures* supplies the
frame for failure mode B: a **fitness function** is any mechanism giving an
objective integrity assessment of some architectural characteristic, and they
categorize them atomic vs. holistic, triggered vs. continual, static vs.
dynamic. The load-bearing idea for our case is that fitness functions act as
**a ratchet on quality degradation** — which is the only form that works in
brownfield, where absolute thresholds fail on day one and get disabled.
([nealford.com](https://nealford.com/books/buildingevolutionaryarchitectures.html),
[ch. 2](https://www.oreilly.com/library/view/building-evolutionary-architectures/9781492097532/ch02.html))

## 3. The thesis

> Don't tell the agent to write better tests. Change what counts as done, and
> give the agent the non-local information *before* it decides.

swe-term does not need a new subsystem for this. It already has the organs;
they are undefined or unwired:

| Existing contract | Where | What's missing |
|---|---|---|
| `Obligation` — "required check with kind, risk policy, status, and **minimum V&V rung**" | §5 (Target) | **The rung ladder is never defined.** This is the single largest gap. |
| Invariant 7 — "obligation kind and risk determine a minimum V&V rung; a model may escalate but never downgrade it" | §10 | Enforcement exists (`control_monitor.go`), rungs don't |
| `VerificationReceipt` — fresh, source-bound evidence an obligation was discharged | §5 (Target) | Validated this session (`evidence-gated-lifecycle`); no test-quality obligations bind to it |
| Invariant 6 — lifecycle claims require a current receipt | §10 | Same |
| §11 — "user-journey tests... **assert observable behavior rather than implementation helpers**" | §11 | **The doctrine is already written. Nothing enforces it.** |
| Analyzer/enrichment adapters + Fleet CPG overlay | §9, Fleet CPG plan | Blast radius is computed but never used as a *debt* signal |
| Compaction preserves a protected spine of "active constraints" | §7 | Architectural constraints aren't in the spine, so they're forgotten mid-task |

The chain that closes the loop: **fitness functions are normally too expensive
to run per-change because they are whole-repo. The Fleet CPG overlay makes
them O(diff). O(diff) makes them gateable per-change. Gateable per-change
makes them an `Obligation`. An `Obligation` is something invariant 7 says the
model cannot downgrade.** That is the payoff from the nine CPG experiments
pointed at failure mode B.

## 4. Mechanisms, ordered by leverage per unit of cost

Ordered so that the cheap ones are useful alone and the expensive ones are
optional. Stop wherever the returns stop.

**Diagnostic first — and this is the most actionable finding in the whole
document.** Applying Böckeler's feedforward/feedback test to swe-term today:

| | Computational | Inferential |
|---|---|---|
| **Guides** | thin — no structural context is fed to the model yet (this is what the Fleet CPG `ContextPacket` would become) | heavy — `AGENTS.md`, `CLAUDE.md`, skills, `ARCHITECTURE.md` |
| **Sensors** | **almost nothing** — `go test`/`vet`, and the `.githooks/pre-commit` summary check added 2026-09-07 | none |

swe-term is **guide-heavy and sensor-poor** — squarely the
*"encodes rules but never finds out whether they worked"* failure. The repo
has an unusually rich set of written constraints (a 14-invariant architecture
contract, a preregistration framework, explicit doctrine in §11 about
asserting observable behaviour) and almost no mechanism that checks whether
any of it held. That imbalance, not a missing technique, is the gap.

Mapping the mechanisms below onto that frame:

| Mechanism | Böckeler category |
|---|---|
| M1 probes vs. specifications | guide (computational affordance) |
| M2 V&V rung ladder | neither — this is the *policy over* sensors, and is the piece her taxonomy doesn't name |
| M3 mutation-gated receipts | computational sensor, post-integration |
| M4 debt signals pre-decision | **computational guide** — the category swe-term is emptiest in |
| M5 protected spine | guide persistence under compaction |
| M6 typed feedback | sensor *output format* — she calls this "a positive kind of prompt injection" |
| M7 approved fixtures (new, below) | computational sensor + review affordance |

### M1. Separate *probes* from *specifications* (cheapest, highest ratio)

Because agents demonstrably use tests as observational feedback channels
(2602.07900), give that behavior a first-class home so it stops contaminating
the suite: a scratch probe affordance that is trivially cheap, explicitly
throwaway, and **never committed** — versus a committed test, which must meet
a rung. Most brittle committed tests are probes that were never deleted.

This costs a convention and a `.gitignore` line, and it removes a whole class
of brittle test by making the throwaway path *easier* than the committed one.

### M2. Define the V&V rung ladder — by what the evidence can catch

The rungs must be properties of the **evidence**, not of the filename. A
function in `*_test.go` is not automatically R1.

| Rung | Evidence | Catches | Blind to |
|---|---|---|---|
| R0 | Compiles / typechecks | Type errors | All behavior |
| R1 | Example-based assertion on **observable output or state at a declared boundary** | The case you thought of | Everything else |
| R2 | Property/invariant over **generated** inputs | Cases within the input domain you didn't think of | Out-of-domain, faults |
| R3 | R2 + **reachability evidence** that the changed code was actually exercised | "The test never touched the new path" | Failure paths |
| R4 | R3 + adversarial inputs / fault injection | Error-path and resource bugs | Nondeterminism |
| R5 | R4 + **seeded deterministic replay** of any counterexample | Flaky/heisenbugs; makes findings reproducible | — |

A mock-heavy interaction test that asserts call order is **not R1** — it
restates the implementation, so it sits below R1 with R0's blindness. Naming
that explicitly is what lets the monitor reject it for a high-risk obligation.

Risk → minimum rung is then policy (hand-authored, closed, per invariant 7 and
§6's "closed, hand-authored rule IDs"), and the agent physically cannot
discharge a high-risk obligation with a cheap brittle test.

### M3. Make the receipt bind to mutation-kill on the diff, not coverage

The metric you gate on is the metric the agent optimizes. Gate on coverage and
you get coverage-shaped tests; this is exactly how you get 100%-coverage /
near-zero-mutation suites. You cannot kill a mutant in the changed lines with
a test that only asserts that a mock was called.

Constraints that make this viable rather than aspirational:
- **Diff-scoped, not whole-repo.** Mutation testing is slow; per-PR viability
  requires scoping to changed lines. (Go options exist — `go-gremlins/gremlins`,
  `go-mutesting` forks — but **verify diff-scoping support and exact flags
  against the tool's current docs at implementation time**; I could not
  confirm them from a live source during this research and will not invent
  flag names.)
- **Ratchet, not threshold.** Brownfield absolute thresholds fail immediately.
  Gate on "mutation score of the diff does not regress against the touched
  files' recorded baseline." This matches Ford's ratchet framing and this
  org's own "no baseline, no credible improvement claim" doctrine.
- **Declare the proxy's limits.** Per 2607.22880, this is a proxy whose
  correlation with real-bug detection is contested once suite size is
  controlled. It should be registered as a measured evaluator under
  `evaluator-validity-audit`, not treated as ground truth.

### M4. Move the debt signal *before* the decision, into the ContextPacket

Failure mode B is an information problem at decision time, not a review
problem. An agent that is told *"this function has 47 callers across 12
modules, historically co-changes with 6 files, and this shape already appears
twice"* plans differently than one that learns it in review.

Deterministic, cheap, and already mostly built or planned:

| Signal | Source | Debt question it answers |
|---|---|---|
| Blast radius of touched symbols | Fleet CPG (built, validated) | How much already depends on this? |
| Change coupling from git history | One script over `git log` | What has historically been forced to change *with* this? |
| Duplication delta | `jscpd` (already in the org's own tooling list) | Is this the third copy? (rule-of-three, mechanically) |
| Public surface delta | CPG defs diff | Did this widen a surface we must now keep compatible? |
| Dependency-direction violation on the **proposed** graph | CPG overlay + layering rules | Does this create a cycle or cross a layer? |

The sharp part: **evaluate these against the CPG *overlay* — the proposed
post-change graph — not the committed one.** Phase 1–3 of the Fleet CPG work
built exactly that and proved it correct and O(diff). This is the highest-value
consumer of that engine identified so far.

### M5. Put architectural constraints in the protected spine

§7 says compaction preserves "active constraints." A fitness-function
violation or a layering rule is precisely the thing that gets evicted at
hour three and re-violated at hour four. There is already a
`protected-spine-compaction` spec (draft) — architectural constraints and
open fitness violations should be in its protected set.

### M6. Typed feedback when a fitness function fails

A failed gate that returns a wall of text gets worked around; one that returns
*failure location, observed value, admissible alternatives* gets fixed. This
is exactly the mechanism the repo's existing `structured-verifier-feedback`
spec (draft) was written to test — fitness-function output should be its
first consumer.

### M7. Approved fixtures — restructure the review, don't just grade the tests

A different attack on failure mode A, from Böckeler's colleagues and
documented as [Approved Fixtures](https://lexler.github.io/augmented-coding-patterns/patterns/approved-fixtures/):
design tests around approval files combining input and expected output in a
domain-specific, easy-to-validate format. Validate the *test execution logic*
once; after that, adding a case means reviewing a fixture, and the runner
regenerates approval files so verification is a **diff review**.

Why it matters here: M3 grades test quality after the fact. This instead
attacks the underlying economics — *reviewing many AI-generated tests with
complex assertions is impractical*, which is a large part of why brittle
tests survive review. Making the review cheap is a different lever than making
the tests better, and the two compose.

Böckeler's own caveat, kept: her colleagues *"use it selectively where it
fits, it's not a wholesale answer to the test quality problem"* — it works
best where there's an intuitive representation that's straightforward to
check. For swe-term specifically, the control-journal and receipt trace
corpora are close to an ideal fit (they already look like fixture tables).

## 5. Build vs. adopt: Hegel, Bombadil, and semi-formal methods

### 5.1 What these actually are

All three names are ambiguous and all three resolve to Antithesis. Stating the
referents precisely, because two have famous unrelated namesakes:

- **Semi-formal methods** — a *stance*, not a tool.
  [*The pragmatic magic of semi-formal methods*](https://antithesis.com/blog/2025/semi_formal_proofs/)
  (Katkoria & Moore, 12 May 2025). Full formal methods (TLA+, P) are the most
  rigorous verification available and impractical for most teams — their
  phrasing: *"it all starts to feel like you're getting a PhD when all you
  want to do is merge a pull request."* Semi-formal methods are the *"secret
  shop"* where you buy rigor **piecemeal** — types, invariants, properties,
  assertions — without adopting a separate specification language.

- **Hegel** — a family of **property-based testing libraries**, announced
  [24 Mar 2026](https://antithesis.com/blog/2026/hegel/) by **David MacIver,
  who wrote [Hypothesis](https://github.com/hypothesisworks/hypothesis)** and
  joined Antithesis in Nov 2025 (followed by Liam DeVoe, another core
  Hypothesis maintainer). The name is a dialectic joke — *Hypothesis,
  Antithesis, synthesis*. It lives under the [`hegeldev`](https://github.com/hegeldev)
  org, **not** `antithesishq`. Not to be confused with Hegel AI (an unrelated
  YC LLM-eval company) or the Hegel JS type checker.

- **Bombadil** — [property-based testing for **web and terminal UIs**](https://github.com/antithesishq/bombadil),
  autonomously exploring and validating correctness properties. Runs locally,
  in CI, and inside Antithesis. Not the dotfile manager of the same name.

### 5.2 Verdicts

| | What it is | Verdict for swe-term | Why |
|---|---|---|---|
| **Semi-formal methods** | A stance | **Adopt the frame. Build nothing.** | It is *already* §4's M2 rung ladder, independently arrived at. "Buy rigor piecemeal" is precisely what "minimum V&V rung" means. Costs nothing; validates the design. |
| **Hegel (`hegel-go`)** | PBT library, MIT, `go get hegel.dev/go/hegel` | **Adopt — after measuring it against Go's native fuzzing.** Do not build. | Verified real and active: ⭐92, created 2026-01-14, pushed 2026-09-17, v0.9.5. PBT is commodity-but-deep — *shrinking* (minimizing a counterexample) is the hard part, and Hegel's lineage is literally Hypothesis. Building our own would be NIH against the reference implementation's author. |
| **Bombadil** | PBT for web **and terminal** UIs | **Strong candidate — swe-term is a TUI.** Do not build. | Autonomous property-based exploration of a *terminal* UI is close to unique in the ecosystem. Building it ourselves means terminal emulation + state extraction + exploration strategy — a large specialized effort against an existing tool. |

### 5.3 The distinction that actually answers the question

These are **test engines**. The original question was about swe-term **as a
harness steering agents on brownfield codebases.** Those are different axes,
and conflating them produces the wrong build/adopt answer:

- **Axis A — testing swe-term itself.** `hegel-go` and Bombadil apply directly
  and should be adopted rather than rebuilt.
- **Axis B — swe-term steering *other* agents.** Hegel and Bombadil are
  *things the harness can require and supply*. They are not the steering
  mechanism. The steering mechanism is the obligation → rung → receipt layer
  in §4, and **nobody else is building that.**

So the answer to *"adopt these, or build something better optimized for
swe-term?"* is **both, on different layers**: adopt the engines, build the
gate. The engines are deep commodity; the gate is the differentiated piece.

And the synthesis worth noticing: **Hegel and Bombadil are not alternatives to
§4's rung ladder — they are the implementations of its rungs.**

| Rung (§4 M2) | Evidence producer |
|---|---|
| R1 example-based | stdlib `testing` |
| R2 property over generated inputs | **`hegel-go`** (or native `testing.F`) |
| R2–R4 for the TUI surface | **Bombadil** (terminal driver) |
| R4 fault injection, R5 seeded replay | Antithesis platform — only if ever justified |
| The ladder existing at all | **Semi-formal methods** supplies the rationale |

This is a case where adopting external tools *strengthens* the in-house
design instead of competing with it: the ladder was the missing piece, and
these fill its rungs.

### 5.4 Honest caveats before adopting anything

- **`hegel-go` is beta and says so.** v0.9.5, and the README states: *"As part
  of our beta, we may make breaking changes... If that instability bothers
  you, please check back in a few months for a stable release."* Fine for a
  test dependency; not something to build a gate's correctness on yet.
- **It is not pure Go.** `hegel-go` drives **`libhegel`, a native Rust
  engine**, loaded as `.so`/`.dylib`/`.dll` — vendored via git-lfs,
  `go:embed`'d, materialized to `~/.cache/hegel-go/`. This is a *test-path*
  dependency, not a shipped-binary one, so it doesn't violate §2's
  single-binary posture directly — but it does affect CI hermeticity, and it
  sits awkwardly beside the deliberate "core stays cgo-free" line in the
  Fleet CPG plan. Name it before adopting, don't discover it in CI.
  Platforms are limited to Linux amd64/arm64, macOS arm64, Windows amd64/arm64.
- **Go already has native fuzzing** (`testing.F`, Go 1.18+). The honest
  question is not "is Hegel good" (it is) but "does it beat the stdlib enough
  to justify a cgo-backed beta dependency *for swe-term's actual bug
  classes*." That is measurable, and this repo preregisters measurements —
  see the proposed `hegel-vs-native-fuzzing` experiment in §8.
- **There is a gravity well.** Hegel is MIT and standalone, but the stated
  design goal is to *"seamlessly integrate with Antithesis to increase its
  bug-finding power,"* and the highest rungs (R4/R5) are the commercial
  platform. That's good product design, not a trap — but adopt with open eyes
  about where the value ramp points.
- **Bombadil is explicitly experimental** (`0.x`, "API might still change").

### 5.5 The one idea worth stealing outright

Antithesis ships its agent tooling as a **Claude Code plugin**
([`antithesishq/antithesis-skills`](https://github.com/antithesishq/antithesis-skills)
— note the `.claude-plugin/`, `CLAUDE.md`, `AGENTS.md`), and one of those
skills is `antithesis-mutation-testing`, whose stated purpose is sharper than
the framing in §4's M3:

> *"Validate the **oracle**, not the system under test. A green run shows
> nothing bad was observed — not that the property would have noticed."*

That is mutation testing pointed one level up: not "do my tests catch bugs"
but **"would my property have fired?"** It even carries a verdict vocabulary
for survivors — *not mutatable, outstanding, withdrawn, refined*.

This is the missing answer to the obvious objection against §4: *once you gate
on fitness functions and properties, what validates the gates?* An agent that
can write a property can write a vacuous one. Mutation-testing the oracle is
the deterministic check, and it should be a standing obligation on any
fitness function swe-term gates on — which also partly answers the
evaluator-validity concern that §2 raised against mutation score itself.

Also worth noting as evidence for §4's M1: Antithesis is shipping **an agent
skill for getting agents to write property-based tests**, and states the bug
examples in the Hegel announcement were written by Claude. The
affordance-over-instruction approach is being independently pursued by the
people with the most data about it.

## 6. What not to do

- **Don't retrofit full DST.** Antithesis' own docs call the pluggable
  approach impractical for existing systems. Adopt properties, reachability,
  seeds, and fault injection; don't pretend a brownfield app can become
  FoundationDB.
- **Don't gate on an LLM judging test brittleness.** The judge is usually the
  same model that wrote the test — circular, and precisely what
  `evaluator-validity-audit` exists to check. Deterministic gates first.
- **Don't set absolute quality thresholds in a brownfield repo.** Ratchet only.
- **Don't ship instructions as the mechanism.** 2602.07900 is direct evidence
  that prompt-level test interventions don't move outcomes. Prompting can
  accompany a gate; it cannot be the gate.
- **Don't let this become a second agent loop.** §11: experiment
  infrastructure is tooling around the core.

## 7. Why this fits the architecture rather than fighting it

- Adds **no new invariant**. M2 gives invariant 7's "V&V rung" a definition it
  currently lacks; M3 gives invariant 6's receipt a concrete evidence kind.
- Reuses the enforcement machinery validated this session
  (`control_monitor.go`, `receipt_gate.go` — see
  `experiments/evidence/{temporal-journal-monitor,evidence-gated-lifecycle}/`).
- Consumes the Fleet CPG analyzer via §9's existing extension point.
- §12 conflicts: none. M2/M3 depend on the "state expansion" refactor (item 3)
  for where obligations live, and on the tool-safety contract (item 2) only
  insofar as fitness functions run as bounded tools.

## 8. Proposed experiments

Claims enter this architecture through preregistered experiments, so these are
proposals, not results.

| Proposed id | Kind | Hypothesis to falsify | Cheapest falsification |
|---|---|---|---|
| `mutation-gated-test-quality` | mechanism-hypothesis | Gating an obligation's receipt on diff-scoped mutation-kill (treatment) produces tests that survive refactoring better than a coverage-gated control, on a frozen brownfield corpus | Refactor-survival: apply behavior-preserving refactors to the corpus; count tests that break. Brittle tests break; behavioral ones don't. |
| `reachability-assertion-efficacy` | mechanism-hypothesis | Requiring reachability evidence (sometimes-style) for changed code eliminates the "test never exercised the change" class | Seed changes whose tests don't touch the new path; measure detection rate vs. a coverage-only control |
| `debt-signal-foresight` | benchmark | Injecting CPG blast radius + change-coupling into the packet *pre-decision* changes the plan an agent produces vs. the same task without it | Paired tasks on real brownfield PRs; compare structural-fitness deltas of resulting patches |
| `vv-rung-ladder` | mechanism-hypothesis | A closed rung ladder can be mechanically assigned from evidence (not filenames), and the monitor rejects under-rung discharge | Extend the existing `control_monitor` trace corpus with rung-downgrade traces; they must fail closed with a stable rule ID |
| `hegel-vs-native-fuzzing` | benchmark | `hegel-go` finds defect classes in swe-term's own code that Go's native `testing.F` fuzzing does not, by enough margin to justify a cgo-backed beta dependency | Seed a frozen defect corpus in `internal/core`; run both engines under equal time budgets; compare detection rate and counterexample minimality. Kills the adoption if native fuzzing is within noise. |

`debt-signal-foresight` is the one I'd run first: it is the cheapest, it
reuses the CPG engine as-is, and it tests the load-bearing claim of §4's M4
(that this is an information problem at decision time).

## 9. Open questions

- **Refactor-survival needs a corpus.** "Brittle" is only mechanically
  definable relative to refactors that *should* preserve tests. Building that
  corpus is the real cost of `mutation-gated-test-quality`, and it may be the
  thing that kills it.
- **Reachability evidence in Go.** Go's native fuzzing (`testing.F`) and
  coverage give partial signal; whether an Antithesis-style
  sometimes-assertion is expressible without a runtime is unresolved.
- **Rung assignment may need judgment.** Distinguishing "asserts observable
  behavior" from "restates the implementation" is mechanical in easy cases
  (does the test reference unexported symbols? does it assert call counts?)
  and genuinely hard in others. The hard cases may need a human or a judge —
  which reopens evaluator validity.
- **Does any of this survive contact with a real PR stream?** Everything here
  is reasoned from published evidence plus this repo's architecture. None of
  it has been run.

## Citations

| Source | Link | Used for |
|---|---|---|
| Hora & Robbes, *Are Coding Agents Generating Over-Mocked Tests?* (MSR 2026) | [arXiv:2602.00409](https://arxiv.org/abs/2602.00409) | Failure mode A is empirically real |
| *Rethinking the Value of Agent-Generated Tests* | [arXiv:2602.07900](https://arxiv.org/abs/2602.07900) | Prompt interventions don't move outcomes; agents use tests as probes |
| Zhao, Zhou & Cohen, *Do Coverage and Mutation Scores... Correlate with Effectiveness?* | [arXiv:2607.22880](https://arxiv.org/abs/2607.22880) | Evaluator-validity caution on mutation score |
| *Mutation-Guided LLM-based Test Generation at Meta* | [arXiv:2501.12862](https://arxiv.org/pdf/2501.12862) | Industrial precedent for mutation-guided generation |
| *Agentic LMs: Hunting Down Test Smells* | [arXiv:2504.07277](https://arxiv.org/html/2504.07277) | Test-smell detection context |
| Antithesis — How Antithesis works | [docs](https://antithesis.com/docs/properties_assertions/properties/) | State-space framing; "an integration test threads a single path" |
| Antithesis — Sometimes Assertions | [docs](https://antithesis.com/docs/best_practices/sometimes_assertions/) | Coverage ≡ `assertSometimes(true)`; locations vs. situations |
| Antithesis — Deterministic simulation testing | [docs](https://antithesis.com/docs/resources/deterministic_simulation_testing/) | DST scope; retrofit impracticality for existing systems |
| Ford, Parsons & Kua, *Building Evolutionary Architectures* | [site](https://nealford.com/books/buildingevolutionaryarchitectures.html), [ch.2](https://www.oreilly.com/library/view/building-evolutionary-architectures/9781492097532/ch02.html) | Fitness functions; ratchet on quality degradation |
| FoundationDB simulation testing | [docs](https://apple.github.io/foundationdb/testing.html#simulation) | Origin of the pluggable-nondeterminism approach |
| Antithesis — *The pragmatic magic of semi-formal methods* (Katkoria & Moore, 2025-05-12) | [blog](https://antithesis.com/blog/2025/semi_formal_proofs/) | §5: semi-formal stance; "buy rigor piecemeal" |
| Antithesis — *Hypothesis, Antithesis, synthesis* (MacIver, 2026-03-24) | [blog](https://antithesis.com/blog/2026/hegel/) | §5: Hegel announcement, lineage, roadmap |
| Hegel for Go | [hegeldev/hegel-go](https://github.com/hegeldev/hegel-go), [hegel.dev](https://hegel.dev) | §5: MIT, v0.9.5 beta, libhegel native engine |
| Bombadil | [antithesishq/bombadil](https://github.com/antithesishq/bombadil), [manual](https://antithesishq.github.io/bombadil/) | §5: PBT for web **and terminal** UIs |
| Antithesis Skills (Claude Code plugin) | [antithesishq/antithesis-skills](https://github.com/antithesishq/antithesis-skills) | §5.5: "validate the oracle, not the SUT"; agent-skill affordance precedent |
| Hypothesis | [hypothesisworks/hypothesis](https://github.com/hypothesisworks/hypothesis) | §5: Hegel's lineage |
| Böckeler, *Harness engineering for coding agent users* (2026-04-02) | [martinfowler.com](https://martinfowler.com/articles/harness-engineering.html) | §2: guides/sensors × computational/inferential; regulation categories; harnessability; Ashby's Law |
| Thoughtworks, *Harness engineering and agent feedback: Exploring AI coding sensors* | [blog](https://www.thoughtworks.com/en-de/insights/blog/generative-ai/harness-engineering-agent-feedback-exploring-ai-coding-sensors) | §2: sensor-suite experiment; situational awareness over automation |
| *Approved Fixtures* (Augmented Coding Patterns) | [pattern](https://lexler.github.io/augmented-coding-patterns/patterns/approved-fixtures/) | §4 M7: make test review a diff review |
| Böckeler, *TDD inside the agent loop — theater or actual value?* | [martinfowler.com](https://martinfowler.com/articles/exploring-gen-ai/tdd-in-the-agent-loop.html) | §2: adjacent, unread at time of writing |
