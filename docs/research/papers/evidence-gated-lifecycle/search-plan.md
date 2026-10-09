# Paper search plan: evidence-gated lifecycle claims

Status: complete

## Decision to inform

Determine whether swe-term needs a minimal, deterministic receipt gate before a
future agent loop can treat `verified`, `done`, or `ready_to_merge` as a lifecycle
state rather than model output.

## Local anchor

- Observed failure or unresolved question: a successful check can become stale
  after source, verifier, configuration, runtime, or lockfile state changes.
- Relevant source, trace, or evidence: root `ARCHITECTURE.md` §7 and invariant 6
  in §10 require a receipt bound to current source and verifier state.
- Architecture seam: target `VerificationReceipt`, `Obligation`, and
  `ControlEvent` contracts; this experiment advances state expansion without
  adding persistence or an agent loop.
- Implemented versus target: provider streaming is implemented; the receipt and
  lifecycle control plane are target contracts.
- Constraints: deterministic Go traces only; no model, network, database,
  container, or mutable workspace.

## Mechanism inventory

| Mechanism family | Synonyms and failure terms | Why it could affect the local failure |
|---|---|---|
| evidence-gated lifecycle control | proof-carrying state, verification receipt, stale evidence, source-bound gate | separates an agent's completion claim from an independently checked current state |
| temporal control | event ordering, lifecycle monitor, approval gate | useful later, but broader than the first source-bound receipt reducer |
| ordinary test status | pass flag, success result, stale result | control condition: lacks an identity that can invalidate stale success |

## Query log

| Date | Query or candidate | Source or endpoint | Result count | Fallback or error | Notes |
|---|---|---|---:|---|---|
| 2026-08-30 | Proof-or-Stop (`2607.14890`) | Hugging Face paper page | 1 direct candidate | arXiv abstract used for independent source check | carried forward from the initial harness mechanism pass; re-read for the narrow receipt claim |

## Candidate ledger

| Paper | Mechanism | Local fit | Evidence | Evaluator | Reproduction | Operational fit | Independent support | Screen decision |
|---|---|---|---|---|---|---|---|---|
| Proof-or-Stop | source-bound, mechanically verified lifecycle gate | high | medium | medium | high | high | low | deep-read |

## Exclusions

- Do not reproduce unattended loops, review-agent topology, self-application,
  or paper-level outcome scores.
- Do not use a model as verifier or introduce a free-form policy language.
- Do not infer broad safety or model-agnostic effectiveness from a single
  preprint with one model family and a self-hosted corpus.

## Search closure

The local decision is now narrow enough for a deterministic falsification test:
does exact identity binding reject stale and tampered receipts without rejecting
unchanged scope? Broader temporal-monitor papers are deferred to Experiment 3.
