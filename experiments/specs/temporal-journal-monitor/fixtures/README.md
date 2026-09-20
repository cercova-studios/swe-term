# Frozen temporal-monitor trace corpus

This directory defines `temporal-monitor-trace-corpus-v2`. Its executable form
is the trace tests in `internal/core/control_monitor_test.go` (one table-driven
illegal-trace test plus sequential legal, replay, retention, and bounds tests),
pinned by `executable-corpus.sha256` (sha256sum format, repository-relative
paths) so the manifest `content_digest` covers the executable traces, not only
this directory. `TestFrozenExperimentCorpusMatchesLock` fails when a pinned
file changes without a new lock. It models only synthetic identifiers and
SHA-256-like identities; it contains no customer code, credentials, provider
access, or executable mutation.

| Trace class | Expected treatment outcome |
|---|---|
| approval → lease → declaration → observed effect → release | accepted |
| lease without matching approval | reject `control.approval.required` |
| competing lease | reject `control.lease.conflict` |
| effect declaration under a lease that is not active | reject `control.lease.required` |
| observed undeclared effect | reject `control.effect.undeclared` |
| lifecycle without a current receipt | reject `control.lifecycle.receipt_required` |
| stale receipt after target change | reject `control.receipt.stale` |
| receipt target without an obligation | reject `control.receipt.invalid` |
| receipt for a different obligation with matching digests | reject `control.receipt.obligation_mismatch` |
| valid failed receipt | recorded as current receipt |
| lifecycle backed only by a failed receipt | reject `control.lifecycle.receipt_failed` |
| observed effect after a recorded receipt, then lifecycle | reject `control.lifecycle.receipt_required` |
| identical target re-declared after a recorded receipt | receipt stays current |
| unsupported event or schema | reject with a stable control rule |
| immediate duplicate event | idempotent |
| sequence gap, zero sequence, or altered duplicate | reject `control.event.sequence` |
| cancellation followed by lifecycle success | reject `control.lifecycle.cancelled` |
| prefix crash/replay | same terminal monitor state |

`v2` supersedes `temporal-monitor-trace-corpus-v1` (the corpus `run-001` was
evaluated against) and has no recorded run yet. It adds, after review of the
v1 reducer: the obligation-scoped target rows and the failed-receipt rows
(the v1 reducer matched receipts by identity alone and rejected valid failed
receipts as invalid), the observed-effect invalidation row (v1 left a recorded
receipt usable after a governed mutation), and the identical-target row (v1
discarded a current receipt when the same target was re-declared). A second
review pass added the declaration-without-active-lease row and the zero-sequence
row so that the same violated invariant always carries the same rule ID (v1
reported both under `control.effect.declaration_invalid` and
`control.event.schema_unsupported` respectively), and replaced the control
trace's fixture-only assertions with a reducer run on the unchanged-target
trace.

Changing a row, executable trace, or rule identifier requires a new fixture
revision, a regenerated `executable-corpus.sha256`
(`sha256sum internal/core/control_monitor_test.go` from the repository root),
and a recalculated manifest digest.
