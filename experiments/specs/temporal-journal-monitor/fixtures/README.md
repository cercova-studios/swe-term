# Frozen temporal-monitor trace corpus

This directory defines `temporal-monitor-trace-corpus-v2`. Its executable form
is the table-driven suite in `internal/core/control_monitor_test.go`, pinned by
`executable-corpus.sha256` (sha256sum format, repository-relative paths) so the
manifest `content_digest` covers the executable traces, not only this
directory. `TestFrozenExperimentCorpusMatchesLock` fails when a pinned file
changes without a new lock. It models only synthetic identifiers and
SHA-256-like identities; it contains no customer code, credentials, provider
access, or executable mutation.

| Trace class | Expected treatment outcome |
|---|---|
| approval → lease → declaration → observed effect → release | accepted |
| lease without matching approval | reject `control.approval.required` |
| competing lease | reject `control.lease.conflict` |
| observed undeclared effect | reject `control.effect.undeclared` |
| lifecycle without a current receipt | reject `control.lifecycle.receipt_required` |
| stale receipt after target change | reject `control.receipt.stale` |
| receipt target without an obligation | reject `control.receipt.invalid` |
| receipt for a different obligation with matching digests | reject `control.receipt.obligation_mismatch` |
| valid failed receipt | recorded as current receipt |
| lifecycle backed only by a failed receipt | reject `control.lifecycle.receipt_failed` |
| unsupported event or schema | reject with a stable control rule |
| immediate duplicate event | idempotent |
| sequence gap or altered duplicate | reject `control.event.sequence` |
| cancellation followed by lifecycle success | reject `control.lifecycle.cancelled` |
| prefix crash/replay | same terminal monitor state |

`v2` supersedes `temporal-monitor-trace-corpus-v1` (the corpus `run-001` was
evaluated against). It adds the obligation-scoped target rows and the
failed-receipt rows after review of the v1 reducer, which matched receipts by
identity alone and rejected valid failed receipts as invalid.

Changing a row, executable trace, or rule identifier requires a new fixture
revision, a regenerated `executable-corpus.sha256`
(`sha256sum internal/core/control_monitor_test.go` from the repository root),
and a recalculated manifest digest.
