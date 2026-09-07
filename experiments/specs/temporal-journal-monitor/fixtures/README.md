# Frozen temporal-monitor trace corpus

This directory defines `temporal-monitor-trace-corpus-v1`. Its executable form
is the table-driven suite in `internal/core/control_monitor_test.go`. It models
only synthetic identifiers and SHA-256-like identities; it contains no customer
code, credentials, provider access, or executable mutation.

| Trace class | Expected treatment outcome |
|---|---|
| approval → lease → declaration → observed effect → release | accepted |
| lease without matching approval | reject `control.approval.required` |
| competing lease | reject `control.lease.conflict` |
| observed undeclared effect | reject `control.effect.undeclared` |
| lifecycle without a current receipt | reject `control.lifecycle.receipt_required` |
| stale receipt after target change | reject `control.receipt.stale` |
| unsupported event or schema | reject with a stable control rule |
| immediate duplicate event | idempotent |
| sequence gap or altered duplicate | reject `control.event.sequence` |
| cancellation followed by lifecycle success | reject `control.lifecycle.cancelled` |
| prefix crash/replay | same terminal monitor state |

Changing a row, executable trace, or rule identifier requires a new fixture
revision and recalculated manifest digest.
