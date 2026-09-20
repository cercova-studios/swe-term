# Frozen receipt-gate trace corpus

This directory defines `receipt-trace-corpus-v1`. The table-driven Go traces in
`internal/core/receipt_gate_test.go` are the executable form of these fixtures,
pinned by `executable-corpus.sha256` (sha256sum format, repository-relative
paths) so the manifest `content_digest` covers the executable traces, not only
this directory. `TestFrozenExperimentCorpusMatchesLock` fails when a pinned
file changes without a new lock. Changing a case requires a new experiment
revision, a regenerated lock (`sha256sum internal/core/receipt_gate_test.go`
from the repository root), and a recalculated source digest.

| Trace | Expected treatment outcome |
|---|---|
| fresh passing receipt | lifecycle claim accepted |
| missing receipt | reject `receipt.missing` |
| failed receipt | reject `receipt.failed` |
| source change inside scope | reject `receipt.stale` |
| verifier change | reject `receipt.stale` |
| verifier arguments change | reject `receipt.stale` |
| configuration change | reject `receipt.stale` |
| runtime change | reject `receipt.stale` |
| lockfile change | reject `receipt.stale` |
| body tampering | reject `receipt.invalid` |
| unchanged scope | lifecycle claim remains accepted |
| replay of same receipt | state is identical |

Fixture facts are synthetic SHA-256 identities. They contain no credentials,
customer code, provider access, or proprietary data.
