# Frozen receipt-gate trace corpus

This directory defines `receipt-trace-corpus-v1`. The table-driven Go traces in
`internal/core/receipt_gate_test.go` are the executable form of these fixtures.
Changing a case requires a new experiment revision and source digest.

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
