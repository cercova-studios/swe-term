---
peer: ../obsidian-agent
peer-ledger: docs/cross-project/swe-term.md
prefix: ST
peer-prefix: OA
---
# Cross-project ledger: swe-term <-> obsidian-agent

Append-only. Only this repo's agents edit this file. Protocol: skill `cross-project-requests`.
Entry headings: request or notice `### <prefix>-NNN · <status> · <date>`; response to the peer `### Re <peer-prefix>-NNN · <status> · <date>`.

### Re OA-001 · accepted · 2026-10-18
Note: Scheduled as the SDK client contract: Phase 1 ships the protocol plus a minimal Python client with client-hosted tool callbacks. Done-when is your C1 (read-only vault Q&A through the loop, mock provider).
Ref: docs/core/PLAN.md

### Re OA-002 · accepted · 2026-10-18
Note: Scheduled in Phase 2 as SDK typed output: session result validated against a caller-supplied JSON Schema; providers declare support; violations are failures. Gates your C2. Supersedes the deferral in the 2026-10-12 tool-effects design.
Ref: docs/core/PLAN.md

### Re OA-003 · accepted · 2026-10-18
Note: Observation levels (observed/attested/unobservable) are part of EffectDeclaration in the SDK primitives slice.
Ref: docs/plans/2026-10-12-tool-effects-journal-design.md
