# Framework Comparison (Compressed)

This document is the terse, explicit comparison layer for architecture decisions.
Detailed evidence remains in each framework's deep-dive and critique docs.

## Executive Takeaways

- **pi-mono** is the primary architectural baseline to port.
- **Codex** has the strongest protocol seam and execution safety model.
- **Claude Code** has production-hardened permissions/orchestration lessons, but not a portable architecture.
- **Flue** is valuable for headless framework/deploy shape, not core agent loop internals.
- **Deep Agents** validates middleware composition and adapter patterns, but not the dependency stack.

## Port / Avoid Matrix

| Framework | Port | Avoid |
|---|---|---|
| **pi-mono** | Layered modularity, provider abstraction, loop/tool separation | God object session, dynamic import extensions, heuristic token counting |
| **Codex** | Protocol-first seam (SQ/EQ), layered execution safety, steerable one-task loop | Responses API lock-in, oversized runtime surface, shared mutable session core |
| **Claude Code** | Permission pipeline patterns, read-parallel/write-serial tool orchestration, startup prefetch | Monolithic query engine/state, feature-flag architecture as core structure |
| **Flue** | Headless-first framework framing, deployable harness lifecycle, host-side tool boundary | Treating borrowed core as original architecture, vendor-coupled durability assumptions |
| **Deep Agents** | Middleware composition model, pluggable backend protocol, adapter strategy (ACP/evals) | Deep framework dependency chain, LangSmith-shaped observability assumptions |

## swe-term Decision Mapping

| Decision Area | swe-term Direction | Primary Source |
|---|---|---|
| Core architecture | Interface-driven Go core with explicit layering | `../research/PI_MONO_*`, `ARCHITECTURE.md` |
| Protocol boundary | Frontend/core boundary should be wire-protocol friendly | `../research/CODEX_*` |
| Safety boundary | Layered approvals + deterministic policy + sandbox | `../research/CLAUDE_CODE_CRITIQUE.md`, `../research/CODEX_CRITIQUE.md` |
| Context management | Token budget + explicit compaction strategy + durable summaries | `../research/CLAUDE_*`, `../research/DEEPAGENTS_*` |
| Extensibility | Plugins/adapters over loop forks | `../research/PI_MONO_*`, `../research/DEEPAGENTS_*` |
| Deployment posture | Single-binary local-first core, optional sidecars/services | `ARCHITECTURE.md`, `../research/FLUE_*` |

## Canonical Rule

When analysis docs disagree, default to:

1. `ARCHITECTURE.md` for architecture truth
2. `PLAN.md` for delivery sequencing
3. Deep-dive/critique docs for supporting evidence
