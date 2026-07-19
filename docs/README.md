# Documentation Index

This directory is organized to keep architecture decisions centralized and reduce repeated rationale.

## Groups

- `core/` — canonical architecture, roadmap, framework synthesis, and backlog
- `research/` — deep-dive and critique evidence packs
- `services/` — service and sidecar architecture docs

## Source-of-Truth Documents

- [`core/ARCHITECTURE.md`](core/ARCHITECTURE.md) — Canonical architecture spec.
- [`core/PLAN.md`](core/PLAN.md) — Concise execution roadmap (phases, milestones, done criteria).
- [`core/FRAMEWORKS.md`](core/FRAMEWORKS.md) — Condensed cross-framework comparison and porting guidance.
- [`core/BACKLOG.md`](core/BACKLOG.md) — Tooling and integration opportunities.
- [`services/AST_SERVICE_ARCHITECTURE.md`](services/AST_SERVICE_ARCHITECTURE.md) — AST/analyzer service architecture.
- [`core/GOLANG_TUI_PLAN.md`](core/GOLANG_TUI_PLAN.md) — Detailed architecture rationale and extended reference material.

## Comparative Analysis Documents

Deep dives and critiques are retained as supporting evidence under `research/`:

- [`research/CLAUDE_DEEP_DIVE.md`](research/CLAUDE_DEEP_DIVE.md), [`research/CLAUDE_CODE_CRITIQUE.md`](research/CLAUDE_CODE_CRITIQUE.md)
- [`research/CODEX_DEEP_DIVE.md`](research/CODEX_DEEP_DIVE.md), [`research/CODEX_CRITIQUE.md`](research/CODEX_CRITIQUE.md)
- [`research/PI_MONO_DEEP_DIVE.md`](research/PI_MONO_DEEP_DIVE.md), [`research/PI_MONO_CRITIQUE.md`](research/PI_MONO_CRITIQUE.md)
- [`research/FLUE_DEEP_DIVE.md`](research/FLUE_DEEP_DIVE.md), [`research/FLUE_CRITIQUE.md`](research/FLUE_CRITIQUE.md)
- [`research/DEEPAGENTS_DEEP_DIVE.md`](research/DEEPAGENTS_DEEP_DIVE.md), [`research/DEEPAGENTS_CRITIQUE.md`](research/DEEPAGENTS_CRITIQUE.md)

Use `core/FRAMEWORKS.md` first; read deep dives/critiques only when detailed evidence is needed.

## Reading Order

1. [`core/ARCHITECTURE.md`](core/ARCHITECTURE.md)
2. [`core/PLAN.md`](core/PLAN.md)
3. [`core/FRAMEWORKS.md`](core/FRAMEWORKS.md)
4. [`core/BACKLOG.md`](core/BACKLOG.md)
5. [`services/AST_SERVICE_ARCHITECTURE.md`](services/AST_SERVICE_ARCHITECTURE.md)
6. [`core/GOLANG_TUI_PLAN.md`](core/GOLANG_TUI_PLAN.md) (deep reference)
7. Specific deep-dive/critique files in [`research/`](research/) as needed

## Group Entrypoints

- [`core/README.md`](core/README.md)
- [`research/README.md`](research/README.md)
- [`services/README.md`](services/README.md)
