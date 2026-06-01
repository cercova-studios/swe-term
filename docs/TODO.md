# TODO: Docs Physical Grouping Pass

This is a handoff checklist for physically moving docs into grouped folders after the initial logical grouping pass.

## Target Structure

- `docs/core/`
  - `ARCHITECTURE.md`
  - `PLAN.md`
  - `FRAMEWORKS.md`
  - `BACKLOG.md`
  - `GOLANG_TUI_PLAN.md`
- `docs/research/`
  - `CLAUDE_DEEP_DIVE.md`
  - `CLAUDE_CODE_CRITIQUE.md`
  - `CODEX_DEEP_DIVE.md`
  - `CODEX_CRITIQUE.md`
  - `PI_MONO_DEEP_DIVE.md`
  - `PI_MONO_CRITIQUE.md`
  - `FLUE_DEEP_DIVE.md`
  - `FLUE_CRITIQUE.md`
  - `DEEPAGENTS_DEEP_DIVE.md`
  - `DEEPAGENTS_CRITIQUE.md`
- `docs/services/`
  - `AST_SERVICE_ARCHITECTURE.md`

## Tasks

- Move the files to the target folders (prefer `git mv` to preserve history).
- Rewrite all references to moved files across repository docs and metadata.
- Update `docs/README.md`, `docs/core/README.md`, `docs/research/README.md`, and `docs/services/README.md` links.
- Update any references in `AGENTS.md` that point to old doc paths.
- Decide whether to leave compatibility stub files at old paths.
  - If yes, create short "moved" docs pointing to new paths.
  - If no, ensure all links are fully rewritten.
- Run a final broken-link/reference scan and fix any misses.

## Acceptance Criteria

- No references remain to old paths unless intentionally preserved via stubs.
- Primary routing docs still clearly direct readers:
  - architecture truth
  - roadmap
  - framework synthesis
  - backlog
- Deep-dive and critique docs remain intact and discoverable under `docs/research/`.
