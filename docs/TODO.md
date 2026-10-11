# Docs Physical Grouping Pass

The grouping pass is complete. The architecture contract subsequently moved to
the repository root so repository-wide preflight instructions have one stable
entrypoint; `docs/core/ARCHITECTURE.md` remains a compatibility pointer.

## Target Structure

- `ARCHITECTURE.md` — authoritative repository contract
- `docs/core/`
  - `ARCHITECTURE.md` — temporary compatibility pointer
  - `PLAN.md`
  - `FRAMEWORKS.md`
  - `BACKLOG.md`
  - `GOLANG_TUI_PLAN.md`
- `docs/research/`
  - `papers/` (discovery packets; general findings live in the source-insights library)
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
- Per-harness verdicts remain discoverable in `docs/core/HARNESS_VERDICTS.md`.
