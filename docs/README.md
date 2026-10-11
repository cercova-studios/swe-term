# Documentation Index

This directory is organized to keep architecture decisions centralized and reduce repeated rationale.

## Groups

- `core/` — roadmap, framework synthesis, backlog, and long-form rationale
- `plans/` — bounded implementation and experiment plans
- `research/` — deep-dive and critique evidence packs
- [`research/papers/`](research/papers/) — paper discovery, research-taste, and
  experiment-selection workflow
- `services/` — service and sidecar architecture docs
- [`../experiments/`](../experiments/) — experiment specifications, schemas,
  raw-run boundary, and curated evidence contract

## Source-of-Truth Documents

- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — How to develop and open stacked PRs (`just`, `jj`, `gh stack link`).
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — Canonical architecture contract.
- [`core/PLAN.md`](core/PLAN.md) — Concise execution roadmap (phases, milestones, done criteria).
- [`core/FRAMEWORKS.md`](core/FRAMEWORKS.md) — Condensed cross-framework comparison and porting guidance.
- [`core/BACKLOG.md`](core/BACKLOG.md) — Tooling and integration opportunities.
- [`services/AST_SERVICE_ARCHITECTURE.md`](services/AST_SERVICE_ARCHITECTURE.md) — AST/analyzer service architecture.
- [`core/GOLANG_TUI_PLAN.md`](core/GOLANG_TUI_PLAN.md) — Detailed architecture rationale and extended reference material.

## Comparative Analysis Documents

- [`core/FRAMEWORKS.md`](core/FRAMEWORKS.md) — compressed port/avoid matrix. Read first.
- [`core/HARNESS_VERDICTS.md`](core/HARNESS_VERDICTS.md) — per-harness verdicts, port and do-not-port lists, each pinned to an upstream commit.

General architecture notes on each harness and the papers behind the research
packets live in the `source-insights` library (`~/.agents/knowledge/sources/`),
not in this repository. Entry IDs are cited where they are used.

## Reading Order

1. [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
2. [`core/PLAN.md`](core/PLAN.md)
3. [`core/FRAMEWORKS.md`](core/FRAMEWORKS.md)
4. [`core/BACKLOG.md`](core/BACKLOG.md)
5. [`services/AST_SERVICE_ARCHITECTURE.md`](services/AST_SERVICE_ARCHITECTURE.md)
6. [`core/GOLANG_TUI_PLAN.md`](core/GOLANG_TUI_PLAN.md) (deep reference)
7. Specific deep-dive/critique files in [`research/`](research/) as needed

Paper-backed experiment work starts with
[`research/papers/README.md`](research/papers/README.md), then moves to the
preregistration contract under [`../experiments/`](../experiments/).

Current focused plans:

- [`plans/2026-08-27-harness-hypothesis-experiments.md`](plans/2026-08-27-harness-hypothesis-experiments.md)
- [`plans/2026-08-29-experiment-infrastructure-design.md`](plans/2026-08-29-experiment-infrastructure-design.md)

## Group Entrypoints

- [`core/README.md`](core/README.md)
- [`research/README.md`](research/README.md)
- [`services/README.md`](services/README.md)
