# Documentation

Start with [usage](../README.md), the [architecture contract](../ARCHITECTURE.md),
and [contributor operations](../CONTRIBUTING.md).
[Distiller](../extensions/distiller/README.md) documents article extraction and ebook conversion.

## Current contracts

- [ARCHITECTURE.md](../ARCHITECTURE.md) owns layers, domain contracts, invariants,
  extension points, and the distinction between implemented behavior and accepted targets.
- [CONTRIBUTING.md](../CONTRIBUTING.md) owns setup, checks, Jujutsu and PR delivery.
- [Mechanism ledger](core/MECHANISM_LEDGER.md) owns mechanism classification,
  shipping conditions, evidence requirements, and review rules.
- [Paper discovery](research/papers/README.md) and its templates define the research workflow.
- [Experiments](../experiments/README.md) owns specification, preregistration,
  isolation, cleanup and evidence-promotion rules. Executable specs, templates,
  runners, fixtures, schemas and tests stay with the code.
- [Drift baseline](reports/drift-baseline.json) is an executable operating baseline.

## Designated development archive

The following existing paths are the project's development archive. Keeping their
paths preserves frozen design references and research-change gates. Records may
contain active plans or completed history; read each record's date and status.
They are not current capability or verification claims. Root ARCHITECTURE.md
retains accepted architectural constraints and identifies implemented behavior.

| Area | Records |
| --- | --- |
| [Plans](plans/) | Implementation designs, experiment plans and the [distiller roadmap](plans/distiller-roadmap.md) |
| [Core roadmap](core/PLAN.md), [backlog](core/BACKLOG.md) | Development milestones and candidate work |
| [Go rationale](core/GOLANG_TUI_PLAN.md), [Rust prototype plan](RUST_TUI_PLAN.md) | Extended rationale and earlier implementation designs |
| [Framework synthesis](core/FRAMEWORKS.md), [verdicts](core/HARNESS_VERDICTS.md) | Comparative decisions and source-pinned research |
| [Service design](services/AST_SERVICE_ARCHITECTURE.md) | Sidecar design rationale; current status comes from root architecture |
| [Paper packets](research/papers/) | Search plans, notes and syntheses; the README/templates remain current workflow contracts |
| [Research summary](reports/research-and-experiments-summary.md) | Dated findings and validation history; updates remain subject to the research-change gate |
| [Experiment evidence](../experiments/evidence/), per-spec RESULTS.md files | Historical measurements, provenance and limitations; manifests, receipts and executable spec contracts keep their paths |
| [Completed grouping pass](TODO.md) | Documentation migration history |

General source-derived research lives in the source-insights library at
`~/.agents/knowledge/sources/`; local entry IDs are retained where cited.

Maintain source identity and caveats when updating archive records. Do not rewrite
frozen manifests to hide path moves. Keep raw runs in their ignored local directory.
Review private paths and material before any public publication; this archive
classification does not authorize a wiki push, commit, or release.
