# Mechanism Ledger

Every harness mechanism is either a **boundary** or a **cognitive aid**
(ARCHITECTURE.md §3). This ledger records which, and what evidence keeps it.

- **Boundary**: authority, state, or ground truth the model cannot supply for
  itself. It stays. Its row tracks the friction it costs on a strong model
  (extra turns, tokens, wall time, false denials).
- **Cognitive aid**: something a capable model may do unaided. It is optional
  and configured per model, and it carries an expiry test: a
  `mechanism-hypothesis` ablation under `experiments/specs/`. When the
  ablation shows no lift for a model, the aid is disabled for that model; when
  no supported model needs it, it is deleted.

A mechanism without a row does not ship by default.

## Columns

| Column | Meaning |
|---|---|
| Mechanism | Name used in code and docs |
| Kind | `boundary` or `aid` |
| Status | `Implemented` or `Target`, as in ARCHITECTURE.md §1 |
| Owner | Source path, or the §5 contract it belongs to |
| Evidence | Experiment spec id and its status |
| Models | Models the evidence covers; `—` if none yet |
| Verdict | Boundary: `keep` plus measured friction. Aid: `default-on`, `per-model`, `off`, or `delete` |
| Review when | Event that forces re-evaluation |

## Ledger

| Mechanism | Kind | Status | Owner | Evidence | Models | Verdict | Review when |
|---|---|---|---|---|---|---|---|
| Receipt-gated lifecycle claims | boundary | Implemented (reducer only) | `internal/core/receipt_gate.go`, invariant 6 | `evidence-gated-lifecycle` (complete, accepted) | — (deterministic traces) | keep; friction unmeasured | First loop-driven task run |
| Temporal control-journal monitor | boundary | Target | `ControlJournal`, §6 | `temporal-journal-monitor` (complete) | — (deterministic traces) | keep; friction unmeasured | First loop-driven task run |
| Effect declarations | boundary | Target | `EffectDeclaration`, in-flight refactor 2 | — | — | keep; measure declaration cost per call | First mutating tool lands |
| Externalized task state | boundary | Target | `TaskSnapshot` | `verified-external-task-state` (draft) | — | keep; measure friction | Spec reaches `ready` |
| Context provenance and freshness | boundary | Target (contract defined) | `internal/core/context_packet.go` | Fleet CPG benchmarks (complete) | — | keep | First packet producer lands |
| Protected-spine compaction | boundary (spine) + aid (summary) | Target | §7 | `protected-spine-compaction` (draft) | — | Spine: keep. Summary content: ablate | Spec reaches `ready` |
| Structured verifier feedback | aid | Target | §11 | `structured-verifier-feedback` (draft) | — | undecided; off by default until evidence | Spec run, then every supported-model change |
| Pre-materialized context packets | aid | Target | `ContextPacket` | — | — | Offered and lazily loaded, never forced into the prompt; needs an ablation spec | First packet producer lands |

## Rules

1. Adding a mechanism adds a row in the same change.
2. An `aid` row with no evidence ships off by default.
3. A supported-model change re-runs every `aid` row's ablation for that model.
4. Changing a row's `Kind` from `boundary` to `aid` needs an explicit
   architecture decision. A boundary is not demoted because it causes friction;
   it is made cheaper.
5. Deleted aids keep their row with verdict `delete` and the commit that
   removed them, so the evidence trail survives.

Not yet built: a `cmd/drift` signal for aid surface area (prompt bytes plus
aid code in the core). Add it when the first aid lands in code, so growth in
aids shows up on the ratchet.
