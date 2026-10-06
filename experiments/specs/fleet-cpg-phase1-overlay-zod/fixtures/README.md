# Frozen fixtures

Unused by this experiment. `source.kind` is `git`, so the frozen input is the
pinned upstream revision named in `manifest.json`. The shared checkout lives
under `experiments/runs/fleet-cpg-phase1-overlay-zod/shared/repo` (raw run
data, not a fixture subtree) and the three merge commits are checked out
in-place per bucket by `../run_overlay.sh`.

Place only reviewed, redistributable experiment inputs here.
- Keep fixtures minimal and deterministic.
- Record their aggregate content digest in `manifest.json` before preregistration.
- Never include credentials, proprietary workplace data, or personal data.
- Do not alter fixtures after inspecting results. Create a new experiment ID or
  revision when the corpus changes.
