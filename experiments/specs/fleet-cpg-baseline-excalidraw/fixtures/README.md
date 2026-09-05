# Frozen fixtures

Unused by this experiment. `source.kind` is `git`, so the frozen input is the
pinned upstream revision named in `manifest.json`, materialized fresh into
`experiments/runs/fleet-cpg-baseline-excalidraw/<run-id>/materialized/` by `../run_prototype.sh` rather
than copied from a fixture subtree here.

Place only reviewed, redistributable experiment inputs here.
- Keep fixtures minimal and deterministic.
- Record their aggregate content digest in `manifest.json` before preregistration.
- Never include credentials, proprietary workplace data, or personal data.
- Do not alter fixtures after inspecting results. Create a new experiment ID or
  revision when the corpus changes.
