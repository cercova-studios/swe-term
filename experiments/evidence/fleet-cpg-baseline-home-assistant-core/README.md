# Evidence: fleet-cpg-baseline-home-assistant-core

Kind: benchmark · Status: complete · Promotion: proposed 2026-09-05, acceptance pending

## Preregistered manifest

- Spec: [`experiments/specs/fleet-cpg-baseline-home-assistant-core/manifest.json`](../../specs/fleet-cpg-baseline-home-assistant-core/manifest.json)
- Source: `https://github.com/home-assistant/core.git` @ `6ad726ba56d517e56536cdd6fa2ba9f358bbc0ef`
  (shallow clone), content digest `sha256:9ceecdf3a83da1c485a14fa13c2f9d2d552491ab4e1f47bd270037cbc9230e49`
- Rubric digest: `sha256:61bcb2cab58042894e754f0aa9818da0bc9acb71875a1919462f065cbefec0c4`
- Environment: `_JAVA_OPTIONS=-Xmx24g`, `memory_bytes` 24GiB declared in the manifest
- Variants: `cold`, `warm` · 3 repetitions · 54 terminal rows, 0 errors

## Component metrics

[`summary.json`](summary.json), from the spec's evaluator.

- `process_wall_time_ms`: cold median **213,006ms**, warm median **216,186ms** —
  acceptance rule not met → hypothesis rejected.
- Cold build (`pysrc2cpg`, ~3.03M LOC in `homeassistant/`): wall 49.79 / 59.72 / 61.28s,
  peak RSS 13.2–15.6GB.
- Result counts stable across all 6 invocations (the excalidraw wobble did not recur).

## Artifact references

Raw runs local only; the verbose build logs are retained gzipped (~56MB) because
they are the only record of the JVM's own timing breakdown at this scale.

## Discordant cases

Scaling. Against fastapi this repo is 144× the LOC but only 37× the cold-build
wall time and 44× the peak RSS — sublinear, contradicting the "roughly linear"
reading taken from the first three (much closer) data points. A plausible
mechanism (≈1,000 structurally similar integration packages) is stated in the
spec README as a hypothesis, not a finding.

## Limitations (draft for human review)

- One huge repo of one shape; a differently-shaped 3M-LOC repo may not scale the same way.
- Heap was capped from pilot data; a different cap changes RSS and possibly wall time.
- One exploratory pilot (57.8s / 12.3GB) preceded preregistration and is excluded.

## Decision

Accept as baseline evidence. This repo is the target the Go engine's slice-1
base-build measurement will be compared against. No invariant affected.
