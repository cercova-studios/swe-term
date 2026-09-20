#!/usr/bin/env sh
# Fails when a change touches research or experiment records without also
# touching the compiled summary report. Reads the changed paths, one per line,
# from stdin so the same check runs from the Git pre-commit hook, from
# `just research-check` on a jj revision, and from `just stack` before push.
# See docs/reports/research-and-experiments-summary.md and
# docs/research/papers/README.md for what belongs in each.
#
# Escape hatch: SKIP_RESEARCH_CHECK=1 when the summary genuinely doesn't need
# an update (e.g. a typo fix in a spec that changes no finding).
set -eu

SUMMARY="docs/reports/research-and-experiments-summary.md"

if [ "${SKIP_RESEARCH_CHECK:-}" = "1" ]; then
    exit 0
fi

changed=$(cat)

touches_tracked_area=$(printf '%s\n' "$changed" | grep -E '^(docs/research/papers/|experiments/(specs|evidence)/)' || true)
touches_summary=$(printf '%s\n' "$changed" | grep -Fx "$SUMMARY" || true)

if [ -n "$touches_tracked_area" ] && [ -z "$touches_summary" ]; then
    printf '\nresearch-check: research/experiment files changed but %s was not updated:\n\n' "$SUMMARY" >&2
    printf '%s\n' "$touches_tracked_area" | sed 's/^/  /' >&2
    printf '\nCompile the finding into %s before committing (see its own\nheader for what belongs there), or set SKIP_RESEARCH_CHECK=1 if this\nchange genuinely has nothing to add to the summary (e.g. a typo fix).\n\n' "$SUMMARY" >&2
    exit 1
fi

exit 0
