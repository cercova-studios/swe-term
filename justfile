# Command surface for humans and agents. Prefer `just <recipe>` over ad hoc flags.
# Go version lives in go.mod. Optional Nix: `nix develop`.

set shell := ["sh", "-eu", "-c"]
export PAGER := "cat"

default:
    @just --list

[private]
need bin:
    command -v {{bin}} >/dev/null || { printf 'missing: %s\n' '{{bin}}' >&2; exit 1; }

doctor: (need "just") (need "go") (need "jj") (need "gh")
    go version
    jj --version
    gh --version
    if gh stack --help >/dev/null 2>&1; then printf 'gh stack: ok\n'; else printf 'warning: gh stack extension missing; run just setup\n' >&2; fi

setup:
    #!/usr/bin/env sh
    set -eu
    if [ ! -d .jj ]; then
        jj git init --colocate
    fi
    # Only colocated checkouts have a Git dir; jj-only checkouts rely on `just research-check`.
    if [ -d .git ]; then git config core.hooksPath .githooks; fi
    if ! gh stack --help >/dev/null 2>&1; then
        gh extension install github/gh-stack
    fi

test:
    go test ./...

vet:
    go vet ./...

fmt:
    go fmt ./...

run *args:
    go run . {{args}}

# Create and validate VCS-neutral research experiment specifications.
experiment-new id:
    go run ./cmd/experimentctl new {{id}}

experiment-validate id:
    go run ./cmd/experimentctl validate {{id}}

experiment-ready id:
    go run ./cmd/experimentctl ready {{id}}

experiment-digest id:
    go run ./cmd/experimentctl digest {{id}}

experiment-list:
    go run ./cmd/experimentctl list

# Continuous drift sensor: compare codebase health signals against the
# recorded baseline. Reports by default; `just drift-strict` fails on
# regression. Not a commit gate — run it on a schedule or on demand.
drift:
    go run ./cmd/drift

drift-strict:
    go run ./cmd/drift -strict

# Accept current values as the new baseline. Explain why in the commit message.
drift-accept:
    go run ./cmd/drift -update

fetch:
    jj git fetch

# Fail if a revision touches research/experiment records without updating the
# compiled summary. jj has no commit hooks, so run this before `jj commit`
# (defaults to the working copy) or on any revision. Bypass: SKIP_RESEARCH_CHECK=1.
research-check rev='@':
    jj diff -r {{rev}} --name-only | sh scripts/check-research-summary.sh

# Push bookmarks (bottom to top) and open/update a GitHub stack against trunk `dev`.
# Runs `research-check` on every bookmark first.
# usage: just stack layer1 layer2
stack *bookmarks:
    #!/usr/bin/env sh
    set -eu
    if [ "$#" -eq 0 ]; then
        printf 'usage: just stack bookmark...\n' >&2
        exit 2
    fi
    for b in "$@"; do
        just research-check "$b"
    done
    for b in "$@"; do
        jj git push --allow-new --bookmark "$b"
    done
    gh stack link --base dev --open "$@"
