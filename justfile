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
    git config core.hooksPath .githooks
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

fetch:
    jj git fetch

# Push bookmarks (bottom to top) and open/update a GitHub stack against trunk `dev`.
# usage: just stack layer1 layer2
stack *bookmarks:
    #!/usr/bin/env sh
    set -eu
    if [ "$#" -eq 0 ]; then
        printf 'usage: just stack bookmark...\n' >&2
        exit 2
    fi
    for b in "$@"; do
        jj git push --allow-new --bookmark "$b"
    done
    gh stack link --base dev --open "$@"
