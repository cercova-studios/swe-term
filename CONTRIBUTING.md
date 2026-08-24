# Contributing

`just` is the command surface. Jujutsu (`jj`) is the VCS. Pull requests are stacked with `gh stack link` against trunk **`dev`**.

Do not use raw `git`, `gh pr create`, or `gh stack init` / `add` / `submit`. Those fight colocated jj.

## Bootstrap

Same bar either path: `just doctor` must pass.

**Optional Nix** (reproducible tools, not a required contributor path):

```sh
nix develop
# or: nix develop -c just doctor
just setup
just doctor
```

The flake provides `go`, `just`, `jj`, `gh`, `rustc`, and `cargo`. It does not install the `gh stack` extension.

**Manual:**

1. Go at the version in `go.mod`
2. `just`, `jj`, and `gh` on `PATH`
3. Then:

```sh
just setup
just doctor
```

`just setup` colocate-inits jj in this Git checkout (creates local `.jj/`, not committed) and installs `github/gh-stack` if missing.

Set identity once (user-level, not in this repo):

```sh
jj config set --user user.name "Your Name"
jj config set --user user.email "you@example.com"
```

## Daily loop

```sh
just fetch
jj new dev@origin
# edit
jj diff
just test
just vet
jj commit -m "why this change exists"
jj bookmark create feat/short-name -r @-
```

Bookmark the immutable commit (`@-` after `jj commit`), not the empty working copy (`@`).

History surgery stays native jj: `describe`, `new`, `squash`, `edit`, `undo`. Do not wrap those in just recipes.

## Stacked pull requests

One concern per bookmark. Chain layers with `jj new`. List bookmarks **bottom to top**.

```sh
jj new dev@origin
# layer 1
jj commit -m "Add config merge denylist"
jj bookmark create feat/config-denylist -r @-

# layer 2
jj commit -m "Stream completion protocol in the TUI"
jj bookmark create feat/tui-stream -r @-

just stack feat/config-denylist feat/tui-stream
```

`just stack` pushes with `--allow-new` and runs `gh stack link --base dev --open`. A single-layer change still goes through `just stack`, never `gh pr create`.

To add layers later, pass the **full** bottom-to-top list (bookmark names and/or existing PR numbers):

```sh
just stack feat/config-denylist feat/tui-stream feat/status-line
# or: gh stack link --base dev --open 12 13 feat/status-line
```

After merge:

```sh
just fetch
jj rebase -d dev@origin
```

Do not force-push `dev`.

## Tests

Land a `*_test.go` only when it locks:

1. **Fail-closed combinatorics** — merge order, denylist, credential rejection, unknown provider, stream completion protocol, usage/cost that must not silently lie.
2. **User-journey e2e** — mock provider, no live keys. Assert observable behavior, not helpers.

Do not land helper, constructor, mapper, formatter, ranking, or string-snapshot tests. Scratch tests locally; delete them unless they meet (1) or (2). `just test` staying green is not a reason to keep a file.

## Anti-patterns

- `git add` / `git commit` / `git push`
- `gh pr create`
- `gh stack init` / `add` / `submit` (git-checkout workflow)
- Bookmarking `@` while it is still the dirty working copy
- Force-pushing `dev`
- Treating the Nix flake as required, or mise as the Go version manager (Go lives in `go.mod`)
