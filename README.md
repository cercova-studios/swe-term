<div align="center">

# swe-term

**A terminal SWE-agent harness where the harness owns authority, state, and ground truth — and the model owns cognition.**

<i>Frontend → Go core → Provider · heavy engines as sidecars</i>
<br></br>
</div>

swe-term (`st`) is a single-binary, local-first coding-agent harness. A small
Go core owns the agent loop, policy, and verification contracts; heavy engines
(such as the Rust CPG analyzer in `engine/`) run as out-of-process sidecars
behind narrow adapters. Safety and completion claims are meant to rest on
deterministic evidence — effect declarations, a durable control journal, and
source-bound verification receipts — not on the model's word.

**Status:** early. Today the runtime streams one provider request per turn,
either one-shot or in a TUI. The multi-step tool-using loop, effect
declarations, and the control journal are accepted `Target` contracts, not
shipped behavior. See [ARCHITECTURE.md §4 and §12](./ARCHITECTURE.md).

See [**ARCHITECTURE.md**](./ARCHITECTURE.md) for the mental model, domain
types, invariants, and extension points — read it before proposing design
changes.

See [**docs/core/MECHANISM_LEDGER.md**](./docs/core/MECHANISM_LEDGER.md) for
every harness mechanism, whether it is a durable boundary or an expiring
cognitive aid, and the evidence behind it.

## Quick start

```bash
# Prerequisites: go 1.26+, just; OPENAI_API_KEY for the default provider
just run                              # TUI (/config, /help, /quit)
just run --provider mock              # TUI without a network call
just run "Explain go.mod"             # one-shot, prints Markdown and exits
just run --reasoning high "Summarize main.go"
```

Providers: `openai` (default, reads `OPENAI_API_KEY`) and `mock`.

## What it does today

| Capability | Status | Source |
|---|---|---|
| Provider-neutral streaming contract (`Provider`, `StreamEvent`, `Usage`) | Implemented | `internal/core/provider.go`, `internal/core/usage.go` |
| One-shot and TUI frontends | Implemented | `main.go`, `internal/tui/` |
| OpenTelemetry `gen_ai.*` spans per provider call | Implemented | `internal/telemetry/` |
| Receipt gate for lifecycle claims (reducer) | Implemented, not yet wired to a store | `internal/core/receipt_gate.go` |
| `ContextPacket` / `Provenance` contract | Defined, no producer yet | `internal/core/context_packet.go` |
| Agent loop, tools, effect declarations, control journal | Target | [ARCHITECTURE.md §5, §12](./ARCHITECTURE.md) |
| Fleet CPG analyzer sidecar | In progress | `engine/`, [implementation plan](docs/plans/2026-09-05-fleet-cpg-engine-implementation.md) |

## Configuration

Precedence: defaults < user config < project config < flags.

- **User config:** `$SWE_TERM_HOME/config.toml` (default
  `$XDG_CONFIG_HOME/swe-term`, else `~/.config/swe-term`). Keys: `provider`,
  `model`, `reasoning`, `env_key`.
- **Project config:** `.swe-term/config.toml`, found from the working
  directory. Keys: `provider`, `model`, `reasoning`. It cannot set `env_key`
  or `base_url`.
- **Flags:** `--provider`, `--model`, `--reasoning`
  (`none|minimal|low|medium|high|xhigh|max`), `--config <path>`.
- **Telemetry:** set `OTEL_EXPORTER_OTLP_ENDPOINT` or
  `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` to export traces; unset means no
  export.

Never put API keys or tokens in config files; `env_key` names the environment
variable that holds the key.

## Local development

`just` is the command surface and Jujutsu (`jj`) is the VCS. Read
[CONTRIBUTING.md](./CONTRIBUTING.md) before your first change.

```bash
just setup                # colocate jj, install hooks and gh-stack
just doctor               # check required tools
just test                 # go test ./...
just vet                  # go vet
just fmt                  # gofmt
just drift                # architecture drift ratchet vs docs/reports/drift-baseline.json
```

Optional: `nix develop` provides `go`, `just`, `jj`, `gh`, `rustc`, and
`cargo`.

Common `just` targets: `run`, `test`, `vet`, `fmt`, `drift`, `drift-strict`,
`drift-accept`, `fetch`, `stack`, `research-check`, `experiment-*`.

## Research and experiments

swe-term is built evidence-first: mechanisms are promoted only after
preregistered experiments.

- [`docs/core/FRAMEWORKS.md`](docs/core/FRAMEWORKS.md) — port/avoid verdicts
  for pi, Codex, Claude Code, opencode, oh-my-pi, Prime Agent, DeepSeek
  Harness, Flue, and Deep Agents, revalidated 2026-10-10. Evidence lives in
  [`docs/research/`](docs/research/).
- [`docs/research/papers/README.md`](docs/research/papers/README.md) — paper
  discovery and triage contract.
- [`experiments/README.md`](experiments/README.md) — preregistered
  experiments (`just experiment-new|validate|ready|digest|list`).
- [`docs/reports/research-and-experiments-summary.md`](docs/reports/research-and-experiments-summary.md)
  — running digest of results. Commits touching research must update it;
  `just research-check` and the pre-commit hook enforce this.

## License

No license file has been added yet — treat the repository as
all-rights-reserved until one lands.
