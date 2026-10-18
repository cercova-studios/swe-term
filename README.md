# swe-term

swe-term (`st`) streams model responses in a terminal UI or a one-shot command.
It supports OpenAI and a local mock provider, layered configuration, and optional
OpenTelemetry tracing.

```sh
just run --provider mock                 # TUI without a network call
just run --provider mock "Hello"         # one-shot response
just run "Explain this error message"    # OpenAI; requires OPENAI_API_KEY
just run --reasoning high "Explain this algorithm"
```

Use `/config`, `/help`, and `/quit` in the TUI. Each turn sends one provider request.
The [architecture contract](ARCHITECTURE.md) describes interfaces and operating
invariants. [Distiller](extensions/distiller/README.md) separately extracts articles
and converts HTML or born-digital PDFs into ebooks.

## Setup

Install Go at the version in [go.mod](go.mod) and `just`. Contributor tooling also
uses Jujutsu (`jj`) and `gh`; see [CONTRIBUTING.md](CONTRIBUTING.md). Optional
`nix develop` provides Go, just, jj, gh, rustc, and cargo.

## Configuration

Precedence: defaults < user config < project config < flags.

- User config: `$SWE_TERM_HOME/config.toml`; by default `$XDG_CONFIG_HOME/swe-term`,
  otherwise `~/.config/swe-term`. Keys: `provider`, `model`, `reasoning`, `env_key`.
- Project config: `.swe-term/config.toml`, found from the working directory.
  Keys: `provider`, `model`, `reasoning`. Project config cannot set `env_key` or
  `base_url`.
- Flags: `--provider`, `--model`, `--reasoning`
  (`none|minimal|low|medium|high|xhigh|max`), and `--config <path>`.
- Providers: `openai` is the default and reads `OPENAI_API_KEY`; `mock` needs no key.
- Tracing: set `OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`.
  With both unset, tracing opens no connection.

Keep key values out of config files. `env_key` names the environment variable
that holds the key.

## Development

```sh
just setup       # initialize contributor tooling and hooks
just doctor      # check required tools
just test
just vet
just fmt
just drift       # compare against the versioned architecture baseline
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) for delivery and research-change checks.
[Documentation](docs/README.md) routes current interfaces and operating contracts;
[experiments/README.md](experiments/README.md) defines the experiment workflow.

## License

No license file is present. Treat this repository as all-rights-reserved.
