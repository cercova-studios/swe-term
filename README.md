# swe-term

Terminal SWE-agent harness (`swe-term` / `st`). Go owns the agent loop; heavy engines run as sidecar binaries.

```sh
just run                  # TUI
just run --provider mock
just run "Explain go.mod" # one-shot
```

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md). How to contribute: [`CONTRIBUTING.md`](CONTRIBUTING.md).

Research starts with the
[`Hugging Face Papers discovery contract`](docs/research/papers/README.md) and
moves into [`preregistered harness experiments`](experiments/README.md).
