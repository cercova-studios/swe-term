# swe-term

Terminal SWE-agent harness (`swe-term` / `st`). Go owns the agent loop; heavy engines run as sidecar binaries.

```sh
just run                  # TUI
just run --provider mock
just run "Explain go.mod" # one-shot
```

Architecture: [`docs/core/ARCHITECTURE.md`](docs/core/ARCHITECTURE.md). How to contribute: [`CONTRIBUTING.md`](CONTRIBUTING.md).
