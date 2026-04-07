# swe_distiller

`swe_distiller` is a Rust CLI and library for extracting readable content from URLs and writing it to Markdown, HTML, or JSON.

Core CLI contract:

```bash
swe_distiller <url> -o webpage.md
```

## Quickstart

### 1. Build

```bash
cargo build
```

### 2. Run (default markdown output)

```bash
cargo run -- "https://example.com/article" -o webpage.md
```

### 3. Choose output mode

```bash
# HTML
cargo run -- "https://example.com/article" --mode html -o webpage.html

# JSON
cargo run -- "https://example.com/article" --mode json -o webpage.json
```

### 4. Enable LLM extraction pipeline

```bash
cargo run -- "https://example.com/article" --llm -o webpage.md
```

Optional environment variables:

```bash
export SWE_DISTILLER_LLM_PROVIDERS="jina,reader-lm:1.5b,gemma4:31b-cloud"
export SWE_DISTILLER_LLM_TIMEOUT_MS=12000
export SWE_DISTILLER_LLM_MAX_INPUT_CHARS=60000
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export JINA_API_KEY="..."
```

### 5. Enable browser fallback support

Browser fallback is behind a feature flag:

```bash
cargo run --features browser -- "https://example.com/article" -o webpage.md
```

### 6. Run tests

```bash
cargo test
```

## Attribution

This project is inspired by the extraction approach and architecture patterns from [`defuddle`](https://github.com/nicholasgasior/defuddle).  
`swe_distiller` is an independent Rust implementation with its own pipeline, CLI, and module layout.

