# Building a Rust TUI Agent using Agent-Client-Protocol
## Learning-Oriented Implementation Plan

---

## Executive Summary

You'll build a fully-featured Rust Terminal User Interface (TUI) that implements the Agent-Client-Protocol (ACP) to communicate with AI coding agents. This project is structured as a progressive learning journey covering Rust fundamentals, systems programming, async/concurrent patterns, type systems, and terminal rendering—all while becoming a better backend engineer.

**Tech Stack:**
- Rust 1.75+ (ownership, lifetimes, async/await)
- Ratatui (TUI framework with immediate-mode rendering)
- Crossterm (cross-platform terminal manipulation)
- Tokio (async runtime for concurrent operations)
- Serde (JSON serialization/deserialization)
- JSON-RPC 2.0 (protocol layer)
- SQLx (async database with compile-time query checking)

**What Makes This Rust-Specific:**
- Master ownership, borrowing, and lifetimes through real-world usage
- Learn Rust's powerful type system with enums and pattern matching
- Understand async/await with Tokio for concurrent I/O
- Experience zero-cost abstractions and compile-time guarantees
- Apply trait-based polymorphism instead of interfaces
- Use Result/Option for explicit error handling

---

## Phase 1: Rust Fundamentals & Protocol Basics (Weeks 1-2)

### Learning Goals
- Rust project structure & Cargo workflow
- Ownership, borrowing, and lifetimes basics
- Structs, enums, and pattern matching
- Error handling with Result and Option
- JSON-RPC 2.0 protocol implementation

### Milestone 1.1: JSON-RPC 2.0 Message Handler
**What You'll Build:**
```
codex-rs/
├── Cargo.toml
├── src/
│   ├── main.rs              # Binary entry point
│   ├── lib.rs               # Library root
│   └── jsonrpc/
│       ├── mod.rs
│       ├── message.rs       # Request/Response/Notification types
│       ├── codec.rs         # JSON encoding/decoding
│       └── error.rs         # JSON-RPC error types
```

**Key Concepts:**
- Rust modules and visibility (`pub`, `pub(crate)`)
- Struct definitions with Serde derive macros
- Enum discriminated unions (tagged unions)
- Pattern matching with `match` and `if let`
- `Result<T, E>` for error propagation with `?` operator
- Trait implementations (Display, Error, From)

**Tasks:**
1. Define JSON-RPC message types using enums and structs:
   ```rust
   #[derive(Debug, Serialize, Deserialize)]
   #[serde(untagged)]
   pub enum Message {
       Request(Request),
       Response(Response),
       Notification(Notification),
   }
   ```
2. Implement serialization/deserialization with serde_json
3. Create custom error types with `thiserror` crate
4. Write unit tests using `#[cfg(test)]` modules
5. Handle malformed JSON with proper error types

**Rust Learning Focus:**
- **Ownership**: Understand move semantics vs. borrowing
- **Lifetimes**: Why references need lifetime annotations
- **Error handling**: `Result<T, E>`, `Option<T>`, and the `?` operator
- **Traits**: Derive macros (Debug, Clone, Serialize) and manual implementations
- **Pattern matching**: Exhaustive matching and destructuring

**System Engineering Concepts:**
- **Type-safe protocols**: Compiler prevents invalid messages at compile time
- **Zero-copy parsing**: serde_json works directly with borrowed data
- **Memory safety**: No null pointers, no dangling references

---

### Milestone 1.2: ACP Type System
**What You'll Build:**
```
src/
├── acp/
│   ├── mod.rs
│   ├── types.rs           # Core ACP types
│   ├── content.rs         # ContentBlock enum
│   ├── session.rs         # Session types
│   ├── tool_call.rs       # ToolCall types
│   └── capabilities.rs    # Capability negotiation
```

**Key Concepts:**
- Enum-based discriminated unions
- Struct field visibility and encapsulation
- Newtype pattern for type safety
- Builder pattern with consuming methods
- Serde attributes for JSON mapping

**Tasks:**
1. Define `ContentBlock` as a Rust enum:
   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize)]
   #[serde(tag = "type", rename_all = "snake_case")]
   pub enum ContentBlock {
       Text { text: String },
       Image { data: Vec<u8>, mime_type: String },
       Resource { uri: String, mime_type: String },
   }
   ```
2. Implement `ToolCall` with lifecycle states
3. Create `SessionMode` enum (Ask, Code, Architect)
4. Add type-safe builders for complex types
5. Write property-based tests with `proptest`

**Rust Learning Focus:**
- **Enums**: Algebraic data types with data attached to variants
- **Pattern matching**: Exhaustive, type-safe variant handling
- **Type safety**: Newtypes prevent mixing incompatible types
- **Derive macros**: Automatic trait implementations
- **Serde attributes**: Fine-grained serialization control

**Distributed Systems Concepts:**
- **Schema as types**: Protocol defined in type system, not docs
- **Compile-time validation**: Invalid messages rejected by compiler
- **Forward compatibility**: Non-exhaustive enums for protocol evolution

---

### Milestone 1.3: Stdio Transport & Process Management
**What You'll Build:**
```
src/
├── transport/
│   ├── mod.rs
│   ├── stdio.rs           # Stdin/stdout transport
│   └── framing.rs         # Newline-delimited framing
├── process/
│   ├── mod.rs
│   └── agent.rs           # Agent process spawner
```

**Key Concepts:**
- `std::process::Command` for spawning subprocesses
- `std::io::{BufReader, BufWriter}` for buffered I/O
- `std::sync::mpsc` channels for thread communication
- Smart pointers: `Box<T>`, `Arc<T>`, `Mutex<T>`
- RAII pattern for resource cleanup

**Tasks:**
1. Implement `StdioTransport` with `BufReader`/`BufWriter`
2. Spawn agent process and capture stdin/stdout
3. Parse newline-delimited JSON messages
4. Implement graceful shutdown with Drop trait
5. Add timeout handling with `std::time::Duration`

**Rust Learning Focus:**
- **Ownership transfer**: Moving values into closures and threads
- **RAII**: Resources cleaned up automatically via Drop
- **Interior mutability**: `Mutex<T>` for shared mutable state
- **Smart pointers**: `Box`, `Rc`, `Arc` and when to use each
- **Threads**: `std::thread::spawn` and message passing

**Systems Engineering Concepts:**
- **Process isolation**: Sandboxing via subprocess
- **IPC patterns**: Pipes, stdin/stdout communication
- **Resource cleanup**: Drop trait ensures no leaks
- **Deadlock prevention**: Structured concurrency patterns

---

## Phase 2: Async/Await & Concurrent Communication (Weeks 3-4)

### Learning Goals
- Tokio async runtime fundamentals
- Async/await syntax and futures
- Channels for async message passing
- Select! macro for concurrent operations
- Request/response correlation

### Milestone 2.1: Async Bidirectional Connection
**What You'll Build:**
```
src/
├── connection/
│   ├── mod.rs
│   ├── manager.rs         # Connection state machine
│   ├── dispatcher.rs      # Route messages
│   └── correlation.rs     # Match responses to requests
```

**Key Concepts:**
- `async fn` and `.await` syntax
- `tokio::spawn` for concurrent tasks
- `tokio::sync::mpsc` for async channels
- `tokio::select!` macro for racing futures
- `Arc<Mutex<T>>` vs `Arc<RwLock<T>>`

**Tasks:**
1. Convert stdio transport to async with tokio::io
2. Spawn separate tasks for reading and writing:
   ```rust
   tokio::spawn(async move {
       while let Some(msg) = rx.recv().await {
           // Send message to agent
       }
   });
   ```
3. Implement request correlation with `HashMap<RequestId, oneshot::Sender>`
4. Create notification dispatcher with `broadcast` channels
5. Handle connection errors and reconnection logic

**Rust Learning Focus:**
- **Async/await**: Futures, executors, and cooperative multitasking
- **Send + Sync traits**: Understanding thread-safety requirements
- **Pin and Unpin**: Why some futures can't be moved
- **Lifetimes in async**: `async fn` with references
- **Cancellation**: Using `tokio::select!` and cancellation tokens

**Concurrent Programming Concepts:**
- **Cooperative multitasking**: async yields control at await points
- **Work stealing**: Tokio's multi-threaded scheduler
- **Structured concurrency**: Tasks as scoped futures
- **Backpressure**: Bounded channels prevent unbounded queues

**Networking Concepts:**
- **Full-duplex async I/O**: Non-blocking reads and writes
- **Zero-copy I/O**: `tokio::io::split` for concurrent access
- **Cancellation-safe code**: Handling dropped futures correctly

---

### Milestone 2.2: Session Management with Async State
**What You'll Build:**
```
src/
├── session/
│   ├── mod.rs
│   ├── manager.rs         # Session lifecycle
│   ├── history.rs         # Message history
│   └── persistence.rs     # Async file I/O
```

**Key Concepts:**
- `tokio::fs` for async file operations
- `Arc<RwLock<T>>` for shared read access
- Atomic operations for session IDs
- Async trait methods (with `async-trait` crate)

**Tasks:**
1. Define `Session` struct with interior mutability
2. Implement async session CRUD operations
3. Add async file persistence with atomic writes
4. Use `tokio::fs::write` with temp files
5. Implement session history with bounded vectors

**Rust Learning Focus:**
- **Interior mutability patterns**: When to use `RwLock` vs `Mutex`
- **Async traits**: Limitations and the `async-trait` crate
- **Atomic types**: `AtomicU64` for lock-free counters
- **Copy-on-write**: Using `Arc::make_mut` for efficiency

**Backend Engineering Concepts:**
- **Read-heavy workloads**: RwLock for concurrent reads
- **State consistency**: Avoiding race conditions with locks
- **Durability**: Atomic file writes (temp + rename)

---

### Milestone 2.3: Event Stream with Tokio Broadcast
**What You'll Build:**
```
src/
├── events/
│   ├── mod.rs
│   ├── stream.rs          # Event types and dispatch
│   ├── subscriber.rs      # Subscription management
│   └── filter.rs          # Event filtering
```

**Key Concepts:**
- `tokio::sync::broadcast` for fan-out messaging
- Enum-based event types
- `tokio_stream::Stream` trait
- Error handling in async contexts

**Tasks:**
1. Define event enum with all event types:
   ```rust
   #[derive(Debug, Clone)]
   pub enum Event {
       SessionUpdate { session_id: String, state: SessionState },
       MessageChunk { content: String },
       ToolCall { call: ToolCall },
   }
   ```
2. Implement broadcast-based event bus
3. Add subscription filtering by session ID
4. Handle slow consumers (lag detection)
5. Create async stream adapters

**Rust Learning Focus:**
- **Clone trait**: Required for broadcast channels
- **Async streams**: `Stream` trait and `StreamExt`
- **Error propagation**: `Result` in async functions
- **Cancellation**: Handling dropped receivers

**Messaging Concepts:**
- **Pub/sub patterns**: Broadcast vs. mpsc channels
- **Backpressure handling**: Channel capacity and lag
- **Message filtering**: Efficient subscription management

---

## Phase 3: TUI Development with Ratatui (Weeks 5-6)

### Learning Goals
- Ratatui immediate-mode rendering
- Crossterm for terminal control
- Component-based architecture
- State management in UI layer
- Terminal event handling

### Milestone 3.1: Ratatui Foundation
**What You'll Build:**
```
src/
├── tui/
│   ├── mod.rs
│   ├── app.rs             # Main application state
│   ├── ui.rs              # Rendering logic
│   ├── events.rs          # Input handling
│   └── components/
│       └── mod.rs
```

**Key Concepts:**
- Immediate-mode rendering (redraw every frame)
- Terminal raw mode with crossterm
- Event loop with tokio
- Framerate control with `tokio::time::interval`

**Tasks:**
1. Set up Ratatui with crossterm backend:
   ```rust
   let backend = CrosstermBackend::new(std::io::stdout());
   let terminal = Terminal::new(backend)?;
   ```
2. Implement event loop with `tokio::select!`:
   ```rust
   loop {
       tokio::select! {
           _ = tick.tick() => { /* render */ }
           Some(event) = event_rx.recv() => { /* handle event */ }
       }
   }
   ```
3. Define `App` struct with application state
4. Implement rendering function with Ratatui widgets
5. Handle keyboard input with crossterm events

**Rust Learning Focus:**
- **Terminal setup**: Raw mode, alternate screen
- **Owned vs borrowed**: When to use `&str` vs `String` in widgets
- **Lifetimes in rendering**: Frame lifetime bounds
- **Error recovery**: Ensuring terminal cleanup on panic

**UI Architecture Concepts:**
- **Immediate-mode rendering**: Stateless render functions
- **Frame timing**: 60 FPS vs event-driven updates
- **Terminal state management**: Enter/exit alternate screen

---

### Milestone 3.2: Component System & Layouts
**What You'll Build:**
```
src/
├── tui/
│   └── components/
│       ├── chat_view.rs   # Message display
│       ├── input_box.rs   # User input
│       ├── status_bar.rs  # Status information
│       ├── tool_card.rs   # Tool execution display
│       └── modal.rs       # Modal dialogs
```

**Key Concepts:**
- Trait-based components
- Ratatui layout system (constraints, directions)
- Widget composition
- Stateful widgets with Ratatui

**Tasks:**
1. Define `Component` trait for reusable UI elements:
   ```rust
   pub trait Component {
       fn render(&mut self, f: &mut Frame, area: Rect);
       fn handle_event(&mut self, event: Event) -> anyhow::Result<()>;
   }
   ```
2. Implement scrollable chat view with `List` widget
3. Create multi-line input with `Paragraph` + state
4. Build status bar with connection indicators
5. Add modal dialogs for approval prompts

**Rust Learning Focus:**
- **Trait objects**: `Box<dyn Component>` for heterogeneous collections
- **Trait bounds**: Generic constraints on component types
- **Mutable references**: Exclusive access during rendering
- **Lifetime bounds**: Ensuring widget data lives long enough

**Rendering Concepts:**
- **Layout constraints**: Flexbox-like positioning
- **Widget state**: Maintaining scroll positions
- **Dirty tracking**: When to skip re-rendering

---

### Milestone 3.3: Streaming Text Display
**What You'll Build:**
```
src/
├── tui/
│   ├── markdown/
│   │   ├── mod.rs
│   │   ├── parser.rs      # Incremental markdown parsing
│   │   └── renderer.rs    # Render to styled text
│   └── streaming/
│       └── buffer.rs      # Incremental text buffer
```

**Key Concepts:**
- Text styling with ratatui::style
- Incremental string building
- ANSI color support
- Performance optimization for large text

**Tasks:**
1. Parse markdown incrementally as chunks arrive
2. Convert markdown to Ratatui `Text` with styles
3. Add syntax highlighting for code blocks (using `syntect`)
4. Implement word wrapping and text reflow
5. Optimize rendering for large message histories

**Rust Learning Focus:**
- **String vs &str**: When to allocate vs borrow
- **Cow<str>**: Clone-on-write for efficiency
- **Iterators**: Lazy processing of text chunks
- **Performance**: Profiling with cargo flamegraph

**Terminal Rendering Deep Dive:**
- **Immediate-mode efficiency**: Minimize allocations per frame
- **Styled text**: Ratatui's `Text` and `Span` types
- **ANSI escape codes**: Terminal color capabilities
- **Buffer management**: Reusing allocations with capacity hints

*Reference*: Unlike OpenTUI's Zig FFI approach, Ratatui uses immediate-mode rendering where you rebuild the UI each frame. This trades some CPU for simpler code and automatic memory management via Rust's ownership system.

---

## Phase 4: Tool Execution & Security (Weeks 7-8)

### Learning Goals
- Secure file system operations
- Process spawning and monitoring
- Permission system design
- Sandboxing techniques

### Milestone 4.1: File System Operations
**What You'll Build:**
```
src/
├── tools/
│   ├── mod.rs
│   ├── filesystem.rs      # read_text_file, write_text_file
│   ├── validation.rs      # Path sanitization
│   └── diff.rs            # Diff generation
```

**Key Concepts:**
- `std::path::PathBuf` for path manipulation
- Path canonicalization with `std::fs::canonicalize`
- Permission checking with `std::fs::metadata`
- Diff generation with `similar` crate

**Tasks:**
1. Implement path validation against workspace root
2. Add `read_text_file` with async file I/O
3. Implement atomic file writes (temp + rename):
   ```rust
   let temp_path = format!("{}.tmp.{}", path, uuid::Uuid::new_v4());
   tokio::fs::write(&temp_path, content).await?;
   tokio::fs::rename(temp_path, path).await?;
   ```
4. Generate unified diffs with `similar` crate
5. Reject path traversal attempts (e.g., `../`)

**Rust Learning Focus:**
- **Path types**: `Path`, `PathBuf`, and `AsRef<Path>`
- **Error types**: Converting between error types with `?`
- **Async I/O**: Understanding tokio::fs internals
- **Security**: Using type system to prevent vulnerabilities

**Security Engineering Concepts:**
- **Path canonicalization**: Resolve symlinks and relative paths
- **TOCTOU prevention**: Minimize time between check and use
- **Principle of least privilege**: Restrict to workspace only
- **Defense in depth**: Multiple validation layers

---

### Milestone 4.2: Process Management with Tokio
**What You'll Build:**
```
src/
├── tools/
│   ├── terminal.rs        # Process spawning
│   ├── output.rs          # Output capture
│   └── lifecycle.rs       # Process lifecycle
```

**Key Concepts:**
- `tokio::process::Command` for async process spawning
- `tokio::io::AsyncReadExt` for output streaming
- Timeout handling with `tokio::time::timeout`
- Signal handling with `nix` crate (Unix) or `windows-sys`

**Tasks:**
1. Implement async process spawning with tokio::process
2. Stream stdout/stderr in real-time to event bus
3. Add timeout enforcement with tokio::time::timeout
4. Implement graceful termination (SIGTERM → SIGKILL)
5. Handle zombie processes with proper cleanup

**Rust Learning Focus:**
- **Async process I/O**: Streaming output as it arrives
- **Timeouts**: `tokio::time::timeout` with futures
- **Error handling**: Process exit codes and signals
- **Platform differences**: Unix vs Windows process management

**Process Management Concepts:**
- **Process groups**: Killing entire process trees
- **Signal handling**: SIGTERM, SIGKILL, SIGCHLD
- **Exit codes**: Distinguishing success from failure
- **Resource limits**: CPU, memory, file descriptor limits

---

### Milestone 4.3: Permission System with Type Safety
**What You'll Build:**
```
src/
├── approval/
│   ├── mod.rs
│   ├── policy.rs          # Permission policies
│   ├── request.rs         # Permission requests
│   └── rules.rs           # Auto-approval rules
```

**Key Concepts:**
- Enum-based permission levels
- Type-state pattern for permission flow
- Channel-based approval requests
- Persistent approval rules

**Tasks:**
1. Define permission policy enum:
   ```rust
   #[derive(Debug, Clone, Copy)]
   pub enum Policy {
       AlwaysAllow,
       AlwaysAsk,
       DenyAll,
   }
   ```
2. Implement approval request flow with oneshot channels
3. Add TUI modal for user approval
4. Persist user decisions with serde
5. Implement rule-based auto-approval

**Rust Learning Focus:**
- **Type-state pattern**: Encoding state in types
- **Enums for state machines**: Representing permissions
- **Oneshot channels**: Single-use communication
- **Serialization**: Persisting rules with serde

**Security Concepts:**
- **Explicit authorization**: User consent required
- **Audit logging**: Record all permission decisions
- **Policy enforcement**: Type system prevents bypasses

---

## Phase 5: Backend Engineering Patterns (Weeks 9-10)

### Learning Goals
- Database integration with SQLx
- Structured logging with tracing
- Error handling strategies
- Observability and metrics

### Milestone 5.1: Database Layer with SQLx
**What You'll Build:**
```
src/
├── database/
│   ├── mod.rs
│   ├── pool.rs            # Connection pool
│   ├── migrations/        # SQL migrations
│   │   └── 001_init.sql
│   ├── models/
│   │   ├── session.rs
│   │   └── history.rs
│   └── repository/
│       ├── session_repo.rs
│       └── history_repo.rs
```

**Key Concepts:**
- SQLx for compile-time checked SQL
- Connection pooling with async
- Database migrations
- Repository pattern in Rust

**Tasks:**
1. Set up SQLite database with SQLx:
   ```rust
   let pool = SqlitePool::connect(&database_url).await?;
   sqlx::migrate!("./migrations").run(&pool).await?;
   ```
2. Define models with `sqlx::FromRow` derive
3. Implement repository pattern with async methods
4. Add compile-time query checking with `sqlx::query!`
5. Write integration tests with test database

**Rust Learning Focus:**
- **Compile-time SQL validation**: Catch SQL errors at compile time
- **Async traits**: Using `#[async_trait]` for repositories
- **Connection pooling**: `Arc<Pool>` for shared access
- **Macro hygiene**: Understanding `query!` and `query_as!` macros

**Database Engineering Concepts:**
- **Prepared statements**: Protection against SQL injection
- **Connection pooling**: Managing expensive resources
- **Migrations**: Versioning database schema
- **Repository pattern**: Abstracting data access

---

### Milestone 5.2: Structured Logging with Tracing
**What You'll Build:**
```
src/
├── logging/
│   ├── mod.rs
│   ├── setup.rs           # Logger initialization
│   ├── middleware.rs      # Request tracing
│   └── context.rs         # Span management
```

**Key Concepts:**
- `tracing` crate for structured logging
- Spans for request correlation
- Log levels (ERROR, WARN, INFO, DEBUG, TRACE)
- Tracing subscribers for output formatting

**Tasks:**
1. Set up tracing with `tracing-subscriber`:
   ```rust
   tracing_subscriber::fmt()
       .with_target(false)
       .with_thread_ids(true)
       .with_level(true)
       .json()
       .init();
   ```
2. Add spans for all async operations:
   ```rust
   #[tracing::instrument(skip(self))]
   async fn send_request(&self, req: Request) -> Result<Response> {
       // Automatically logged with context
   }
   ```
3. Implement request ID propagation
4. Add error logging with backtraces
5. Configure multiple outputs (stdout + file)

**Rust Learning Focus:**
- **Tracing spans**: Hierarchical context management
- **Procedural macros**: `#[instrument]` attribute
- **Subscribers**: Configuring tracing output
- **Error context**: Using `anyhow` with tracing

**Observability Concepts:**
- **Structured logging**: Machine-parseable logs
- **Trace context**: Following requests across boundaries
- **Span relationships**: Parent-child relationships
- **Log aggregation**: JSON output for centralized logging

---

### Milestone 5.3: Error Handling & Resilience
**What You'll Build:**
```
src/
├── errors/
│   ├── mod.rs
│   ├── types.rs           # Error types
│   ├── retry.rs           # Retry logic
│   └── circuit_breaker.rs # Circuit breaker
```

**Key Concepts:**
- Error type design with `thiserror`
- Retry logic with exponential backoff
- Circuit breaker pattern
- Error context with `anyhow`

**Tasks:**
1. Define domain-specific error types:
   ```rust
   #[derive(Debug, thiserror::Error)]
   pub enum ToolError {
       #[error("File not found: {path}")]
       FileNotFound { path: String },
       #[error("Permission denied")]
       PermissionDenied,
   }
   ```
2. Implement retry logic with `tokio-retry`:
   ```rust
   Retry::spawn(
       ExponentialBackoff::from_millis(10).take(3),
       || async { /* operation */ }
   ).await?
   ```
3. Add circuit breaker for agent connection
4. Provide rich error context to users
5. Implement panic handling in spawned tasks

**Rust Learning Focus:**
- **Error trait**: Implementing custom error types
- **Error composition**: `thiserror` for derives
- **Error context**: `anyhow` for application errors
- **Panic handling**: `catch_unwind` in FFI boundaries

**Distributed Systems Concepts:**
- **Retry strategies**: Exponential backoff, jitter
- **Circuit breaker**: Fail fast when backend is down
- **Timeout propagation**: Cascading timeouts
- **Graceful degradation**: Partial functionality during failures

---

### Milestone 5.4: Worker Pool & Concurrency Patterns
**What You'll Build:**
```
src/
├── pool/
│   ├── mod.rs
│   ├── worker.rs          # Worker task
│   ├── queue.rs           # Task queue
│   └── semaphore.rs       # Rate limiting
```

**Key Concepts:**
- `tokio::task::JoinSet` for task management
- Semaphores for resource limiting
- Bounded channels for backpressure
- Cancellation with `CancellationToken`

**Tasks:**
1. Implement worker pool with JoinSet:
   ```rust
   let mut set = JoinSet::new();
   for _ in 0..num_workers {
       set.spawn(worker_task(rx.clone()));
   }
   ```
2. Add semaphore-based rate limiting
3. Implement priority queue with heap
4. Handle graceful shutdown with CancellationToken
5. Add metrics for pool utilization

**Rust Learning Focus:**
- **Task spawning**: `tokio::spawn` vs `tokio::task::spawn_blocking`
- **Cancellation**: `CancellationToken` and `select!`
- **Semaphores**: `tokio::sync::Semaphore` for limiting
- **JoinSet**: Managing multiple concurrent tasks

**Backend Engineering Concepts:**
- **Worker pool pattern**: Fixed concurrency limit
- **Bounded queues**: Backpressure mechanism
- **Graceful shutdown**: Draining in-flight tasks
- **Rate limiting**: Token bucket, semaphore patterns

---

## Phase 6: Production Readiness (Weeks 11-12)

### Learning Goals
- Cross-platform builds
- Performance profiling
- Integration testing
- Distribution and packaging

### Milestone 6.1: Cross-Platform Support
**What You'll Build:**
```
Build artifacts for:
- linux (x86_64, aarch64, musl)
- macOS (x86_64, aarch64)
- Windows (x86_64)
```

**Key Concepts:**
- Cross-compilation with cargo
- Platform-specific code with `cfg` attributes
- Static linking with musl
- CI/CD with GitHub Actions

**Tasks:**
1. Add platform-specific code with conditional compilation:
   ```rust
   #[cfg(unix)]
   fn get_terminal_size() -> (u16, u16) {
       // Unix implementation
   }
   
   #[cfg(windows)]
   fn get_terminal_size() -> (u16, u16) {
       // Windows implementation
   }
   ```
2. Set up cross-compilation targets
3. Configure static linking for Linux (musl)
4. Embed version info with `build.rs`
5. Automate builds with cargo-dist

**Rust Learning Focus:**
- **Conditional compilation**: `#[cfg]` attributes
- **Build scripts**: build.rs for codegen
- **Cross-compilation**: cargo cross and targets
- **Static vs dynamic linking**: Trade-offs

---

### Milestone 6.2: Performance Profiling & Optimization
**What You'll Build:**
```
Profiling setup:
- CPU profiling with perf/Instruments
- Memory profiling with valgrind/heaptrack
- Flamegraphs for hotspot analysis
```

**Key Concepts:**
- Cargo bench for benchmarking
- Flamegraph generation
- Memory leak detection
- Allocation profiling

**Tasks:**
1. Write benchmarks with criterion.rs:
   ```rust
   fn bench_message_parse(c: &mut Criterion) {
       c.bench_function("parse json-rpc", |b| {
           b.iter(|| parse_message(black_box(JSON_DATA)))
       });
   }
   ```
2. Profile with cargo flamegraph
3. Reduce allocations in hot paths
4. Optimize string handling (use `&str` where possible)
5. Measure and optimize startup time

**Rust Learning Focus:**
- **Zero-cost abstractions**: Verifying optimization
- **Inline assembly**: Understanding codegen
- **Allocation tracking**: Where allocations happen
- **Benchmarking**: Criterion.rs for statistical analysis

**Performance Engineering:**
- **Flamegraphs**: Visualizing CPU time
- **Allocation profiling**: Finding memory hotspots
- **Inlining**: When to use `#[inline]`
- **SIMD**: Using portable_simd for data processing

---

### Milestone 6.3: Testing Strategy
**What You'll Build:**
```
tests/
├── integration/
│   ├── agent_test.rs      # End-to-end tests
│   └── fixtures/
└── common/
    └── mock_agent.rs      # Test doubles
```

**Key Concepts:**
- Unit tests with `#[cfg(test)]`
- Integration tests in `tests/`
- Property-based testing with `proptest`
- Test fixtures and mocking

**Tasks:**
1. Write unit tests for all modules (target 80%+ coverage)
2. Create integration tests for full workflows
3. Add property-based tests for parsers:
   ```rust
   proptest! {
       #[test]
       fn parse_any_valid_request(req in arb_request()) {
           let json = serde_json::to_string(&req)?;
           let parsed = parse_message(&json)?;
           assert_eq!(req, parsed);
       }
   }
   ```
4. Mock external dependencies with traits
5. Measure test coverage with cargo-tarpaulin

**Rust Learning Focus:**
- **Test organization**: Unit vs integration tests
- **Mocking**: Trait objects for test doubles
- **Property testing**: Generating random test cases
- **Coverage**: cargo-tarpaulin integration

**Testing Concepts:**
- **Test pyramid**: Unit → integration → e2e
- **Hermetic tests**: No external dependencies
- **Property-based testing**: Exploring input space
- **Test fixtures**: Reusable test data

---

## Learning Resources

### Rust Fundamentals
- **The Rust Programming Language** (The Book) - Official guide
- **Rust for Rustaceans** (Jon Gjengset) - Intermediate/advanced Rust
- **Programming Rust** (Blandy, Orendorff, Tindall) - Comprehensive reference
- **Rust by Example** - Learn by doing exercises
- **Rustlings** - Interactive exercises for learning Rust

### Async Rust
- **Asynchronous Programming in Rust** (Official async book)
- **Tokio Tutorial** - Official Tokio guide
- Jon Gjengset's "Crust of Rust" YouTube series on async
- **Async Rust** blog posts by Tyler Mandry

### Backend Engineering
- **Zero to Production in Rust** (Luca Palmieri) - Building production web services
- **Designing Data-Intensive Applications** (Kleppmann) - Essential backend reading
- **Database Internals** (Alex Petrov) - How databases work
- **Building Microservices** (Sam Newman)

### TUI Development
- [Ratatui Documentation](https://ratatui.rs/) - Official docs and tutorials
- [Ratatui Examples](https://github.com/ratatui/ratatui/tree/main/examples) - 40+ examples
- [Crossterm Documentation](https://docs.rs/crossterm/) - Terminal manipulation
- Building a Rust TUI with Ratatui (various blog posts)
- OpenTUI Architecture (opentui/ARCHITECTURE.md) - TypeScript/Zig, but concepts apply

### Rust-Specific Patterns
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) - API design
- [Rust Design Patterns](https://rust-unofficial.github.io/patterns/) - Common patterns
- [Effective Rust](https://www.lurklurk.org/effective-rust/) - Best practices
- [Common Rust Lifetime Misconceptions](https://github.com/pretzelhammer/rust-blog) - Deep dive

### Performance & Profiling
- [The Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Criterion.rs Guide](https://bheisler.github.io/criterion.rs/book/) - Benchmarking
- [Flamegraph](https://github.com/flamegraph-rs/flamegraph) - Profiling visualization

### Error Handling
- [Error Handling in Rust](https://blog.burntsushi.net/rust-error-handling/) - By BurntSushi
- thiserror and anyhow documentation
- [Error Handling Survey](https://blog.yoshuawuyts.com/error-handling-survey/) - Patterns overview

---

## Project Structure (Final)

```
codex-rs/
├── Cargo.toml
├── Cargo.lock
├── README.md
├── LICENSE
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── src/
│   ├── main.rs              # Binary entry
│   ├── lib.rs               # Library root
│   │
│   ├── jsonrpc/             # JSON-RPC 2.0
│   │   ├── mod.rs
│   │   ├── message.rs
│   │   ├── codec.rs
│   │   └── error.rs
│   │
│   ├── acp/                 # ACP protocol
│   │   ├── mod.rs
│   │   ├── types.rs
│   │   ├── content.rs
│   │   ├── session.rs
│   │   └── tool_call.rs
│   │
│   ├── transport/           # Transport layer
│   │   ├── mod.rs
│   │   └── stdio.rs
│   │
│   ├── connection/          # Connection management
│   │   ├── mod.rs
│   │   ├── manager.rs
│   │   └── correlation.rs
│   │
│   ├── session/             # Session state
│   │   ├── mod.rs
│   │   ├── manager.rs
│   │   └── persistence.rs
│   │
│   ├── events/              # Event system
│   │   ├── mod.rs
│   │   └── stream.rs
│   │
│   ├── tools/               # Tool implementations
│   │   ├── mod.rs
│   │   ├── filesystem.rs
│   │   └── terminal.rs
│   │
│   ├── approval/            # Permission system
│   │   ├── mod.rs
│   │   └── policy.rs
│   │
│   ├── database/            # Database layer
│   │   ├── mod.rs
│   │   ├── pool.rs
│   │   ├── models/
│   │   └── repository/
│   │
│   ├── pool/                # Worker pool
│   │   ├── mod.rs
│   │   └── worker.rs
│   │
│   ├── tui/                 # Terminal UI
│   │   ├── mod.rs
│   │   ├── app.rs
│   │   ├── ui.rs
│   │   ├── events.rs
│   │   ├── components/
│   │   ├── markdown/
│   │   └── streaming/
│   │
│   └── logging/             # Logging setup
│       ├── mod.rs
│       └── setup.rs
│
├── tests/                   # Integration tests
│   ├── integration/
│   └── common/
│
├── benches/                 # Benchmarks
│   └── message_parsing.rs
│
├── migrations/              # Database migrations
│   └── 001_init.sql
│
└── docs/                    # Documentation
    ├── architecture.md
    └── examples/
```

---

## Weekly Breakdown

| Week | Phase | Focus | Deliverable | Rust Skills |
|------|-------|-------|-------------|-------------|
| 1 | Phase 1 | JSON-RPC & Types | Message encoder/decoder | Ownership, enums, Result |
| 2 | Phase 1 | ACP Types & Stdio | Can spawn agent and communicate | Lifetimes, traits, I/O |
| 3 | Phase 2 | Async Basics | Tokio runtime setup | async/await, futures |
| 4 | Phase 2 | Async Connection | Bidirectional communication | Channels, select!, Arc |
| 5 | Phase 3 | Ratatui Setup | Basic TUI rendering | Terminal APIs, lifetimes |
| 6 | Phase 3 | Components | Polished UI with widgets | Traits, composition |
| 7 | Phase 4 | File Tools | Secure file operations | Path types, validation |
| 8 | Phase 4 | Process Management | Command execution | Process APIs, signals |
| 9 | Phase 5 | Database & Logging | SQLx + tracing integrated | Macros, spans, queries |
| 10 | Phase 5 | Errors & Workers | Resilience patterns | Error types, JoinSet |
| 11 | Phase 6 | Cross-platform | Builds on all targets | cfg attributes, linking |
| 12 | Phase 6 | Polish & Testing | Production-ready | Benchmarks, coverage |

---

## Success Criteria

By the end of this project, you should be able to:

✅ Write idiomatic Rust with proper ownership and borrowing  
✅ Handle lifetimes confidently in complex scenarios  
✅ Build async applications with Tokio  
✅ Implement type-safe protocols with enums and traits  
✅ Use the compiler to catch bugs at compile time  
✅ Build cross-platform TUI applications  
✅ Integrate databases with compile-time query checking  
✅ Apply structured logging and tracing  
✅ Write comprehensive tests (unit + integration + property)  
✅ Profile and optimize Rust code  
✅ Deploy cross-compiled binaries  
✅ Design resilient backend systems  

**Core Skills Gained:**
- **Rust**: Ownership, lifetimes, traits, async/await, error handling
- **Systems**: Process management, I/O, memory safety, zero-cost abstractions
- **Backend**: Database integration, logging, resilience, concurrency
- **Async**: Tokio runtime, futures, channels, select!, cancellation
- **Type Safety**: Enums, pattern matching, type-state pattern
- **Performance**: Zero-cost abstractions, profiling, optimization

**Interview-Ready Knowledge:**
After completing this project, you can confidently discuss:
- "Explain Rust's ownership system" → References, borrowing, lifetimes
- "How does async Rust differ from Go?" → Compile-time futures vs goroutines
- "What are zero-cost abstractions?" → Monomorphization, inlining examples
- "How do you handle errors in Rust?" → Result, ?, thiserror, anyhow
- "What's the difference between Arc and Rc?" → Thread safety, atomic refcounting
- "How do you optimize Rust code?" → Profiling workflow, allocation reduction
- "Explain Send and Sync traits" → Thread safety guarantees

**Rust vs Go Comparison:**
- **Memory safety**: Compile-time guarantees vs runtime checks
- **Concurrency**: async/await vs goroutines (cooperative vs preemptive)
- **Error handling**: Result/Option vs multiple returns
- **Performance**: Zero-cost abstractions vs garbage collection
- **Learning curve**: Steeper initially, but compiler teaches best practices
- **Type system**: More powerful (enums, traits) vs simpler (interfaces)

---

## Next Steps

1. **Set up development environment:**
   ```bash
   # Install Rust via rustup
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   
   # Create project
   cargo new codex-rs --bin
   cd codex-rs
   
   # Add dependencies to Cargo.toml (see below)
   ```

2. **Initial dependencies (Cargo.toml):**
   ```toml
   [dependencies]
   # Async runtime
   tokio = { version = "1.40", features = ["full"] }
   
   # Serialization
   serde = { version = "1.0", features = ["derive"] }
   serde_json = "1.0"
   
   # TUI
   ratatui = "0.28"
   crossterm = "0.28"
   
   # Error handling
   thiserror = "1.0"
   anyhow = "1.0"
   
   # Logging
   tracing = "0.1"
   tracing-subscriber = { version = "0.3", features = ["json"] }
   
   # Database (added later)
   sqlx = { version = "0.8", features = ["runtime-tokio", "sqlite"] }
   
   [dev-dependencies]
   criterion = "0.5"
   proptest = "1.4"
   tokio-test = "0.4"
   ```

3. **Start with Milestone 1.1** and work sequentially.

4. **Study Rust-specific resources:**
   - Work through [The Rust Book](https://doc.rust-lang.org/book/) chapters 1-10 first
   - Do [Rustlings](https://rustlings.cool/) exercises for ownership/borrowing
   - Read [Async Book](https://rust-lang.github.io/async-book/) before Phase 2
   - Study [Ratatui examples](https://github.com/ratatui/ratatui/tree/main/examples) before Phase 3

5. **Join communities:**
   - [Rust Users Forum](https://users.rust-lang.org/)
   - [Rust Discord](https://discord.gg/rust-lang)
   - [r/rust](https://reddit.com/r/rust)
   - [Ratatui Discord](https://discord.gg/pMCEU9hNEj)

6. **Set up your learning journal:**
   ```bash
   mkdir -p docs/learning-journal
   # Document: "Today I learned about lifetimes because..."
   # Track: Compiler error → understanding → solution
   ```

## How This Makes You a Better Backend Engineer

This project uniquely teaches:

✅ **Memory safety without GC** - Understanding low-level control with high-level safety  
✅ **Type-driven development** - Compiler as pair programmer  
✅ **Zero-cost abstractions** - Performance without sacrificing readability  
✅ **Fearless concurrency** - Async patterns without data races  
✅ **Compile-time guarantees** - Catch bugs before runtime  
✅ **Robust error handling** - Explicit error propagation  
✅ **Systems thinking** - Close to the metal, but safe  

**Real-world parallels:**
- JSON-RPC implementation → gRPC services, REST APIs
- Async Tokio patterns → High-performance web servers
- SQLx compile-time queries → Type-safe database access
- Worker pools → Task queues, job processing
- Tracing integration → Production observability
- Process management → Orchestration, sandboxing

**Why Rust for backend:**
- **Performance**: C/C++ speed with memory safety
- **Reliability**: Compiler catches most bugs before production
- **Concurrency**: Async I/O without data races
- **Ecosystem**: Growing adoption (AWS, Discord, Cloudflare use Rust)
- **Career value**: Rust skills are increasingly in demand

By building this TUI in Rust, you'll gain transferable skills for:
- **Web services**: Actix, Axum, Rocket frameworks
- **Databases**: Implementing storage engines
- **CLI tools**: Ripgrep, fd, bat are all Rust
- **Systems programming**: OS components, drivers
- **WebAssembly**: Rust is first-class WASM citizen

Good luck building! This project will give you a deep understanding of Rust and systems programming. 🦀
