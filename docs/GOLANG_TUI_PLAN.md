# Building a Golang TUI Agent using Agent-Client-Protocol
## Learning-Oriented Implementation Plan

---

## Executive Summary

You'll build a fully-featured Golang Terminal User Interface (TUI) that implements the Agent-Client-Protocol (ACP) to communicate with AI coding agents. This project is structured as a progressive learning journey covering core software engineering, systems programming, distributed systems, networking, and concurrent messaging patterns.

**Tech Stack:**
- Go 1.21+ (concurrency, channels, goroutines)
- Bubble Tea (TUI framework, inspired by Elm architecture)
- JSON-RPC 2.0 (protocol layer)
- stdio/IPC (process communication)
- Optional: gRPC for future extensions

---

## Phase 1: Foundations & Protocol Basics (Weeks 1-2)

### Learning Goals
- Go project structure & module management
- JSON-RPC 2.0 protocol fundamentals
- stdin/stdout as transport layer
- Basic process spawning & IPC

### Milestone 1.1: JSON-RPC 2.0 Message Handler
**What You'll Build:**
```
codex-go/
├── go.mod
├── cmd/
│   └── codex-client/
│       └── main.go          # Entry point
├── pkg/
│   ├── jsonrpc/
│   │   ├── message.go       # Request/Response/Notification types
│   │   ├── codec.go         # JSON encoding/decoding
│   │   └── codec_test.go
│   └── transport/
│       ├── stdio.go         # stdin/stdout transport
│       └── stdio_test.go
```

**Key Concepts:**
- Go structs with JSON tags (`json:"jsonrpc"`)
- Interface design for transport abstraction
- `io.Reader` and `io.Writer` interfaces
- Newline-delimited JSON parsing
- Error handling patterns (errors.Is, errors.As)

**Tasks:**
1. Define JSON-RPC message types (Request, Response, Notification, Error)
2. Implement `Codec` interface for encoding/decoding
3. Build `StdioTransport` with newline-delimited JSON
4. Write unit tests using Go's testing package
5. Handle malformed JSON gracefully

**System Engineering Concepts:**
- **Transport layer abstraction**: Why stdin/stdout? (Language-agnostic, cross-platform)
- **Framing protocols**: Newline-delimited vs. length-prefixed messages
- **Buffered I/O**: `bufio.Scanner` for line-by-line reading

---

### Milestone 1.2: ACP Type System
**What You'll Build:**
```
pkg/
├── acp/
│   ├── types.go           # Core ACP types
│   ├── content.go         # ContentBlock types
│   ├── session.go         # Session types
│   ├── toolcall.go        # ToolCall types
│   └── capabilities.go    # Capability negotiation
```

**Key Concepts:**
- Go struct embedding for polymorphism
- JSON unmarshaling with discriminated unions
- Enum patterns in Go (iota, string constants)
- Builder pattern for complex types

**Tasks:**
1. Translate ACP TypeScript/Rust types to Go
2. Implement `ContentBlock` as a discriminated union:
   - Text, Image, Audio, ResourceLink, Resource
3. Define `ToolCall` with status lifecycle (pending → in_progress → completed/failed)
4. Create `SessionMode` enum (ask, code, architect)
5. Add custom `UnmarshalJSON` for union types

**Distributed Systems Concepts:**
- **Capability negotiation**: How clients/servers discover features
- **Protocol versioning**: Forward/backward compatibility
- **Schema evolution**: Adding fields without breaking changes

---

### Milestone 1.3: Initialization Handshake
**What You'll Build:**
```
pkg/
├── acp/
│   ├── client.go          # Client-side ACP implementation
│   └── handshake.go       # Initialize request/response
├── agent/
│   └── spawn.go           # Agent process spawner
```

**Key Concepts:**
- Process spawning with `os/exec`
- Bidirectional pipe communication
- Context cancellation for timeouts
- Structured logging (slog or zap)

**Tasks:**
1. Implement `AgentProcess` struct to manage spawned agent
2. Create `Initialize()` method with capability negotiation
3. Add timeout handling with `context.WithTimeout`
4. Verify protocol version compatibility
5. Log initialization steps

**Systems Engineering Concepts:**
- **Process lifecycle**: spawn → initialize → session → shutdown
- **Timeouts & deadlocks**: Why every network call needs a timeout
- **Graceful shutdown**: Signal handling (SIGTERM, SIGINT)

---

## Phase 2: Core Agent Communication (Weeks 3-4)

### Learning Goals
- Concurrent programming with goroutines & channels
- Request/response correlation (request ID tracking)
- Event-driven architecture
- Stateful connection management

### Milestone 2.1: Bidirectional Connection Manager
**What You'll Build:**
```
pkg/
├── connection/
│   ├── manager.go         # Connection state machine
│   ├── dispatcher.go      # Route requests/responses
│   └── correlation.go     # Match responses to requests
```

**Key Concepts:**
- Goroutine per connection for reading/writing
- Channels for message passing
- `sync.Mutex` for shared state
- `sync.WaitGroup` for goroutine coordination
- Request ID generation (UUID or atomic counter)

**Tasks:**
1. Create `Connection` struct with read/write goroutines
2. Implement request correlation table (`map[string]chan Response`)
3. Build notification dispatcher (publish-subscribe pattern)
4. Handle connection errors and reconnection logic
5. Add metrics: requests sent/received, latency

**Concurrent Programming Concepts:**
- **Producer-consumer pattern**: Writer goroutine consumes from send channel
- **Fan-out pattern**: Single reader goroutine dispatches to multiple handlers
- **Synchronization primitives**: When to use channels vs. mutexes
- **Deadlock prevention**: Never hold a lock while waiting on a channel

**Networking Concepts:**
- **Full-duplex communication**: Both sides can initiate requests
- **Head-of-line blocking**: Why multiplexing matters
- **Flow control**: Buffered channels as backpressure mechanism

---

### Milestone 2.2: Session Management
**What You'll Build:**
```
pkg/
├── session/
│   ├── session.go         # Session state
│   ├── history.go         # Conversation history
│   └── persistence.go     # Save/load sessions
```

**Key Concepts:**
- Stateful vs. stateless design
- JSON serialization for persistence
- File I/O and atomic writes
- CRUD operations in memory

**Tasks:**
1. Define `Session` struct with conversation history
2. Implement `session/new` request handling
3. Add `session/prompt` for user messages
4. Store session state in `~/.codex-go/sessions/`
5. Implement atomic file writes (write temp → rename)

**Distributed Systems Concepts:**
- **State management**: Where does state live? (Client vs. agent)
- **Durability**: Write-ahead logging for crash recovery
- **Idempotency**: Handling duplicate requests safely

---

### Milestone 2.3: Event Stream Processing
**What You'll Build:**
```
pkg/
├── events/
│   ├── stream.go          # SSE-style event handling
│   ├── handlers.go        # Event type handlers
│   └── subscriber.go      # Event subscription system
```

**Key Concepts:**
- Observer pattern
- Type-safe event dispatching
- Buffered channels to prevent blocking

**Tasks:**
1. Define event types (session/update, agent_message_chunk, tool_call, etc.)
2. Implement `EventStream` with subscription API
3. Create handlers for each event type
4. Add filtering by session ID
5. Handle slow consumers (drop or buffer?)

**Messaging Concepts:**
- **Pub/Sub architecture**: Decoupling producers from consumers
- **At-most-once vs. at-least-once delivery**: Trade-offs
- **Message ordering**: FIFO guarantees within a session
- **Backpressure handling**: What happens when consumer is slow?

---

## Phase 3: TUI Development (Weeks 5-6)

### Learning Goals
- Event-driven UI architecture (Elm/Redux pattern)
- Terminal rendering & layout
- State management in UI layer
- Reactive programming patterns

### Milestone 3.1: Bubble Tea Foundation
**What You'll Build:**
```
cmd/
├── codex-client/
│   └── main.go
internal/
├── tui/
│   ├── app.go             # Main Bubble Tea model
│   ├── update.go          # Event handlers
│   ├── view.go            # Rendering logic
│   └── commands.go        # Side effects (I/O)
```

**Key Concepts:**
- Model-View-Update (MVU) architecture
- Immutable state updates
- Commands as side effects
- Messages as events

**Tasks:**
1. Set up Bubble Tea application skeleton
2. Define `Model` struct with application state
3. Implement `Init()`, `Update()`, and `View()` methods
4. Handle keyboard input (tea.KeyMsg)
5. Render basic UI with lipgloss for styling

**UI Architecture Concepts:**
- **Unidirectional data flow**: Events → Update → Model → View
- **Pure functions**: View should be deterministic
- **Command pattern**: Wrapping side effects
- **Component composition**: Nesting sub-models

---

### Milestone 3.2: Layout & Component System
**What You'll Build:**
```
internal/
├── tui/
│   ├── components/
│   │   ├── chatview.go    # Message display
│   │   ├── inputbox.go    # User input
│   │   ├── statusbar.go   # Bottom status
│   │   ├── toolcall.go    # Tool execution UI
│   │   └── approval.go    # Permission prompts
│   └── layout/
│       └── flexbox.go     # Flexible layout
```

**Key Concepts:**
- Component-based architecture
- Nested state management
- Event bubbling and delegation
- Viewport for scrolling content

**Tasks:**
1. Implement scrollable chat view with message history
2. Create multi-line input box with syntax highlighting
3. Build status bar with session info & connection state
4. Add tool call cards with progress indicators
5. Design approval prompt modal for permissions

**Rendering Concepts:**
- **Retained mode vs. immediate mode**: Terminal is immediate mode
- **Dirty regions**: Only re-render what changed
- **Layout algorithms**: Flexbox-style sizing
- **ANSI escape sequences**: Colors, cursor control

---

### Milestone 3.3: Streaming Message Display
**What You'll Build:**
```
internal/
├── tui/
│   ├── markdown/
│   │   ├── renderer.go    # Markdown → ANSI
│   │   └── codeblock.go   # Syntax highlighting
│   └── streaming/
│       └── buffer.go      # Incremental rendering
```

**Key Concepts:**
- Incremental parsing
- Markdown rendering (glamour library)
- Syntax highlighting (chroma library)
- Double buffering for flicker-free updates

**Tasks:**
1. Parse markdown incrementally as chunks arrive
2. Render inline code, bold, italics, headers
3. Add syntax highlighting for code blocks (Go, Python, Bash, etc.)
4. Implement word wrapping and reflow
5. Handle ANSI color codes from agent

**Real-Time Systems Concepts:**
- **Latency vs. throughput**: Prioritize low latency for UX
- **Buffering strategies**: Line buffering vs. byte streaming
- **Frame rate limiting**: Don't redraw faster than 60 FPS

**Terminal Rendering Deep Dive:**
Learn how high-performance TUI frameworks optimize rendering:
- **Double buffering**: Maintain previous and current frame buffers
- **Diff-based updates**: Only output ANSI codes for changed cells
- **Cell-based buffers**: Store (char, fg_color, bg_color, attributes) per cell
- **Zero-copy techniques**: Reuse buffers to minimize allocations
- **ANSI optimization**: Batch cursor movements, minimize escape sequences

*Reference*: Study OpenTUI's architecture (opentui/ARCHITECTURE.md) which uses:
- TypeScript → Zig FFI for zero-copy buffer sharing
- OptimizedBuffer with typed arrays (Uint32Array for chars, Float32Array for colors)
- Native Zig renderer for ANSI generation and screen diffing

---

## Phase 4: Tool Execution & Permissions (Weeks 7-8)

### Learning Goals
- Implementing agent capabilities
- Security & sandboxing
- File system operations
- Command execution patterns

### Milestone 4.1: File System Operations
**What You'll Build:**
```
pkg/
├── tools/
│   ├── filesystem.go      # fs/read_text_file, fs/write_text_file
│   ├── validation.go      # Path validation
│   └── sandbox.go         # Path restrictions
```

**Key Concepts:**
- Path traversal prevention
- Atomic file operations
- Permission checks
- Diff generation

**Tasks:**
1. Implement `fs/read_text_file` handler
2. Add `fs/write_text_file` with atomic writes
3. Generate unified diffs (go-diff library)
4. Validate paths against workspace root
5. Reject paths outside sandbox (no `../` escapes)

**Security Engineering Concepts:**
- **Least privilege principle**: Only grant necessary permissions
- **Path canonicalization**: Resolve symlinks and `.` / `..`
- **TOCTOU vulnerabilities**: Time-of-check-to-time-of-use races
- **Defense in depth**: Multiple layers of validation

---

### Milestone 4.2: Terminal Operations
**What You'll Build:**
```
pkg/
├── tools/
│   ├── terminal.go        # Process management
│   └── output_buffer.go   # Command output capture
```

**Key Concepts:**
- Process lifecycle management
- stdout/stderr multiplexing
- Exit code handling
- Timeout enforcement

**Tasks:**
1. Implement `terminal/create` to spawn processes
2. Capture output with `io.MultiWriter`
3. Add `terminal/output` to poll captured output
4. Implement `terminal/wait_for_exit` with timeouts
5. Handle `terminal/kill` and `terminal/release`

**Process Management Concepts:**
- **Zombie processes**: Always wait on children
- **Signal handling**: SIGTERM vs. SIGKILL
- **Process groups**: Killing entire process trees
- **PTY vs. pipe**: When to allocate a pseudo-terminal

---

### Milestone 4.3: Permission System
**What You'll Build:**
```
pkg/
├── approval/
│   ├── policy.go          # Approval policy engine
│   ├── prompt.go          # User prompts
│   └── rules.go           # Auto-approve rules
internal/
├── tui/
│   └── components/
│       └── approval_dialog.go
```

**Key Concepts:**
- Policy-based access control
- Modal dialogs in TUI
- User interaction state machines

**Tasks:**
1. Define approval policies: none, ask, review
2. Implement permission request flow (session/request_permission)
3. Create TUI modal for user approval
4. Add auto-approval rules (e.g., "allow all reads")
5. Store user preferences

**Security Concepts:**
- **Explicit authorization**: User must opt-in to dangerous operations
- **Audit logging**: Record all approvals/denials
- **Revocation**: Ability to cancel in-flight operations

---

## Phase 5: Advanced Features (Weeks 9-10)

### Learning Goals
- Distributed tracing & observability
- Configuration management
- Error recovery strategies
- Testing async systems

### Milestone 5.1: Observability & Logging
**What You'll Build:**
```
pkg/
├── observability/
│   ├── tracing.go         # OpenTelemetry integration
│   ├── metrics.go         # Prometheus metrics
│   └── logging.go         # Structured logging
```

**Key Concepts:**
- Distributed tracing (OpenTelemetry)
- Structured logging (zerolog or zap)
- Metrics collection (request counts, latencies)
- Context propagation

**Tasks:**
1. Add OpenTelemetry tracing to all RPC calls
2. Emit span events for key operations
3. Collect metrics: request rate, error rate, latency percentiles
4. Implement structured logging with correlation IDs
5. Export traces to Jaeger or Zipkin

**Observability Concepts:**
- **Three pillars**: Logs, metrics, traces
- **Trace context**: Propagating trace IDs across boundaries
- **Sampling strategies**: Head-based vs. tail-based sampling
- **SLIs/SLOs**: Measuring service health

---

### Milestone 5.2: Configuration Management
**What You'll Build:**
```
pkg/
├── config/
│   ├── loader.go          # Multi-source config
│   ├── types.go           # Config struct
│   └── validation.go      # Config validation
```

**Key Concepts:**
- Configuration cascade (files → env → flags)
- TOML/YAML parsing
- Environment variable overrides
- Config hot-reloading (optional)

**Tasks:**
1. Define config schema (model, sandbox mode, MCP servers, etc.)
2. Load from `~/.codex-go/config.toml`
3. Override with environment variables
4. Override with CLI flags (cobra library)
5. Validate config at startup

**Software Engineering Concepts:**
- **12-factor app**: Configuration via environment
- **Configuration as code**: Version-controlled defaults
- **Fail-fast principle**: Validate early, fail early

---

### Milestone 5.3: Error Handling & Recovery
**What You'll Build:**
```
pkg/
├── errors/
│   ├── types.go           # ACP error codes
│   ├── retry.go           # Retry logic with backoff
│   └── recovery.go        # Panic recovery
```

**Key Concepts:**
- Exponential backoff
- Circuit breaker pattern
- Panic recovery in goroutines
- Error wrapping and context

**Tasks:**
1. Map Go errors to JSON-RPC error codes
2. Implement retry logic with exponential backoff
3. Add circuit breaker for agent connection
4. Recover from panics in goroutines (with defer/recover)
5. Provide rich error context to users

**Distributed Systems Concepts:**
- **Partial failures**: Network can fail independently
- **Idempotency**: Safe to retry operations
- **Timeout propagation**: Deadlines across call chains
- **Failure isolation**: Prevent cascading failures

---

### Milestone 5.4: Testing Strategy
**What You'll Build:**
```
test/
├── integration/
│   ├── agent_test.go      # End-to-end tests
│   └── mock_agent.go      # Fake agent for testing
pkg/
└── */
    └── *_test.go          # Unit tests
```

**Key Concepts:**
- Table-driven tests
- Test fixtures
- Mocking interfaces
- Integration vs. unit tests

**Tasks:**
1. Write unit tests for all packages (aim for 80%+ coverage)
2. Create mock agent that implements ACP protocol
3. Add integration tests for session lifecycle
4. Test error scenarios (disconnection, timeouts, malformed JSON)
5. Benchmark critical paths (message encoding, rendering)

**Testing Concepts:**
- **Test pyramid**: Unit → Integration → E2E
- **Hermetic tests**: No external dependencies
- **Test doubles**: Mocks, stubs, fakes
- **Property-based testing**: Generate random inputs (gopter)

---

## Phase 6: Production Polish (Weeks 11-12)

### Learning Goals
- Cross-platform support
- Packaging & distribution
- Documentation
- Performance optimization

### Milestone 6.1: Cross-Platform Support
**What You'll Build:**
```
Build artifacts for:
- linux/amd64, linux/arm64
- darwin/amd64, darwin/arm64
- windows/amd64
```

**Key Concepts:**
- Build tags for platform-specific code
- Cross-compilation with Go
- Static linking
- Release automation (goreleaser)

**Tasks:**
1. Add platform-specific sandbox code (Linux: seccomp, macOS: sandbox-exec)
2. Use goreleaser for multi-platform builds
3. Embed version info at build time
4. Test on all target platforms
5. Package as single static binary

---

### Milestone 6.2: Performance Optimization
**What You'll Build:**
```
Profiling and optimization:
- CPU profiling
- Memory profiling
- Benchmarking critical paths
```

**Key Concepts:**
- pprof for profiling
- Escape analysis
- Reducing allocations
- Optimizing hot paths

**Tasks:**
1. Profile application with pprof (CPU, memory, goroutines)
2. Reduce allocations in hot paths (reuse buffers, sync.Pool)
3. Benchmark rendering performance
4. Optimize JSON parsing (ffjson or jsoniter if needed)
5. Measure startup time and reduce it

**Performance Engineering Concepts:**
- **Premature optimization**: Measure before optimizing
- **Allocation profiling**: Stack vs. heap allocations
- **Cache-friendly data structures**: Locality of reference
- **Lock contention**: Minimize critical sections

**Rendering Performance Techniques (inspired by OpenTUI):**
While OpenTUI uses Zig for native performance, apply these concepts in Go:
- **Buffer reuse**: Use `sync.Pool` for terminal buffers
- **String builder**: Use `strings.Builder` instead of string concatenation
- **Byte slices**: Prefer `[]byte` over `string` for ANSI generation
- **Inline functions**: Small functions get inlined by compiler
- **Avoid interface boxing**: Direct struct methods are faster
- **Benchmark rendering**: Target < 16ms per frame (60 FPS)

---

### Milestone 6.3: Documentation & Examples
**What You'll Build:**
```
docs/
├── architecture.md        # This document
├── getting-started.md     # Quick start guide
├── configuration.md       # Config reference
├── contributing.md        # Dev guide
└── examples/
    ├── basic-session.md
    ├── mcp-integration.md
    └── custom-tools.md
```

**Tasks:**
1. Write comprehensive README with installation instructions
2. Document all configuration options
3. Add code examples for common use cases
4. Create architecture diagrams (mermaid)
5. Record demo GIF/video of TUI in action

---

## Optional Extensions (Post-MVP)

### Extension 1: MCP Client Integration
Implement MCP client to connect to external MCP servers (filesystem, database, APIs).

**Learning:**
- HTTP/SSE client implementation
- Tool discovery and schema validation
- Dynamic tool registration

### Extension 2: Agent as MCP Server
Expose your TUI client as an MCP server that other tools can connect to.

**Learning:**
- Implementing server-side protocol
- Bidirectional transformation (client ↔ server)

### Extension 3: Collaborative Sessions
Multiple users sharing the same session over network.

**Learning:**
- Operational transformation (OT) or CRDTs
- WebSocket server
- Conflict resolution

### Extension 4: Plugin System
Load custom tools as Go plugins.

**Learning:**
- Go plugin system (`plugin` package)
- Dynamic loading and versioning
- Sandboxing untrusted plugins

### Extension 5: Advanced Sandboxing
Implement OS-specific sandboxing.

**Learning:**
- Linux: landlock, seccomp-bpf, namespaces
- macOS: Seatbelt (sandbox-exec)
- Windows: AppContainer

---

## Backend Engineering Deep Dives

### Phase 2.5: Advanced Concurrency Patterns (Insert Between Phase 2 & 3)

#### Learning Goals
- Worker pool patterns
- Context-based cancellation propagation
- Rate limiting & throttling
- Semaphore patterns for resource management

#### Milestone 2.5: Concurrent Task Execution
**What You'll Build:**
```
pkg/
├── pool/
│   ├── worker_pool.go      # Worker pool implementation
│   ├── task_queue.go       # Bounded task queue
│   └── rate_limiter.go     # Token bucket rate limiter
```

**Key Concepts:**
- Bounded worker pools with `sync.WaitGroup`
- Select statement for multiple channel operations
- `sync.Pool` for object reuse
- Rate limiting with token buckets or sliding windows

**Tasks:**
1. Implement a worker pool for concurrent tool execution
2. Add task queue with priority handling
3. Build rate limiter for API calls
4. Implement graceful shutdown with context cancellation
5. Add metrics for pool utilization

**Backend Engineering Concepts:**
- **Worker pool pattern**: Fixed number of goroutines processing tasks
- **Bounded queues**: Backpressure through queue limits
- **Context propagation**: Cancellation across goroutine boundaries
- **Resource pooling**: Reusing expensive objects (connections, buffers)

---

### Phase 4.5: Database & Persistence (Insert Between Phase 4 & 5)

#### Learning Goals
- SQL database design & optimization
- Transaction management
- Connection pooling
- Query performance optimization

#### Milestone 4.5: SQLite Integration
**What You'll Build:**
```
pkg/
├── database/
│   ├── db.go              # Database connection
│   ├── migrations/        # Schema migrations
│   │   ├── 001_init.sql
│   │   └── migrate.go
│   ├── models/
│   │   ├── session.go     # Session model
│   │   └── history.go     # History model
│   └── repository/
│       ├── session_repo.go
│       └── history_repo.go
```

**Key Concepts:**
- Database/sql interface
- Prepared statements for security
- Transaction isolation levels
- Connection pooling configuration
- Query builders vs. raw SQL

**Tasks:**
1. Set up SQLite with database/sql
2. Implement schema migrations system
3. Create repository pattern for data access
4. Add transaction management
5. Implement connection pool tuning
6. Add query logging and slow query detection

**Database Engineering Concepts:**
- **Repository pattern**: Abstracting data access
- **Unit of Work**: Managing transactions
- **Connection pooling**: MaxOpenConns, MaxIdleConns, ConnMaxLifetime
- **N+1 query problem**: Eager loading vs. lazy loading
- **Database normalization**: 3NF design
- **Indexing strategies**: B-tree vs. hash indexes

---

### Phase 5.5: Structured Logging & Instrumentation (Enhance Phase 5)

#### Milestone 5.5: Production-Grade Logging
**What You'll Build:**
```
pkg/
├── logging/
│   ├── logger.go          # Structured logger wrapper
│   ├── middleware.go      # HTTP/RPC logging middleware
│   ├── context.go         # Request ID propagation
│   └── sampler.go         # Log sampling for high-volume
```

**Key Concepts:**
- Structured logging with slog (Go 1.21+)
- Log levels (DEBUG, INFO, WARN, ERROR)
- Contextual fields (request_id, session_id, user_id)
- Log sampling and rate limiting
- Log aggregation (stdout → log shipper → central store)

**Tasks:**
1. Replace fmt.Println with structured logger (slog)
2. Add request ID generation and propagation
3. Implement log sampling for high-volume events
4. Add caller information (file, line) in logs
5. Create logging middleware for all RPC calls
6. Output logs in JSON format for parsing

**Observability Concepts:**
- **Structured logging**: Key-value pairs for machine parsing
- **Correlation IDs**: Tracing requests across services
- **Log sampling**: Reducing volume while maintaining visibility
- **Log levels**: Filtering by severity
- **Log aggregation**: ELK stack, Loki, CloudWatch

---

## Learning Resources

### Go Fundamentals
- **The Go Programming Language** (Donovan & Kernighan)
- **Concurrency in Go** (Katherine Cox-Buday) - Essential for goroutines & channels
- **Go Design Patterns** (Mario Castro Contreras)
- **100 Go Mistakes and How to Avoid Them** (Teiva Harsanyi)
- **Learning Go** (Jon Bodner) - Modern Go practices

### Backend Engineering
- **Designing Data-Intensive Applications** (Martin Kleppmann) - Must read for backend engineers
- **Software Engineering at Google** - Google's engineering practices
- **Building Microservices** (Sam Newman)
- **Database Internals** (Alex Petrov) - Deep dive into database design
- **Web Scalability for Startup Engineers** (Artur Ejsmont)

### Systems Programming
- **The Linux Programming Interface** (Michael Kerrisk)
- **Advanced Programming in the UNIX Environment** (Stevens & Rago)
- **Systems Performance** (Brendan Gregg) - Performance engineering

### Distributed Systems
- **Distributed Systems: Principles and Paradigms** (Tanenbaum)
- **Designing Distributed Systems** (Brendan Burns)
- MIT 6.824 Distributed Systems course (free online)

### Networking
- **UNIX Network Programming** (Stevens)
- **Computer Networks** (Tanenbaum)
- **HTTP: The Definitive Guide** (Gourley & Totty)

### TUI Development
- [Bubble Tea Documentation](https://github.com/charmbracelet/bubbletea) - Official docs & examples
- [Charm Libraries](https://charm.sh/) - lipgloss, bubbles, glamour
- OpenTUI Architecture (see opentui/ARCHITECTURE.md in this repo)
- "Building TUIs in Go" by [packagemain](https://packagemain.tech/p/terminal-ui-bubble-tea)
- [Tips for Building Bubble Tea Programs](https://leg100.github.io/en/posts/building-bubbletea-programs/)

### Go-Specific Backend Patterns
- [How to Architecture Good Go Backend REST API Services](https://medium.com/@janishar.ali/how-to-architecture-good-go-backend-rest-api-services-14cc4730c05b)
- [Uber Go Style Guide](https://github.com/uber-go/guide/blob/master/style.md)
- [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments)
- [Effective Go](https://go.dev/doc/effective_go) - Official Go best practices

### Performance & Profiling
- [Go's pprof profiling guide](https://go.dev/blog/pprof)
- [Profiling Go Programs](https://go.dev/blog/profiling-go-programs)
- Dave Cheney's [High Performance Go Workshop](https://dave.cheney.net/high-performance-go-workshop/dotgo-paris.html)

---

## Project Structure (Final)

```
codex-go/
├── go.mod
├── go.sum
├── README.md
├── ARCHITECTURE.md
├── LICENSE
├── Makefile
├── .goreleaser.yml
│
├── cmd/
│   ├── codex-client/          # Main TUI binary
│   │   └── main.go
│   └── codex-test-agent/      # Mock agent for testing
│       └── main.go
│
├── pkg/                        # Public API
│   ├── jsonrpc/               # JSON-RPC 2.0 implementation
│   ├── transport/             # Transport abstractions
│   ├── acp/                   # ACP protocol types
│   ├── connection/            # Connection management
│   ├── session/               # Session state
│   ├── events/                # Event stream
│   ├── tools/                 # Tool implementations
│   ├── approval/              # Permission system
│   ├── config/                # Configuration
│   ├── observability/         # Tracing, metrics, logging
│   └── errors/                # Error handling
│
├── internal/                   # Private implementation
│   ├── tui/                   # Bubble Tea TUI
│   │   ├── app.go
│   │   ├── update.go
│   │   ├── view.go
│   │   ├── commands.go
│   │   ├── components/
│   │   ├── layout/
│   │   ├── markdown/
│   │   └── streaming/
│   └── agent/                 # Agent process management
│
├── test/
│   ├── integration/           # Integration tests
│   └── fixtures/              # Test data
│
├── docs/                       # Documentation
│   ├── getting-started.md
│   ├── configuration.md
│   ├── architecture.md
│   └── examples/
│
└── scripts/                    # Build scripts
    ├── build.sh
    ├── test.sh
    └── release.sh
```

---

## Weekly Breakdown

| Week | Phase | Focus | Deliverable | Backend Skills |
|------|-------|-------|-------------|----------------|
| 1 | Phase 1 | JSON-RPC & Transport | Working message encoder/decoder | Protocol design, serialization |
| 2 | Phase 1 | ACP Types & Handshake | Can initialize connection to agent | Type systems, API contracts |
| 3 | Phase 2 | Connection Manager | Bidirectional communication working | Concurrency, channels |
| 4 | Phase 2 | Session & Events | Can send prompts, receive responses | State management, pub/sub |
| 4.5 | Phase 2.5 | Worker Pools | Concurrent task execution | Worker patterns, rate limiting |
| 5 | Phase 3 | Bubble Tea Foundation | Basic TUI with input/output | Event-driven architecture |
| 6 | Phase 3 | Layout & Streaming | Polished UI with streaming display | Real-time systems, buffers |
| 7 | Phase 4 | File System Tools | Can read/write files via agent | Sandboxing, validation |
| 8 | Phase 4 | Terminal & Permissions | Can execute commands with approval | Process mgmt, security |
| 8.5 | Phase 4.5 | Database Layer | SQLite integration, persistence | SQL, transactions, repos |
| 9 | Phase 5 | Observability | Tracing and logging integrated | Structured logging, metrics |
| 9.5 | Phase 5.5 | Production Logging | Request IDs, sampling | Correlation, observability |
| 10 | Phase 5 | Config & Error Handling | Robust error recovery | Resilience patterns |
| 11 | Phase 6 | Cross-Platform | Builds on all targets | Build systems, distribution |
| 12 | Phase 6 | Polish & Documentation | Production-ready release | Documentation, profiling |

---

## Success Criteria

By the end of this project, you should be able to:

✅ Implement a complete JSON-RPC 2.0 client from scratch  
✅ Manage bidirectional communication over stdio  
✅ Build a responsive TUI with complex state management  
✅ Handle concurrent operations safely with goroutines & channels  
✅ Design and implement worker pools with backpressure  
✅ Integrate databases with proper connection pooling  
✅ Implement security policies and sandboxing  
✅ Write comprehensive tests (unit + integration)  
✅ Add production-grade logging and observability  
✅ Deploy a cross-platform CLI tool  
✅ Debug distributed systems with tracing  
✅ Profile and optimize hot paths  
✅ Design extensible protocols and clean APIs  
✅ Understand agent-editor interaction patterns  

**Core Skills Gained:**
- **Go**: Concurrency, interfaces, modules, testing, performance
- **Systems**: Process management, IPC, sandboxing, resource pooling
- **Networking**: JSON-RPC, request/response correlation, streaming
- **Distributed Systems**: State management, error handling, observability, resilience
- **Database**: SQL, migrations, transactions, connection pooling
- **UI**: Event-driven architecture, rendering, layout, real-time updates
- **Backend Engineering**: Worker patterns, rate limiting, structured logging, profiling

**Interview-Ready Knowledge:**
After completing this project, you can confidently discuss:
- "How do you handle backpressure in Go services?" → Worker pools, buffered channels
- "Explain your approach to error handling" → Wrapping, retries, circuit breakers
- "How do you manage database connections?" → Pooling, prepared statements, repos
- "Describe your logging strategy" → Structured logging, correlation IDs, sampling
- "How do you test concurrent code?" → Mocks, race detector, integration tests
- "What's your profiling workflow?" → pprof, benchmark tests, optimization strategies

---

## Next Steps

1. **Set up development environment:**
   ```bash
   mkdir codex-go && cd codex-go
   go mod init github.com/yourusername/codex-go
   mkdir -p cmd/codex-client pkg/jsonrpc internal/tui
   ```

2. **Install dependencies:**
   ```bash
   go get github.com/charmbracelet/bubbletea
   go get github.com/charmbracelet/lipgloss
   go get github.com/charmbracelet/glamour    # Markdown rendering
   go get github.com/charmbracelet/bubbles    # Pre-built components
   go get go.opentelemetry.io/otel
   go get github.com/mattn/go-sqlite3         # SQLite driver (CGO)
   # OR use modernc.org/sqlite for pure Go (no CGO)
   ```

3. **Start with Milestone 1.1** and work sequentially through the phases.

4. **Study OpenTUI as reference:**
   - Read [opentui/ARCHITECTURE.md](opentui/ARCHITECTURE.md) to understand:
     - Component-based rendering architecture
     - Event-driven UI patterns (Model-View-Update)
     - Terminal rendering pipeline (layout → render → buffer → ANSI)
     - FFI patterns for performance (adapt to Go's strengths)
   - Examine [opentui/GOLANG_REFACTOR.md](opentui/GOLANG_REFACTOR.md) for Go-specific considerations
   - Key takeaways for Go implementation:
     - Use interfaces for Renderable abstraction
     - Leverage goroutines for async rendering
     - Apply channels for event communication
     - Use struct embedding for component composition

5. **Join communities:**
   - [Charm Bubble Tea Discord](https://charm.sh/community) (for TUI questions)
   - [Gophers Slack](https://gophers.slack.com) (for Go questions)
   - [ACP GitHub Discussions](https://github.com/sourcegraph/agent-client-protocol) (for protocol questions)
   - [r/golang](https://reddit.com/r/golang) (Reddit community)

6. **Set up your learning journal:**
   ```bash
   mkdir docs/learning-journal
   # Document challenges, solutions, and insights as you build
   # This becomes your personal reference and portfolio piece
   ```

## How This Makes You a Better Backend Engineer

This project uniquely combines:

✅ **Protocol Implementation** - JSON-RPC, ACP → understanding RPC systems  
✅ **Concurrency Mastery** - Goroutines, channels, context → essential for backend  
✅ **State Management** - Sessions, persistence → applicable to web services  
✅ **IPC & Process Management** - Spawning, pipes, signals → systems programming  
✅ **Error Handling** - Retries, circuit breakers → production resilience  
✅ **Observability** - Logging, tracing, metrics → debugging production systems  
✅ **API Design** - Clean interfaces, clear contracts → maintainable codebases  
✅ **Database Integration** - SQLite, migrations, repos → data persistence patterns  
✅ **Testing** - Unit, integration, mocking → confidence in code  
✅ **Performance** - Profiling, optimization → efficient systems  

**Real-world parallels:**
- JSON-RPC communication → gRPC services, REST APIs
- Session management → User session handling in web apps
- Event streams → Kafka consumers, WebSocket servers
- Tool execution → Job queues, worker services
- Permission system → RBAC in APIs
- Connection management → Database connection pools, service meshes
- Rate limiting → API throttling, load shedding

By building this TUI agent, you'll develop muscle memory for patterns used in:
- **Microservices**: Service-to-service communication, observability
- **APIs**: Request handling, validation, error responses
- **Data pipelines**: Stream processing, buffering, backpressure
- **Distributed systems**: State management, failure handling, consistency

Good luck building! This project will give you a deep understanding of building production-grade systems in Go. 🚀
