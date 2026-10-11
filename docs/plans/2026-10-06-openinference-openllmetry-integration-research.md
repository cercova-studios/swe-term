# Instrumenting swe-term with GenAI tracing conventions

Status: research + design proposal. Nothing here is promoted into
[`ARCHITECTURE.md`](../../ARCHITECTURE.md). §9's extension-point rules and §10's
invariants constrain everything below; §6 governs what may and may not be
inferred by a model at runtime. The recommendation in §8 is a *staged* one whose
first increment is deliberately smaller than "adopt an SDK", and §10 names what
not to build.

All version numbers, package paths, attribute keys, and environment variable
names below were verified against primary sources on **2026-10-06**. Anything
that could not be verified is marked explicitly in §11.

---

## 1. Problem statement

swe-term's current runtime slice is one `StreamRequest` to one `Provider`,
collected into a `Response` and rendered — ARCHITECTURE.md §4 ("Current runtime
slice") states plainly that "there is not yet a multi-step tool-using agent
loop," and §12 refactor 1 holds that loop as accepted direction, not shipped
behavior.

The question is whether to emit OpenTelemetry spans carrying GenAI semantic
conventions from that path, and if so, under which of the three competing
conventions.

Three facts about the repo frame the answer and are worth stating before the
landscape, because they do most of the work in §8:

1. **The harness already has per-turn structured facts.** `core.Usage` carries
   model, reasoning effort, four token classes, and `CostUSD *float64`
   ([`internal/core/usage.go`](../../internal/core/usage.go)). `UsageTotals`
   aggregates them with an explicit `CostKnown` flag — honest-unknown accounting
   that predates this proposal and already satisfies §10 invariant 12 for the
   facts it covers.
2. **The repo already runs a continuous sensor that ratchets dependency count.**
   `cmd/drift` measures `direct_dependencies` against a checked-in baseline
   ([`docs/reports/drift-baseline.json`](../reports/drift-baseline.json),
   recorded 2026-09-25: `direct_dependencies: 6`). Any OTel adoption is a
   measured regression in a signal this repo chose to watch. That is a cost, not
   a veto — but it has to be named.
3. **The repo already models obligations, receipts, rungs, and provenance as
   event-sourced reducers** (`control_monitor.go`, `receipt_gate.go`,
   `vv_rung.go`, `context_packet.go`). Per ARCHITECTURE.md §6, "Approval,
   mutation lease, observed effect, snapshot, and verification events are written
   to a durable ordered control journal. **Lossy telemetry is separate.**" That
   sentence is the boundary this proposal must respect: tracing is the lossy
   lane, and it must not become a second home for control facts.

In the vocabulary of
[`2026-09-20-test-quality-and-architectural-fitness-steering.md`](2026-09-20-test-quality-and-architectural-fitness-steering.md)
§2, tracing is a **computational sensor** — deterministic, post-hoc, machine
readable. It is not a guide: nothing a span records reaches the model before it
acts. That classification matters for §9: the repo's documented gap is
*computational guides* (§3 of that doc: "almost nothing" in computational
sensors, "thin" in computational guides), and tracing fills the sensor cell, not
the empty guide cell.

---

## 2. Landscape, with dates and versions

### 2.1 The three things are not three competing specs

They are **one convention plus two vendor vocabularies that predate it**, and
the OTel convention is the one in motion.

| | Owner | What it specifies | Current state (2026-10-06) |
|---|---|---|---|
| **OTel GenAI semconv** (`gen_ai.*`) | OpenTelemetry (CNCF) | Span kinds, span names, attributes, metrics, events for inference, embeddings, retrieval, memory, tool execution, agents | **Development** (experimental). Moved out of the main semconv repo into [`open-telemetry/semantic-conventions-genai`](https://github.com/open-telemetry/semantic-conventions-genai). That repo has **no tags or releases**, and its README's "Schema URL" section reads literally `TODO`. |
| **OpenInference** (`openinference.span.kind`, `llm.*`) | Arize | Span kinds + LLM/retriever/reranker/embedding/tool attributes, transport-agnostic; the native vocabulary of Phoenix | Active. Go modules published, latest `v0.1.11` (2026-10-01). Also now specifies non-LLM "decision spans" (`decision.*`), i.e. it is still *expanding*, not converging onto `gen_ai.*`. |
| **OpenLLMetry** (`traceloop.*` / older `llm.*`) | Traceloop | OTel-based instrumentation distribution; its own `semconv-ai` package | Python/TS healthy; **Go is effectively dormant** — see §3.2. |

### 2.2 The move out of the main semconv repo is the headline

The main [`open-telemetry/semantic-conventions`](https://github.com/open-telemetry/semantic-conventions)
repo is at **v1.44.0**. In it, the GenAI attributes are now rendered as
["GenAI Attributes (Moved)"](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/),
and the older keys carry deprecation notes of the form *"Replaced by
`gen_ai.usage.output_tokens`, which has moved to the OpenTelemetry GenAI
semantic conventions repository."* The `gen-ai-spans` page on opentelemetry.io
now returns a **"Moved"** stub.

What has already been renamed, and must not be used on new work:

| Deprecated | Replacement |
|---|---|
| `gen_ai.system` | `gen_ai.provider.name` |
| `gen_ai.usage.prompt_tokens` | `gen_ai.usage.input_tokens` |
| `gen_ai.usage.completion_tokens` | `gen_ai.usage.output_tokens` |
| `gen_ai.prompt` / `gen_ai.completion` | Removed, no replacement; use `gen_ai.input.messages` / `gen_ai.output.messages` or the Event API |

Everything in the new repo is stamped
![Development]. Nothing in `gen_ai.*` is Stable. This is the single most
important staleness risk in this document.

### 2.3 Has `gen_ai.*` absorbed or superseded either vendor spec?

**No — but the convergence is real on one side and absent on the other.**

- OpenInference remains its own vocabulary. Its Go semconv package is
  self-contained string constants (`llm.model_name`, `llm.token_count.prompt`,
  `openinference.span.kind`) with no `gen_ai.*` aliasing, and it is *adding*
  namespaces (`decision.*`) rather than deferring to OTel. Arize's Phoenix is
  an OTLP receiver, so it ingests `gen_ai.*` spans at the transport level; what
  it does *not* do is interpret them with the same fidelity as its native
  attributes. Treating Phoenix as a "compatible consumer of `gen_ai.*`" is true
  about transport and optimistic about rendering — see §11.
- OpenLLMetry's whole premise is OTel-native emission, so it is the closer of
  the two; but its Go surface is too thin to matter here (§3.2).

The honest summary: **`gen_ai.*` is the convention with institutional
ownership and no stability guarantee; OpenInference is the vocabulary with
stability in practice and no institutional ownership.**

### 2.4 What `gen_ai.*` says about inference spans

From [`docs/gen-ai/gen-ai-spans.md`](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)
(Status: Development):

- **Span kind** SHOULD be `CLIENT`; MAY be `INTERNAL` for in-process models.
- **Span name** SHOULD be `{gen_ai.operation.name} {gen_ai.request.model}`.
- **Required**: `gen_ai.operation.name`, `gen_ai.provider.name`.
- **Conditionally required**: `gen_ai.request.model` (if available),
  `error.type` (if the operation ended in an error).
- Content attributes (`gen_ai.input.messages`, `gen_ai.output.messages`,
  `gen_ai.system_instructions`) are **opt-in**, carry an explicit
  *"likely to contain sensitive information including user/PII data"* warning,
  and — because structured attributes are not yet supported on spans —
  "SHOULD be serialized to JSON string on spans."

`gen_ai.operation.name` well-known values include `chat`, `text_completion`,
`generate_content`, `embeddings`, `execute_tool`, `invoke_agent`,
`invoke_workflow`, `create_agent`, `retrieval`, `fetch_response`. Tool spans get
`INTERNAL` kind and name `execute_tool {gen_ai.tool.name}` — directly relevant to
§12 refactor 1.

---

## 3. Go reality check

This is the crux, and the answer is better than the Python/TS-first reputation
suggests — with one sharp disappointment.

### 3.1 Is there a Go OpenInference package? **Yes — three of them.**

[`github.com/Arize-ai/openinference`](https://github.com/Arize-ai/openinference)
has a `go/` tree (a `go.work` workspace), publishing:

| Module | Version | Date | License | Imported by |
|---|---|---|---|---|
| `github.com/Arize-ai/openinference/go/openinference-semantic-conventions` | `v0.1.11` | 2026-10-01 | Apache-2.0 | 5 |
| `github.com/Arize-ai/openinference/go/openinference-instrumentation` | `v0.1.2` | 2026-09-15 | Apache-2.0 | 4 |
| `github.com/Arize-ai/openinference/go/openinference-instrumentation-openai-go` | (in-repo; see below) | — | Apache-2.0 | — |
| `github.com/Arize-ai/openinference/go/openinference-instrumentation-anthropic-sdk-go` | (in-repo) | — | Apache-2.0 | — |

Verified on [pkg.go.dev](https://pkg.go.dev/search?q=openinference) and the
[repo contents API](https://api.github.com/repos/Arize-ai/openinference/contents/go).

**The semantic-conventions module is plain string constants** — no OTel types in
its API — which makes it trivially adoptable or trivially copyable:

```go
import semconv "github.com/Arize-ai/openinference/go/openinference-semantic-conventions"

span.SetAttributes(
    attribute.String(semconv.OpenInferenceSpanKind, semconv.SpanKindLLM),
    attribute.String(semconv.LLMModelName, "gpt-4o"),
    attribute.String(semconv.LLMProvider, semconv.LLMProviderOpenAI),
)
```

**The instrumentation module is the PII opt-out layer**, and it is the most
directly reusable idea in either ecosystem. `traceconfig.go` defines
`RedactedValue = "__REDACTED__"` plus ten environment variables that match the
Python/JS SDKs byte-for-byte: `OPENINFERENCE_HIDE_INPUTS`,
`OPENINFERENCE_HIDE_OUTPUTS`, `OPENINFERENCE_HIDE_INPUT_MESSAGES`,
`OPENINFERENCE_HIDE_OUTPUT_MESSAGES`, `OPENINFERENCE_HIDE_INPUT_TEXT`,
`OPENINFERENCE_HIDE_OUTPUT_TEXT`, `OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS`,
`OPENINFERENCE_HIDE_LLM_TOOLS`, `OPENINFERENCE_HIDE_PROMPTS`,
`OPENINFERENCE_HIDE_INPUT_IMAGES`, read by `TraceConfigFromEnv()`.

**The disappointment: the `openai-go` instrumentor does not fit this repo.**
Its [`go.mod`](https://github.com/Arize-ai/openinference/blob/main/go/openinference-instrumentation-openai-go/go.mod)
requires `github.com/openai/openai-go v1.12.0`. swe-term is on
`github.com/openai/openai-go/v3 v3.51.0` — a different Go module path, so the
middleware is not merely version-skewed, it is a *different package* that cannot
be wired into this client. Worse, its README states it traces
"`/v1/chat/completions`" calls, hooked via `option.WithMiddleware`; swe-term uses
`client.Responses.NewStreaming` (the Responses API), which that middleware does
not model. **The ready-made instrumentor is unusable here on two independent
grounds.**

### 3.2 Is there a Go OpenLLMetry SDK? **Technically yes; practically no.**

[`github.com/traceloop/go-openllmetry`](https://github.com/traceloop/go-openllmetry)
exists with a `traceloop-sdk` module and a `semconv-ai` package. Highest tag is
`traceloop-sdk/v0.1.3`; the most recent commit on `main` is dated
**2026-01-17** — roughly nine months stale as of this writing. Its quickstart
centers on `sdk.Config{APIKey: os.Getenv("TRACELOOP_API_KEY")}`, i.e. a
vendor-keyed path. **Not a candidate.**

### 3.3 What does vanilla OTel Go offer for `gen_ai.*` today?

`go.opentelemetry.io/otel` is at **v1.47.0**. Its in-tree `semconv/` directory
ships frozen per-version packages. The GenAI picture across them, verified by
grepping the v1.47.0 source tree:

| Package | `gen_ai.*` keys | Notes |
|---|---|---|
| `go.opentelemetry.io/otel/semconv/v1.38.0` | 40 | has `usage.input_tokens`/`output_tokens`; **no** cache or reasoning keys |
| `go.opentelemetry.io/otel/semconv/v1.41.0` | **50** | adds `usage.cache_read.input_tokens`, `usage.cache_creation.input_tokens`, `usage.reasoning.output_tokens`, `request.stream`, `response.time_to_first_chunk`, `workflow.name`, `prompt.name` |
| `go.opentelemetry.io/otel/semconv/v1.42.0` | **0** | |
| `go.opentelemetry.io/otel/semconv/v1.43.0` | **0** | |

**`semconv/v1.41.0` is the high-water mark and the last Go package that carries
`gen_ai.*` at all.** The zero counts in v1.42.0 and v1.43.0 are the downstream
consequence of §2.2: opentelemetry-go's codegen tracks the main semconv repo,
and GenAI left it. A `pkg.go.dev` search for a Go binding of the new
`semantic-conventions-genai` repo returns **0 modules**.

So the Go story is:

> **Pin `go.opentelemetry.io/otel/semconv/v1.41.0` for `gen_ai.*` constants
> (it is frozen and will not move under you, and it is shipped inside the
> current otel v1.47.0 release), and expect to hand-write any attribute added
> to the convention after main-repo v1.41.0 until a Go codegen of the new repo
> appears.**

Note also what `gen_ai.*` does **not** have, at any version: **no cost
attribute.** Grepping v1.41.0 for `cost` returns zero hits. OpenInference does:
`llm.cost.prompt`, `llm.cost.completion`, `llm.cost.total`, plus
`llm.cost.prompt_details.cache_read` / `.cache_write`. Since swe-term computes
cost at the provider boundary on purpose
([`internal/provider/openai/pricing.go`](../../internal/provider/openai/pricing.go),
per [`2026-08-23-tui-telemetry-design.md`](2026-08-23-tui-telemetry-design.md)),
cost is a first-class local fact with no home in `gen_ai.*`.

### 3.4 Exporters and the dependency bill

The OTLP trace exporters in otel v1.47.0 are
`go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc` and
`.../otlptracehttp` (both present in the
[v1.47.0 tree](https://github.com/open-telemetry/opentelemetry-go/tree/v1.47.0/exporters/otlp/otlptrace)).

A minimal exporting setup adds these **direct** requires: `go.opentelemetry.io/otel`,
`go.opentelemetry.io/otel/sdk`, `go.opentelemetry.io/otel/trace`, and one
exporter module — four, against a current baseline of six. `cmd/drift` would
report `direct_dependencies` going 6 → ~10, a ~67% regression on a watched
signal. OTLP/gRPC additionally drags in `google.golang.org/grpc`
transitively; **OTLP/HTTP is the cheaper of the two** and is the right default
for a local CLI.

---

## 4. Span and attribute mapping for this harness

Design decisions taken before the table, so the table reads as consequences:

- **One span per provider call.** Not one per chunk. A per-chunk span is
  thousands of spans per turn carrying no information a timestamp does not;
  `gen_ai.response.time_to_first_chunk` (new in v1.41.0) captures the only
  genuinely interesting streaming fact, and a span *event* captures the second.
- **The span ends where `CollectResponse` returns**, not where `Stream` returns.
  `Stream` returns a channel immediately; its return is not the end of the
  operation. Ending the span at `Stream`'s return would record a few
  microseconds and attach usage to nothing.
- **Usage attributes are written from the `EventComplete` payload only**, which
  is the one place §10 invariant 1 guarantees a terminal completion exists.

### 4.1 Mapping table

`oi:` prefixes OpenInference keys; unprefixed keys are `gen_ai.*` from
`semconv/v1.41.0`.

| Code site | Span | Kind | Starts / ends | Attributes & events |
|---|---|---|---|---|
| `Provider.Stream` entry (`internal/core/provider.go`) | `chat {model}` (per convention `{gen_ai.operation.name} {gen_ai.request.model}`) | `CLIENT` | Starts immediately before the provider call; **ends when the stream is fully drained** (see `CollectResponse`) | At start: `gen_ai.operation.name=chat`, `gen_ai.provider.name=openai`, `gen_ai.request.model` (`StreamRequest.Model`, or the provider default — see §4.3), `gen_ai.request.stream=true`. Reasoning effort: see §4.4. |
| `Stream` setup-failure return (`return nil, err`) | same span | `CLIENT` | Ends on the error return | `error.type` = a low-cardinality class string; `span.SetStatus(codes.Error, msg)`; `span.RecordError(err)`. **No usage attributes.** |
| Streaming chunk loop (`EventText` fan-out in `openai.go`) | *no span* | — | — | On the first text delta only: `gen_ai.response.time_to_first_chunk` (duration). Optionally one span event `gen_ai.first_chunk`. Per-chunk text is never recorded. |
| `EventComplete` → `CollectResponse` (`internal/core/usage.go`) | same span | — | Span ends after the channel closes and `CollectResponse` returns | `gen_ai.response.model` (`Usage.Model`), `gen_ai.usage.input_tokens` (**reconstructed — see §4.2**), `gen_ai.usage.output_tokens` (`Usage.OutputTokens`), `gen_ai.usage.cache_read.input_tokens` (`Usage.CachedInputTokens`), `gen_ai.usage.cache_creation.input_tokens` (`Usage.CacheWriteTokens`), `gen_ai.usage.reasoning.output_tokens` (`Usage.ReasoningTokens`) |
| `CollectResponse` protocol violations (`text after completion`, `duplicate completion`, `closed without completion`) | same span | — | Ends on return | `error.type="protocol_violation"` + `span.SetStatus(codes.Error, …)`. **Critical: these must never leave the span `Ok` with a partial `output.value`** — see §5.2. |
| `EventError` mid-stream | same span | — | Ends on return | `error.type`, `RecordError`, `SetStatus(codes.Error)`. Any text accumulated before the error is **not** promoted to a success output. |
| `costForUsage` (`internal/provider/openai/pricing.go`) | same span | — | — | `oi:llm.cost.total` (float, USD) **only when `CostUSD != nil`**. When nil, set `swe_term.cost.known=false` and **omit the cost attribute entirely.** No zero. (§10 invariant 12.) |
| Long-context tier branch (`inputTokens > 272_000`) | same span | — | — | `swe_term.pricing.tier` = `"short"` \| `"long"`. Local, non-standard, and worth having: it is the one pricing input that silently doubles the bill. |
| A TUI turn (`internal/tui/tui.go`, user submit → `streamDoneMsg`) | `invoke_agent swe-term` | `INTERNAL` | Starts on submit; ends on `streamDoneMsg`; parent of the provider span | `gen_ai.operation.name=invoke_agent`, `gen_ai.conversation.id` = TUI session id. **Not** `UsageTotals` — see §4.5. |
| `runOnce` one-shot (`main.go`) | *no extra span* | — | — | The provider span **is** the trace. A one-shot parent with exactly one child is a wrapper with no information in it. |
| Prompt / completion content (opt-in only) | provider span | — | — | `gen_ai.input.messages` / `gen_ai.output.messages`, JSON-serialized per the convention's note that structured span attributes are not yet supported. **Off by default.** See §4.6. |

### 4.2 The token-accounting trap (the most important row above)

`usageFromResponse` computes:

```go
cached     := usage.InputTokensDetails.CachedTokens
cacheWrite := usage.InputTokensDetails.CacheWriteTokens
input      := usage.InputTokens - cached - cacheWrite   // ← disjoint
```

So `core.Usage.InputTokens` is **uncached input only** — the doc comment on
`Usage` says exactly this. Both conventions define their input-token field the
*other* way:

- `gen_ai.usage.input_tokens`: *"This value SHOULD include all types of input
  tokens, including cached tokens"*, and `cache_read` / `cache_creation` are
  each defined as *"SHOULD be included in `gen_ai.usage.input_tokens`."*
- OpenInference's `llm.token_count.prompt` is the provider's `prompt_tokens`
  total, with `prompt_details.cache_read` as a *breakdown* of it.

Emitting `Usage.InputTokens` directly into either key therefore **understates
input tokens by the entire cache volume** — silently, and most on exactly the
cache-heavy sessions where the number matters. Required:

```go
inputTokens := u.InputTokens + u.CachedInputTokens + u.CacheWriteTokens
```

which is also precisely the expression `costForUsage` already uses for its
272,000-token tier test. One shared helper, not two call sites.

### 4.3 Model identity: request vs response

`StreamRequest.Model` may be empty (`openai.go` falls back to `p.model`, and
`mock.go` has a three-level fallback chain). `gen_ai.request.model` is
*Conditionally Required "If available"* — so when the caller specified nothing,
record the model the provider actually resolved to and let
`gen_ai.response.model` carry the authoritative value from `Usage.Model`. Do not
backfill `gen_ai.request.model` from the response: that fabricates a request
field the caller never sent.

### 4.4 Reasoning effort has no `gen_ai.*` home

`Usage.Reasoning` / `StreamRequest.Reasoning` (`none`…`max`, per the TUI
telemetry design) maps to neither convention's attribute registry. Options, in
preference order:

1. `oi:llm.invocation_parameters` as a JSON object `{"reasoning_effort":"high"}`
   — this is what the OpenInference `openai-go` instrumentor does for
   `reasoning_effort`, so it renders in Phoenix.
2. A local key `swe_term.request.reasoning_effort`.

Do **not** invent `gen_ai.request.reasoning_effort`. It does not exist in
v1.41.0 and squatting on the namespace guarantees a future collision.

### 4.5 `UsageTotals` does not belong on a span

Session totals are a *reduction over* turns. Putting them on each turn's span
duplicates derivable state and makes every span's meaning depend on its
position. The backend sums `gen_ai.usage.*` across the trace; that is what
backends are for. `UsageTotals` stays where it is — the TUI status line.

### 4.6 Cardinality and PII

| Attribute | Concern | Default |
|---|---|---|
| `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions` | **PII-bearing.** The convention carries an explicit sensitive-data warning and makes them opt-in. Unbounded size. | **Off** |
| `gen_ai.conversation.id` | High cardinality (one value per session) — fine as a span attribute, never as a metric dimension | On |
| `gen_ai.response.id` | High cardinality; the convention's own `fetch_response` guidance keeps it out of span *names* for this reason | On (attribute only) |
| `error.type` | Must stay low cardinality. Map to a closed set (`provider_error`, `protocol_violation`, `incomplete`, `context_canceled`, `setup_failure`) — never raw `err.Error()` | On |
| `gen_ai.request.model` / `.response.model` | Low cardinality | On |
| `oi:llm.cost.total` | Not sensitive; may be commercially so in shared backends | On |

**Secret handling is non-negotiable.** ARCHITECTURE.md §10 invariant 14: secret
values are absent from prompts, journals, receipts, and artifacts. A span is an
artifact. `option.WithAPIKey` must never be reflected into
`llm.invocation_parameters`; the only safe rule is an allowlist of fields to
record, never a denylist of fields to strip.

**Reuse, don't reinvent, the opt-out.** The `OPENINFERENCE_HIDE_*` env var set
and `RedactedValue = "__REDACTED__"` are already a thought-through,
cross-language-consistent contract (§3.1). Adopting those names costs nothing
and gives users one vocabulary across tools.

---

## 5. Where the instrumentation boundary belongs

### 5.1 The three options

| | Option | Verdict |
|---|---|---|
| (a) | **Decorator `Provider` wrapping any provider** | **Recommended.** |
| (b) | Inside each provider implementation | Reject. |
| (c) | At `runOnce` / TUI level | Reject as the primary seam; keep as the optional parent span. |

**Why (a).** ARCHITECTURE.md §9 names "Provider adapters — implemented. Add
model backends without changing core stream semantics," and §8 calls
provider/model a capability port with swappable backends. A decorator satisfying
`core.Provider` *is* the extension mechanism the document already blesses — and
it is the only option that instruments the `mock` provider too, which means the
instrumentation is testable without a network. §4's "core/extension interaction
is interface-driven" is satisfied exactly.

**Why not (b).** It duplicates identical attribute logic in every provider, and
guarantees the second provider drifts from the first. It also pushes OTel's
types into `internal/provider/*`, where §10 invariant 13 ("extension internals
and vendor-specific types do not enter core state") is at least adjacent: OTel
types in a provider are not core state, but the attribute vocabulary becomes a
per-provider decision rather than one owned module. The Fleet CPG plan already
committed to the opposite pattern — a *"shared span taxonomy"* in one place
(`docs/plans/2026-09-05-fleet-cpg-engine-implementation.md`, Observability row).
One owned constants module, one decorator.

**Why not (c) alone.** `runOnce` and the TUI both call `Stream` and both would
need the same instrumentation — the duplication (b) has, relocated. The TUI
`invoke_agent` parent span from §4.1 is still worth having, but it is *in
addition to* the decorator, not instead of it.

### 5.2 The invariants a tracing wrapper can violate

This is where the recommendation earns or loses its keep. The relevant text:

- **§10 invariant 9:** *"Truncation, timeout, cancellation, denial, and failure
  are never flattened into successful text output."*
- **§10 invariant 12:** *"Unknown pricing, provenance, freshness, or confinement
  remains explicit; it is never converted to a reassuring zero or success."*
- **§10 invariant 1:** *"Exactly one terminal completion exists for a successful
  provider stream."*
- **§6:** *"Lossy telemetry is separate"* from the control journal.

Four concrete failure modes a careless wrapper exhibits, each mapped to the
invariant it breaks:

1. **Swallowing errors to keep the span clean.** A decorator that catches
   `EventError`, records a span event, and forwards an `EventComplete` has
   converted failure into success — invariant 9, and invariant 1 as collateral
   (it manufactures a terminal completion). **Rule: the decorator is a
   pass-through. It must forward every `StreamEvent` unchanged, in order, and
   must not originate, drop, reorder, or rewrite one.** That is a property worth
   a table-driven test per §11 of ARCHITECTURE.md.
2. **Emitting `llm.cost.total = 0` when `CostUSD == nil`.** Invariant 12, and it
   is the single easiest mistake to make because `*float64` dereferences so
   naturally. The existing `UsageTotals.CostKnown` flag exists *because* this
   repo already decided unknown cost is not zero cost; the span must inherit that
   decision, by omitting the attribute.
3. **Under-reporting input tokens** per §4.2. Not strictly an invariant breach —
   but it is a reassuring-looking wrong number, which is the behavior invariant
   12 exists to prevent.
4. **Letting a failed export fail a turn.** A tracing SDK that blocks or errors
   the user's request inverts the cost/benefit entirely. The span-export path
   must be best-effort and non-blocking; `otel.SetErrorHandler` should route SDK
   errors to the existing stderr path, visibly (§3 "failures must be visible")
   but harmlessly.

The decorator must also **not** write control-journal facts. Obligations,
receipts, rungs, and leases have reducers and a durable ordered journal; spans
are the lossy lane per §6. A span may *carry* a receipt digest as a correlation
key. It may never be the place a receipt is recorded.

### 5.3 Sketch (illustrative only — not for the repo)

```go
// Illustrative. Error-class mapping, TraceConfig, and the attribute set are
// elided; the point is the shape: pass-through, span ends on channel close.
type tracingProvider struct {
    next   core.Provider
    tracer trace.Tracer
}

func (p *tracingProvider) Stream(ctx context.Context, req core.StreamRequest) (<-chan core.StreamEvent, error) {
    model := req.Model // may be ""; see §4.3
    ctx, span := p.tracer.Start(ctx, "chat "+model, trace.WithSpanKind(trace.SpanKindClient))
    span.SetAttributes(
        semconv.GenAIOperationNameChat,
        semconv.GenAIProviderNameOpenAI,
        semconv.GenAIRequestStream(true),
    )

    in, err := p.next.Stream(ctx, req)
    if err != nil {
        span.SetAttributes(semconv.ErrorTypeKey.String("setup_failure"))
        span.SetStatus(codes.Error, err.Error())
        span.End()
        return nil, err // unchanged
    }

    out := make(chan core.StreamEvent, cap(in))
    go func() {
        defer close(out)
        defer span.End()
        start, first := time.Now(), true
        for ev := range in {
            switch ev.Kind {
            case core.EventText:
                if first {
                    span.SetAttributes(semconv.GenAIResponseTimeToFirstChunk(time.Since(start).Seconds()))
                    first = false
                }
            case core.EventComplete:
                setUsageAttrs(span, ev.Usage) // §4.2 reconstruction; omits unknown cost
            case core.EventError:
                span.SetStatus(codes.Error, errText(ev.Err))
            }
            out <- ev // never modified, never dropped
        }
    }()
    return out, nil
}
```

Two things this sketch gets right and a naive version gets wrong: the span ends
when the **channel closes** (not when `Stream` returns), and every event is
forwarded verbatim.

---

## 6. Backend and configuration story

### 6.1 What a user actually runs

Any OTLP-receiving backend works; the span attributes decide how well it renders.

| Backend | Run it | Endpoint | Renders `gen_ai.*`? |
|---|---|---|---|
| **Phoenix** (self-hosted) | `docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix` | OTLP/HTTP `http://localhost:6006/v1/traces`; OTLP/gRPC `localhost:4317` | Native for `llm.*` / `openinference.*`. `gen_ai.*` is accepted at the transport layer; fidelity of its LLM-specific views on `gen_ai.*` alone is **unverified** (§11). |
| **Jaeger** | `docker run -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one` | OTLP/HTTP `:4318`, gRPC `:4317` | Generic — shows attributes as a key/value list. No LLM views, but complete and vendor-neutral. |
| **Traceloop** (hosted) | vendor account | vendor endpoint + `TRACELOOP_API_KEY` | Not a candidate here (§3.2) |
| **Any OTLP collector** | user's choice | `OTEL_EXPORTER_OTLP_ENDPOINT` | Generic |

Phoenix's ports are its own env vars: `PHOENIX_PORT` (defaults to 6006) and
`PHOENIX_GRPC_PORT` (defaults to 4317), and its docs state that port 6006
accepts traces at `/v1/traces` in OTLP Protobuf form. Note that `6006` is *not*
the OTLP default `4318`, so a Phoenix user must set the endpoint explicitly.

### 6.2 Environment variables (verified against the exporter source)

The OTLP spec defines, with these defaults:

- `OTEL_EXPORTER_OTLP_ENDPOINT` — default `http://localhost:4318` for OTLP/HTTP,
  `http://localhost:4317` for OTLP/gRPC. When this non-signal-specific variable
  is used, the HTTP exporter appends `/v1/traces`.
- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` — per-signal override; **takes precedence
  and is used as-is, with no path appended.** (This asymmetry is the #1 OTLP
  misconfiguration and worth one line in any README.)
- `OTEL_EXPORTER_OTLP_PROTOCOL` / `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`,
  `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_EXPORTER_OTLP_TIMEOUT`,
  `OTEL_EXPORTER_OTLP_COMPRESSION`, `OTEL_EXPORTER_OTLP_INSECURE`,
  `OTEL_EXPORTER_OTLP_CERTIFICATE` — all confirmed present in the
  opentelemetry-go v1.47.0 source tree.

### 6.3 Zero-config default: off, and why `OTEL_SDK_DISABLED` cannot be the mechanism

**Verified negative: the string `OTEL_SDK_DISABLED` does not appear anywhere in
the opentelemetry-go v1.47.0 source tree.** A grep across the extracted release
archive returns zero files. Whatever other languages do, **a Go program cannot
rely on that variable to turn tracing off.**

So off-by-default must be a code decision, which for a local CLI is the right
answer anyway:

1. No `TracerProvider` is installed unless tracing is explicitly requested. The
   global default in `go.opentelemetry.io/otel` is a no-op provider, so an
   un-configured binary opens no socket, spawns no exporter goroutine, and
   resolves no DNS.
2. The trigger is an explicit opt-in — a config key (`[telemetry] otlp_endpoint`
   in the existing TOML config, which `internal/config` already parses) or a
   `--otlp-endpoint` flag. **Not** the mere presence of
   `OTEL_EXPORTER_OTLP_ENDPOINT`: that variable is commonly exported
   machine-wide, and a CLI that starts phoning a collector because an unrelated
   service set an env var is a surprise, bordering on a privacy incident.
3. Once opted in, the standard `OTEL_*` variables configure the exporter
   normally, because the exporter reads them itself.
4. Default protocol **OTLP/HTTP** (fewer transitive dependencies than gRPC,
   §3.4).
5. Content capture (`gen_ai.input.messages` / `.output.messages`) requires a
   *second*, separate opt-in. Turning on tracing must not turn on prompt
   capture.

---

## 7. Attacking this proposal before recommending it

The strongest case against building this at all, stated as strongly as I can
make it:

**swe-term has no agent loop, and tracing's entire value is the loop.** A trace
is worth more than a log line precisely when there is causal structure —
parent/child, fan-out, retries, tool calls, a loop that went sideways on turn
7. ARCHITECTURE.md §4 says the loop does not exist yet; §12 lists it as
direction. For today's one-shot path, the recommended instrumentation produces
**a single span per turn**, which is a structured log record with a worse
ergonomic story: it needs a collector, a backend, a Docker container, and four
new dependencies to read what `fmt.Fprintln(os.Stderr, string(jsonBytes))` would
hand you with `jq`. Judged against "the best code is no code," a trace of a
one-span trace is a dashboard for a number you already print in the status line.

Three supporting arguments:

- **The dependency cost lands on a watched signal.** 6 → ~10 direct deps, a
  ~67% regression in `cmd/drift`'s `direct_dependencies`. The repo built that
  ratchet on 2026-09-20 specifically to notice this class of change. Tripping
  your own newest sensor for a feature with no current consumer is a bad trade.
- **The convention is mid-move.** `gen_ai.*` has just relocated repositories,
  the new repo has no tags and a `TODO` schema URL, everything is stamped
  Development, and the Go binding stopped at `semconv/v1.41.0` with v1.42.0 and
  v1.43.0 shipping zero `gen_ai.*` keys. Writing an integration against a
  convention whose Go codegen just went dark is adopting someone else's churn.
- **The facts tracing would carry are already modeled better elsewhere.** The
  repo's interesting per-turn facts — obligations, receipts, rungs, provenance —
  live in deterministic reducers with closed vocabularies and fail-closed
  semantics. §6 explicitly segregates lossy telemetry from that journal. Tracing
  cannot carry those facts (correctly), and the facts it *can* carry are
  `core.Usage`, which is already typed, aggregated, and rendered.

**Could the chosen standard be wrong?** The strongest counter-pick is
**OpenInference**, and it is not a weak case: it has an actual published Go
module (`v0.1.11`, Apache-2.0, maintained *this month*), a Go PII opt-out layer,
native cost attributes that `gen_ai.*` lacks entirely, token-detail keys that map
one-to-one onto `core.Usage`'s cache/reasoning split, and a local backend
(Phoenix) that renders it natively. Against that, `gen_ai.*` offers a CNCF
governance story and vendor neutrality — both of which are *future* value, and
neither of which prints a token count today.

**What survives the attack.** Two things:

1. The **attribute vocabulary** is worth adopting now even with no SDK, because
   naming things consistently is nearly free and makes a later exporter a small
   change rather than a rewrite.
2. The **§4.2 token-accounting mismatch** is a real latent hazard, though it is
   *not* an active bug today. Verified 2026-10-06: the repo's only consumer of
   the field, `costForUsage` (`internal/provider/openai/pricing.go:82`), already
   reconstructs the inclusive total as
   `InputTokens + CachedInputTokens + CacheWriteTokens` for its tier test, and
   prices the three buckets at three distinct rates. The arithmetic is correct
   and internally consistent; `Usage.InputTokens` simply means "uncached input"
   and is documented as such.

   What is true is that the field name invites the convention's reading, and the
   moment anything emits it as `gen_ai.usage.input_tokens` it understates input
   by the whole cache volume. So this is worth extracting into a shared helper as
   mismatch-proofing ahead of an emitter — not billed as a bug fix.

Everything else should wait for the loop.

---

## 8. Staged proposal

### Stage 0 — Vocabulary, no dependency (do this)

Add **one owned constants file** fixing the span and attribute taxonomy for this
repo: operation names, provider names, the error-type closed set, the token keys,
and the local `swe_term.*` keys. Copy the `gen_ai.*` and `llm.cost.*` key strings
in as constants with a comment citing `semconv/v1.41.0` and the OpenInference
spec. **Zero new dependencies.** This is the shared-span-taxonomy commitment the
Fleet CPG plan already made, honored in one place instead of two.

Alongside it, extract the token-reconstruction helper from §4.2 and point both it
and `costForUsage` at the same expression, with a test asserting
`input + cache_read + cache_creation` equals the provider's reported
`InputTokens` total. This changes no behaviour today (see §7) — it names the
inclusive total once so a future emitter cannot pick the disjoint field by
mistake — and it does not need OTel to be worth landing.

**Guides/sensors classification:** this is sensor *preparation* — no sensor fires
yet. Honest about it.

### Stage 1 — One NDJSON sensor line per turn (smallest useful increment)

A decorator `core.Provider` that, when enabled, writes **one JSON object per
completed turn** to a file or stderr, keyed with the Stage 0 attribute names.
Roughly 80 lines, still zero new dependencies, composes with `jq`, and runs in
CI. It answers the questions a single-span trace would answer — what model, how
many tokens, what cost, how long to first chunk, did it fail and how — while
sitting in the same family as `cmd/drift`: a local sensor emitting JSON a human
or a script reads.

Prove its value before Stage 2. Concretely: if nobody has grepped that file
within a month, Stage 2 is unjustified.

**The pass-through property gets its table-driven test here** (§5.2 rule 1),
using `mock.Provider`, because that property is what makes the decorator safe
under §10 invariants 1 and 9 — and a sensor never observed to fail is not yet
known to be a sensor
([steering doc](2026-09-20-test-quality-and-architectural-fitness-steering.md) §5.5).

### Stage 2 — OTLP export (only if Stage 1 proves itself *and* the loop lands)

Swap the NDJSON writer behind the same decorator for a real `TracerProvider`
plus `otlptracehttp`. Trigger conditions, both required:

1. §12 refactor 1 has landed far enough that a turn is **multiple** spans
   (tool dispatch, sub-agent fan-out, retries) — i.e. there is causal structure
   a flat log genuinely cannot show; and
2. Stage 1's output has been used often enough that its ergonomic limits are a
   felt pain, not a predicted one.

Stage 2 is where the four dependencies get paid for, and where
`cmd/drift -update` gets run with an explanatory note.

### Stage 3 — Only if someone asks

OpenInference `llm.*` keys emitted *in addition to* `gen_ai.*`, behind a config
switch, for users who specifically want Phoenix's native LLM views. Dual-emission
is cheap at the attribute layer (same span, extra keys) and should never be the
default.

---

## 9. Compliance with ARCHITECTURE.md

| Section | Requirement | How this proposal complies |
|---|---|---|
| §9 Extension points | Provider adapters are the implemented extension point | Stage 1/2 ships as a `core.Provider` decorator — the blessed seam |
| §9 | "A capability moves into core only when independent extensions must agree on it for safety, replay, or correctness" | The attribute taxonomy (Stage 0) is exactly such an agreement, so it lives in core; the exporter does not |
| §10 inv. 1 | Exactly one terminal completion | Decorator is pass-through; never originates `EventComplete` |
| §10 inv. 9 | Failure never flattened into success | Error-class attributes + `codes.Error`; accumulated text is never promoted on failure |
| §10 inv. 12 | Unknown stays unknown | Unknown cost omits the attribute and sets `swe_term.cost.known=false`; no zeros |
| §10 inv. 13 | Vendor types out of core state | OTel types live in the decorator package; `core.Usage` is unchanged |
| §10 inv. 14 | No secret values in artifacts | Allowlist-only attribute recording; API keys never reflected into invocation parameters |
| §6 | Lossy telemetry separate from the control journal | The decorator writes no control events; spans may carry correlation keys only |
| §12 refactor 1 | One-shot → agent loop | **Advances, does not conflict.** Stage 2 is explicitly gated on the loop; the `invoke_agent` / `execute_tool` span shapes are chosen to be the loop's natural parents |
| §12 refactor 4 | Frontend protocol seam | Neutral. Tracing attaches below the frontend boundary |
| §14 | Update the doc in the same change | Stage 0 adds no contract. **Stage 1 does**: a decorator is a new extension-point instance and the span taxonomy is a schema, so §9 and (if it names a new domain contract) §5 must be updated in that same change |

---

## 10. Do not build

- **Do not build a per-chunk span.** Thousands of spans per turn, no information.
  `gen_ai.response.time_to_first_chunk` and one span event cover it.
- **Do not build a metrics pipeline.** `gen_ai.*` defines metrics; this is a
  local CLI with one user. Histograms of one are not insight.
- **Do not build prompt/completion capture on by default**, and do not couple it
  to the tracing switch. Two switches.
- **Do not build an OTel bridge for the control journal.** §6 separates them, and
  a lossy transport is structurally the wrong home for events invariant 8 says
  must not drop.
- **Do not build a `UsageTotals` span attribute.** Derivable; the backend sums.
- **Do not depend on `go-openllmetry`.** Dormant since 2026-01-17, vendor-keyed,
  `v0.x`.
- **Do not depend on `openinference-instrumentation-openai-go`.** Wrong SDK major
  version (`openai-go v1` vs this repo's `/v3`) *and* wrong API surface (Chat
  Completions middleware vs the streaming Responses API).
- **Do not invent `gen_ai.*` keys.** Reasoning effort has no `gen_ai` home; use
  `llm.invocation_parameters` or a `swe_term.*` key. Squatting on a namespace
  mid-migration is how you earn a rename.
- **Do not rely on `OTEL_SDK_DISABLED`.** Verified absent from opentelemetry-go
  v1.47.0.
- **Do not chase `semconv/v1.42.0`+ for `gen_ai.*`.** Those packages have zero
  `gen_ai` keys. Pin `v1.41.0`.
- **Do not let span export block or fail a turn.**

---

## 11. Which standard to pick

> **Vanilla OpenTelemetry Go, with `gen_ai.*` attributes pinned to
> `go.opentelemetry.io/otel/semconv/v1.41.0`, supplemented by exactly three
> OpenInference keys (`llm.cost.total`, `llm.cost.prompt`, `llm.cost.completion`)
> and `llm.invocation_parameters`, for the facts `gen_ai.*` does not model.**

The reason, in one sentence: `gen_ai.*` is the only vocabulary with
vendor-neutral governance *and* complete coverage of swe-term's token model
(`cache_read`, `cache_creation`, `reasoning.output_tokens` all exist in
v1.41.0), and the one thing it is missing — cost — is a four-key patch from a
spec whose Go package is Apache-2.0 and actively maintained.

**On the "both are compatible consumers" claim — partially provable, and I will
not overstate it.** What the specs prove:

- OpenLLMetry is explicitly OTel-based and exports over OTLP; its backend
  consumes OTLP spans, so `gen_ai.*` spans transit it by construction.
- Phoenix is an OTLP receiver (ports 6006/HTTP and 4317/gRPC per its own docs),
  so it *ingests* `gen_ai.*` spans.

What the specs do **not** prove, and I could not verify from primary sources:
that Phoenix's LLM-specific UI (token/cost panels, message rendering) populates
from `gen_ai.*` keys rather than only from `openinference.*` / `llm.*`. Arize's
Go semconv package is self-contained and does not alias `gen_ai.*`, and
OpenInference is still *adding* namespaces (`decision.*`) rather than deferring
to OTel — which is weak evidence against deep `gen_ai.*` interpretation. **This
is the one open question whose answer could flip the recommendation to
OpenInference**, and Stage 3's dual-emission switch exists precisely so the
answer is cheap to act on.

Had the OpenInference `openai-go` instrumentor actually fit this repo's SDK, the
calculus would be different — "free, maintained, renders natively" beats "right
in principle." It does not fit (§3.1), so both paths require hand-written
attributes, and once you are writing them by hand the governance argument wins.

---

## 12. Open questions and unverified items

| # | Item | Status |
|---|---|---|
| 1 | Does Phoenix's LLM UI render `gen_ai.*`-only spans with full fidelity? | **Unverified.** Could not confirm from primary docs. Decides Stage 3's priority; see §11. |
| 2 | Will a Go codegen of `semantic-conventions-genai` appear? | **Unknown.** `pkg.go.dev` search returns 0 modules today. Until then, post-v1.41.0 attributes are hand-written. |
| 3 | Exact published version of `openinference-instrumentation-openai-go` | **Not verified** (read from in-repo `go.mod`, not a pkg.go.dev release page). Does not affect the recommendation — the module is rejected on SDK-major and API-surface grounds either way. |
| 4 | `semantic-conventions-genai` schema URL | **Literally `TODO`** in its README. No `Schema URL` can be set on a `Resource` for GenAI conventions yet. |
| 5 | Does any `gen_ai.*` attribute reach Stable before Stage 2? | **Unknown.** All are Development as of 2026-10-06. A rename between now and Stage 2 is plausible — which is an argument for Stage 0's single constants file, where a rename is one diff. |
| 6 | `openinference-semantic-conventions` v0.1.x stability policy | **Not documented** that I could find. `v0.x` implies none. Copying the four cost key *strings* rather than taking the dependency avoids the question entirely at Stage 0. |
| 7 | Whether swe-term should emit `gen_ai.conversation.id` for the one-shot path | **Open design question.** A single-turn conversation id is arguably noise. |
| 8 | Freshness | Every version above is as of **2026-10-06**. The `gen_ai.*` move is weeks-fresh; re-verify §2.2 and §3.3 before acting on Stage 2. |

---

## 13. Citations

**OpenTelemetry GenAI semantic conventions**
- New home repo: https://github.com/open-telemetry/semantic-conventions-genai
- Inference span contract (Status: Development): https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md
- Attribute registry, incl. "Moved" and deprecated keys: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
- Main semconv repo, latest release v1.44.0: https://github.com/open-telemetry/semantic-conventions/releases/tag/v1.44.0
- Moved spans stub: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/

**OpenTelemetry Go**
- Latest release v1.47.0: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.47.0
- `semconv/v1.41.0` (last package with `gen_ai.*`, 50 keys): https://github.com/open-telemetry/opentelemetry-go/tree/v1.47.0/semconv/v1.41.0
- `semconv/v1.38.0` (40 `gen_ai.*` keys, no cache/reasoning): https://pkg.go.dev/go.opentelemetry.io/otel/semconv/v1.38.0
- OTLP trace exporters: https://github.com/open-telemetry/opentelemetry-go/tree/v1.47.0/exporters/otlp/otlptrace
- OTLP exporter env vars and defaults: https://opentelemetry.io/docs/specs/otel/protocol/exporter/

**OpenInference (Arize)**
- Spec: https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md
- Repo README / instrumentation inventory: https://github.com/Arize-ai/openinference/blob/main/README.md
- Go semconv module v0.1.11, Apache-2.0: https://pkg.go.dev/github.com/Arize-ai/openinference/go/openinference-semantic-conventions
- Go instrumentation config module v0.1.2 (`OPENINFERENCE_HIDE_*`): https://pkg.go.dev/github.com/Arize-ai/openinference/go/openinference-instrumentation
- `openai-go` instrumentor README and `go.mod` (`openai-go v1.12.0`, chat completions): https://github.com/Arize-ai/openinference/tree/main/go/openinference-instrumentation-openai-go

**OpenLLMetry (Traceloop)**
- Go SDK, latest tag `traceloop-sdk/v0.1.3`, last commit 2026-01-17: https://github.com/traceloop/go-openllmetry

**Phoenix**
- Docker ports 6006 / 4317: https://arize.com/docs/phoenix/self-hosting/deployment-options/docker
- Configuration (`PHOENIX_PORT` 6006, `PHOENIX_GRPC_PORT` 4317, `/v1/traces`): https://arize.com/docs/phoenix/self-hosting/configuration

**In-repo sources**
- [`ARCHITECTURE.md`](../../ARCHITECTURE.md) §4, §5, §6, §8, §9, §10, §11, §12, §14
- [`internal/core/provider.go`](../../internal/core/provider.go), [`internal/core/usage.go`](../../internal/core/usage.go)
- [`internal/provider/openai/openai.go`](../../internal/provider/openai/openai.go), [`internal/provider/openai/pricing.go`](../../internal/provider/openai/pricing.go)
- [`main.go`](../../main.go), [`internal/tui/tui.go`](../../internal/tui/tui.go)
- [`cmd/drift/main.go`](../../cmd/drift/main.go), [`docs/reports/drift-baseline.json`](../reports/drift-baseline.json)
- [`docs/plans/2026-09-20-test-quality-and-architectural-fitness-steering.md`](2026-09-20-test-quality-and-architectural-fitness-steering.md) §2, §3, §5.5
- [`docs/plans/2026-08-23-tui-telemetry-design.md`](2026-08-23-tui-telemetry-design.md)
- [`docs/plans/2026-09-05-fleet-cpg-engine-implementation.md`](2026-09-05-fleet-cpg-engine-implementation.md) (Observability row: shared span taxonomy)


---

## Appendix A — Independent verification (2026-10-06)

The load-bearing claims above were re-checked by the delegating session against
the downloaded modules rather than taken from the report. Method: `go mod
download`, then grep of the extracted module in `GOMODCACHE`.

| Claim | Verdict | Measured |
|---|---|---|
| `gen_ai.*` keys vanish from Go semconv after v1.41.0 | **Confirmed** | Distinct `gen_ai.*` attribute strings in `go.opentelemetry.io/otel@v1.47.0`: `semconv/v1.40.0` = 51, `v1.41.0` = **57**, `v1.42.0` = **0**, `v1.43.0` = **0** |
| `gen_ai.*` defines no cost attribute | **Confirmed** | No key matching `cost`/`price`/`usd` under `semconv/v1.41.0` |
| `OTEL_SDK_DISABLED` absent from opentelemetry-go | **Confirmed** for `otel@v1.47.0` and `otel/sdk@v1.47.0` (no match in either module tree). Not checked in `contrib`/`autoexport`. |
| OpenInference Go semconv module exists at v0.1.11 | **Confirmed** | `go list -m -versions` returns … v0.1.9, v0.1.10, v0.1.11 |
| `usageFromResponse` is a pre-existing bug | **Not confirmed — corrected in §7/§8** | `costForUsage` already reconstructs the inclusive total; no current consumer is wrong |

The report's own count of "50" `gen_ai.*` keys at v1.41.0 differs from the 57
measured here; the likely cause is counting exported Go constants rather than
distinct attribute strings. The cliff to zero at v1.42.0 — the finding the
pinning recommendation rests on — reproduces exactly.

One detail worth pulling forward: `semconv/v1.41.0` does define
`gen_ai.usage.cache_read.input_tokens` and
`gen_ai.usage.cache_creation.input_tokens`, so this repo's three-way input split
maps onto the convention **1:1** with no information loss. The only adjustment
the mapping needs is that `gen_ai.usage.input_tokens` carries the inclusive
total.
