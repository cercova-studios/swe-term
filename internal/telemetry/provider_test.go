package telemetry

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"go.opentelemetry.io/otel"

	"swe-term/internal/core"
)

// scriptedProvider emits an exact event sequence so the decorator can be
// compared against it byte for byte.
type scriptedProvider struct {
	events  []core.StreamEvent
	initErr error
}

func (s *scriptedProvider) Models(context.Context) ([]core.Model, error) { return nil, nil }

func (s *scriptedProvider) Stream(context.Context, core.StreamRequest) (<-chan core.StreamEvent, error) {
	if s.initErr != nil {
		return nil, s.initErr
	}
	ch := make(chan core.StreamEvent, len(s.events))
	for _, event := range s.events {
		ch <- event
	}
	close(ch)
	return ch, nil
}

func tracingProvider(inner core.Provider) *Provider {
	// otel.Tracer with no global provider installed returns a no-op tracer,
	// so this needs no exporter, no endpoint, and no network.
	return &Provider{Inner: inner, Name: "test", tracer: otel.Tracer(tracerName)}
}

func drain(t *testing.T, ch <-chan core.StreamEvent) []core.StreamEvent {
	t.Helper()
	var got []core.StreamEvent
	for event := range ch {
		got = append(got, event)
	}
	return got
}

// The decorator's whole safety argument is that it observes without altering.
// ARCHITECTURE.md §10 invariant 1 (exactly one terminal completion) and
// invariant 9 (failure is never flattened into successful output) are
// properties of the event sequence, so they are checked as a sequence
// equality over the shapes a provider can produce — including the malformed
// ones, because masking those would hide a core violation rather than
// surface it.
func TestTracingProviderForwardsEveryStreamUnchanged(t *testing.T) {
	boom := errors.New("upstream failed")
	usage := core.Usage{Model: "m", InputTokens: 3, CachedInputTokens: 2, CacheWriteTokens: 1, OutputTokens: 4, TotalTokens: 10}

	streams := []struct {
		name   string
		events []core.StreamEvent
	}{
		{"text then completion", []core.StreamEvent{
			{Kind: core.EventText, Text: "hello "},
			{Kind: core.EventText, Text: "world"},
			{Kind: core.EventComplete, Usage: usage},
		}},
		{"error mid-stream", []core.StreamEvent{
			{Kind: core.EventText, Text: "partial"},
			{Kind: core.EventError, Err: boom},
		}},
		{"error without details", []core.StreamEvent{
			{Kind: core.EventError},
		}},
		{"duplicate completion is not repaired", []core.StreamEvent{
			{Kind: core.EventComplete, Usage: usage},
			{Kind: core.EventComplete, Usage: usage},
		}},
		{"text after completion is not reordered", []core.StreamEvent{
			{Kind: core.EventComplete, Usage: usage},
			{Kind: core.EventText, Text: "late"},
		}},
		{"no completion at all", []core.StreamEvent{
			{Kind: core.EventText, Text: "truncated"},
		}},
		{"empty stream", nil},
	}

	for _, stream := range streams {
		t.Run(stream.name, func(t *testing.T) {
			ch, err := tracingProvider(&scriptedProvider{events: stream.events}).Stream(context.Background(), core.StreamRequest{Model: "m"})
			if err != nil {
				t.Fatal(err)
			}
			got := drain(t, ch)
			if !reflect.DeepEqual(got, stream.events) {
				t.Fatalf("decorator altered the stream:\n got %+v\nwant %+v", got, stream.events)
			}
		})
	}
}

// The decorated stream must reach core.CollectResponse with the same verdict
// it would have reached undecorated: tracing an invariant violation must not
// make it pass.
func TestTracingProviderPreservesCollectResponseVerdict(t *testing.T) {
	usage := core.Usage{Model: "m", TotalTokens: 1}
	cases := []struct {
		name    string
		events  []core.StreamEvent
		wantErr string
	}{
		{"well formed", []core.StreamEvent{{Kind: core.EventText, Text: "ok"}, {Kind: core.EventComplete, Usage: usage}}, ""},
		{"duplicate completion", []core.StreamEvent{{Kind: core.EventComplete}, {Kind: core.EventComplete}}, "duplicate completion"},
		{"text after completion", []core.StreamEvent{{Kind: core.EventComplete}, {Kind: core.EventText, Text: "late"}}, "text after completion"},
		{"missing completion", []core.StreamEvent{{Kind: core.EventText, Text: "x"}}, "closed without completion"},
		{"error surfaces", []core.StreamEvent{{Kind: core.EventError, Err: errors.New("upstream failed")}}, "upstream failed"},
	}

	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			bare, _ := (&scriptedProvider{events: test.events}).Stream(context.Background(), core.StreamRequest{})
			_, bareErr := core.CollectResponse(bare)

			traced, _ := tracingProvider(&scriptedProvider{events: test.events}).Stream(context.Background(), core.StreamRequest{})
			_, tracedErr := core.CollectResponse(traced)

			if (bareErr == nil) != (tracedErr == nil) {
				t.Fatalf("tracing changed the verdict: bare=%v traced=%v", bareErr, tracedErr)
			}
			if test.wantErr == "" {
				if tracedErr != nil {
					t.Fatalf("unexpected error: %v", tracedErr)
				}
				return
			}
			if tracedErr == nil {
				t.Fatalf("expected an error mentioning %q", test.wantErr)
			}
			if bareErr.Error() != tracedErr.Error() {
				t.Fatalf("error text changed: bare=%q traced=%q", bareErr, tracedErr)
			}
		})
	}
}

func TestTracingProviderPropagatesSetupError(t *testing.T) {
	boom := errors.New("no credentials")
	ch, err := tracingProvider(&scriptedProvider{initErr: boom}).Stream(context.Background(), core.StreamRequest{})
	if !errors.Is(err, boom) {
		t.Fatalf("setup error = %v, want %v", err, boom)
	}
	if ch != nil {
		t.Fatal("a failed Stream must not return a channel")
	}
}

// With no endpoint configured the decorator must not exist at all: an
// unconfigured local CLI opens no connection and pays no wrapper.
func TestWrapProviderIsIdentityWhenDisabled(t *testing.T) {
	for _, key := range []string{"OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"} {
		t.Setenv(key, "")
	}
	inner := &scriptedProvider{}
	if got := WrapProvider(inner, "test"); got != core.Provider(inner) {
		t.Fatalf("WrapProvider wrapped the provider with tracing disabled: %T", got)
	}
}

func TestWrapProviderWrapsWhenEndpointIsSet(t *testing.T) {
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
	inner := &scriptedProvider{}
	if _, ok := WrapProvider(inner, "test").(*Provider); !ok {
		t.Fatal("WrapProvider did not wrap with an endpoint configured")
	}
}

// Invariant 12: unknown pricing stays unknown. A model absent from the price
// table reports no cost, which must not become a cost of zero.
func TestUnknownCostEmitsNoAttribute(t *testing.T) {
	for _, attr := range usageAttributes(core.Usage{InputTokens: 1}) {
		if string(attr.Key) == costTotalUSDKey {
			t.Fatalf("unknown cost emitted %v", attr.Value.AsFloat64())
		}
	}

	usd := 0.25
	var got float64
	found := false
	for _, attr := range usageAttributes(core.Usage{InputTokens: 1, CostUSD: &usd}) {
		if string(attr.Key) == costTotalUSDKey {
			got, found = attr.Value.AsFloat64(), true
		}
	}
	if !found || got != usd {
		t.Fatalf("known cost = %v (found=%v), want %v", got, found, usd)
	}
}

// gen_ai.usage.input_tokens is cache-inclusive by definition, while
// core.Usage.inputTokens is uncached only. Emitting the field directly would
// understate input by the whole cache volume.
func TestInputTokensAttributeIsCacheInclusive(t *testing.T) {
	attrs := usageAttributes(core.Usage{InputTokens: 5_000, CachedInputTokens: 4_000, CacheWriteTokens: 1_000})

	found := map[string]int64{}
	for _, attr := range attrs {
		found[string(attr.Key)] = attr.Value.AsInt64()
	}
	if got := found["gen_ai.usage.input_tokens"]; got != 10_000 {
		t.Fatalf("gen_ai.usage.input_tokens = %d, want 10000 (5000 uncached + 4000 cached + 1000 cache-write)", got)
	}
	if got := found["gen_ai.usage.cache_read.input_tokens"]; got != 4_000 {
		t.Fatalf("cache_read = %d, want 4000", got)
	}
	if got := found["gen_ai.usage.cache_creation.input_tokens"]; got != 1_000 {
		t.Fatalf("cache_creation = %d, want 1000", got)
	}
}
