package telemetry

import (
	"context"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	// Pinned to v1.41.0 deliberately; upgrading this path is not a routine
	// bump. The GenAI conventions moved to an untagged
	// semantic-conventions-genai repository and the Go codegen stopped with
	// them: against otel@v1.47.0, semconv/v1.41.0 carries 57 distinct gen_ai.*
	// attribute strings while v1.42.0 and v1.43.0 carry zero. The otel module
	// may be upgraded freely; this import may not, until that repository tags a
	// release. See
	// docs/plans/2026-10-06-openinference-openllmetry-integration-research.md
	semconv "go.opentelemetry.io/otel/semconv/v1.41.0"
	"go.opentelemetry.io/otel/trace"

	"swe-term/internal/core"
)

const tracerName = "swe-term/internal/telemetry"

// Keys with no gen_ai.* equivalent at v1.41.0. Cost follows OpenInference,
// which defines it; gen_ai.* defines no cost attribute at any version.
// Reasoning effort has no gen_ai.* request-side key either, and inventing one
// would put a name into a reserved namespace that no consumer reads.
const (
	costTotalUSDKey = "llm.cost.total"
	reasoningKey    = "swe_term.reasoning_effort"
)

// Provider wraps a core.Provider and traces each Stream call.
//
// It is a strict pass-through. Every StreamEvent the inner provider emits is
// forwarded unchanged, in order, exactly once, and the channel closes when
// the inner channel closes. The decorator observes; it never repairs,
// reorders, filters, or synthesises an event. That is what makes it safe
// under ARCHITECTURE.md §10 invariant 1 (exactly one terminal completion) and
// invariant 9 (failure is never flattened into successful output).
type Provider struct {
	Inner  core.Provider
	Name   string
	tracer trace.Tracer
}

// WrapProvider returns inner unchanged when tracing is off, so a build with
// no endpoint configured carries no decorator at all.
func WrapProvider(inner core.Provider, name string) core.Provider {
	if !tracingEnabled() {
		return inner
	}
	return &Provider{Inner: inner, Name: name, tracer: otel.Tracer(tracerName)}
}

func (p *Provider) Models(ctx context.Context) ([]core.Model, error) {
	return p.Inner.Models(ctx)
}

func (p *Provider) Stream(ctx context.Context, req core.StreamRequest) (<-chan core.StreamEvent, error) {
	// Span name is the convention's "{operation} {model}".
	ctx, span := p.tracer.Start(ctx, "chat "+req.Model,
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(
			semconv.GenAIOperationNameChat,
			semconv.GenAIProviderNameKey.String(p.Name),
			semconv.GenAIRequestModelKey.String(req.Model),
			semconv.GenAIRequestStreamKey.Bool(true),
			attribute.String(reasoningKey, req.Reasoning),
		),
	)

	inner, err := p.Inner.Stream(ctx, req)
	if err != nil {
		// A setup failure ends the span here; no stream exists to observe.
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		span.End()
		return nil, err
	}

	out := make(chan core.StreamEvent)
	go func() {
		defer close(out)
		defer span.End()

		for event := range inner {
			switch event.Kind {
			case core.EventComplete:
				span.SetAttributes(usageAttributes(event.Usage)...)
			case core.EventError:
				// Recorded, not consumed: the event is still forwarded below
				// so the caller fails exactly as it would without tracing.
				if event.Err != nil {
					span.RecordError(event.Err)
					span.SetStatus(codes.Error, event.Err.Error())
				} else {
					span.SetStatus(codes.Error, "provider stream: error without details")
				}
			}

			select {
			case out <- event:
			case <-ctx.Done():
				// The consumer stopped draining. Report the abandoned stream
				// rather than blocking forever; a cancelled stream is not a
				// successful one (invariant 9).
				span.SetStatus(codes.Error, ctx.Err().Error())
				return
			}
		}
	}()
	return out, nil
}

// usageAttributes maps core.Usage onto the convention.
//
// InputTokens is the one field that cannot be copied straight across.
// core.Usage.InputTokens is uncached input only, while
// gen_ai.usage.input_tokens is defined to include cached and cache-creation
// tokens, so the inclusive total is reconstructed here. The expression is the
// same one internal/provider/openai.costForUsage uses for its tier test;
// emitting the disjoint field directly would understate input by the whole
// cache volume on exactly the cache-heavy sessions where it matters.
func usageAttributes(usage core.Usage) []attribute.KeyValue {
	attrs := []attribute.KeyValue{
		semconv.GenAIUsageInputTokensKey.Int64(usage.InputTokens + usage.CachedInputTokens + usage.CacheWriteTokens),
		semconv.GenAIUsageOutputTokensKey.Int64(usage.OutputTokens),
		semconv.GenAIUsageReasoningOutputTokensKey.Int64(usage.ReasoningTokens),
		semconv.GenAIUsageCacheReadInputTokensKey.Int64(usage.CachedInputTokens),
		semconv.GenAIUsageCacheCreationInputTokensKey.Int64(usage.CacheWriteTokens),
	}
	if usage.Model != "" {
		attrs = append(attrs, semconv.GenAIResponseModelKey.String(usage.Model))
	}
	// ARCHITECTURE.md §10 invariant 12: unknown pricing stays explicit and is
	// never converted to a reassuring zero. An absent CostUSD means the model
	// is not in the price table, which is not the same as a free call.
	if usage.CostUSD != nil {
		attrs = append(attrs, attribute.Float64(costTotalUSDKey, *usage.CostUSD))
	}
	return attrs
}
