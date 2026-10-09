package telemetry

import (
	"context"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"swe-term/internal/core"
)

const tracerName = "swe-term/internal/telemetry"

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
	return &Provider{
		Inner:  inner,
		Name:   name,
		tracer: otel.Tracer(tracerName),
	}
}

func (p *Provider) Models(ctx context.Context) ([]core.Model, error) {
	return p.Inner.Models(ctx)
}

func (p *Provider) Stream(ctx context.Context, req core.StreamRequest) (<-chan core.StreamEvent, error) {
	ctx, span := p.tracer.Start(ctx, spanName(req.Model),
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(
			operationChat,
			providerName(p.Name),
			requestModel(req.Model),
			requestStream(true),
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

		var events, textEvents, responseBytes int64
		for event := range inner {
			events++
			switch event.Kind {
			case core.EventText:
				textEvents++
				responseBytes += int64(len(event.Text))
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
				// rather than blocking this goroutine forever; a cancelled
				// stream is not a successful one (invariant 9).
				span.SetStatus(codes.Error, ctx.Err().Error())
				span.SetAttributes(streamShapeAttributes(events, textEvents, responseBytes)...)
				return
			}
		}
		span.SetAttributes(streamShapeAttributes(events, textEvents, responseBytes)...)
	}()
	return out, nil
}

func streamShapeAttributes(events, textEvents, responseBytes int64) []attribute.KeyValue {
	return []attribute.KeyValue{
		attribute.Int64(streamEventsKey, events),
		attribute.Int64(textEventsKey, textEvents),
		attribute.Int64(responseBytesKey, responseBytes),
	}
}

// usageAttributes maps core.Usage onto the convention.
//
// inputTokens is the one field that cannot be copied straight across.
// core.Usage.inputTokens is uncached input only, while
// gen_ai.usage.input_tokens is defined to include cached and cache-creation
// tokens, so the inclusive total is reconstructed here. The expression is the
// same one internal/provider/openai.costForUsage uses for its tier test;
// emitting the disjoint field directly would understate input by the whole
// cache volume on exactly the cache-heavy sessions where it matters.
func usageAttributes(usage core.Usage) []attribute.KeyValue {
	attrs := []attribute.KeyValue{
		inputTokens(usage.InputTokens + usage.CachedInputTokens + usage.CacheWriteTokens),
		outputTokens(usage.OutputTokens),
		reasoningToks(usage.ReasoningTokens),
		cacheReadToks(usage.CachedInputTokens),
		cacheWriteToks(usage.CacheWriteTokens),
		attribute.Int64(totalTokensKey, usage.TotalTokens),
	}
	if usage.Model != "" {
		attrs = append(attrs, responseModel(usage.Model))
	}
	return append(attrs, cost(usage.CostUSD)...)
}
