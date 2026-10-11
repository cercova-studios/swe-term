package telemetry

import (
	"context"
	"fmt"
	"os"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.41.0"
)

// tracingEnabled reports whether tracing is configured.
//
// Tracing is off unless an OTLP endpoint is set, and this is a code decision
// rather than a convention one: OTEL_SDK_DISABLED is specified by
// OpenTelemetry but is not implemented anywhere in opentelemetry-go v1.47.0
// (verified by grepping the otel and otel/sdk module trees), so there is no
// standard switch to honour. swe-term is a local terminal program; it must
// not open a network connection because a library was linked in.
//
// OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_TRACES_ENDPOINT are the
// standard variables, read here by name because the exporter only reads them
// once it has already been constructed.
func tracingEnabled() bool {
	return os.Getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") != "" ||
		os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT") != ""
}

// Setup installs a global tracer provider when an OTLP endpoint is
// configured, and returns a shutdown function that flushes pending spans.
//
// When tracing is off it returns a no-op shutdown and no error, so callers
// need no conditional. A configuration failure is returned rather than
// swallowed: starting a run that silently drops its telemetry is the kind of
// reassuring falsehood ARCHITECTURE.md §10 invariant 12 forbids. The caller
// decides whether that is fatal.
func Setup(ctx context.Context, version string) (func(context.Context) error, error) {
	if !tracingEnabled() {
		return func(context.Context) error { return nil }, nil
	}

	exporter, err := otlptracehttp.New(ctx)
	if err != nil {
		return nil, fmt.Errorf("telemetry: OTLP exporter: %w", err)
	}

	// NewSchemaless, not NewWithAttributes(semconv.SchemaURL, ...): the SDK's
	// default resource is built against the module's newest schema (1.43.0 at
	// otel v1.47.0) while the gen_ai.* attributes are pinned to 1.41.0, and
	// resource.Merge refuses two different non-empty schema URLs. A
	// schemaless resource carries the service attributes — whose names are
	// stable across these versions — and lets the merge adopt the default's
	// schema. Found by exporting to a real OTLP receiver; the merge error is
	// invisible to a test that never builds a resource.
	res, err := resource.Merge(
		resource.Default(),
		resource.NewSchemaless(
			semconv.ServiceName("swe-term"),
			semconv.ServiceVersion(version),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("telemetry: resource: %w", err)
	}

	provider := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
	)
	otel.SetTracerProvider(provider)
	return provider.Shutdown, nil
}
