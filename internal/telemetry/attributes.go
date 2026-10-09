// Package telemetry emits OpenTelemetry traces for provider calls.
//
// It exists as its own package so that no OpenTelemetry type reaches
// internal/core: ARCHITECTURE.md §10 invariant 13 forbids vendor-specific
// types in core state. The instrumentation attaches as a core.Provider
// decorator (§9, "Provider adapters"), so core stream semantics are unchanged
// and an uninstrumented build behaves identically.
package telemetry

import semconv "go.opentelemetry.io/otel/semconv/v1.41.0"

// The span taxonomy lives here, in one owned file, rather than being spelled
// at each call site.
//
// The semconv import is pinned to v1.41.0 deliberately, and upgrading it is
// not a routine bump. The GenAI conventions moved out of
// open-telemetry/semantic-conventions into a separate, as yet untagged
// semantic-conventions-genai repository, and the Go code generation stopped
// with them: measured against otel@v1.47.0, semconv/v1.40.0 carries 51
// distinct gen_ai.* attribute strings and v1.41.0 carries 57, while v1.42.0
// and v1.43.0 carry none at all. The otel module itself may be upgraded
// freely; this import path may not, until the GenAI repository tags a release
// and the Go generator catches up.
//
// Verification and sources:
// docs/plans/2026-10-06-openinference-openllmetry-integration-research.md
// gen_ai.* attributes, all from semconv v1.41.0.
var (
	operationChat  = semconv.GenAIOperationNameChat
	requestStream  = semconv.GenAIRequestStreamKey.Bool
	providerName   = semconv.GenAIProviderNameKey.String
	requestModel   = semconv.GenAIRequestModelKey.String
	responseModel  = semconv.GenAIResponseModelKey.String
	inputTokens    = semconv.GenAIUsageInputTokensKey.Int64
	outputTokens   = semconv.GenAIUsageOutputTokensKey.Int64
	reasoningToks  = semconv.GenAIUsageReasoningOutputTokensKey.Int64
	cacheReadToks  = semconv.GenAIUsageCacheReadInputTokensKey.Int64
	cacheWriteToks = semconv.GenAIUsageCacheCreationInputTokensKey.Int64
)

// Keys with no gen_ai.* equivalent at v1.41.0.
//
// Cost: gen_ai.* defines no cost attribute at any published version, so these
// follow OpenInference, which does. Reasoning effort: gen_ai.* has no
// request-side reasoning or effort key either, and inventing a gen_ai.* name
// for it would put a key into a reserved namespace that no consumer reads, so
// it goes under this repository's own prefix instead.
const (
	costTotalUSDKey  = "llm.cost.total"
	reasoningKey     = "swe_term.reasoning_effort"
	totalTokensKey   = "swe_term.usage.total_tokens"
	streamEventsKey  = "swe_term.stream.events"
	textEventsKey    = "swe_term.stream.text_events"
	responseBytesKey = "swe_term.response.bytes"
)

// spanName follows the convention's "{operation} {model}" shape.
func spanName(model string) string {
	if model == "" {
		return operationChat.Value.AsString()
	}
	return operationChat.Value.AsString() + " " + model
}
