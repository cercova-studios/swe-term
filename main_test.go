package main

import (
	"context"
	"testing"

	"swe-term/internal/config"
	"swe-term/internal/core"
)

func TestMockProviderEmitsDeterministicTelemetry(t *testing.T) {
	provider, err := newProvider(config.Config{Provider: "mock", Model: "mock-model", Reasoning: "high"})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := provider.Stream(context.Background(), core.StreamRequest{
		Messages:  []core.Message{{Role: core.RoleUser, Content: "hello"}},
		Model:     "mock-model",
		Reasoning: "high",
	})
	if err != nil {
		t.Fatal(err)
	}
	response, err := core.CollectResponse(stream)
	if err != nil {
		t.Fatal(err)
	}
	if response.Usage.Model != "mock-model" || response.Usage.Reasoning != "high" || response.Usage.TotalTokens == 0 {
		t.Fatalf("usage = %+v", response.Usage)
	}
}
