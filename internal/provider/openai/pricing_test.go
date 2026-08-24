package openai

import (
	"math"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"

	"swe-term/internal/core"
)

func TestCostForUsageUsesInputOutputAndCacheRates(t *testing.T) {
	cost := costForUsage("gpt-5.6-luna", core.Usage{
		InputTokens:       10_000,
		OutputTokens:      2_000,
		CachedInputTokens: 4_000,
		CacheWriteTokens:  1_000,
	})
	if cost == nil {
		t.Fatal("cost = nil")
	}
	// Luna catalog rates: $0.20/$1.20/$0.02/$0.25 per million tokens.
	want := 0.01*0.20 + 0.004*0.02 + 0.001*0.25 + 0.002*1.20
	if math.Abs(*cost-want) > 1e-12 {
		t.Fatalf("cost = %.10f, want %.10f", *cost, want)
	}
}

func TestCostForUsageUsesLongContextRatesAbove272KInput(t *testing.T) {
	cost := costForUsage("gpt-5.6-sol", core.Usage{
		InputTokens:  272_001,
		OutputTokens: 10_000,
	})
	if cost == nil {
		t.Fatal("cost = nil")
	}
	want := 0.272001*8.00 + 0.01*30.00
	if math.Abs(*cost-want) > 1e-12 {
		t.Fatalf("cost = %.10f, want %.10f", *cost, want)
	}
}

func TestUsageFromResponseSeparatesCachedAndCacheWriteInput(t *testing.T) {
	got := usageFromResponse(responses.Response{
		Model:     "gpt-5.6-luna",
		Reasoning: shared.Reasoning{Effort: shared.ReasoningEffortMedium},
		Usage: responses.ResponseUsage{
			InputTokens:  10_000,
			OutputTokens: 2_000,
			TotalTokens:  12_000,
			InputTokensDetails: responses.ResponseUsageInputTokensDetails{
				CachedTokens:     4_000,
				CacheWriteTokens: 1_000,
			},
			OutputTokensDetails: responses.ResponseUsageOutputTokensDetails{
				ReasoningTokens: 500,
			},
		},
	}, "fallback-model")
	if got.Model != "gpt-5.6-luna" || got.Reasoning != "medium" ||
		got.InputTokens != 5_000 || got.CachedInputTokens != 4_000 || got.CacheWriteTokens != 1_000 ||
		got.OutputTokens != 2_000 || got.ReasoningTokens != 500 || got.TotalTokens != 12_000 || got.CostUSD == nil {
		t.Fatalf("usage = %+v", got)
	}
}

func TestCostForUsageReturnsUnavailableForUnknownModel(t *testing.T) {
	if costForUsage("unknown-model", core.Usage{InputTokens: 1, OutputTokens: 1}) != nil {
		t.Fatal("unknown model unexpectedly has a cost")
	}
}
