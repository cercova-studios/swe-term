package openai

import (
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"

	"swe-term/internal/core"
)

type tokenRates struct {
	input      float64
	output     float64
	cacheRead  float64
	cacheWrite float64
}

type modelPricing struct {
	short tokenRates
	long  tokenRates
}

// Rates are standard USD per million tokens from OpenAI's API pricing table,
// verified 2026-08-23: https://developers.openai.com/api/docs/pricing
// Requests with more than 272K input tokens use the long-context rates.
var modelPrices = map[string]modelPricing{
	"gpt-5.6": {
		short: tokenRates{input: 4.00, output: 20.00, cacheRead: 0.40, cacheWrite: 5.00},
		long:  tokenRates{input: 8.00, output: 30.00, cacheRead: 0.80, cacheWrite: 10.00},
	},
	"gpt-5.6-sol": {
		short: tokenRates{input: 4.00, output: 20.00, cacheRead: 0.40, cacheWrite: 5.00},
		long:  tokenRates{input: 8.00, output: 30.00, cacheRead: 0.80, cacheWrite: 10.00},
	},
	"gpt-5.6-terra": {
		short: tokenRates{input: 2.00, output: 12.00, cacheRead: 0.20, cacheWrite: 2.50},
		long:  tokenRates{input: 4.00, output: 18.00, cacheRead: 0.40, cacheWrite: 5.00},
	},
	"gpt-5.6-luna": {
		short: tokenRates{input: 0.20, output: 1.20, cacheRead: 0.02, cacheWrite: 0.25},
		long:  tokenRates{input: 0.40, output: 1.80, cacheRead: 0.04, cacheWrite: 0.50},
	},
}

func reasoningParam(value string) (shared.ReasoningParam, bool) {
	if value == "" {
		return shared.ReasoningParam{}, false
	}
	return shared.ReasoningParam{Effort: shared.ReasoningEffort(value)}, true
}

func usageFromResponse(response responses.Response, fallbackModel string) core.Usage {
	model := string(response.Model)
	if model == "" {
		model = fallbackModel
	}
	usage := response.Usage
	cached := usage.InputTokensDetails.CachedTokens
	cacheWrite := usage.InputTokensDetails.CacheWriteTokens
	input := usage.InputTokens - cached - cacheWrite
	if input < 0 {
		input = 0
	}
	result := core.Usage{
		Model:             model,
		Reasoning:         string(response.Reasoning.Effort),
		InputTokens:       input,
		OutputTokens:      usage.OutputTokens,
		ReasoningTokens:   usage.OutputTokensDetails.ReasoningTokens,
		CachedInputTokens: cached,
		CacheWriteTokens:  cacheWrite,
		TotalTokens:       usage.TotalTokens,
	}
	result.CostUSD = costForUsage(model, result)
	return result
}

func costForUsage(model string, usage core.Usage) *float64 {
	pricing, ok := modelPrices[model]
	if !ok {
		return nil
	}
	rates := pricing.short
	inputTokens := usage.InputTokens + usage.CachedInputTokens + usage.CacheWriteTokens
	if inputTokens > 272_000 {
		rates = pricing.long
	}
	usd := (float64(usage.InputTokens)*rates.input +
		float64(usage.CachedInputTokens)*rates.cacheRead +
		float64(usage.CacheWriteTokens)*rates.cacheWrite +
		float64(usage.OutputTokens)*rates.output) / 1_000_000
	return &usd
}
