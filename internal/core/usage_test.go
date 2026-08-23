package core

import "testing"

func TestUsageTotalsAddAccumulatesTokensAndKnownCosts(t *testing.T) {
	firstCost := 0.0012
	secondCost := 0.0034
	var totals UsageTotals
	totals.Add(Usage{InputTokens: 1200, OutputTokens: 340, ReasoningTokens: 90, TotalTokens: 1540, CostUSD: &firstCost})
	totals.Add(Usage{InputTokens: 800, OutputTokens: 200, ReasoningTokens: 40, TotalTokens: 1000, CostUSD: &secondCost})

	if totals.InputTokens != 2000 || totals.OutputTokens != 540 || totals.ReasoningTokens != 130 || totals.TotalTokens != 2540 {
		t.Fatalf("totals = %+v", totals)
	}
	if !totals.CostKnown || totals.CostUSD != 0.0046 {
		t.Fatalf("cost = %v (known=%v), want 0.0046 known", totals.CostUSD, totals.CostKnown)
	}
}

func TestUsageTotalsCostIsUnavailableWhenAnyTurnHasUnknownPricing(t *testing.T) {
	cost := 0.0012
	var totals UsageTotals
	totals.Add(Usage{TotalTokens: 10, CostUSD: &cost})
	totals.Add(Usage{TotalTokens: 20})

	if totals.CostKnown {
		t.Fatalf("cost unexpectedly known: %+v", totals)
	}
}

func TestCollectResponseRejectsInvalidCompletionSequence(t *testing.T) {
	tests := []struct {
		name   string
		events []StreamEvent
	}{
		{
			name:   "missing completion",
			events: []StreamEvent{{Kind: EventText, Text: "partial"}},
		},
		{
			name: "duplicate completion",
			events: []StreamEvent{
				{Kind: EventComplete},
				{Kind: EventComplete},
			},
		},
		{
			name: "text after completion",
			events: []StreamEvent{
				{Kind: EventComplete},
				{Kind: EventText, Text: "late"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := CollectResponse(events(tt.events...)); err == nil {
				t.Fatal("CollectResponse returned nil error")
			}
		})
	}
}

func events(values ...StreamEvent) <-chan StreamEvent {
	ch := make(chan StreamEvent, len(values))
	for _, value := range values {
		ch <- value
	}
	close(ch)
	return ch
}
