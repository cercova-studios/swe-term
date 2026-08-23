package tui

import (
	"fmt"
	"strconv"
	"strings"

	"swe-term/internal/core"
)

type statusState struct {
	Model     string
	Reasoning string
	Last      core.Usage
	Session   core.UsageTotals
}

func formatStatusLine(state statusState) string {
	model := state.Model
	if state.Last.Model != "" {
		model = state.Last.Model
	}
	if model == "" {
		model = "no-model"
	}
	reasoning := state.Reasoning
	if state.Last.Reasoning != "" {
		reasoning = state.Last.Reasoning
	}
	if reasoning == "" {
		reasoning = "auto"
	}
	parts := []string{model, "reasoning: " + reasoning}
	turnInput := state.Last.InputTokens + state.Last.CachedInputTokens + state.Last.CacheWriteTokens
	if turnInput > 0 || state.Last.OutputTokens > 0 || state.Last.CostUSD != nil {
		parts = append(parts, "turn: "+formatTokens(turnInput)+" in / "+formatTokens(state.Last.OutputTokens)+" out")
	}
	if state.Session.TotalTokens > 0 {
		parts = append(parts, "session: "+formatTokens(state.Session.TotalTokens)+" tokens")
	}
	if state.Session.Turns > 0 {
		if state.Session.CostKnown {
			parts = append(parts, formatCost(state.Session.CostUSD))
		} else {
			parts = append(parts, "$—")
		}
	}
	return strings.Join(parts, " · ")
}

func formatCost(cost float64) string {
	if cost > 0 && cost < 0.0001 {
		return "$" + strconv.FormatFloat(cost, 'f', -1, 64)
	}
	return fmt.Sprintf("$%.4f", cost)
}

func formatTokens(value int64) string {
	if value < 1000 {
		return strconv.FormatInt(value, 10)
	}
	units := []string{"", "k", "M", "B"}
	amount := float64(value)
	unit := 0
	for amount >= 1000 && unit < len(units)-1 {
		amount /= 1000
		unit++
	}
	text := strconv.FormatFloat(amount, 'f', 1, 64)
	text = strings.TrimSuffix(text, ".0")
	return text + units[unit]
}
