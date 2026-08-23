package core

import (
	"fmt"
	"strings"
)

// Usage is provider-reported usage for one completed model response.
// InputTokens is uncached input; cached and cache-write tokens are tracked separately.
type Usage struct {
	Model             string
	Reasoning         string
	InputTokens       int64
	OutputTokens      int64
	ReasoningTokens   int64
	CachedInputTokens int64
	CacheWriteTokens  int64
	TotalTokens       int64
	CostUSD           *float64
}

// UsageTotals accumulates provider usage for the current TUI session.
type UsageTotals struct {
	InputTokens       int64
	OutputTokens      int64
	ReasoningTokens   int64
	CachedInputTokens int64
	CacheWriteTokens  int64
	TotalTokens       int64
	CostUSD           float64
	CostKnown         bool
	Turns             int64
}

func (t *UsageTotals) Add(u Usage) {
	first := t.Turns == 0
	t.Turns++
	t.InputTokens += u.InputTokens
	t.OutputTokens += u.OutputTokens
	t.ReasoningTokens += u.ReasoningTokens
	t.CachedInputTokens += u.CachedInputTokens
	t.CacheWriteTokens += u.CacheWriteTokens
	t.TotalTokens += u.TotalTokens
	if u.CostUSD == nil {
		t.CostKnown = false
		return
	}
	t.CostUSD += *u.CostUSD
	if first {
		t.CostKnown = true
	}
}

type Response struct {
	Text  string
	Usage Usage
}

func CollectResponse(ch <-chan StreamEvent) (Response, error) {
	var response Response
	var b strings.Builder
	completed := false
	for ev := range ch {
		switch ev.Kind {
		case EventText:
			if completed {
				response.Text = b.String()
				return response, fmt.Errorf("provider stream: text after completion")
			}
			b.WriteString(ev.Text)
		case EventComplete:
			if completed {
				response.Text = b.String()
				return response, fmt.Errorf("provider stream: duplicate completion")
			}
			completed = true
			response.Usage = ev.Usage
		case EventError:
			response.Text = b.String()
			return response, ev.Err
		}
	}
	response.Text = b.String()
	if !completed {
		return response, fmt.Errorf("provider stream: closed without completion")
	}
	return response, nil
}
