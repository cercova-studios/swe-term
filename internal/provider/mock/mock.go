package mock

import (
	"context"
	"fmt"

	"swe-term/internal/core"
)

const Name = "mock"

// Provider is an in-process core.Provider for tests. No network.
type Provider struct {
	Text  string
	Err   error
	Model string
	Usage core.Usage
}

func (p *Provider) Stream(ctx context.Context, req core.StreamRequest) (<-chan core.StreamEvent, error) {
	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("mock: empty StreamRequest.Messages")
	}
	ch := make(chan core.StreamEvent, 3)
	go func() {
		defer close(ch)
		if p.Err != nil {
			select {
			case ch <- core.StreamEvent{Kind: core.EventError, Err: p.Err}:
			case <-ctx.Done():
			}
			return
		}
		text := p.Text
		if text == "" {
			text = "mock response"
		}
		for _, chunk := range chunkRunes(text, 4) {
			select {
			case ch <- core.StreamEvent{Kind: core.EventText, Text: chunk}:
			case <-ctx.Done():
				return
			}
		}
		usage := p.Usage
		if usage.Model == "" {
			usage.Model = req.Model
			if usage.Model == "" {
				usage.Model = p.Model
			}
			if usage.Model == "" {
				usage.Model = "mock-model"
			}
		}
		if usage.Reasoning == "" {
			usage.Reasoning = req.Reasoning
			if usage.Reasoning == "" {
				usage.Reasoning = "auto"
			}
		}
		select {
		case ch <- core.StreamEvent{Kind: core.EventComplete, Usage: usage}:
		case <-ctx.Done():
		}
	}()
	return ch, nil
}

func (p *Provider) Models(ctx context.Context) ([]core.Model, error) {
	_ = ctx
	id := p.Model
	if id == "" {
		id = "mock-model"
	}
	return []core.Model{{ID: id, Provider: Name}}, nil
}

func chunkRunes(s string, n int) []string {
	if n <= 0 {
		return []string{s}
	}
	r := []rune(s)
	var out []string
	for len(r) > 0 {
		k := n
		if k > len(r) {
			k = len(r)
		}
		out = append(out, string(r[:k]))
		r = r[k:]
	}
	return out
}
