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
}

func (p *Provider) Stream(ctx context.Context, req core.StreamRequest) (<-chan core.StreamEvent, error) {
	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("mock: empty StreamRequest.Messages")
	}
	ch := make(chan core.StreamEvent, 2)
	go func() {
		defer close(ch)
		if p.Err != nil {
			select {
			case ch <- core.StreamEvent{Kind: core.EventError, Err: p.Err}:
			case <-ctx.Done():
			}
			return
		}
		select {
		case ch <- core.StreamEvent{Kind: core.EventText, Text: p.Text}:
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
