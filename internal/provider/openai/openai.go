package openai

import (
	"context"
	"fmt"
	"strings"

	"swe-term/internal/core"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
)

const (
	Name         = "openai"
	DefaultModel = "gpt-5.6-luna"
)

type Provider struct {
	client openai.Client
	model  string
}

// New builds a Provider. Empty model falls back to DefaultModel.
// apiKey is taken from the caller
func New(apiKey, model string) *Provider {
	if model == "" {
		model = DefaultModel
	}
	return &Provider{
		client: openai.NewClient(option.WithAPIKey(apiKey)),
		model:  model,
	}
}

func (p *Provider) Stream(ctx context.Context, req core.StreamRequest) (<-chan core.StreamEvent, error) {
	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("openai: empty StreamRequest.Messages")
	}
	model := req.Model
	if model == "" {
		model = p.model
	}

	ch := make(chan core.StreamEvent, 16)
	go func() {
		defer close(ch)

		resp, err := p.client.Responses.New(ctx, responses.ResponseNewParams{
			Input: responses.ResponseNewParamsInputUnion{
				OfString: openai.String(flattenInput(req.Messages)),
			},
			Model: openai.ChatModel(model),
		})
		if err != nil {
			select {
			case ch <- core.StreamEvent{Kind: core.EventError, Err: err}:
			case <-ctx.Done():
			}
			return
		}

		select {
		case ch <- core.StreamEvent{Kind: core.EventText, Text: resp.OutputText()}:
		case <-ctx.Done():
		}
	}()
	return ch, nil
}

func (p *Provider) Models(ctx context.Context) ([]core.Model, error) {
	_ = ctx
	return []core.Model{{ID: p.model, Provider: Name}}, nil
}

func flattenInput(messages []core.Message) string {
	if len(messages) == 1 {
		return messages[0].Content
	}
	var b strings.Builder
	for i, m := range messages {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(string(m.Role))
		b.WriteString(": ")
		b.WriteString(m.Content)
	}
	return b.String()
}
