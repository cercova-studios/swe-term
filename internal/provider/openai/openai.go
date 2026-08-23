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

	params := responses.ResponseNewParams{
		Input: responses.ResponseNewParamsInputUnion{
			OfString: openai.String(flattenInput(req.Messages)),
		},
		Model: openai.ChatModel(model),
	}
	if reasoning, ok := reasoningParam(req.Reasoning); ok {
		params.Reasoning = reasoning
	}

	ch := make(chan core.StreamEvent, 64)
	go func() {
		defer close(ch)
		stream := p.client.Responses.NewStreaming(ctx, params)
		defer stream.Close()

		send := func(ev core.StreamEvent) bool {
			select {
			case ch <- ev:
				return true
			case <-ctx.Done():
				return false
			}
		}

		for stream.Next() {
			switch v := stream.Current().AsAny().(type) {
			case responses.ResponseTextDeltaEvent:
				if v.Delta == "" {
					continue
				}
				if !send(core.StreamEvent{Kind: core.EventText, Text: v.Delta}) {
					return
				}
			case responses.ResponseCompletedEvent:
				if !send(core.StreamEvent{Kind: core.EventComplete, Usage: usageFromResponse(v.Response, model)}) {
					return
				}
			case responses.ResponseErrorEvent:
				err := fmt.Errorf("openai: %s", v.Message)
				if v.Code != "" {
					err = fmt.Errorf("openai: %s (%s)", v.Message, v.Code)
				}
				send(core.StreamEvent{Kind: core.EventError, Err: err})
				return
			case responses.ResponseFailedEvent:
				send(core.StreamEvent{Kind: core.EventError, Err: fmt.Errorf("openai: response failed")})
				return
			}
		}
		if err := stream.Err(); err != nil {
			send(core.StreamEvent{Kind: core.EventError, Err: err})
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
