package core

import "context"

type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

type Message struct {
	Role    Role
	Content string
}

type StreamRequest struct {
	Messages []Message
	Model    string
}

type EventKind string

const (
	EventText  EventKind = "text"
	EventError EventKind = "error"
)

type StreamEvent struct {
	Kind EventKind
	Text string
	Err  error
}

type Model struct {
	ID       string
	Provider string
}

// Provider streams completions. Implementations live in internal/provider/*, not here.
//
// Stream returns a channel that must be drained. Setup failures may return an error;
// failures after the stream starts arrive as EventError. The channel is always closed.
type Provider interface {
	Stream(ctx context.Context, req StreamRequest) (<-chan StreamEvent, error)
	Models(ctx context.Context) ([]Model, error)
}
