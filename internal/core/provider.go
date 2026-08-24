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
	Messages  []Message
	Model     string
	Reasoning string
}

type EventKind string

const (
	EventText     EventKind = "text"
	EventComplete EventKind = "complete"
	EventError    EventKind = "error"
)

type StreamEvent struct {
	Kind  EventKind
	Text  string
	Err   error
	Usage Usage
}

type Model struct {
	ID       string
	Provider string
}

// Provider streams completions. Implementations live in internal/provider/*, not here.
//
// Stream returns a channel that must be drained. Setup failures may return an
// error; failures after the stream starts arrive as EventError. Successful
// streams emit one EventComplete with provider-reported usage before the
// channel closes. The channel is always closed.
type Provider interface {
	Stream(ctx context.Context, req StreamRequest) (<-chan StreamEvent, error)
	Models(ctx context.Context) ([]Model, error)
}
