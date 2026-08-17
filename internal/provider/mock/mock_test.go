package mock

import (
	"context"
	"errors"
	"testing"

	"swe-term/internal/core"
)

func TestStreamText(t *testing.T) {
	t.Parallel()

	p := &Provider{Text: "formal verification is..."}
	ch, err := p.Stream(context.Background(), core.StreamRequest{
		Messages: []core.Message{{Role: core.RoleUser, Content: "explain"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := core.CollectText(ch)
	if err != nil {
		t.Fatal(err)
	}
	if got != p.Text {
		t.Fatalf("got %q", got)
	}
}

func TestStreamRequiresMessages(t *testing.T) {
	t.Parallel()

	p := &Provider{Text: "unused"}
	_, err := p.Stream(context.Background(), core.StreamRequest{})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestStreamErrorEvent(t *testing.T) {
	t.Parallel()

	want := errors.New("boom")
	p := &Provider{Err: want}
	ch, err := p.Stream(context.Background(), core.StreamRequest{
		Messages: []core.Message{{Role: core.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = core.CollectText(ch)
	if !errors.Is(err, want) {
		t.Fatalf("err = %v", err)
	}
}
