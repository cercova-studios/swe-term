package core

import (
	"errors"
	"testing"
)

func TestCollectText(t *testing.T) {
	t.Parallel()

	ch := make(chan StreamEvent, 3)
	ch <- StreamEvent{Kind: EventText, Text: "hello "}
	ch <- StreamEvent{Kind: EventText, Text: "world"}
	close(ch)

	got, err := CollectText(ch)
	if err != nil {
		t.Fatalf("CollectText: %v", err)
	}
	if got != "hello world" {
		t.Fatalf("got %q", got)
	}
}

func TestCollectTextError(t *testing.T) {
	t.Parallel()

	want := errors.New("provider down")
	ch := make(chan StreamEvent, 2)
	ch <- StreamEvent{Kind: EventText, Text: "partial"}
	ch <- StreamEvent{Kind: EventError, Err: want}
	close(ch)

	got, err := CollectText(ch)
	if !errors.Is(err, want) {
		t.Fatalf("err = %v, want %v", err, want)
	}
	if got != "partial" {
		t.Fatalf("got %q", got)
	}
}
