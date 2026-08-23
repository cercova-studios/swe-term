package tui

import (
	"strings"
	"testing"

	"swe-term/internal/core"
)

func TestStreamDoneWithoutCompletionShowsError(t *testing.T) {
	m := newModel(Options{})
	m.busy = true

	updated, _ := m.Update(streamDoneMsg{})
	got := updated.(model)
	if got.busy || len(got.lines) == 0 || !strings.Contains(got.lines[len(got.lines)-1], "without completion") {
		t.Fatalf("busy=%v lines=%q", got.busy, got.lines)
	}
}

func TestDuplicateStreamCompletionDoesNotDoubleCountUsage(t *testing.T) {
	m := newModel(Options{})
	usage := core.Usage{TotalTokens: 10}

	updated, _ := m.Update(streamDeltaMsg{ev: core.StreamEvent{Kind: core.EventComplete, Usage: usage}})
	got := updated.(model)
	updated, _ = got.Update(streamDeltaMsg{ev: core.StreamEvent{Kind: core.EventComplete, Usage: usage}})
	got = updated.(model)

	if got.sessionUsage.TotalTokens != 10 || len(got.lines) == 0 || !strings.Contains(got.lines[len(got.lines)-1], "duplicate completion") {
		t.Fatalf("totalTokens=%d lines=%q", got.sessionUsage.TotalTokens, got.lines)
	}
}

func TestTextAfterStreamCompletionShowsError(t *testing.T) {
	m := newModel(Options{})
	m.streamCompleted = true

	updated, _ := m.Update(streamDeltaMsg{ev: core.StreamEvent{Kind: core.EventText, Text: "late"}})
	got := updated.(model)
	if got.streamRaw != "" || len(got.lines) == 0 || !strings.Contains(got.lines[len(got.lines)-1], "text after completion") {
		t.Fatalf("streamRaw=%q lines=%q", got.streamRaw, got.lines)
	}
}
