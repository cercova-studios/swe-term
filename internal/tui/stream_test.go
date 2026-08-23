package tui

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"

	"swe-term/internal/core"
	"swe-term/internal/provider/mock"
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

func TestStaleStreamDoneDoesNotAbortNewTurn(t *testing.T) {
	m := newModel(Options{})
	m.busy = true
	m.streamID = 2
	m.streamRaw = "new"

	updated, _ := m.Update(streamDoneMsg{id: 1})
	got := updated.(model)
	if !got.busy || got.streamRaw != "new" {
		t.Fatalf("stale done mutated busy=%v streamRaw=%q", got.busy, got.streamRaw)
	}
}

func TestConversationKeepsPriorTurns(t *testing.T) {
	m := newModel(Options{Provider: &mock.Provider{Text: "ok"}})
	updated, _ := m.handleLine("one")
	got := updated.(model)
	if len(got.history) != 1 || got.history[0].Content != "one" {
		t.Fatalf("history after user = %+v", got.history)
	}

	got.busy = true
	got.streamRaw = "reply"
	got.streamCompleted = true
	updated, _ = got.Update(streamDoneMsg{id: got.streamID})
	got = updated.(model)
	if len(got.history) != 2 || got.history[1].Role != core.RoleAssistant || got.history[1].Content != "reply" {
		t.Fatalf("history after complete = %+v", got.history)
	}

	updated, _ = got.handleLine("two")
	got = updated.(model)
	if len(got.history) != 3 || got.history[2].Content != "two" {
		t.Fatalf("history after follow-up = %+v", got.history)
	}
}

func TestEscMarksPartialInterrupted(t *testing.T) {
	m := newModel(Options{})
	m.busy = true
	m.streamID = 1
	m.streamRaw = "partial"

	updated, _ := m.Update(tea.KeyPressMsg{Code: tea.KeyEsc})
	got := updated.(model)
	if got.busy {
		t.Fatal("esc left busy set")
	}
	found := false
	for _, line := range got.lines {
		if strings.Contains(line, "interrupted") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("partial interrupt missing marker: %q", got.lines)
	}
}
