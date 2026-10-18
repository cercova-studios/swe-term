package core

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestFileControlJournalReplayMatchesLiveState(t *testing.T) {
	path := filepath.Join(t.TempDir(), "journal.jsonl")
	journal, err := NewFileControlJournal(path)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	defer journal.Close()

	identity := testControlReceiptIdentity("a")
	receipt := testPassingReceipt(identity)
	events := []ControlEvent{
		testControlEvent(0, ControlSetReceiptTarget, func(event *ControlEvent) { event.Obligation, event.Identity = "tests", identity }),
		testControlEvent(0, ControlApprovalGranted, func(event *ControlEvent) { event.Action = "edit" }),
		testControlEvent(0, ControlLeaseAcquired, func(event *ControlEvent) { event.Action, event.Lease = "edit", "lease-1" }),
		testControlEvent(0, ControlEffectsDeclared, func(event *ControlEvent) {
			event.Lease, event.Declared = "lease-1", testWriteDeclaration("workspace/main.go")
		}),
		testControlEvent(0, ControlEffectObserved, func(event *ControlEvent) {
			event.Lease, event.Observed = "lease-1", testObservedWrite("workspace/main.go")
		}),
		testControlEvent(0, ControlLeaseReleased, func(event *ControlEvent) { event.Lease = "lease-1" }),
		testControlEvent(0, ControlReceiptRecorded, func(event *ControlEvent) { event.Receipt = receipt }),
		testControlEvent(0, ControlLifecycleClaimed, func(event *ControlEvent) { event.Claim = ClaimVerified }),
	}

	live := ControlMonitorState{}
	for index, event := range events {
		stored, err := journal.Append(event)
		if err != nil {
			t.Fatalf("append event %d: %v", index, err)
		}
		if stored.Sequence != uint64(index+1) {
			t.Fatalf("stored sequence = %d, want %d", stored.Sequence, index+1)
		}
		var decision ControlDecision
		live, decision = ApplyControlEvent(live, stored)
		if !decision.Accepted {
			t.Fatalf("live event %d rejected: %#v", stored.Sequence, decision)
		}
	}

	replayed, err := ReplayControlMonitor(journal)
	if err != nil {
		t.Fatalf("replay journal: %v", err)
	}
	if !reflect.DeepEqual(replayed, live) {
		t.Fatalf("replayed state differs from live:\n got: %#v\nwant: %#v", replayed, live)
	}

	// Reopening an existing journal resumes its sequence instead of
	// restarting or overwriting the ordered record.
	reopened, err := NewFileControlJournal(path)
	if err != nil {
		t.Fatalf("reopen journal: %v", err)
	}
	defer reopened.Close()
	stored, err := reopened.Append(testControlEvent(0, ControlCancelled, nil))
	if err != nil {
		t.Fatalf("append after reopen: %v", err)
	}
	if stored.Sequence != uint64(len(events)+1) {
		t.Fatalf("sequence after reopen = %d, want %d", stored.Sequence, len(events)+1)
	}
}

func TestFileControlJournalRejectsSequenceMismatchWithoutWriting(t *testing.T) {
	path := filepath.Join(t.TempDir(), "journal.jsonl")
	journal, err := NewFileControlJournal(path)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	defer journal.Close()

	if _, err := journal.Append(testControlEvent(0, ControlApprovalGranted, func(event *ControlEvent) { event.Action = "edit" })); err != nil {
		t.Fatalf("append first event: %v", err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read journal: %v", err)
	}
	if _, err := journal.Append(testControlEvent(7, ControlCancelled, nil)); err == nil {
		t.Fatal("append with mismatched sequence succeeded; want rejection")
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("re-read journal: %v", err)
	}
	if string(after) != string(before) {
		t.Fatalf("rejected append wrote to the journal:\nbefore: %q\nafter:  %q", before, after)
	}
}

func TestFileControlJournalRejectsCorruptTail(t *testing.T) {
	tests := []struct {
		name    string
		corrupt func(t *testing.T, path string)
		want    string
	}{
		{
			name: "unterminated last line",
			corrupt: func(t *testing.T, path string) {
				t.Helper()
				file, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
				if err != nil {
					t.Fatalf("open for corruption: %v", err)
				}
				defer file.Close()
				if _, err := file.WriteString(`{"SchemaVersion":1,"Sequence":2`); err != nil {
					t.Fatalf("write corruption: %v", err)
				}
			},
			want: "line 2",
		},
		{
			name: "corrupt JSON line",
			corrupt: func(t *testing.T, path string) {
				t.Helper()
				if err := os.WriteFile(path, []byte("{\"SchemaVersion\":1,\"Sequence\":1,\"Kind\":\"approval_granted\",\"Action\":\"edit\"}\nnot-json\n"), 0o644); err != nil {
					t.Fatalf("write corruption: %v", err)
				}
			},
			want: "line 2",
		},
		{
			name: "sequence gap",
			corrupt: func(t *testing.T, path string) {
				t.Helper()
				if err := os.WriteFile(path, []byte("{\"SchemaVersion\":1,\"Sequence\":2,\"Kind\":\"cancelled\"}\n"), 0o644); err != nil {
					t.Fatalf("write corruption: %v", err)
				}
			},
			want: "line 1",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "journal.jsonl")
			journal, err := NewFileControlJournal(path)
			if err != nil {
				t.Fatalf("open journal: %v", err)
			}
			if _, err := journal.Append(testControlEvent(0, ControlApprovalGranted, func(event *ControlEvent) { event.Action = "edit" })); err != nil {
				t.Fatalf("append first event: %v", err)
			}
			if err := journal.Close(); err != nil {
				t.Fatalf("close journal: %v", err)
			}

			test.corrupt(t, path)

			if _, err := NewFileControlJournal(path); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("reopen corrupt journal error = %v, want mention of %q", err, test.want)
			}
		})
	}
}
