package core

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"
)

// ControlJournal is the durable, ordered record the control monitor reduces.
// Control events required for replay and safety do not drop (Section 6): the
// journal owns history, ordering, and durability, while the monitor stays a
// pure reducer over the events the journal replays.
type ControlJournal interface {
	// Append stores one event and returns it with the store-assigned
	// Sequence (last + 1). A caller-supplied non-zero Sequence that does not
	// equal last + 1 is rejected without writing.
	Append(event ControlEvent) (ControlEvent, error)
	// Replay visits events in sequence order from line 1. A corrupt,
	// unterminated, or out-of-sequence line is a loud error naming the line,
	// never a silently skipped record.
	Replay(visit func(ControlEvent) error) error
}

// FileControlJournal is the JSONL ControlJournal: one compact JSON object per
// line, appended under a mutex (single writer), flushed and synced before
// Append returns.
type FileControlJournal struct {
	mu   sync.Mutex
	path string
	file *os.File
	last uint64
}

func NewFileControlJournal(path string) (*FileControlJournal, error) {
	last, err := scanControlJournal(path)
	if err != nil {
		return nil, err
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("journal: open %s: %w", path, err)
	}
	return &FileControlJournal{path: path, file: file, last: last}, nil
}

func (journal *FileControlJournal) Append(event ControlEvent) (ControlEvent, error) {
	journal.mu.Lock()
	defer journal.mu.Unlock()

	want := journal.last + 1
	if event.Sequence != 0 && event.Sequence != want {
		return ControlEvent{}, fmt.Errorf("journal: sequence %d does not match next sequence %d; nothing written", event.Sequence, want)
	}
	event.Sequence = want

	line, err := json.Marshal(event)
	if err != nil {
		return ControlEvent{}, fmt.Errorf("journal: encode event %d: %w", want, err)
	}
	if _, err := journal.file.Write(append(line, '\n')); err != nil {
		return ControlEvent{}, fmt.Errorf("journal: append event %d: %w", want, err)
	}
	if err := journal.file.Sync(); err != nil {
		return ControlEvent{}, fmt.Errorf("journal: sync event %d: %w", want, err)
	}
	journal.last = want
	return event, nil
}

func (journal *FileControlJournal) Replay(visit func(ControlEvent) error) error {
	journal.mu.Lock()
	defer journal.mu.Unlock()
	return replayControlJournal(journal.path, visit)
}

func (journal *FileControlJournal) Close() error {
	journal.mu.Lock()
	defer journal.mu.Unlock()
	return journal.file.Close()
}

// ReplayControlMonitor folds a journal from genesis through the control
// monitor. The replayed state must equal the live state built by applying the
// same events as they were appended.
func ReplayControlMonitor(journal ControlJournal) (ControlMonitorState, error) {
	state := ControlMonitorState{}
	err := journal.Replay(func(event ControlEvent) error {
		var decision ControlDecision
		state, decision = ApplyControlEvent(state, event)
		if !decision.Accepted {
			return fmt.Errorf("journal: event %d (%s) rejected by monitor: %s", event.Sequence, event.Kind, decision.RuleID)
		}
		return nil
	})
	if err != nil {
		return ControlMonitorState{}, err
	}
	return state, nil
}

// scanControlJournal validates an existing journal file and returns its last
// sequence (0 when the file is absent or empty).
func scanControlJournal(path string) (uint64, error) {
	var last uint64
	err := replayControlJournal(path, func(event ControlEvent) error {
		last = event.Sequence
		return nil
	})
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, err
	}
	return last, nil
}

func replayControlJournal(path string, visit func(ControlEvent) error) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	var lineNumber uint64
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			lineNumber++
			if line[len(line)-1] != '\n' {
				return fmt.Errorf("journal: line %d: unterminated record", lineNumber)
			}
			var event ControlEvent
			if decodeErr := json.Unmarshal(bytes.TrimSuffix(line, []byte("\n")), &event); decodeErr != nil {
				return fmt.Errorf("journal: line %d: corrupt record: %w", lineNumber, decodeErr)
			}
			if event.Sequence != lineNumber {
				return fmt.Errorf("journal: line %d: sequence %d breaks the ordered record", lineNumber, event.Sequence)
			}
			if visitErr := visit(event); visitErr != nil {
				return visitErr
			}
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("journal: read: %w", err)
		}
	}
}
