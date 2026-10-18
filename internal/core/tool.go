package core

import (
	"context"
	"encoding/json"
)

// Tool is one bounded action the loop may invoke. Scheduling and policy read
// Invocation.Declared, never the tool's name: there is deliberately no
// ReadOnly() bool, because a single bool conflates approval risk with
// scheduling concurrency, and name-based policy can be bypassed by renaming.
type Tool interface {
	Name() string
	Version() string
	// Schema is the JSON Schema for Invocation.Args.
	Schema() json.RawMessage
	Execute(ctx context.Context, inv Invocation) (ToolResult, error)
}

// Invocation is one tool call as executed. Declared is computed from Args
// after any pre-tool hook rewrite; RewrittenFrom preserves the original
// arguments whenever a hook or shim rewrote them, so the rewrite is auditable
// and the effective arguments can be disclosed back to the model.
type Invocation struct {
	ID            string
	Args          json.RawMessage
	RewrittenFrom json.RawMessage
	Declared      EffectDeclaration
}

// ToolOutcome is the closed result vocabulary. Truncation, timeout,
// cancellation, denial, and failure are distinct states and are never
// flattened into successful text output (invariant 9).
type ToolOutcome string

const (
	ToolOutcomeOK        ToolOutcome = "ok"
	ToolOutcomeTruncated ToolOutcome = "truncated"
	ToolOutcomeTimeout   ToolOutcome = "timeout"
	ToolOutcomeCancelled ToolOutcome = "cancelled"
	ToolOutcomeDenied    ToolOutcome = "denied"
	ToolOutcomeFailed    ToolOutcome = "failed"
)

// ArtifactHandle points at the full, un-truncated payload behind a truncated
// result, so omitted output stays recoverable.
type ArtifactHandle struct {
	ID     string
	URI    string
	SHA256 string
	Bytes  int64
}

func (handle ArtifactHandle) Valid() bool {
	return handle.ID != "" && handle.URI != "" && handle.SHA256 != "" && handle.Bytes > 0
}

// ToolResult is the outcome of one Execute call. EffectiveArgs echoes the
// post-rewrite arguments whenever the invocation was rewritten, so the model
// never reasons from stale pre-hook beliefs.
type ToolResult struct {
	Outcome         ToolOutcome
	Output          string
	TruncatedFields []string
	Artifact        *ArtifactHandle
	EffectiveArgs   json.RawMessage
}

// Valid enforces the closed outcome vocabulary and the truncation contract:
// a truncated result must name the truncated fields and carry a valid handle
// to the full payload.
func (result ToolResult) Valid() bool {
	switch result.Outcome {
	case ToolOutcomeOK, ToolOutcomeTimeout, ToolOutcomeCancelled, ToolOutcomeDenied, ToolOutcomeFailed:
		return true
	case ToolOutcomeTruncated:
		return len(result.TruncatedFields) > 0 && result.Artifact != nil && result.Artifact.Valid()
	}
	return false
}
