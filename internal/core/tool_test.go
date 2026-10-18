package core

import "testing"

func TestToolResultValidFailsClosed(t *testing.T) {
	artifact := &ArtifactHandle{ID: "art-1", URI: "file:///tmp/art-1", SHA256: "abc123", Bytes: 42}
	tests := []struct {
		name   string
		result ToolResult
		want   bool
	}{
		{"ok", ToolResult{Outcome: ToolOutcomeOK, Output: "done"}, true},
		{"failed is a valid distinct outcome", ToolResult{Outcome: ToolOutcomeFailed}, true},
		{"denied is a valid distinct outcome", ToolResult{Outcome: ToolOutcomeDenied}, true},
		{"unknown outcome", ToolResult{Outcome: ToolOutcome("vanished")}, false},
		{"empty outcome", ToolResult{}, false},
		{"truncated with artifact and fields", ToolResult{Outcome: ToolOutcomeTruncated, Output: "partial", TruncatedFields: []string{"output"}, Artifact: artifact}, true},
		{"truncated without artifact", ToolResult{Outcome: ToolOutcomeTruncated, Output: "partial", TruncatedFields: []string{"output"}}, false},
		{"truncated without fields", ToolResult{Outcome: ToolOutcomeTruncated, Output: "partial", Artifact: artifact}, false},
		{"truncated with empty artifact", ToolResult{Outcome: ToolOutcomeTruncated, TruncatedFields: []string{"output"}, Artifact: &ArtifactHandle{}}, false},
		{"truncated with zero-byte artifact", ToolResult{Outcome: ToolOutcomeTruncated, TruncatedFields: []string{"output"}, Artifact: &ArtifactHandle{ID: "a", URI: "file:///tmp/a", SHA256: "abc"}}, false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.result.Valid(); got != test.want {
				t.Fatalf("Valid() = %v, want %v for %#v", got, test.want, test.result)
			}
		})
	}
}
