package core

import "testing"

func TestEffectDeclarationValidateFailsClosed(t *testing.T) {
	tests := []struct {
		name        string
		declaration EffectDeclaration
		wantValid   bool
	}{
		{"read observed", EffectDeclaration{Reads: []PathEffect{{Path: "docs/a.md"}}, Observation: ObservationObserved}, true},
		{"read unobservable allowed", EffectDeclaration{Reads: []PathEffect{{Path: "docs/a.md"}}, Network: []string{"example.com"}, Observation: ObservationUnobservable}, true},
		{"secret name", EffectDeclaration{Secrets: []string{"OPENAI_API_KEY"}, Observation: ObservationAttested}, true},
		{"empty declaration", EffectDeclaration{Observation: ObservationObserved}, false},
		{"invalid observation", EffectDeclaration{Reads: []PathEffect{{Path: "docs/a.md"}}, Observation: ObservationLevel("guessed")}, false},
		{"missing observation", EffectDeclaration{Reads: []PathEffect{{Path: "docs/a.md"}}}, false},
		{"duplicate paths", EffectDeclaration{Reads: []PathEffect{{Path: "docs/a.md"}, {Path: "docs/a.md"}}, Observation: ObservationObserved}, false},
		{"duplicate hosts", EffectDeclaration{Network: []string{"example.com", "example.com"}, Observation: ObservationObserved}, false},
		{"empty path", EffectDeclaration{Reads: []PathEffect{{Path: ""}}, Observation: ObservationObserved}, false},
		{"empty process", EffectDeclaration{Processes: []string{""}, Observation: ObservationObserved}, false},
		{"traversal", EffectDeclaration{Writes: []PathEffect{{Path: "docs/../secret"}}, Observation: ObservationObserved}, false},
		{"unobservable write", EffectDeclaration{Writes: []PathEffect{{Path: "docs/a.md"}}, Observation: ObservationUnobservable}, false},
		{"unobservable process", EffectDeclaration{Processes: []string{"make"}, Observation: ObservationUnobservable}, false},
		{"secret assignment value", EffectDeclaration{Secrets: []string{"OPENAI_API_KEY=sk-live"}, Observation: ObservationObserved}, false},
		{"secret bearer value", EffectDeclaration{Secrets: []string{"Bearer abc123"}, Observation: ObservationObserved}, false},
		{"secret token prefix", EffectDeclaration{Secrets: []string{"sk-livevalue"}, Observation: ObservationObserved}, false},
		{"secret github prefix", EffectDeclaration{Secrets: []string{"ghp_value"}, Observation: ObservationObserved}, false},
		{"secret with space", EffectDeclaration{Secrets: []string{"MY SECRET"}, Observation: ObservationObserved}, false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.declaration.Validate(); got != test.wantValid {
				t.Fatalf("Validate() = %v, want %v for %#v", got, test.wantValid, test.declaration)
			}
		})
	}
}

func TestEffectDeclarationCovers(t *testing.T) {
	declaration := EffectDeclaration{
		Reads:       []PathEffect{{Path: "docs", Glob: true}, {Path: "README.md"}},
		Writes:      []PathEffect{{Path: "docs/out", Glob: true}},
		Processes:   []string{"make test"},
		Network:     []string{"example.com"},
		Secrets:     []string{"OPENAI_API_KEY"},
		Observation: ObservationAttested,
	}
	tests := []struct {
		name     string
		observed ObservedEffect
		want     bool
	}{
		{"glob read inside", ObservedEffect{Class: EffectClassRead, Target: "docs/a/b.md", Observation: ObservationAttested}, true},
		{"glob boundary rejects sibling", ObservedEffect{Class: EffectClassRead, Target: "docs-other/a.md", Observation: ObservationAttested}, false},
		{"exact read", ObservedEffect{Class: EffectClassRead, Target: "README.md", Observation: ObservationObserved}, true},
		{"non-glob does not cover child", ObservedEffect{Class: EffectClassRead, Target: "README.md/x", Observation: ObservationObserved}, false},
		{"write under glob", ObservedEffect{Class: EffectClassWrite, Target: "docs/out/a.md", Observation: ObservationAttested}, true},
		{"read does not cover write", ObservedEffect{Class: EffectClassWrite, Target: "docs/a.md", Observation: ObservationAttested}, false},
		{"process exact", ObservedEffect{Class: EffectClassProcess, Target: "make test", Observation: ObservationAttested}, true},
		{"process mismatch", ObservedEffect{Class: EffectClassProcess, Target: "make build", Observation: ObservationAttested}, false},
		{"network exact", ObservedEffect{Class: EffectClassNetwork, Target: "example.com", Observation: ObservationAttested}, true},
		{"secret exact", ObservedEffect{Class: EffectClassSecret, Target: "OPENAI_API_KEY", Observation: ObservationAttested}, true},
		{"observed covered by attested declaration", ObservedEffect{Class: EffectClassRead, Target: "README.md", Observation: ObservationObserved}, true},
		{"unobservable not covered by attested declaration", ObservedEffect{Class: EffectClassRead, Target: "README.md", Observation: ObservationUnobservable}, false},
		{"invalid class", ObservedEffect{Class: EffectClass("teleport"), Target: "README.md", Observation: ObservationObserved}, false},
		{"empty target", ObservedEffect{Class: EffectClassRead, Target: "", Observation: ObservationObserved}, false},
		{"invalid observed level", ObservedEffect{Class: EffectClassRead, Target: "README.md", Observation: ObservationLevel("guessed")}, false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := declaration.Covers(test.observed); got != test.want {
				t.Fatalf("Covers(%#v) = %v, want %v", test.observed, got, test.want)
			}
		})
	}
}

func TestEffectDeclarationCoversObservationMatrix(t *testing.T) {
	levels := []ObservationLevel{ObservationObserved, ObservationAttested, ObservationUnobservable}
	// want[declared][observed]: an event is covered only when the declaration
	// promised at least the uncertainty the event carries.
	want := map[ObservationLevel]map[ObservationLevel]bool{
		ObservationObserved:     {ObservationObserved: true, ObservationAttested: false, ObservationUnobservable: false},
		ObservationAttested:     {ObservationObserved: true, ObservationAttested: true, ObservationUnobservable: false},
		ObservationUnobservable: {ObservationObserved: true, ObservationAttested: true, ObservationUnobservable: true},
	}
	for _, declared := range levels {
		for _, observed := range levels {
			declaration := EffectDeclaration{Reads: []PathEffect{{Path: "a.md"}}, Observation: declared}
			event := ObservedEffect{Class: EffectClassRead, Target: "a.md", Observation: observed}
			if got := declaration.Covers(event); got != want[declared][observed] {
				t.Fatalf("declared=%s observed=%s: Covers = %v, want %v", declared, observed, got, want[declared][observed])
			}
		}
	}
}
