package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPreregistrationGateFailsClosed(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*manifest)
		want   string
	}{
		{
			name: "draft status",
			mutate: func(m *manifest) {
				m.Status = "draft"
			},
			want: "status must be preregistered",
		},
		{
			name: "duplicate variant",
			mutate: func(m *manifest) {
				m.Variants[1].Name = m.Variants[0].Name
			},
			want: "variant name \"control\" is duplicated",
		},
		{
			name: "network without justification",
			mutate: func(m *manifest) {
				m.Environment.Network = "inherit"
			},
			want: "network_justification is required",
		},
		{
			name: "jj source without revision",
			mutate: func(m *manifest) {
				m.Source.Kind = "jj"
				m.Source.Revision = ""
			},
			want: "source.revision is required",
		},
		{
			name: "model experiment under repeated",
			mutate: func(m *manifest) {
				m.Model.Required = true
				m.Model.Provider = "test"
				m.Model.ID = "model"
				m.Model.PromptDigest = testDigest("b")
				m.Model.ToolSchemaDigest = testDigest("c")
				m.Repetitions = 2
			},
			want: "model-dependent experiments require at least three repetitions",
		},
		{
			name: "seed schedule does not cover repetitions",
			mutate: func(m *manifest) {
				m.Randomness.Mode = "seeded"
				m.Randomness.Seeds = []int{1}
				m.Repetitions = 2
			},
			want: "seeded randomness mode requires one unique seed per repetition",
		},
		{
			name: "evidence escapes repository",
			mutate: func(m *manifest) {
				m.Evidence.RawRuns = "../runs"
			},
			want: "evidence.raw_runs must be a safe repository-relative path",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			m := validReadyManifest("test-experiment")
			test.mutate(&m)
			errs := validateManifest(m, "test-experiment", true)
			if !hasErrorContaining(errs, test.want) {
				t.Fatalf("expected error containing %q, got %v", test.want, errs)
			}
		})
	}
}

func TestNewExperimentProducesValidDraftAndBlocksPrematureRun(t *testing.T) {
	root := t.TempDir()
	projectRoot, err := findRepositoryRoot()
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"manifest.json", "README.md", "FIXTURES.md"} {
		source := filepath.Join(projectRoot, "experiments", "templates", name)
		target := filepath.Join(root, "experiments", "templates", name)
		copyTestFile(t, source, target)
	}

	created, err := newExperiment(root, "test-experiment")
	if err != nil {
		t.Fatal(err)
	}
	if !created {
		t.Fatal("expected a new experiment directory")
	}
	created, err = newExperiment(root, "test-experiment")
	if err != nil {
		t.Fatal(err)
	}
	if created {
		t.Fatal("expected repeated creation to be an idempotent no-op")
	}

	draftErrors := validateExperiment(root, "test-experiment", false)
	if len(draftErrors) != 0 {
		t.Fatalf("new draft should pass structural validation, got %v", draftErrors)
	}
	readyErrors := validateExperiment(root, "test-experiment", true)
	if !hasErrorContaining(readyErrors, "status must be preregistered") {
		t.Fatalf("new draft should fail the preregistration gate, got %v", readyErrors)
	}

	fixtures := filepath.Join(root, "experiments", "specs", "test-experiment", "fixtures")
	digest, err := digestDirectory(fixtures)
	if err != nil {
		t.Fatal(err)
	}
	ready := validReadyManifest("test-experiment")
	ready.Source.ContentDigest = digest
	content, err := json.MarshalIndent(ready, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(root, "experiments", "specs", "test-experiment", "manifest.json")
	if err := os.WriteFile(manifestPath, append(content, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	readyErrors = validateExperiment(root, "test-experiment", true)
	if len(readyErrors) != 0 {
		t.Fatalf("completed preregistration should pass, got %v", readyErrors)
	}
}

func TestDirectoryDigestBindsRelativePaths(t *testing.T) {
	root := t.TempDir()
	first := filepath.Join(root, "first.txt")
	if err := os.WriteFile(first, []byte("same content"), 0o644); err != nil {
		t.Fatal(err)
	}
	before, err := digestDirectory(root)
	if err != nil {
		t.Fatal(err)
	}
	second := filepath.Join(root, "second.txt")
	if err := os.Rename(first, second); err != nil {
		t.Fatal(err)
	}
	after, err := digestDirectory(root)
	if err != nil {
		t.Fatal(err)
	}
	if before == after {
		t.Fatal("directory digest must change when a fixture path changes")
	}
}

func TestDecodeManifestRejectsUnknownFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manifest.json")
	content := `{"schema_version":1,"unexpected":true}`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := decodeManifest(path)
	if err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("expected unknown field error, got %v", err)
	}
}

func validReadyManifest(id string) manifest {
	return manifest{
		SchemaVersion:       1,
		ExperimentID:        id,
		Status:              "preregistered",
		Papers:              []paper{{Title: "Paper", URL: "https://example.com/paper", Claim: "Mechanism claim"}},
		Hypothesis:          "The treatment changes the primary metric.",
		NullHypothesis:      "The treatment does not change the primary metric.",
		IndependentVariable: "Structured feedback is enabled only in the treatment.",
		Source: source{
			Kind:          "directory",
			Locator:       "fixtures",
			ContentDigest: testDigest("a"),
		},
		Environment: environment{
			Adapter:    "process",
			Network:    "disabled",
			AllowedEnv: []string{"PATH"},
			Resources: resources{
				TimeoutSeconds: 300,
				MaxParallelism: 1,
				MaxOutputBytes: 1 << 20,
			},
		},
		Model: model{Required: false, Parameters: map[string]any{}},
		Variants: []variant{
			{Name: "control", Description: "Baseline", Command: []string{"true"}, Config: map[string]any{}},
			{Name: "treatment", Description: "Mechanism enabled", Command: []string{"true"}, Config: map[string]any{}},
		},
		Repetitions: 1,
		Randomness:  randomness{Mode: "deterministic", Seeds: []int{}},
		Metrics: metrics{
			Primary:        "completion",
			Secondary:      []string{"elapsed"},
			AcceptanceRule: "Treatment completes every frozen fixture.",
		},
		Evaluator: evaluator{
			Kind:         "manual",
			Rubric:       "Score completion from the frozen expected result.",
			RubricDigest: testDigest("d"),
		},
		Evidence: evidence{
			RawRuns: "experiments/runs/" + id,
			Curated: "experiments/evidence/" + id,
		},
	}
}

func testDigest(character string) string {
	return "sha256:" + strings.Repeat(character, 64)
}

func hasErrorContaining(errs []string, expected string) bool {
	for _, err := range errs {
		if strings.Contains(err, expected) {
			return true
		}
	}
	return false
}

func copyTestFile(t *testing.T, source, target string) {
	t.Helper()
	content, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, content, 0o644); err != nil {
		t.Fatal(err)
	}
}
