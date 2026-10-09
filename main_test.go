package main

import (
	"bytes"
	"errors"
	"regexp"
	"strings"
	"testing"

	"swe-term/internal/config"
	"swe-term/internal/core"
	"swe-term/internal/provider/mock"
)

func mockResult() config.Result {
	return config.Result{
		Query:  "hello",
		Config: config.Config{Provider: mock.Name, Model: "mock-model", Reasoning: "high"},
	}
}

// End to end through the one-shot path, with real wiring at every step:
// provider construction, streaming, chunk reassembly, markdown rendering, and
// the bytes a user actually sees. The mock emits its text in 4-rune chunks, so
// a reassembly regression shows up here as mangled output rather than as a
// passing unit test over a helper.
func TestRunOnceRendersStreamedResponse(t *testing.T) {
	res := mockResult()
	provider, err := newProvider(res.Config)
	if err != nil {
		t.Fatal(err)
	}

	var out bytes.Buffer
	if err := runOnce(&out, provider, res); err != nil {
		t.Fatal(err)
	}

	rendered := out.String()
	// Assert on what a user actually sees. glamour styles each span
	// separately, so the raw bytes interleave ANSI escapes between words
	// ("mock" ESC " response" ESC …) and the literal substring is absent even
	// though the text rendered correctly. Collapsing whitespace too, because
	// the renderer pads to terminal width.
	visible := strings.Join(strings.Fields(stripANSI(rendered)), " ")
	if !strings.Contains(visible, "mock response") {
		t.Fatalf("rendered output lost the response text.\n visible: %q\n raw: %q", visible, rendered)
	}
	if !strings.HasSuffix(rendered, "\n") {
		t.Fatalf("rendered output must end in a newline, got %q", rendered)
	}
}

// ARCHITECTURE.md §10 invariant 9: failure is never flattened into successful
// output. A provider error must surface as an error and leave nothing rendered.
func TestRunOnceRendersNothingOnProviderError(t *testing.T) {
	provider := &mock.Provider{Err: errors.New("upstream failed")}

	var out bytes.Buffer
	err := runOnce(&out, provider, mockResult())
	if err == nil {
		t.Fatal("provider error did not surface")
	}
	if out.Len() != 0 {
		t.Fatalf("rendered %q despite a provider error", out.String())
	}
}

// Usage never reaches runOnce's output, so the telemetry the cost path depends
// on is checked where it is actually produced: the real provider built by the
// real constructor, drained through the real collector.
func TestMockProviderEmitsDeterministicTelemetry(t *testing.T) {
	res := mockResult()
	provider, err := newProvider(res.Config)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := provider.Stream(t.Context(), core.StreamRequest{
		Messages:  []core.Message{{Role: core.RoleUser, Content: res.Query}},
		Model:     res.Config.Model,
		Reasoning: res.Config.Reasoning,
	})
	if err != nil {
		t.Fatal(err)
	}
	response, err := core.CollectResponse(stream)
	if err != nil {
		t.Fatal(err)
	}
	if response.Usage.Model != "mock-model" || response.Usage.Reasoning != "high" {
		t.Fatalf("usage identity = %+v", response.Usage)
	}
	if response.Usage.TotalTokens == 0 {
		t.Fatalf("usage reported no tokens: %+v", response.Usage)
	}
}

// Only CSI sequences; glamour emits no hyperlinks for this input. Hand-rolled
// rather than pulling charmbracelet/x/ansi up from indirect to a direct
// dependency for two lines in one test.
var ansiPattern = regexp.MustCompile(`\x1b\[[0-9;]*[a-zA-Z]`)

func stripANSI(s string) string { return ansiPattern.ReplaceAllString(s, "") }
