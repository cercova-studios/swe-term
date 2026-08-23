package config

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestReasoningConfigLoadsAndFlagsOverride(t *testing.T) {
	home := t.TempDir()
	cwd := t.TempDir()
	t.Setenv("SWE_TERM_HOME", home)
	t.Chdir(cwd)
	write(t, filepath.Join(home, "config.toml"), "reasoning = \"high\"\n")

	got, err := Load([]string{"--reasoning", "low", "q"})
	if err != nil {
		t.Fatal(err)
	}
	if got.Config.Reasoning != "low" {
		t.Fatalf("reasoning = %q, want low", got.Config.Reasoning)
	}
}

func TestReasoningConfigRejectsUnsupportedValue(t *testing.T) {
	home := t.TempDir()
	cwd := t.TempDir()
	t.Setenv("SWE_TERM_HOME", home)
	t.Chdir(cwd)
	write(t, filepath.Join(home, "config.toml"), "reasoning = \"extreme\"\n")

	_, err := Load(nil)
	if err == nil || !strings.Contains(err.Error(), "reasoning") {
		t.Fatalf("err = %v, want reasoning validation error", err)
	}
}
