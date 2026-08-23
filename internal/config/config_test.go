package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDefaults(t *testing.T) {
	home := t.TempDir()
	cwd := t.TempDir()
	t.Setenv("SWE_TERM_HOME", home)
	t.Chdir(cwd)

	got, err := Load([]string{"hello"})
	if err != nil {
		t.Fatal(err)
	}
	if got.Config.Provider != "openai" || got.Config.Model != "gpt-5.6-luna" || got.Config.EnvKey != "OPENAI_API_KEY" {
		t.Fatalf("defaults: %+v", got.Config)
	}
	if got.Query != "hello" {
		t.Fatalf("query = %q", got.Query)
	}
}

func TestUserThenProjectThenFlags(t *testing.T) {
	home := t.TempDir()
	cwd := t.TempDir()
	t.Setenv("SWE_TERM_HOME", home)
	t.Chdir(cwd)

	write(t, filepath.Join(home, "config.toml"), "provider = \"openai\"\nmodel = \"from-user\"\n")
	write(t, filepath.Join(cwd, ".swe-term", "config.toml"), "model = \"from-project\"\n")

	got, err := Load([]string{"--model", "from-flag", "q"})
	if err != nil {
		t.Fatal(err)
	}
	if got.Config.Model != "from-flag" {
		t.Fatalf("model = %q, want from-flag", got.Config.Model)
	}
	if got.Query != "q" {
		t.Fatalf("query = %q", got.Query)
	}
}

func TestRejectCredentialInUserConfig(t *testing.T) {
	home := t.TempDir()
	cwd := t.TempDir()
	t.Setenv("SWE_TERM_HOME", home)
	t.Chdir(cwd)

	write(t, filepath.Join(home, "config.toml"), "api_key = \"sk-secret\"\n")
	_, err := Load(nil)
	if err == nil || !strings.Contains(err.Error(), "credential") {
		t.Fatalf("err = %v, want credential rejection", err)
	}
}

func TestProjectDenylist(t *testing.T) {
	home := t.TempDir()
	cwd := t.TempDir()
	t.Setenv("SWE_TERM_HOME", home)
	t.Chdir(cwd)

	write(t, filepath.Join(cwd, ".swe-term", "config.toml"), "env_key = \"OPENAI_API_KEY\"\n")
	_, err := Load(nil)
	if err == nil || !strings.Contains(err.Error(), "project config cannot set") {
		t.Fatalf("err = %v, want project denylist", err)
	}
}

func TestUserEnvKeyName(t *testing.T) {
	home := t.TempDir()
	cwd := t.TempDir()
	t.Setenv("SWE_TERM_HOME", home)
	t.Chdir(cwd)

	write(t, filepath.Join(home, "config.toml"), "env_key = \"MY_OPENAI_KEY\"\n")
	got, err := Load([]string{"q"})
	if err != nil {
		t.Fatal(err)
	}
	if got.Config.EnvKey != "MY_OPENAI_KEY" {
		t.Fatalf("EnvKey = %q", got.Config.EnvKey)
	}
}

func TestUnknownProvider(t *testing.T) {
	home := t.TempDir()
	cwd := t.TempDir()
	t.Setenv("SWE_TERM_HOME", home)
	t.Chdir(cwd)

	_, err := Load([]string{"--provider", "acme", "q"})
	if err == nil || !strings.Contains(err.Error(), "unknown provider") {
		t.Fatalf("err = %v", err)
	}
}

func TestUserConfigLocation(t *testing.T) {
	cwd := t.TempDir()
	t.Chdir(cwd)
	t.Setenv("SWE_TERM_HOME", "")

	t.Run("xdg", func(t *testing.T) {
		xdg := t.TempDir()
		t.Setenv("XDG_CONFIG_HOME", xdg)
		write(t, filepath.Join(xdg, "swe-term", "config.toml"), "model = \"from-xdg\"\n")

		got, err := Load(nil)
		if err != nil {
			t.Fatal(err)
		}
		want := filepath.Join(xdg, "swe-term", "config.toml")
		if got.UserPath != want {
			t.Fatalf("UserPath = %q, want %q", got.UserPath, want)
		}
		if got.Config.Model != "from-xdg" {
			t.Fatalf("model = %q", got.Config.Model)
		}
	})

	t.Run("dotconfig", func(t *testing.T) {
		home := t.TempDir()
		t.Setenv("XDG_CONFIG_HOME", "")
		t.Setenv("HOME", home)
		write(t, filepath.Join(home, ".config", "swe-term", "config.toml"), "model = \"from-dotconfig\"\n")

		got, err := Load(nil)
		if err != nil {
			t.Fatal(err)
		}
		want := filepath.Join(home, ".config", "swe-term", "config.toml")
		if got.UserPath != want {
			t.Fatalf("UserPath = %q, want %q", got.UserPath, want)
		}
		if got.Config.Model != "from-dotconfig" {
			t.Fatalf("model = %q", got.Config.Model)
		}
	})
}

func TestExplicitConfigFlag(t *testing.T) {
	home := t.TempDir()
	cwd := t.TempDir()
	t.Setenv("SWE_TERM_HOME", home)
	t.Chdir(cwd)

	alt := filepath.Join(t.TempDir(), "alt.toml")
	write(t, alt, "model = \"from-alt\"\n")

	got, err := Load([]string{"--config", alt, "q"})
	if err != nil {
		t.Fatal(err)
	}
	if got.Config.Model != "from-alt" {
		t.Fatalf("model = %q", got.Config.Model)
	}
}

func write(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
