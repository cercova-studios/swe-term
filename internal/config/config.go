package config

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/pelletier/go-toml/v2"
)

const (
	DefaultProvider = "openai"
	appName         = "swe-term"
	projectDir      = ".swe-term"
	configFile      = "config.toml"
)

type Config struct {
	Provider  string
	Model     string
	EnvKey    string
	Reasoning string
}

type Result struct {
	Config      Config
	Query       string
	Help        bool
	UserPath    string
	ProjectPath string
}

type layer struct {
	Provider  string
	Model     string
	EnvKey    string
	Reasoning string
}

type builtin struct {
	EnvKey string
	Model  string
}

var builtins = map[string]builtin{
	"openai": {EnvKey: "OPENAI_API_KEY", Model: "gpt-5.6-luna"},
	"mock":   {EnvKey: "", Model: "mock-model"},
}

var (
	userKeys    = map[string]struct{}{"provider": {}, "model": {}, "env_key": {}, "reasoning": {}}
	projectKeys = map[string]struct{}{"provider": {}, "model": {}, "reasoning": {}}
)

const usageText = `usage: go run . [flags] [query]

With a query, runs one shot and exits. With no query, opens a TUI
(/config, /help, /quit).

Flags:
  --provider string    provider id (openai, mock)
  --model string       model id
  --reasoning string   reasoning effort (none, minimal, low, medium, high, xhigh, max)
  --config path        user config file (default: $SWE_TERM_HOME/config.toml)

Config files are TOML. Allowed keys: provider, model, reasoning, env_key (user file only).
Do not put API keys or tokens in config. Project .swe-term/config.toml cannot
set env_key or base_url.

Environment:
  SWE_TERM_HOME       user config directory
                      (default: $XDG_CONFIG_HOME/swe-term, else ~/.config/swe-term)
`

func Usage() string { return usageText }

// Dump is the /config view: on-disk TOML tagged by where it was found.
func Dump(userPath, projectPath string) string {
	var parts []string
	if b, ok := taggedTOML("user", userPath); ok {
		parts = append(parts, b)
	}
	if projectPath != "" {
		if b, ok := taggedTOML("project", projectPath); ok {
			parts = append(parts, b)
		}
	}
	if len(parts) == 0 {
		return "(no config.toml found)"
	}
	return strings.Join(parts, "\n\n")
}

func taggedTOML(tag, path string) (string, bool) {
	if path == "" {
		return "", false
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "", false
		}
		return fmt.Sprintf("[%s]\n<read error: %v>", tag, err), true
	}
	body := strings.TrimRight(string(raw), "\n")
	if body == "" {
		return "[" + tag + "]", true
	}
	return "[" + tag + "]\n" + body, true
}

// Load merges defaults < user config < project config < flags.
// args should be os.Args[1:].
func Load(args []string) (Result, error) {
	home, err := userHome()
	if err != nil {
		return Result{}, err
	}
	cwd, err := os.Getwd()
	if err != nil {
		return Result{}, err
	}
	return load(home, cwd, args)
}

func userHome() (string, error) {
	if v := os.Getenv("SWE_TERM_HOME"); v != "" {
		return v, nil
	}
	if v := os.Getenv("XDG_CONFIG_HOME"); v != "" {
		return filepath.Join(v, appName), nil
	}
	dir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("config: user home: %w", err)
	}
	return filepath.Join(dir, ".config", appName), nil
}

func load(home, cwd string, args []string) (Result, error) {
	flags, query, help, err := parseFlags(args)
	if err != nil {
		return Result{}, err
	}
	if help {
		return Result{Help: true}, nil
	}

	userPath := filepath.Join(home, configFile)
	if flags.configPath != "" {
		userPath = flags.configPath
	}

	var cfg Config
	if err := applyFile(&cfg, userPath, false); err != nil {
		return Result{}, err
	}
	proj := findProjectConfig(cwd)
	if proj != "" {
		if err := applyFile(&cfg, proj, true); err != nil {
			return Result{}, err
		}
	}
	if flags.provider != "" {
		cfg.Provider = flags.provider
	}
	if flags.model != "" {
		cfg.Model = flags.model
	}
	if flags.reasoning != "" {
		cfg.Reasoning = flags.reasoning
	}

	if err := cfg.applyDefaults(); err != nil {
		return Result{}, err
	}
	return Result{
		Config:      cfg,
		Query:       query,
		UserPath:    userPath,
		ProjectPath: proj,
	}, nil
}

type flagValues struct {
	provider   string
	model      string
	reasoning  string
	configPath string
}

func parseFlags(args []string) (flagValues, string, bool, error) {
	var v flagValues
	fs := flag.NewFlagSet(appName, flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	fs.Usage = func() {
		fmt.Fprint(os.Stderr, usageText)
	}
	fs.StringVar(&v.provider, "provider", "", "provider id")
	fs.StringVar(&v.model, "model", "", "model id")
	fs.StringVar(&v.reasoning, "reasoning", "", "reasoning effort")
	fs.StringVar(&v.configPath, "config", "", "user config file")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return v, "", true, nil
		}
		return v, "", false, err
	}
	return v, strings.Join(fs.Args(), " "), false, nil
}

func (c *Config) applyDefaults() error {
	if c.Provider == "" {
		c.Provider = DefaultProvider
	}
	b, ok := builtins[c.Provider]
	if !ok {
		known := make([]string, 0, len(builtins))
		for id := range builtins {
			known = append(known, id)
		}
		sort.Strings(known)
		return fmt.Errorf("config: unknown provider %q (known: %s)", c.Provider, strings.Join(known, ", "))
	}
	if c.Model == "" {
		c.Model = b.Model
	}
	if c.EnvKey == "" {
		c.EnvKey = b.EnvKey
	}
	c.Reasoning = strings.ToLower(strings.TrimSpace(c.Reasoning))
	if c.Reasoning != "" {
		if _, ok := reasoningLevels[c.Reasoning]; !ok {
			return fmt.Errorf("config: unsupported reasoning %q (known: %s)", c.Reasoning, strings.Join(reasoningLevelNames(), ", "))
		}
	}
	return nil
}

func applyFile(cfg *Config, path string, project bool) error {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("config: read %s: %w", path, err)
	}
	layer, err := parseTOML(path, b, project)
	if err != nil {
		return err
	}
	merge(cfg, layer)
	return nil
}

func merge(cfg *Config, l layer) {
	if l.Provider != "" {
		cfg.Provider = l.Provider
	}
	if l.Model != "" {
		cfg.Model = l.Model
	}
	if l.EnvKey != "" {
		cfg.EnvKey = l.EnvKey
	}
	if l.Reasoning != "" {
		cfg.Reasoning = l.Reasoning
	}
}

func parseTOML(path string, data []byte, project bool) (layer, error) {
	var raw map[string]any
	if err := toml.Unmarshal(data, &raw); err != nil {
		return layer{}, fmt.Errorf("config: parse %s: %w", path, err)
	}
	allowed := userKeys
	if project {
		allowed = projectKeys
	}
	for k, v := range raw {
		lk := strings.ToLower(k)
		if _, ok := v.(map[string]any); ok {
			return layer{}, fmt.Errorf("config: %s: nested tables are not supported", path)
		}
		if _, ok := allowed[lk]; ok {
			continue
		}
		if secretKey(lk) {
			return layer{}, fmt.Errorf("config: %s: %q is a credential; set it in the environment, not in config", path, k)
		}
		if project {
			return layer{}, fmt.Errorf("config: %s: project config cannot set %q", path, k)
		}
		return layer{}, fmt.Errorf("config: %s: unknown key %q (allowed: provider, model, reasoning, env_key)", path, k)
	}

	var file struct {
		Provider  string `toml:"provider"`
		Model     string `toml:"model"`
		EnvKey    string `toml:"env_key"`
		Reasoning string `toml:"reasoning"`
	}
	if err := toml.Unmarshal(data, &file); err != nil {
		return layer{}, fmt.Errorf("config: parse %s: %w", path, err)
	}
	return layer{Provider: file.Provider, Model: file.Model, EnvKey: file.EnvKey, Reasoning: file.Reasoning}, nil
}

func secretKey(k string) bool {
	k = strings.ToLower(strings.ReplaceAll(k, "-", "_"))
	if k == "env_key" {
		return false
	}
	return strings.Contains(k, "key") ||
		strings.Contains(k, "token") ||
		strings.Contains(k, "secret") ||
		strings.Contains(k, "password")
}

func findProjectConfig(cwd string) string {
	dir := cwd
	for {
		p := filepath.Join(dir, projectDir, configFile)
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}
