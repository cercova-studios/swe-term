package main

import (
	"context"
	"fmt"
	"os"

	"swe-term/internal/config"
	"swe-term/internal/core"
	"swe-term/internal/provider/mock"
	openaiprovider "swe-term/internal/provider/openai"
	"swe-term/internal/tui"
)

func main() {
	res, err := config.Load(os.Args[1:])
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(2)
	}
	if res.Help {
		fmt.Fprint(os.Stderr, config.Usage())
		return
	}

	provider, perr := newProvider(res.Config)

	if res.Query == "" {
		if err := tui.Run(tui.Options{
			Result:      res,
			Provider:    provider,
			ProviderErr: perr,
			LoadArgs:    os.Args[1:],
		}); err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
		return
	}
	if perr != nil {
		fmt.Fprintf(os.Stderr, "%v\n", perr)
		os.Exit(2)
	}
	if err := runOnce(provider, res); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func runOnce(provider core.Provider, res config.Result) error {
	ch, err := provider.Stream(context.Background(), core.StreamRequest{
		Messages:  []core.Message{{Role: core.RoleUser, Content: res.Query}},
		Model:     res.Config.Model,
		Reasoning: res.Config.Reasoning,
	})
	if err != nil {
		return err
	}
	for ev := range ch {
		switch ev.Kind {
		case core.EventText:
			fmt.Print(ev.Text)
		case core.EventError:
			if ev.Err != nil {
				return ev.Err
			}
			return fmt.Errorf("stream error")
		}
	}
	fmt.Print("\n")
	return nil
}

func newProvider(cfg config.Config) (core.Provider, error) {
	switch cfg.Provider {
	case openaiprovider.Name:
		if cfg.EnvKey == "" {
			return nil, fmt.Errorf("config: provider %s is missing env_key", cfg.Provider)
		}
		key := os.Getenv(cfg.EnvKey)
		if key == "" {
			return nil, fmt.Errorf("%s is not set", cfg.EnvKey)
		}
		return openaiprovider.New(key, cfg.Model), nil
	case mock.Name:
		return &mock.Provider{
			Text:  "mock response",
			Model: cfg.Model,
			Usage: core.Usage{
				Model:        cfg.Model,
				Reasoning:    cfg.Reasoning,
				InputTokens:  8,
				OutputTokens: 2,
				TotalTokens:  10,
			},
		}, nil
	default:
		return nil, fmt.Errorf("config: unknown provider %q", cfg.Provider)
	}
}
