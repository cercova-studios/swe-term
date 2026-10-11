package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"charm.land/glamour/v2"

	"swe-term/internal/config"
	"swe-term/internal/core"
	"swe-term/internal/provider/mock"
	openaiprovider "swe-term/internal/provider/openai"
	"swe-term/internal/telemetry"
	"swe-term/internal/tui"
)

// version labels telemetry resources. It is not read from build info because
// nothing else in the repository reports a version yet; when a release
// process exists this should come from it.
const version = "0.0.0-dev"

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

	// Tracing is off unless an OTLP endpoint is configured, so this is a no-op
	// for an ordinary local run. A misconfigured endpoint is reported and
	// fatal rather than ignored: a run that believes it is traced and is not
	// is the reassuring falsehood ARCHITECTURE.md §10 invariant 12 forbids.
	shutdownTracing, err := telemetry.Setup(context.Background(), version)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(2)
	}
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := shutdownTracing(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "telemetry: flush failed: %v\n", err)
		}
	}()

	provider, perr := newProvider(res.Config)
	if perr == nil {
		// Returns the provider unchanged when tracing is off.
		provider = telemetry.WrapProvider(provider, res.Config.Provider)
	}

	if res.Query == "" {
		if err := tui.Run(tui.Options{
			Result:      res,
			Provider:    provider,
			ProviderErr: perr,
			LoadArgs:    os.Args[1:],
			NewProvider: newProvider,
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
	if err := runOnce(os.Stdout, provider, res); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		flushTracing(shutdownTracing)
		os.Exit(1)
	}
}

// flushTracing exists because os.Exit does not run deferred functions, so
// every exit path that reports a failure has to flush the spans describing it
// before the process dies.
func flushTracing(shutdown func(context.Context) error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := shutdown(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "telemetry: flush failed: %v\n", err)
	}
}

// runOnce takes its writer so the one-shot path is drivable end to end in a
// test; main passes os.Stdout. ARCHITECTURE.md §11 asks user-journey tests to
// assert observable behaviour, which means the rendered bytes have to be
// reachable without redirecting the process's stdout.
func runOnce(w io.Writer, provider core.Provider, res config.Result) error {
	ch, err := provider.Stream(context.Background(), core.StreamRequest{
		Messages:  []core.Message{{Role: core.RoleUser, Content: res.Query}},
		Model:     res.Config.Model,
		Reasoning: res.Config.Reasoning,
	})
	if err != nil {
		return err
	}
	response, err := core.CollectResponse(ch)
	if err != nil {
		return err
	}
	out, err := glamour.Render(response.Text, "dark")
	if err != nil {
		return err
	}
	if !strings.HasSuffix(out, "\n") {
		out += "\n"
	}
	_, err = fmt.Fprint(w, out)
	return err
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
