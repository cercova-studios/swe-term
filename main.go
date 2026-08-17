package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"charm.land/glamour/v2"

	"swe-term/internal/core"
	openaiprovider "swe-term/internal/provider/openai"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: go run . <query>")
		os.Exit(2)
	}
	query := strings.Join(os.Args[1:], " ")

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "OPENAI_API_KEY is not set")
		os.Exit(2)
	}

	var provider core.Provider = openaiprovider.New(apiKey, "")

	ch, err := provider.Stream(context.Background(), core.StreamRequest{
		Messages: []core.Message{{Role: core.RoleUser, Content: query}},
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	text, err := core.CollectText(ch)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	rendered, err := glamour.Render(text, "dark")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Print(rendered)
}
