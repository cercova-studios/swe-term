package tui

import (
	"sort"
	"strings"
)

type slashCmd struct {
	name string
	help string
}

func (c slashCmd) token() string { return "/" + c.name }

var slashCommands = []slashCmd{
	{name: "config", help: "print user and project config.toml"},
	{name: "help", help: "this text"},
	{name: "quit", help: "exit"},
}

// filterSlash ranks commands for the slash dropdown. Empty unless the input is
// an in-progress /command (no args). Exact matches hide the menu.
func filterSlash(typed string) []slashCmd {
	typed = strings.ToLower(strings.TrimSpace(typed))
	if !strings.HasPrefix(typed, "/") || strings.ContainsAny(typed, " \t") {
		return nil
	}
	for _, c := range slashCommands {
		if typed == c.token() {
			return nil
		}
	}
	q := strings.TrimPrefix(typed, "/")
	if q == "" {
		out := make([]slashCmd, len(slashCommands))
		copy(out, slashCommands)
		return out
	}

	type scored struct {
		cmd   slashCmd
		score int
	}
	var hits []scored
	for _, c := range slashCommands {
		if s, ok := slashScore(q, c.name); ok {
			hits = append(hits, scored{c, s})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].score != hits[j].score {
			return hits[i].score > hits[j].score
		}
		return hits[i].cmd.name < hits[j].cmd.name
	})
	out := make([]slashCmd, len(hits))
	for i, h := range hits {
		out[i] = h.cmd
	}
	return out
}

func slashScore(q, name string) (int, bool) {
	switch {
	case strings.HasPrefix(name, q):
		return 1000 - (len(name) - len(q)), true
	case strings.Contains(name, q):
		return 500, true
	default:
		d := editDistance(q, name)
		if d <= 2 {
			return 100 - d, true
		}
		return 0, false
	}
}

func editDistance(a, b string) int {
	ra, rb := []rune(a), []rune(b)
	if len(ra) == 0 {
		return len(rb)
	}
	prev := make([]int, len(rb)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ra); i++ {
		cur := make([]int, len(rb)+1)
		cur[0] = i
		for j := 1; j <= len(rb); j++ {
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}
			cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
		}
		prev = cur
	}
	return prev[len(rb)]
}
