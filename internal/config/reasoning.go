package config

import "sort"

var reasoningLevels = map[string]struct{}{
	"none":    {},
	"minimal": {},
	"low":     {},
	"medium":  {},
	"high":    {},
	"xhigh":   {},
	"max":     {},
}

func reasoningLevelNames() []string {
	names := make([]string, 0, len(reasoningLevels))
	for name := range reasoningLevels {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
