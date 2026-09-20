package main

import (
	"go/parser"
	"go/token"
	"testing"
)

func TestVerdictForRespectsSignalDirection(t *testing.T) {
	tests := []struct {
		name           string
		before, after  float64
		higherIsBetter bool
		want           string
	}{
		// lower-is-better signals: debt markers, exported surface, fan-in
		{"lower-is-better rising is a regression", 3, 5, false, verdictRegressed},
		{"lower-is-better falling is an improvement", 5, 3, false, verdictImproved},
		{"lower-is-better flat is unchanged", 4, 4, false, verdictUnchanged},

		// higher-is-better signals: test-to-source ratio
		{"higher-is-better falling is a regression", 0.5, 0.4, true, verdictRegressed},
		{"higher-is-better rising is an improvement", 0.4, 0.5, true, verdictImproved},
		{"higher-is-better flat is unchanged", 0.5, 0.5, true, verdictUnchanged},

		// a signal ratcheting from zero must still catch the first offender
		{"first debt marker is caught", 0, 1, false, verdictRegressed},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := verdictFor(test.before, test.after, test.higherIsBetter); got != test.want {
				t.Fatalf("verdictFor(%v, %v, higherIsBetter=%v) = %q, want %q",
					test.before, test.after, test.higherIsBetter, got, test.want)
			}
		})
	}
}

func TestCountExportedSurface(t *testing.T) {
	const src = `package sample

type Exported struct{}
type unexported struct{}

const ExportedConst = 1
const unexportedConst = 2

var ExportedVar, unexportedVar = 1, 2

func ExportedFunc() {}
func unexportedFunc() {}

// Reachable: a method on an exported type is part of the surface.
func (e Exported) ExportedMethod() {}
func (e *Exported) ExportedPointerMethod() {}

// Not reachable from another package: the receiver type is unexported.
func (u unexported) ExportedMethodOnUnexportedType() {}

// Not surface: unexported method on an exported type.
func (e Exported) unexportedMethod() {}
`

	file, err := parser.ParseFile(token.NewFileSet(), "sample.go", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}

	// Exported, ExportedConst, ExportedVar, ExportedFunc, ExportedMethod,
	// ExportedPointerMethod = 6.
	if got, want := countExported(file), 6; got != want {
		t.Fatalf("countExported = %d, want %d", got, want)
	}
}
