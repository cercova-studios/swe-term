// Command drift is a continuous drift sensor: it measures slow-accumulating
// codebase health signals and compares them against a recorded baseline.
//
// It is deliberately NOT a commit gate. Per-change gates live in tests (see
// internal/architecture/fitness_test.go); this runs outside the change
// lifecycle, on a schedule or on demand, and answers a different question —
// not "is this change legal" but "what has been slowly getting worse."
// See docs/plans/2026-09-20-test-quality-and-architectural-fitness-steering.md.
//
// Drift is a delta, not a state, so every signal is judged against a
// checked-in baseline rather than an absolute threshold. Absolute thresholds
// fail on day one in an existing codebase and then get disabled; a ratchet
// starts wherever the code already is and only resists getting worse.
//
// Usage:
//
//	drift                 report current values against the baseline
//	drift -strict         exit non-zero if any signal regressed (for CI)
//	drift -update         accept current values as the new baseline
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	modulePrefix = "swe-term"
	// One baseline, one path. Add a flag when a second one exists.
	baselinePath = "docs/reports/drift-baseline.json"
)

// signal is one measured health value. HigherIsBetter records which direction
// counts as a regression, so the comparison logic never has to special-case a
// signal by name.
type signal struct {
	Name           string
	Value          float64
	HigherIsBetter bool
	Note           string
}

type baseline struct {
	RecordedAt string             `json:"recorded_at"`
	Signals    map[string]float64 `json:"signals"`
}

func main() {
	update := flag.Bool("update", false, "accept current values as the new baseline")
	strict := flag.Bool("strict", false, "exit non-zero if any signal regressed")
	flag.Parse()

	root, err := moduleRoot()
	if err != nil {
		fail(err)
	}

	signals, err := collect(root)
	if err != nil {
		fail(err)
	}

	if *update {
		if err := writeBaseline(filepath.Join(root, baselinePath), signals); err != nil {
			fail(err)
		}
		fmt.Printf("baseline updated: %s\n", baselinePath)
		for _, s := range signals {
			fmt.Printf("  %-32s %10s\n", s.Name, format(s.Value))
		}
		return
	}

	previous, err := readBaseline(filepath.Join(root, baselinePath))
	if err != nil {
		fmt.Fprintf(os.Stderr, "no usable baseline at %s (%v)\n", baselinePath, err)
		fmt.Fprintf(os.Stderr, "record one with: go run ./cmd/drift -update\n")
		os.Exit(2)
	}

	regressions := report(previous, signals)
	if regressions > 0 && *strict {
		os.Exit(1)
	}
}

func report(previous baseline, signals []signal) int {
	fmt.Printf("drift report — baseline recorded %s\n\n", previous.RecordedAt)
	fmt.Printf("  %-32s %10s %10s %9s  %s\n", "SIGNAL", "BASELINE", "CURRENT", "DELTA", "VERDICT")

	regressions := 0
	for _, s := range signals {
		before, known := previous.Signals[s.Name]
		if !known {
			fmt.Printf("  %-32s %10s %10s %9s  NEW (not in baseline)\n", s.Name, "—", format(s.Value), "—")
			continue
		}
		delta := s.Value - before
		verdict := verdictFor(before, s.Value, s.HigherIsBetter)
		if verdict == verdictRegressed {
			regressions++
		}
		fmt.Printf("  %-32s %10s %10s %+9s  %s\n",
			s.Name, format(before), format(s.Value), format(delta), verdict)
	}

	fmt.Println()
	for _, s := range signals {
		if s.Note != "" {
			fmt.Printf("  %s: %s\n", s.Name, s.Note)
		}
	}

	fmt.Println()
	switch regressions {
	case 0:
		fmt.Println("no regressions.")
	default:
		// Typed feedback: say what to do, not just that something is wrong.
		fmt.Printf("%d signal(s) regressed.\n", regressions)
		fmt.Println("Either address the drift, or accept it deliberately with:")
		fmt.Println("  go run ./cmd/drift -update    # and explain why in the commit message")
	}
	return regressions
}

const (
	verdictUnchanged = "unchanged"
	verdictImproved  = "improved"
	verdictRegressed = "REGRESSED"
)

// verdictFor is the whole comparison rule, isolated because getting the
// direction backwards would silently invert every signal.
func verdictFor(before, current float64, higherIsBetter bool) string {
	switch delta := current - before; {
	case delta == 0:
		return verdictUnchanged
	case (delta > 0) == higherIsBetter:
		return verdictImproved
	default:
		return verdictRegressed
	}
}

func collect(root string) ([]signal, error) {
	sourceLines, testLines, exported, markers, err := walkGo(root)
	if err != nil {
		return nil, err
	}
	deps, err := directDependencies(root)
	if err != nil {
		return nil, err
	}
	fanIn, hub, err := maxPackageFanIn(root)
	if err != nil {
		return nil, err
	}

	ratio := 0.0
	if sourceLines > 0 {
		ratio = float64(testLines) / float64(sourceLines)
	}

	return []signal{
		{
			Name: "internal_exported_decls", Value: float64(exported),
			Note: "exported names reachable from outside their package in internal/; growth is future compatibility obligation (ARCHITECTURE.md §3)",
		},
		{
			Name: "direct_dependencies", Value: float64(deps),
			Note: "non-indirect requires in go.mod; each one is a portability and supply-chain commitment (§8)",
		},
		{
			Name: "max_package_fan_in", Value: float64(fanIn),
			Note: fmt.Sprintf("most-depended-upon internal package is %s; a rising hub means coupling is concentrating", hub),
		},
		{
			Name: "debt_markers", Value: float64(markers),
			Note: "TODO/FIXME/XXX/HACK comments in Go source",
		},
		{
			Name: "test_to_source_line_ratio", Value: math.Round(ratio*1000) / 1000, HigherIsBetter: true,
			Note: "test lines per source line; falling means code is outgrowing its tests",
		},
	}, nil
}

// walkGo parses every Go file in the module once, collecting the signals that
// need the AST or the raw text.
func walkGo(root string) (sourceLines, testLines, exported, markers int, err error) {
	fset := token.NewFileSet()
	markerWords := []string{"TODO", "FIXME", "XXX", "HACK"}

	err = filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			// The vendored Glean tree is a third-party mirror, not our code.
			if name := entry.Name(); name == "Glean" || name == ".git" || name == "node_modules" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}

		content, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		lines := strings.Count(string(content), "\n")
		isTest := strings.HasSuffix(path, "_test.go")
		if isTest {
			testLines += lines
		} else {
			sourceLines += lines
		}

		file, err := parser.ParseFile(fset, path, content, parser.ParseComments)
		if err != nil {
			// A file that doesn't parse is a build failure, not drift; leave
			// that to the compiler rather than reporting a bogus number.
			return fmt.Errorf("parsing %s: %w", path, err)
		}

		for _, group := range file.Comments {
			for _, comment := range group.List {
				for _, word := range markerWords {
					markers += strings.Count(comment.Text, word)
				}
			}
		}

		// Exported surface is counted for non-test files under internal/ only:
		// cmd/ binaries have no importers, and test files are not surface.
		rel, relErr := filepath.Rel(root, path)
		if relErr != nil || isTest || !strings.HasPrefix(rel, "internal"+string(filepath.Separator)) {
			return nil
		}
		exported += countExported(file)
		return nil
	})
	return sourceLines, testLines, exported, markers, err
}

// countExported counts exported top-level names. Methods count only when
// their receiver type is itself exported, since a method on an unexported
// type is not reachable from another package.
func countExported(file *ast.File) int {
	count := 0
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if !d.Name.IsExported() {
				continue
			}
			if d.Recv == nil {
				count++
				continue
			}
			if receiverIsExported(d.Recv) {
				count++
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					if s.Name.IsExported() {
						count++
					}
				case *ast.ValueSpec:
					for _, name := range s.Names {
						if name.IsExported() {
							count++
						}
					}
				}
			}
		}
	}
	return count
}

func receiverIsExported(recv *ast.FieldList) bool {
	if recv == nil || len(recv.List) == 0 {
		return false
	}
	expr := recv.List[0].Type
	if star, ok := expr.(*ast.StarExpr); ok {
		expr = star.X
	}
	// A generic receiver arrives as Type[T]; the name is on the base.
	if index, ok := expr.(*ast.IndexExpr); ok {
		expr = index.X
	}
	ident, ok := expr.(*ast.Ident)
	return ok && ident.IsExported()
}

func directDependencies(root string) (int, error) {
	content, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		return 0, err
	}
	count, inBlock := 0, false
	for _, line := range strings.Split(string(content), "\n") {
		trimmed := strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(trimmed, "require ("):
			inBlock = true
		case inBlock && trimmed == ")":
			inBlock = false
		case inBlock && trimmed != "" && !strings.HasPrefix(trimmed, "//"):
			if !strings.Contains(trimmed, "// indirect") {
				count++
			}
		case strings.HasPrefix(trimmed, "require ") && !strings.Contains(trimmed, "// indirect"):
			count++
		}
	}
	return count, nil
}

// maxPackageFanIn reports how many internal packages import the single
// most-depended-upon internal package. Direct imports, not transitive:
// transitive fan-in is dominated by whatever the binary wires together and
// says little about where coupling is concentrating.
func maxPackageFanIn(root string) (int, string, error) {
	cmd := exec.Command("go", "list", "-json", modulePrefix+"/...")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return 0, "", fmt.Errorf("go list: %w", err)
	}

	type goPackage struct {
		ImportPath string
		Imports    []string
	}

	fanIn := map[string]int{}
	decoder := json.NewDecoder(strings.NewReader(string(out)))
	for decoder.More() {
		var pkg goPackage
		if err := decoder.Decode(&pkg); err != nil {
			return 0, "", err
		}
		if strings.Contains(pkg.ImportPath, "/Glean/") {
			continue
		}
		for _, imported := range pkg.Imports {
			if strings.HasPrefix(imported, modulePrefix) && !strings.Contains(imported, "/Glean/") {
				fanIn[imported]++
			}
		}
	}

	names := make([]string, 0, len(fanIn))
	for name := range fanIn {
		names = append(names, name)
	}
	// Sort for determinism: the same graph must always report the same hub.
	sort.Slice(names, func(i, j int) bool {
		if fanIn[names[i]] != fanIn[names[j]] {
			return fanIn[names[i]] > fanIn[names[j]]
		}
		return names[i] < names[j]
	})
	if len(names) == 0 {
		return 0, "(none)", nil
	}
	return fanIn[names[0]], names[0], nil
}

func moduleRoot() (string, error) {
	out, err := exec.Command("go", "env", "GOMOD").Output()
	if err != nil {
		return "", fmt.Errorf("locating module: %w", err)
	}
	path := strings.TrimSpace(string(out))
	if path == "" || path == os.DevNull {
		return "", fmt.Errorf("not inside a Go module")
	}
	return filepath.Dir(path), nil
}

func readBaseline(path string) (baseline, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return baseline{}, err
	}
	var b baseline
	if err := json.Unmarshal(content, &b); err != nil {
		return baseline{}, err
	}
	if len(b.Signals) == 0 {
		return baseline{}, fmt.Errorf("baseline contains no signals")
	}
	return b, nil
}

func writeBaseline(path string, signals []signal) error {
	b := baseline{
		RecordedAt: time.Now().UTC().Format(time.RFC3339),
		Signals:    map[string]float64{},
	}
	for _, s := range signals {
		b.Signals[s.Name] = s.Value
	}
	content, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, append(content, '\n'), 0o644)
}

func format(value float64) string {
	if value == float64(int(value)) {
		return fmt.Sprintf("%d", int(value))
	}
	return fmt.Sprintf("%.3f", value)
}

func fail(err error) {
	fmt.Fprintf(os.Stderr, "drift: %v\n", err)
	os.Exit(2)
}
