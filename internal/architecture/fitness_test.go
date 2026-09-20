// Package architecture holds computational fitness functions: deterministic
// sensors that check whether the dependency rules asserted in ARCHITECTURE.md
// actually hold in the package graph.
//
// These are the repository's first architecture sensors. Before them,
// ARCHITECTURE.md's boundaries were feedforward-only — written constraints
// with nothing checking whether they survived contact with the code. See
// docs/plans/2026-09-20-test-quality-and-architectural-fitness-steering.md.
//
// Rules are hand-authored and cite the contract they enforce. A rule is not a
// style preference; each one restates a boundary the architecture already
// claims, so a failure means either the code drifted or the document is now
// wrong. Both are worth stopping for.
package architecture

import (
	"encoding/json"
	"os/exec"
	"strings"
	"testing"
)

// modulePrefix is this module's path, per go.mod.
const modulePrefix = "swe-term"

// goPackage is the subset of `go list -json` this test consumes.
type goPackage struct {
	ImportPath string
	Deps       []string // full transitive dependency list
}

// rule is one architectural boundary, expressed as: packages matching Subject
// must not transitively depend on anything Forbidden reports true for.
//
// Forbidden receives the subject package as well as the dependency so that
// relative rules — "no sibling may import another sibling" — can express
// themselves directly instead of being special-cased by name at match time.
//
// Transitive rather than direct is deliberate — a boundary violation laundered
// through an intermediate package is still a violation.
type rule struct {
	Name      string
	Contract  string // the ARCHITECTURE.md clause this enforces
	Subject   func(pkg string) bool
	Forbidden func(subject, dep string) bool
	Because   string // what breaks if this is violated
}

func isInternal(pkg string) bool { return strings.HasPrefix(pkg, modulePrefix) }

// isThirdParty reports whether an import path belongs to an external module.
// Standard-library paths have no dot in their first segment ("os", "net/http");
// module paths are domain-qualified ("github.com/x/y", "charm.land/z").
func isThirdParty(pkg string) bool {
	if isInternal(pkg) {
		return false
	}
	first, _, _ := strings.Cut(pkg, "/")
	return strings.Contains(first, ".")
}

func exactly(pkg string) func(string) bool {
	return func(candidate string) bool { return candidate == pkg }
}

func under(prefix string) func(string) bool {
	return func(candidate string) bool {
		return candidate == prefix || strings.HasPrefix(candidate, prefix+"/")
	}
}

func rules() []rule {
	core := modulePrefix + "/internal/core"
	tui := modulePrefix + "/internal/tui"
	providers := modulePrefix + "/internal/provider"
	experimentctl := modulePrefix + "/cmd/experimentctl"
	drift := modulePrefix + "/cmd/drift"

	return []rule{
		{
			Name:      "core_is_vendor_free",
			Contract:  "ARCHITECTURE.md §10 invariant 13; §5 provider-neutral vocabulary",
			Subject:   exactly(core),
			Forbidden: func(_, dep string) bool { return isThirdParty(dep) },
			Because: "core state must not carry vendor-specific types; a third-party " +
				"type reachable from core leaks an extension's internals into the core model",
		},
		{
			Name:      "core_depends_on_nothing_internal",
			Contract:  "ARCHITECTURE.md §3 'keep the core small'; §4 boundaries",
			Subject:   exactly(core),
			Forbidden: func(_, dep string) bool { return isInternal(dep) && dep != core },
			Because: "core defines the contracts others depend on; if it depends back on " +
				"config, tui, or a provider, the dependency direction has inverted",
		},
		{
			Name:      "frontend_does_not_own_provider_semantics",
			Contract:  "ARCHITECTURE.md §4 'frontends render state; they do not own provider pricing, policy, verification, or persistence semantics'",
			Subject:   exactly(tui),
			Forbidden: func(_, dep string) bool { return under(providers)(dep) },
			Because: "the TUI must consume the provider-neutral event contract, not a " +
				"concrete provider; importing one puts vendor semantics in the frontend",
		},
		{
			Name:     "providers_do_not_import_each_other",
			Contract: "ARCHITECTURE.md §8 capability ports with swappable backends",
			Subject:  under(providers),
			Forbidden: func(subject, dep string) bool {
				return under(providers)(dep) && dep != subject
			},
			Because: "each provider adapter is an independent backend; a cross-import " +
				"couples two supposedly swappable implementations",
		},
		{
			Name:     "experiment_tooling_is_not_a_second_state_model",
			Contract: "ARCHITECTURE.md §11 'experiment infrastructure is tooling around the core, not a second agent loop or state model'",
			Subject:  exactly(experimentctl),
			Forbidden: func(_, dep string) bool {
				return isInternal(dep) && dep != experimentctl
			},
			Because: "experimentctl validates manifests; depending on core state would " +
				"make the research substrate part of the runtime contract it is meant to study",
		},
		{
			Name:     "drift_sensor_is_decoupled_from_what_it_measures",
			Contract: "docs/plans/2026-09-20-test-quality-and-architectural-fitness-steering.md §4; same reasoning as experiment tooling",
			Subject:  exactly(drift),
			Forbidden: func(_, dep string) bool {
				return isInternal(dep) && dep != drift
			},
			Because: "a sensor that imports the packages it measures starts reporting on " +
				"itself; keeping it decoupled means its numbers describe the system, not the instrument",
		},
	}
}

func loadPackages(t *testing.T) []goPackage {
	t.Helper()

	// The module pattern, not "./...", because `go test` runs this binary with
	// the package directory as its working directory — "./..." would resolve to
	// this test package alone and every rule would pass vacuously.
	out, err := exec.Command("go", "list", "-deps", "-json", modulePrefix+"/...").Output()
	if err != nil {
		t.Fatalf("go list failed: %v", err)
	}

	var pkgs []goPackage
	decoder := json.NewDecoder(strings.NewReader(string(out)))
	for decoder.More() {
		var pkg goPackage
		if err := decoder.Decode(&pkg); err != nil {
			t.Fatalf("decoding go list output: %v", err)
		}
		// The vendored Glean tree is a third-party mirror, not part of this
		// module's architecture.
		if strings.Contains(pkg.ImportPath, "/Glean/") {
			continue
		}
		pkgs = append(pkgs, pkg)
	}
	if len(pkgs) == 0 {
		t.Fatal("go list returned no packages")
	}
	return pkgs
}

func TestArchitectureDependencyRules(t *testing.T) {
	pkgs := loadPackages(t)

	for _, r := range rules() {
		t.Run(r.Name, func(t *testing.T) {
			matched := 0
			for _, pkg := range pkgs {
				if !isInternal(pkg.ImportPath) || !r.Subject(pkg.ImportPath) {
					continue
				}
				matched++

				for _, dep := range pkg.Deps {
					if r.Forbidden(pkg.ImportPath, dep) {
						// Typed feedback: name the rule, the offending edge, the
						// contract, and the consequence — not just "assertion failed".
						t.Errorf(
							"architecture rule %q violated\n"+
								"  offending edge: %s -> %s\n"+
								"  contract:       %s\n"+
								"  why it matters: %s\n"+
								"  resolve by:     removing the dependency, or amending the contract in ARCHITECTURE.md in the same change",
							r.Name, pkg.ImportPath, dep, r.Contract, r.Because)
					}
				}
			}

			// A rule that matches no package is silently vacuous — the sensor
			// equivalent of a test that never runs. Fail loudly instead.
			if matched == 0 {
				t.Fatalf("rule %q matched no packages; its subject pattern is stale", r.Name)
			}
		})
	}
}
