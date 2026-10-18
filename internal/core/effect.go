package core

import (
	"slices"
	"strings"
)

// EffectClass is the closed vocabulary of effect kinds a tool invocation can
// declare or be observed performing.
type EffectClass string

const (
	EffectClassRead    EffectClass = "read"
	EffectClassWrite   EffectClass = "write"
	EffectClassProcess EffectClass = "process"
	EffectClassNetwork EffectClass = "network"
	EffectClassSecret  EffectClass = "secret"
)

// ObservationLevel states how an effect is known: directly observed by the
// harness, attested by the tool, or unobservable by construction (for example
// a provider-hosted tool whose effects no harness can see).
type ObservationLevel string

const (
	ObservationObserved     ObservationLevel = "observed"
	ObservationAttested     ObservationLevel = "attested"
	ObservationUnobservable ObservationLevel = "unobservable"
)

func (level ObservationLevel) Valid() bool {
	return level == ObservationObserved || level == ObservationAttested || level == ObservationUnobservable
}

// PathEffect is one declared filesystem target. Glob marks Path as a
// directory prefix matched on a clean path boundary.
type PathEffect struct {
	Path string
	Glob bool
}

// EffectDeclaration is the bounded set of effects one tool invocation may
// perform, declared before execution and bound to the active lease. Secrets
// holds names only, never values (invariant 14). Observation is the ceiling
// observation level for the declaration.
type EffectDeclaration struct {
	Reads       []PathEffect
	Writes      []PathEffect
	Processes   []string
	Network     []string
	Secrets     []string
	Observation ObservationLevel
}

// ObservedEffect is one effect seen at runtime, tagged with how it is known.
type ObservedEffect struct {
	Class       EffectClass
	Target      string
	Observation ObservationLevel
}

func (effect ObservedEffect) Valid() bool {
	switch effect.Class {
	case EffectClassRead, EffectClassWrite, EffectClassProcess, EffectClassNetwork, EffectClassSecret:
	default:
		return false
	}
	return effect.Target != "" && effect.Observation.Valid()
}

// HasMutation reports whether the declaration governs workspace mutation:
// writes or spawned processes.
func (declaration EffectDeclaration) HasMutation() bool {
	return len(declaration.Writes) > 0 || len(declaration.Processes) > 0
}

// Validate fails closed. A declaration must name at least one effect, stay
// within the declared-effects bound, carry a valid observation level, contain
// no duplicates, empty targets, or traversal paths, and name secrets without
// embedding values. A mutating declaration may never be unobservable: no
// harness could then check the mutation against the declaration (OA-003).
func (declaration EffectDeclaration) Validate() bool {
	if !declaration.Observation.Valid() {
		return false
	}
	total := len(declaration.Reads) + len(declaration.Writes) + len(declaration.Processes) + len(declaration.Network) + len(declaration.Secrets)
	if total == 0 || total > maxDeclaredEffects {
		return false
	}
	if declaration.HasMutation() && declaration.Observation == ObservationUnobservable {
		return false
	}
	return validPathEffects(declaration.Reads) && validPathEffects(declaration.Writes) &&
		validTargets(declaration.Processes) && validTargets(declaration.Network) &&
		validSecretNames(declaration.Secrets)
}

// Covers is the structural membership test the monitor applies to an observed
// effect. Observation levels must be compatible: an unobservable event is
// covered only by a declaration that itself declared unobservable, an
// attested event by an attested or unobservable declaration, and an observed
// event by any valid declared level. The target must then belong to the
// matching effect class.
func (declaration EffectDeclaration) Covers(observed ObservedEffect) bool {
	if !observed.Valid() || !declaration.Observation.Valid() {
		return false
	}
	switch observed.Observation {
	case ObservationUnobservable:
		if declaration.Observation != ObservationUnobservable {
			return false
		}
	case ObservationAttested:
		if declaration.Observation == ObservationObserved {
			return false
		}
	}
	switch observed.Class {
	case EffectClassRead:
		return pathEffectsCover(declaration.Reads, observed.Target)
	case EffectClassWrite:
		return pathEffectsCover(declaration.Writes, observed.Target)
	case EffectClassProcess:
		return slices.Contains(declaration.Processes, observed.Target)
	case EffectClassNetwork:
		return slices.Contains(declaration.Network, observed.Target)
	case EffectClassSecret:
		return slices.Contains(declaration.Secrets, observed.Target)
	}
	return false
}

// Equal compares declarations structurally; nil and empty slices are equal.
func (declaration EffectDeclaration) Equal(other EffectDeclaration) bool {
	return declaration.Observation == other.Observation &&
		slices.Equal(declaration.Reads, other.Reads) &&
		slices.Equal(declaration.Writes, other.Writes) &&
		slices.Equal(declaration.Processes, other.Processes) &&
		slices.Equal(declaration.Network, other.Network) &&
		slices.Equal(declaration.Secrets, other.Secrets)
}

func validPathEffects(effects []PathEffect) bool {
	for index, effect := range effects {
		if effect.Path == "" || slices.Contains(effects[:index], effect) {
			return false
		}
		for _, segment := range strings.Split(effect.Path, "/") {
			if segment == ".." {
				return false
			}
		}
	}
	return true
}

func validTargets(targets []string) bool {
	for index, target := range targets {
		if target == "" || slices.Contains(targets[:index], target) {
			return false
		}
	}
	return true
}

// validSecretNames accepts identifier names only (for example
// OPENAI_API_KEY). Anything shaped like a value — an assignment, a scheme
// prefix, whitespace, or a known token prefix — is rejected so a secret value
// cannot be declared, journaled, or receipted by accident.
func validSecretNames(names []string) bool {
	for index, name := range names {
		if slices.Contains(names[:index], name) || !isSecretName(name) {
			return false
		}
	}
	return true
}

func isSecretName(name string) bool {
	if name == "" || strings.HasPrefix(name, "sk-") || strings.HasPrefix(name, "ghp_") || strings.HasPrefix(name, "Bearer ") {
		return false
	}
	for index, r := range name {
		switch {
		case r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r == '_':
		case r >= '0' && r <= '9' && index > 0:
		default:
			return false
		}
	}
	return true
}

func pathEffectsCover(effects []PathEffect, target string) bool {
	for _, effect := range effects {
		if effect.Path == target {
			return true
		}
		if effect.Glob && strings.HasPrefix(target, strings.TrimSuffix(effect.Path, "/")+"/") {
			return true
		}
	}
	return false
}
