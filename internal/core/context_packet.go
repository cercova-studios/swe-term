package core

import (
	"fmt"
	"slices"
)

// ContextPacket is what an analyzer/enrichment adapter hands the core:
// materialized pre-model context plus the provenance needed to judge it.
//
// ARCHITECTURE.md §9 makes analyzer adapters an extension point and §10
// invariant 13 keeps extension internals out of core state — so the core sees
// a path and a provenance envelope, never the analyzer's own types. §7
// requires that every injected context item carry its source, snapshot
// version, and freshness, and that staleness be checked rather than assumed
// away. This type is the first concrete shape of that contract.
//
// Status: Target. These types exist and validate, but nothing produces a
// packet at runtime yet — same standing as VerificationReceipt and the V&V
// gate, which are also reducers without a loop around them.
type ContextPacket struct {
	// Path is the root of the materialized file tree. Empty when the adapter
	// produced nothing, which is a legitimate, honest state — see Unavailable.
	Path       string
	Provenance Provenance
}

// Provenance answers "how much should the core trust this, and why".
type Provenance struct {
	Source         string // which adapter produced this, e.g. "fleet-cpg"
	BaseRevision   string // the snapshot the analysis was computed against
	HeadRevision   string // the revision the caller asked about
	Freshness      Freshness
	Scope          ScopeCompleteness
	ResolutionTier ResolutionTier
	ProfileVersion string // snapshot manifest digest; joins to telemetry
}

// Freshness records whether the analysis was computed against the state the
// caller actually asked about. It is a measured fact, never an assumption:
// fleet-cpg-phase1-commit-zod demonstrated that a base one unrelated commit
// away from the diff's parent silently corrupts facts for untouched files.
type Freshness string

const (
	FreshnessFresh   Freshness = "fresh"
	FreshnessStale   Freshness = "stale"
	FreshnessUnknown Freshness = "unknown"
)

// ScopeCompleteness records whether the analyzer saw everything it would need
// to see for absence to mean anything.
type ScopeCompleteness string

const (
	ScopeComplete ScopeCompleteness = "complete"
	ScopePartial  ScopeCompleteness = "partial"
	ScopeUnknown  ScopeCompleteness = "unknown"
)

// ResolutionTier records how precisely references were resolved. The
// heuristic tier resolves callees by name, which is known to produce false
// edges — fleet-cpg-phase2 caught a real one, where a touched function named
// `map` absorbed every `Array.prototype.map` call site.
type ResolutionTier string

const (
	TierSyntacticHeuristic ResolutionTier = "syntactic-heuristic"
	TierCompiler           ResolutionTier = "compiler"
)

// Reachability is the three-valued answer required by ARCHITECTURE.md §10
// invariant 11: incomplete graph absence is `unknown`, not proof of
// non-existence.
type Reachability string

const (
	ReachabilityNotFoundInCompleteScope Reachability = "not_found_in_complete_scope"
	ReachabilityUnknown                 Reachability = "unknown"
)

// A third value, "found", belongs to this vocabulary but is not declared
// until something produces it — no lookup path exists yet, and ConcludeAbsence
// structurally cannot return it.

// DetermineFreshness compares the revision an analysis was computed against
// with the parent of the diff under review. It never guesses: a missing
// revision yields unknown rather than an optimistic "probably fine".
func DetermineFreshness(baseRevision, diffParentRevision string) Freshness {
	if baseRevision == "" || diffParentRevision == "" {
		return FreshnessUnknown
	}
	if baseRevision == diffParentRevision {
		return FreshnessFresh
	}
	return FreshnessStale
}

// Unavailable builds the packet to return when an adapter is missing,
// unreachable, or failed. ARCHITECTURE.md §10 invariant 12: unknown
// provenance stays explicit and is never converted to a reassuring zero, so
// this is a well-formed packet carrying honest unknowns — not an empty
// success and not an error the caller might drop.
func Unavailable(source string) ContextPacket {
	return ContextPacket{
		Provenance: Provenance{
			Source:         source,
			Freshness:      FreshnessUnknown,
			Scope:          ScopeUnknown,
			ResolutionTier: TierSyntacticHeuristic,
		},
	}
}

// ConcludeAbsence reports what an absent result means under this provenance.
//
// Absence is only evidence of non-existence when the analyzer both saw
// everything (complete scope) and resolved references exactly (compiler
// tier). A complete scan with heuristic resolution still cannot support the
// claim, because a missed edge is a resolution failure, not a coverage gap.
// Everything else is unknown.
func (p Provenance) ConcludeAbsence() Reachability {
	if p.Scope == ScopeComplete && p.ResolutionTier == TierCompiler {
		return ReachabilityNotFoundInCompleteScope
	}
	return ReachabilityUnknown
}

// Validate fails closed on any provenance that claims more than it can
// support. It is deliberately strict about the Fresh claim in particular:
// asserting freshness without the two revisions that justify it is exactly
// the assumption §7 says to check rather than assume away.
func (p Provenance) Validate() error {
	if p.Source == "" {
		return fmt.Errorf("provenance.source is required: an unattributed context item cannot be judged")
	}
	if !slices.Contains([]Freshness{FreshnessFresh, FreshnessStale, FreshnessUnknown}, p.Freshness) {
		return fmt.Errorf("provenance.freshness %q is outside the closed vocabulary", p.Freshness)
	}
	if !slices.Contains([]ScopeCompleteness{ScopeComplete, ScopePartial, ScopeUnknown}, p.Scope) {
		return fmt.Errorf("provenance.scope %q is outside the closed vocabulary", p.Scope)
	}
	if !slices.Contains([]ResolutionTier{TierSyntacticHeuristic, TierCompiler}, p.ResolutionTier) {
		return fmt.Errorf("provenance.resolution_tier %q is outside the closed vocabulary", p.ResolutionTier)
	}
	if p.Freshness == FreshnessFresh || p.Freshness == FreshnessStale {
		if p.BaseRevision == "" || p.HeadRevision == "" {
			return fmt.Errorf("freshness %q requires both base and head revisions as evidence", p.Freshness)
		}
	}
	return nil
}

// Validate checks the packet as a whole. A packet carrying data must also
// carry the snapshot version that identifies it (§7: every injected context
// item carries its source, snapshot/version, and freshness).
func (c ContextPacket) Validate() error {
	if err := c.Provenance.Validate(); err != nil {
		return err
	}
	if c.Path != "" && c.Provenance.ProfileVersion == "" {
		return fmt.Errorf("a packet with materialized content requires a profile_version to identify the snapshot it came from")
	}
	return nil
}

// UsableAsAuthoritative reports whether this packet may be relied on without
// qualification, and if not, why. A stale or unknown-freshness packet is not
// worthless — it is context the caller must label — so this returns a reason
// rather than an error, and callers are expected to degrade rather than drop.
func (c ContextPacket) UsableAsAuthoritative() (bool, string) {
	if err := c.Validate(); err != nil {
		return false, err.Error()
	}
	switch c.Provenance.Freshness {
	case FreshnessFresh:
		if c.Path == "" {
			return false, "packet is fresh but carries no materialized content"
		}
		return true, ""
	case FreshnessStale:
		return false, fmt.Sprintf(
			"analysis was computed against %s but the diff's parent is %s; "+
				"facts for files this diff never touched may be wrong",
			c.Provenance.BaseRevision, c.Provenance.HeadRevision)
	default:
		return false, "freshness is unknown; treat all facts as unverified"
	}
}
