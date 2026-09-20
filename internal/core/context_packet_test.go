package core

import (
	"strings"
	"testing"
)

func TestDetermineFreshness(t *testing.T) {
	tests := []struct {
		name       string
		base, head string
		want       Freshness
	}{
		{"same revision is fresh", "abc123", "abc123", FreshnessFresh},
		{"different revision is stale", "abc123", "def456", FreshnessStale},
		{"missing base is unknown, not fresh", "", "def456", FreshnessUnknown},
		{"missing head is unknown, not fresh", "abc123", "", FreshnessUnknown},
		{"both missing is unknown", "", "", FreshnessUnknown},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := DetermineFreshness(test.base, test.head); got != test.want {
				t.Fatalf("DetermineFreshness(%q, %q) = %q, want %q", test.base, test.head, got, test.want)
			}
		})
	}
}

func TestConcludeAbsenceIsThreeValued(t *testing.T) {
	// Invariant 11: absence proves non-existence only in a complete scope, and
	// only when resolution was exact. Anything else is unknown.
	tests := []struct {
		name  string
		scope ScopeCompleteness
		tier  ResolutionTier
		want  Reachability
	}{
		{"complete scope with compiler resolution proves absence",
			ScopeComplete, TierCompiler, ReachabilityNotFoundInCompleteScope},
		{"complete scope with heuristic resolution does not",
			ScopeComplete, TierSyntacticHeuristic, ReachabilityUnknown},
		{"partial scope never proves absence",
			ScopePartial, TierCompiler, ReachabilityUnknown},
		{"unknown scope never proves absence",
			ScopeUnknown, TierCompiler, ReachabilityUnknown},
		{"partial scope and heuristic tier is unknown",
			ScopePartial, TierSyntacticHeuristic, ReachabilityUnknown},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p := Provenance{Scope: test.scope, ResolutionTier: test.tier}
			if got := p.ConcludeAbsence(); got != test.want {
				t.Fatalf("ConcludeAbsence() = %q, want %q", got, test.want)
			}
		})
	}
}

func TestProvenanceValidationFailsClosed(t *testing.T) {
	valid := Provenance{
		Source: "fleet-cpg", BaseRevision: "abc", HeadRevision: "abc",
		Freshness: FreshnessFresh, Scope: ScopePartial,
		ResolutionTier: TierSyntacticHeuristic, ProfileVersion: "sha256:x",
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid provenance rejected: %v", err)
	}

	tests := []struct {
		name   string
		mutate func(*Provenance)
	}{
		{"missing source", func(p *Provenance) { p.Source = "" }},
		{"unknown freshness value", func(p *Provenance) { p.Freshness = Freshness("probably") }},
		{"unknown scope value", func(p *Provenance) { p.Scope = ScopeCompleteness("mostly") }},
		{"unknown resolution tier", func(p *Provenance) { p.ResolutionTier = ResolutionTier("vibes") }},
		// The claim that matters most: fresh without the evidence for it.
		{"fresh without base revision", func(p *Provenance) { p.BaseRevision = "" }},
		{"fresh without head revision", func(p *Provenance) { p.HeadRevision = "" }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p := valid
			test.mutate(&p)
			if err := p.Validate(); err == nil {
				t.Fatal("invalid provenance accepted")
			}
		})
	}
}

func TestPacketWithContentRequiresAProfileVersion(t *testing.T) {
	packet := ContextPacket{
		Path: "/tmp/packet",
		Provenance: Provenance{
			Source: "fleet-cpg", BaseRevision: "abc", HeadRevision: "abc",
			Freshness: FreshnessFresh, Scope: ScopePartial,
			ResolutionTier: TierSyntacticHeuristic,
		},
	}
	if err := packet.Validate(); err == nil {
		t.Fatal("packet with content but no profile version was accepted")
	}
	packet.Provenance.ProfileVersion = "sha256:x"
	if err := packet.Validate(); err != nil {
		t.Fatalf("packet with profile version rejected: %v", err)
	}
}

func TestUnavailableIsAnHonestPacketNotAnEmptySuccess(t *testing.T) {
	// Invariant 12: a missing adapter must not be converted into a reassuring
	// zero. The packet is well-formed, carries no content, and says so.
	packet := Unavailable("fleet-cpg")
	if err := packet.Validate(); err != nil {
		t.Fatalf("unavailable packet must still be well-formed: %v", err)
	}
	if packet.Path != "" {
		t.Fatal("unavailable packet must not claim materialized content")
	}
	if packet.Provenance.Freshness != FreshnessUnknown || packet.Provenance.Scope != ScopeUnknown {
		t.Fatalf("unavailable packet must report unknowns, got %#v", packet.Provenance)
	}
	if got := packet.Provenance.ConcludeAbsence(); got != ReachabilityUnknown {
		t.Fatalf("absence under an unavailable adapter = %q, want %q", got, ReachabilityUnknown)
	}
	usable, reason := packet.UsableAsAuthoritative()
	if usable || reason == "" {
		t.Fatalf("unavailable packet reported usable=%v reason=%q", usable, reason)
	}
}

func TestUsableAsAuthoritativeExplainsRefusal(t *testing.T) {
	base := Provenance{
		Source: "fleet-cpg", Scope: ScopePartial,
		ResolutionTier: TierSyntacticHeuristic, ProfileVersion: "sha256:x",
	}

	t.Run("fresh packet with content is usable", func(t *testing.T) {
		p := base
		p.Freshness, p.BaseRevision, p.HeadRevision = FreshnessFresh, "abc", "abc"
		usable, reason := ContextPacket{Path: "/tmp/p", Provenance: p}.UsableAsAuthoritative()
		if !usable {
			t.Fatalf("fresh packet refused: %s", reason)
		}
	})

	t.Run("fresh packet without content is refused", func(t *testing.T) {
		p := base
		p.Freshness, p.BaseRevision, p.HeadRevision = FreshnessFresh, "abc", "abc"
		usable, reason := ContextPacket{Provenance: p}.UsableAsAuthoritative()
		if usable || reason == "" {
			t.Fatal("a fresh packet with no content should not be authoritative")
		}
	})

	t.Run("stale packet names both revisions", func(t *testing.T) {
		p := base
		p.Freshness, p.BaseRevision, p.HeadRevision = FreshnessStale, "abc", "def"
		usable, reason := ContextPacket{Path: "/tmp/p", Provenance: p}.UsableAsAuthoritative()
		if usable {
			t.Fatal("stale packet reported as authoritative")
		}
		// Typed feedback: the caller must be able to act on the refusal.
		if !strings.Contains(reason, "abc") || !strings.Contains(reason, "def") {
			t.Fatalf("stale reason must name both revisions, got %q", reason)
		}
	})
}
