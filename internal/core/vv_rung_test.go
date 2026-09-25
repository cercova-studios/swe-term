package core

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestMinimumRungPolicyIsClosed(t *testing.T) {
	// An unknown kind or risk must fail closed rather than defaulting to the
	// weakest rung, so that adding an obligation kind forces a policy decision.
	if _, ok := MinimumRung("invented_by_a_model", RiskHigh); ok {
		t.Fatal("unknown obligation kind resolved to a rung")
	}
	if _, ok := MinimumRung(ObligationBehaviourChange, RiskLevel("trivial")); ok {
		t.Fatal("unknown risk level resolved to a rung")
	}

	// Every declared kind must cover every risk level; a hole would be an
	// accidental fail-closed that looks like a policy decision.
	for _, kind := range []ObligationKind{
		ObligationBehaviourChange, ObligationRefactor, ObligationDependencyBump,
		ObligationSchemaMigration, ObligationDocumentation,
	} {
		for _, risk := range []RiskLevel{RiskLow, RiskMedium, RiskHigh} {
			rung, ok := MinimumRung(kind, risk)
			if !ok {
				t.Errorf("policy hole: %s at %s has no rung", kind, risk)
				continue
			}
			if !rung.Valid() {
				t.Errorf("policy for %s/%s yields invalid rung %d", kind, risk, int(rung))
			}
		}
	}
}

func TestMinimumRungIsMonotonicInRisk(t *testing.T) {
	// Escalating risk must never *lower* the bar. If it did, an agent could
	// downgrade its obligations by claiming a change is riskier.
	for kind := range minimumRungPolicy {
		low, _ := MinimumRung(kind, RiskLow)
		medium, _ := MinimumRung(kind, RiskMedium)
		high, _ := MinimumRung(kind, RiskHigh)
		if medium < low || high < medium {
			t.Errorf("%s: rung decreases as risk rises (%s, %s, %s)", kind, low, medium, high)
		}
	}
}

func TestVVGateTraces(t *testing.T) {
	tests := []struct {
		name           string
		events         []VVGateEvent
		wantRule       string
		wantDischarged bool
	}{
		{
			name: "sufficient evidence discharges the obligation",
			events: []VVGateEvent{
				declareObligation("o1", ObligationBehaviourChange, RiskLow),
				submitEvidence("o1", RungExample),
				claimDischarged("o1"),
			},
			wantDischarged: true,
		},
		{
			name: "under-rung evidence fails closed",
			events: []VVGateEvent{
				declareObligation("o1", ObligationBehaviourChange, RiskHigh), // needs reachability
				submitEvidence("o1", RungExample),
			},
			wantRule: "vv.rung_insufficient",
		},
		{
			name: "escalated evidence is accepted",
			events: []VVGateEvent{
				declareObligation("o1", ObligationBehaviourChange, RiskLow), // needs example
				submitEvidence("o1", RungReplayable),                        // far above
				claimDischarged("o1"),
			},
			wantDischarged: true,
		},
		{
			name: "discharge without evidence fails closed",
			events: []VVGateEvent{
				declareObligation("o1", ObligationBehaviourChange, RiskLow),
				claimDischarged("o1"),
			},
			wantRule: "vv.undischarged",
		},
		{
			name: "risk downgrade is rejected",
			events: []VVGateEvent{
				declareObligation("o1", ObligationBehaviourChange, RiskHigh),
				reclassify("o1", ObligationBehaviourChange, RiskLow),
			},
			wantRule: "vv.risk_downgrade",
		},
		{
			name: "reclassifying to a laxer kind at equal risk is rejected",
			events: []VVGateEvent{
				// refactor at low risk requires property; documentation
				// requires only build. Same risk, weaker bar.
				declareObligation("o1", ObligationRefactor, RiskLow),
				reclassify("o1", ObligationDocumentation, RiskLow),
			},
			wantRule: "vv.rung_downgrade",
		},
		{
			name: "risk escalation is allowed and raises the bar",
			events: []VVGateEvent{
				declareObligation("o1", ObligationBehaviourChange, RiskLow),
				reclassify("o1", ObligationBehaviourChange, RiskHigh),
				submitEvidence("o1", RungReachability),
				claimDischarged("o1"),
			},
			wantDischarged: true,
		},
		{
			name: "escalation invalidates evidence that only cleared the old bar",
			events: []VVGateEvent{
				declareObligation("o1", ObligationBehaviourChange, RiskLow),
				submitEvidence("o1", RungExample), // satisfies low
				reclassify("o1", ObligationBehaviourChange, RiskHigh),
				claimDischarged("o1"), // stale evidence must not carry over
			},
			wantRule: "vv.undischarged",
		},
		{
			name: "unknown policy fails closed rather than defaulting",
			events: []VVGateEvent{
				declareObligation("o1", ObligationKind("vibes"), RiskHigh),
			},
			wantRule: "vv.policy_unknown",
		},
		{
			name: "evidence for an undeclared obligation fails closed",
			events: []VVGateEvent{
				submitEvidence("ghost", RungReplayable),
			},
			wantRule: "vv.obligation_unknown",
		},
		{
			name: "invalid rung fails closed",
			events: []VVGateEvent{
				declareObligation("o1", ObligationBehaviourChange, RiskLow),
				submitEvidence("o1", VVRung(99)),
			},
			wantRule: "vv.rung_invalid",
		},
		{
			name: "escalation after discharge invalidates discharge",
			events: []VVGateEvent{
				declareObligation("o1", ObligationBehaviourChange, RiskLow),
				submitEvidence("o1", RungExample), // satisfies low
				claimDischarged("o1"),
				reclassify("o1", ObligationBehaviourChange, RiskHigh), // requires higher rung
			},
			wantDischarged: false,
		},
		{
			name: "missing obligation ID fails closed",
			events: []VVGateEvent{
				{Kind: VVDeclareObligation, Obligation: ObligationBehaviourChange, Risk: RiskLow},
			},
			wantRule: "vv.obligation_missing",
		},
		{
			name: "declaring an already declared obligation fails closed",
			events: []VVGateEvent{
				declareObligation("o1", ObligationBehaviourChange, RiskLow),
				declareObligation("o1", ObligationBehaviourChange, RiskLow),
			},
			wantRule: "vv.obligation_exists",
		},
		{
			name: "unsupported event kind fails closed",
			events: []VVGateEvent{
				{Kind: VVGateEventKind("model_policy"), ObligationID: "o1"},
			},
			wantRule: "vv.event_unknown",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state := NewVVGateState()
			var gotRule string
			for _, event := range test.events {
				next, violation := ApplyVVGateEvent(state, event)
				if violation != nil {
					gotRule = violation.RuleID
					break
				}
				state = next
			}
			if gotRule != test.wantRule {
				t.Fatalf("rule = %q, want %q", gotRule, test.wantRule)
			}
			if got := state.Obligations["o1"].Discharged; got != test.wantDischarged {
				t.Fatalf("discharged = %v, want %v", got, test.wantDischarged)
			}
		})
	}
}

func TestVVGateDoesNotMutateInputState(t *testing.T) {
	state := NewVVGateState()
	next, violation := ApplyVVGateEvent(state, declareObligation("o1", ObligationRefactor, RiskLow))
	if violation != nil {
		t.Fatal(violation)
	}
	if !reflect.DeepEqual(state, NewVVGateState()) {
		t.Fatalf("input state mutated: %#v", state)
	}
	if _, ok := next.Obligations["o1"]; !ok {
		t.Fatal("next state is missing the declared obligation")
	}
}

func TestVVGateReplayIsByteEquivalent(t *testing.T) {
	trace := []VVGateEvent{
		declareObligation("o1", ObligationSchemaMigration, RiskMedium),
		submitEvidence("o1", RungAdversarial),
		claimDischarged("o1"),
	}
	once := applyVVTrace(t, trace)
	// Re-submitting identical evidence must not change the terminal state.
	replayed := applyVVTrace(t, append(append([]VVGateEvent{}, trace...), submitEvidence("o1", RungAdversarial)))

	onceJSON, err := json.Marshal(once)
	if err != nil {
		t.Fatal(err)
	}
	replayedJSON, err := json.Marshal(replayed)
	if err != nil {
		t.Fatal(err)
	}
	if string(onceJSON) != string(replayedJSON) {
		t.Fatalf("replayed state differs\nwant: %s\n got: %s", onceJSON, replayedJSON)
	}
}

func applyVVTrace(t *testing.T, trace []VVGateEvent) VVGateState {
	t.Helper()
	state := NewVVGateState()
	for _, event := range trace {
		next, violation := ApplyVVGateEvent(state, event)
		if violation != nil {
			t.Fatalf("unexpected violation %v on %s", violation, event.Kind)
		}
		state = next
	}
	return state
}

func declareObligation(id string, kind ObligationKind, risk RiskLevel) VVGateEvent {
	return VVGateEvent{Kind: VVDeclareObligation, ObligationID: id, Obligation: kind, Risk: risk}
}

func reclassify(id string, kind ObligationKind, risk RiskLevel) VVGateEvent {
	return VVGateEvent{Kind: VVReclassify, ObligationID: id, Obligation: kind, Risk: risk}
}

func submitEvidence(id string, rung VVRung) VVGateEvent {
	return VVGateEvent{Kind: VVSubmitEvidence, ObligationID: id, EvidenceRung: rung}
}

func claimDischarged(id string) VVGateEvent {
	return VVGateEvent{Kind: VVClaimDischarged, ObligationID: id}
}
