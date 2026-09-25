package core

import "fmt"

// VVRung is the verification-and-validation strength of a piece of evidence.
//
// Rungs are ordered by *what class of defect the evidence can catch*, not by
// what the test is called or which file it lives in. A mock-heavy test that
// asserts a call sequence restates the implementation rather than observing
// behaviour, so it does not reach RungExample no matter what its filename is.
//
// ARCHITECTURE.md §5 gives `Obligation` a "minimum V&V rung" and §10
// invariant 7 says a model may escalate but never downgrade it. This type is
// the first concrete definition of that ladder; see
// docs/plans/2026-09-20-test-quality-and-architectural-fitness-steering.md §4.
type VVRung int

const (
	// RungBuild: it compiles and typechecks. No behavioural claim at all.
	RungBuild VVRung = iota
	// RungExample: an assertion on observable output or state at a declared
	// boundary. Catches the case the author thought of.
	RungExample
	// RungProperty: an invariant checked over generated inputs. Catches cases
	// within the input domain that nobody enumerated.
	RungProperty
	// RungReachability: RungProperty plus evidence that the changed code was
	// actually exercised. Catches "the test ran but never touched the change".
	RungReachability
	// RungAdversarial: RungReachability plus faults or adversarial inputs.
	// Catches error-path and resource-exhaustion defects.
	RungAdversarial
	// RungReplayable: RungAdversarial plus a seeded, deterministically
	// replayable counterexample. Makes findings reproducible.
	RungReplayable
)

var rungNames = map[VVRung]string{
	RungBuild:        "build",
	RungExample:      "example",
	RungProperty:     "property",
	RungReachability: "reachability",
	RungAdversarial:  "adversarial",
	RungReplayable:   "replayable",
}

func (rung VVRung) String() string {
	if name, ok := rungNames[rung]; ok {
		return name
	}
	return fmt.Sprintf("vvrung(%d)", int(rung))
}

func (rung VVRung) Valid() bool {
	_, ok := rungNames[rung]
	return ok
}

// ObligationKind and RiskLevel are closed vocabularies. They are inputs to the
// hand-authored policy table below; a model may not extend either, because
// introducing a new kind would let it route around the table entirely.
type ObligationKind string

const (
	ObligationBehaviourChange ObligationKind = "behaviour_change"
	ObligationRefactor        ObligationKind = "refactor"
	ObligationDependencyBump  ObligationKind = "dependency_bump"
	ObligationSchemaMigration ObligationKind = "schema_migration"
	ObligationDocumentation   ObligationKind = "documentation"
)

type RiskLevel string

const (
	RiskLow    RiskLevel = "low"
	RiskMedium RiskLevel = "medium"
	RiskHigh   RiskLevel = "high"
)

var riskOrder = map[RiskLevel]int{RiskLow: 0, RiskMedium: 1, RiskHigh: 2}

// minimumRungPolicy is closed and hand-authored, per ARCHITECTURE.md §6: the
// model may explain these rows but cannot create or weaken them. An absent
// (kind, risk) pair is not a default — it is a failure, so that adding a new
// obligation kind forces an explicit policy decision rather than silently
// inheriting the weakest rung.
var minimumRungPolicy = map[ObligationKind]map[RiskLevel]VVRung{
	ObligationBehaviourChange: {
		RiskLow:    RungExample,
		RiskMedium: RungProperty,
		RiskHigh:   RungReachability,
	},
	ObligationRefactor: {
		// A refactor asserts behaviour did not change, which example-based
		// tests are weak at: they pass precisely because they encode the old
		// shape. Properties are the floor even at low risk.
		RiskLow:    RungProperty,
		RiskMedium: RungProperty,
		RiskHigh:   RungReachability,
	},
	ObligationDependencyBump: {
		RiskLow:    RungExample,
		RiskMedium: RungProperty,
		RiskHigh:   RungAdversarial,
	},
	ObligationSchemaMigration: {
		// Migrations are hard to reverse; ARCHITECTURE.md §8 calls state
		// formats outliving implementations. Replayability is the floor at
		// high risk so a failure can be reproduced exactly.
		RiskLow:    RungProperty,
		RiskMedium: RungAdversarial,
		RiskHigh:   RungReplayable,
	},
	ObligationDocumentation: {
		RiskLow:    RungBuild,
		RiskMedium: RungBuild,
		RiskHigh:   RungExample,
	},
}

// MinimumRung reports the required rung for an obligation. It is a pure
// function of the closed policy table: there is deliberately no event that
// sets a minimum rung directly, which makes "a model may never downgrade it"
// structurally true rather than a rule that must be separately enforced.
func MinimumRung(kind ObligationKind, risk RiskLevel) (VVRung, bool) {
	byRisk, ok := minimumRungPolicy[kind]
	if !ok {
		return 0, false
	}
	rung, ok := byRisk[risk]
	return rung, ok
}

type VVGateEventKind string

const (
	VVDeclareObligation VVGateEventKind = "declare_obligation"
	VVReclassify        VVGateEventKind = "reclassify_obligation"
	VVSubmitEvidence    VVGateEventKind = "submit_evidence"
	VVClaimDischarged   VVGateEventKind = "claim_discharged"
)

// VVGateEvent is a closed event vocabulary. Persistence and journal ordering
// are deferred, matching the scope of the other reducers in this package.
type VVGateEvent struct {
	Kind         VVGateEventKind
	ObligationID string
	Obligation   ObligationKind
	Risk         RiskLevel
	EvidenceRung VVRung
}

type ObligationRecord struct {
	Kind        ObligationKind
	Risk        RiskLevel
	MinimumRung VVRung
	SatisfiedAt VVRung
	Satisfied   bool
	Discharged  bool
}

// VVGateState is an immutable snapshot; Apply does not mutate its input map.
type VVGateState struct {
	Obligations map[string]ObligationRecord
}

func NewVVGateState() VVGateState {
	return VVGateState{Obligations: make(map[string]ObligationRecord)}
}

type VVGateViolation struct {
	RuleID  string
	Message string
}

func (violation *VVGateViolation) Error() string {
	return violation.RuleID + ": " + violation.Message
}

func vvViolation(ruleID, message string) *VVGateViolation {
	return &VVGateViolation{RuleID: ruleID, Message: message}
}

func cloneVVGateState(state VVGateState) VVGateState {
	next := NewVVGateState()
	for key, value := range state.Obligations {
		next.Obligations[key] = value
	}
	return next
}

// ApplyVVGateEvent applies one closed V&V-gate transition. Unknown policy,
// risk downgrade, and under-rung discharge all fail closed and return the
// original state unchanged.
func ApplyVVGateEvent(state VVGateState, event VVGateEvent) (VVGateState, *VVGateViolation) {
	if event.ObligationID == "" {
		return state, vvViolation("vv.obligation_missing", "every event must name an obligation")
	}
	next := cloneVVGateState(state)

	switch event.Kind {
	case VVDeclareObligation:
		if _, exists := next.Obligations[event.ObligationID]; exists {
			return state, vvViolation("vv.obligation_exists", "obligation is already declared")
		}
		minimum, ok := MinimumRung(event.Obligation, event.Risk)
		if !ok {
			return state, vvViolation("vv.policy_unknown",
				fmt.Sprintf("no hand-authored policy for kind %q at risk %q; add a policy row rather than defaulting",
					event.Obligation, event.Risk))
		}
		next.Obligations[event.ObligationID] = ObligationRecord{
			Kind: event.Obligation, Risk: event.Risk, MinimumRung: minimum,
		}
		return next, nil

	case VVReclassify:
		record, exists := next.Obligations[event.ObligationID]
		if !exists {
			return state, vvViolation("vv.obligation_unknown", "cannot reclassify an undeclared obligation")
		}
		minimum, ok := MinimumRung(event.Obligation, event.Risk)
		if !ok {
			return state, vvViolation("vv.policy_unknown", "no hand-authored policy for the requested classification")
		}
		// Reclassification is the real downgrade vector: lowering risk, or
		// switching to a laxer kind, would lower the bar after the fact.
		// Invariant 7 permits escalation only.
		if riskOrder[event.Risk] < riskOrder[record.Risk] {
			return state, vvViolation("vv.risk_downgrade",
				fmt.Sprintf("risk may be escalated but never downgraded (%s -> %s)", record.Risk, event.Risk))
		}
		if minimum < record.MinimumRung {
			return state, vvViolation("vv.rung_downgrade",
				fmt.Sprintf("reclassification would lower the required rung (%s -> %s)",
					record.MinimumRung, minimum))
		}
		record.Kind, record.Risk, record.MinimumRung = event.Obligation, event.Risk, minimum
		// A raised bar invalidates evidence that only cleared the old one.
		if record.Satisfied && record.SatisfiedAt < minimum {
			record.Satisfied = false
			record.Discharged = false // a raised bar invalidates the discharge too
		}
		next.Obligations[event.ObligationID] = record
		return next, nil

	case VVSubmitEvidence:
		record, exists := next.Obligations[event.ObligationID]
		if !exists {
			return state, vvViolation("vv.obligation_unknown", "cannot submit evidence for an undeclared obligation")
		}
		if !event.EvidenceRung.Valid() {
			return state, vvViolation("vv.rung_invalid", "evidence rung is outside the closed ladder")
		}
		if event.EvidenceRung < record.MinimumRung {
			return state, vvViolation("vv.rung_insufficient",
				fmt.Sprintf("evidence at rung %s does not satisfy required rung %s for %s/%s",
					event.EvidenceRung, record.MinimumRung, record.Kind, record.Risk))
		}
		// Escalation is explicitly allowed: keep the strongest evidence seen.
		if !record.Satisfied || event.EvidenceRung > record.SatisfiedAt {
			record.SatisfiedAt = event.EvidenceRung
		}
		record.Satisfied = true
		next.Obligations[event.ObligationID] = record
		return next, nil

	case VVClaimDischarged:
		record, exists := next.Obligations[event.ObligationID]
		if !exists {
			return state, vvViolation("vv.obligation_unknown", "cannot discharge an undeclared obligation")
		}
		if !record.Satisfied {
			return state, vvViolation("vv.undischarged",
				fmt.Sprintf("obligation requires rung %s and has no satisfying evidence", record.MinimumRung))
		}
		record.Discharged = true
		next.Obligations[event.ObligationID] = record
		return next, nil

	default:
		return state, vvViolation("vv.event_unknown", fmt.Sprintf("unsupported event kind %q", event.Kind))
	}
}
