package core

import "reflect"

// ControlRuleID is a stable, machine-readable reason for a rejected event.
// Rule IDs deliberately name invariants rather than the reducer's implementation.
type ControlRuleID string

const (
	ControlAccepted                 ControlRuleID = ""
	ControlEventInvalid             ControlRuleID = "control.event.invalid"
	ControlEventSequence            ControlRuleID = "control.event.sequence"
	ControlEventSchemaUnsupported   ControlRuleID = "control.event.schema_unsupported"
	ControlApprovalRequired         ControlRuleID = "control.approval.required"
	ControlLeaseConflict            ControlRuleID = "control.lease.conflict"
	ControlLeaseRequired            ControlRuleID = "control.lease.required"
	ControlEffectDeclarationInvalid ControlRuleID = "control.effect.declaration_invalid"
	ControlEffectUndeclared         ControlRuleID = "control.effect.undeclared"
	ControlReceiptInvalid           ControlRuleID = "control.receipt.invalid"
	ControlReceiptStale             ControlRuleID = "control.receipt.stale"
	ControlLifecycleReceiptRequired ControlRuleID = "control.lifecycle.receipt_required"
	ControlLifecycleCancelled       ControlRuleID = "control.lifecycle.cancelled"
)

// ControlEventKind is intentionally closed. Extending the journal protocol
// requires a reducer change and an explicit schema version, rather than an
// interpretation authored by a model at runtime.
type ControlEventKind string

const (
	ControlSetReceiptTarget ControlEventKind = "set_receipt_target"
	ControlApprovalGranted  ControlEventKind = "approval_granted"
	ControlLeaseAcquired    ControlEventKind = "lease_acquired"
	ControlEffectsDeclared  ControlEventKind = "effects_declared"
	ControlEffectObserved   ControlEventKind = "effect_observed"
	ControlReceiptRecorded  ControlEventKind = "receipt_recorded"
	ControlLifecycleClaimed ControlEventKind = "lifecycle_claimed"
	ControlLeaseReleased    ControlEventKind = "lease_released"
	ControlCancelled        ControlEventKind = "cancelled"
)

const (
	controlMonitorSchemaVersion = 1
	maxDeclaredEffects          = 32
)

// ControlEvent is one ordered journal entry. Sequence is supplied by the
// journal; replaying an event with the same sequence and content is idempotent.
// Only fields relevant to a given Kind are consulted by the reducer.
type ControlEvent struct {
	SchemaVersion int
	Sequence      uint64
	Kind          ControlEventKind
	Action        string
	Lease         string
	Effect        string
	Effects       []string
	Identity      ReceiptIdentity
	Receipt       VerificationReceipt
	Claim         LifecycleClaim
}

// ControlMonitorState is a deliberately bounded in-memory monitor. The durable
// journal owns history and duplicate detection beyond the immediately replayed
// event; this state stores only the active authorization, lease, declarations,
// receipt target, and last accepted event.
type ControlMonitorState struct {
	LastSequence uint64
	LastEvent    ControlEvent

	ApprovedAction  string
	ActiveLease     string
	DeclaredEffects []string

	ReceiptTarget  ReceiptIdentity
	CurrentReceipt VerificationReceipt
	Cancelled      bool
}

// ControlDecision describes the result without putting failure into mutable
// state. A rejected event leaves the preceding state intact, which makes an
// interrupted or replayed trace easy to reason about.
type ControlDecision struct {
	Accepted bool
	RuleID   ControlRuleID
}

func AcceptedControlDecision() ControlDecision {
	return ControlDecision{Accepted: true}
}

func RejectedControlDecision(ruleID ControlRuleID) ControlDecision {
	return ControlDecision{RuleID: ruleID}
}

// ApplyControlEvent enforces the four experimental ordering rules: approval
// before governed mutation, a single lease, declared effects, and a current
// receipt before a lifecycle claim. It is a pure reducer; it does not execute
// tools, persist a journal, or interpret policy text.
func ApplyControlEvent(state ControlMonitorState, event ControlEvent) (ControlMonitorState, ControlDecision) {
	if event.SchemaVersion != controlMonitorSchemaVersion || event.Sequence == 0 {
		return state, RejectedControlDecision(ControlEventSchemaUnsupported)
	}

	if event.Sequence == state.LastSequence {
		if controlEventsEqual(event, state.LastEvent) {
			return state, AcceptedControlDecision()
		}
		return state, RejectedControlDecision(ControlEventSequence)
	}
	if event.Sequence != state.LastSequence+1 {
		return state, RejectedControlDecision(ControlEventSequence)
	}

	next := state
	decision := AcceptedControlDecision()

	switch event.Kind {
	case ControlSetReceiptTarget:
		if !event.Identity.Valid() {
			decision = RejectedControlDecision(ControlReceiptInvalid)
			break
		}
		next.ReceiptTarget = event.Identity
		next.CurrentReceipt = VerificationReceipt{}
	case ControlApprovalGranted:
		if event.Action == "" || next.ActiveLease != "" || next.Cancelled {
			decision = RejectedControlDecision(ControlEventInvalid)
			break
		}
		next.ApprovedAction = event.Action
	case ControlLeaseAcquired:
		if event.Lease == "" || event.Action == "" || next.Cancelled {
			decision = RejectedControlDecision(ControlEventInvalid)
			break
		}
		if next.ActiveLease != "" {
			decision = RejectedControlDecision(ControlLeaseConflict)
			break
		}
		if next.ApprovedAction != event.Action {
			decision = RejectedControlDecision(ControlApprovalRequired)
			break
		}
		next.ActiveLease = event.Lease
		next.ApprovedAction = ""
	case ControlEffectsDeclared:
		if event.Lease == "" || event.Lease != next.ActiveLease || !validDeclaredEffects(event.Effects) {
			decision = RejectedControlDecision(ControlEffectDeclarationInvalid)
			break
		}
		next.DeclaredEffects = append([]string(nil), event.Effects...)
	case ControlEffectObserved:
		if event.Lease == "" || event.Lease != next.ActiveLease {
			decision = RejectedControlDecision(ControlLeaseRequired)
			break
		}
		if !containsEffect(next.DeclaredEffects, event.Effect) {
			decision = RejectedControlDecision(ControlEffectUndeclared)
			break
		}
	case ControlReceiptRecorded:
		if !event.Receipt.Valid() || event.Receipt.Outcome != ReceiptPassed {
			decision = RejectedControlDecision(ControlReceiptInvalid)
			break
		}
		if !next.ReceiptTarget.Valid() || !event.Receipt.Identity.Equal(next.ReceiptTarget) {
			decision = RejectedControlDecision(ControlReceiptStale)
			break
		}
		next.CurrentReceipt = event.Receipt
	case ControlLifecycleClaimed:
		if event.Claim != ClaimVerified && event.Claim != ClaimDone && event.Claim != ClaimReadyToMerge {
			decision = RejectedControlDecision(ControlEventInvalid)
			break
		}
		if next.Cancelled {
			decision = RejectedControlDecision(ControlLifecycleCancelled)
			break
		}
		if !next.CurrentReceipt.Valid() || next.CurrentReceipt.Outcome != ReceiptPassed || !next.CurrentReceipt.Identity.Equal(next.ReceiptTarget) {
			decision = RejectedControlDecision(ControlLifecycleReceiptRequired)
			break
		}
	case ControlLeaseReleased:
		if event.Lease == "" || event.Lease != next.ActiveLease {
			decision = RejectedControlDecision(ControlLeaseRequired)
			break
		}
		next.ActiveLease = ""
		next.DeclaredEffects = nil
	case ControlCancelled:
		if next.ActiveLease != "" && event.Lease != next.ActiveLease {
			decision = RejectedControlDecision(ControlLeaseRequired)
			break
		}
		next.ApprovedAction = ""
		next.ActiveLease = ""
		next.DeclaredEffects = nil
		next.Cancelled = true
	default:
		decision = RejectedControlDecision(ControlEventSchemaUnsupported)
	}

	if !decision.Accepted {
		return state, decision
	}
	next.LastSequence = event.Sequence
	next.LastEvent = copyControlEvent(event)
	return next, decision
}

func validDeclaredEffects(effects []string) bool {
	if len(effects) == 0 || len(effects) > maxDeclaredEffects {
		return false
	}
	for index, effect := range effects {
		if effect == "" || containsEffect(effects[:index], effect) {
			return false
		}
	}
	return true
}

func containsEffect(effects []string, target string) bool {
	for _, effect := range effects {
		if effect == target {
			return true
		}
	}
	return false
}

func controlEventsEqual(left, right ControlEvent) bool {
	return left.SchemaVersion == right.SchemaVersion &&
		left.Sequence == right.Sequence &&
		left.Kind == right.Kind &&
		left.Action == right.Action &&
		left.Lease == right.Lease &&
		left.Effect == right.Effect &&
		left.Identity.Equal(right.Identity) &&
		left.Receipt == right.Receipt &&
		left.Claim == right.Claim &&
		reflect.DeepEqual(left.Effects, right.Effects)
}

func copyControlEvent(event ControlEvent) ControlEvent {
	event.Effects = append([]string(nil), event.Effects...)
	return event
}
