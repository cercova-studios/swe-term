package core

// ControlRuleID is a stable, machine-readable reason for a rejected event.
// Rule IDs deliberately name invariants rather than the reducer's implementation.
type ControlRuleID string

const (
	ControlAccepted                  ControlRuleID = ""
	ControlEventInvalid              ControlRuleID = "control.event.invalid"
	ControlEventSequence             ControlRuleID = "control.event.sequence"
	ControlEventSchemaUnsupported    ControlRuleID = "control.event.schema_unsupported"
	ControlApprovalRequired          ControlRuleID = "control.approval.required"
	ControlLeaseConflict             ControlRuleID = "control.lease.conflict"
	ControlLeaseRequired             ControlRuleID = "control.lease.required"
	ControlEffectDeclarationInvalid  ControlRuleID = "control.effect.declaration_invalid"
	ControlEffectUndeclared          ControlRuleID = "control.effect.undeclared"
	ControlReceiptInvalid            ControlRuleID = "control.receipt.invalid"
	ControlReceiptObligationMismatch ControlRuleID = "control.receipt.obligation_mismatch"
	ControlReceiptStale              ControlRuleID = "control.receipt.stale"
	ControlLifecycleReceiptRequired  ControlRuleID = "control.lifecycle.receipt_required"
	ControlLifecycleReceiptFailed    ControlRuleID = "control.lifecycle.receipt_failed"
	ControlLifecycleCancelled        ControlRuleID = "control.lifecycle.cancelled"
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
	Declared      EffectDeclaration
	Observed      ObservedEffect
	Obligation    string
	Identity      ReceiptIdentity
	Receipt       VerificationReceipt
	Claim         LifecycleClaim
}

// ControlMonitorState is a deliberately bounded in-memory monitor. The durable
// journal owns history and duplicate detection beyond the immediately replayed
// event; this state stores only the active authorization, lease, declarations,
// receipt target, and last accepted event.
//
// The monitor is scoped to one receipt obligation at a time: ReceiptObligation
// names the check a lifecycle claim must be backed by, and ReceiptTarget is
// the identity that check must have run against. A receipt for a different
// obligation never satisfies the target, even when its digests match.
type ControlMonitorState struct {
	LastSequence uint64
	LastEvent    ControlEvent

	ApprovedAction string
	ActiveLease    string
	Declared       EffectDeclaration

	ReceiptObligation string
	ReceiptTarget     ReceiptIdentity
	CurrentReceipt    VerificationReceipt
	Cancelled         bool
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
	if event.SchemaVersion != controlMonitorSchemaVersion {
		return state, RejectedControlDecision(ControlEventSchemaUnsupported)
	}
	if event.Sequence == 0 {
		return state, RejectedControlDecision(ControlEventSequence)
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
		if event.Obligation == "" || !event.Identity.Valid() {
			decision = RejectedControlDecision(ControlReceiptInvalid)
			break
		}
		// Re-declaring the same target is not a change of verification
		// inputs, so still-current evidence survives it.
		if next.ReceiptObligation != event.Obligation || !next.ReceiptTarget.Equal(event.Identity) {
			next.CurrentReceipt = VerificationReceipt{}
		}
		next.ReceiptObligation = event.Obligation
		next.ReceiptTarget = event.Identity
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
		if event.Lease == "" || event.Lease != next.ActiveLease {
			decision = RejectedControlDecision(ControlLeaseRequired)
			break
		}
		if !event.Declared.Validate() {
			decision = RejectedControlDecision(ControlEffectDeclarationInvalid)
			break
		}
		next.Declared = copyEffectDeclaration(event.Declared)
	case ControlEffectObserved:
		if event.Lease == "" || event.Lease != next.ActiveLease {
			decision = RejectedControlDecision(ControlLeaseRequired)
			break
		}
		if !next.Declared.Covers(event.Observed) {
			decision = RejectedControlDecision(ControlEffectUndeclared)
			break
		}
		// The monitor cannot tell whether a mutating effect touched the
		// verified scope, so an observed write, process, or unobservable
		// effect fails closed: the receipt is no longer current and a fresh
		// one must be recorded before a claim. A declared read within the
		// lease does not clobber an already-recorded receipt.
		if mutatingObserved(event.Observed) {
			next.CurrentReceipt = VerificationReceipt{}
		}
	case ControlReceiptRecorded:
		// A valid failed receipt is evidence too: it is retained so the
		// failure stays auditable, and the lifecycle rule below refuses it.
		if !event.Receipt.Valid() {
			decision = RejectedControlDecision(ControlReceiptInvalid)
			break
		}
		if next.ReceiptObligation == "" || event.Receipt.Obligation != next.ReceiptObligation {
			decision = RejectedControlDecision(ControlReceiptObligationMismatch)
			break
		}
		if !event.Receipt.Identity.Equal(next.ReceiptTarget) {
			decision = RejectedControlDecision(ControlReceiptStale)
			break
		}
		next.CurrentReceipt = event.Receipt
	case ControlLifecycleClaimed:
		if !validLifecycleClaim(event.Claim) {
			decision = RejectedControlDecision(ControlEventInvalid)
			break
		}
		if next.Cancelled {
			decision = RejectedControlDecision(ControlLifecycleCancelled)
			break
		}
		if !next.CurrentReceipt.Valid() || next.CurrentReceipt.Obligation != next.ReceiptObligation || !next.CurrentReceipt.Identity.Equal(next.ReceiptTarget) {
			decision = RejectedControlDecision(ControlLifecycleReceiptRequired)
			break
		}
		if next.CurrentReceipt.Outcome != ReceiptPassed {
			decision = RejectedControlDecision(ControlLifecycleReceiptFailed)
			break
		}
	case ControlLeaseReleased:
		if event.Lease == "" || event.Lease != next.ActiveLease {
			decision = RejectedControlDecision(ControlLeaseRequired)
			break
		}
		next.ActiveLease = ""
		next.Declared = EffectDeclaration{}
	case ControlCancelled:
		if next.ActiveLease != "" && event.Lease != next.ActiveLease {
			decision = RejectedControlDecision(ControlLeaseRequired)
			break
		}
		next.ApprovedAction = ""
		next.ActiveLease = ""
		next.Declared = EffectDeclaration{}
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

// mutatingObserved reports whether an accepted observed effect can have
// changed the verified scope: a write, a spawned process, or any effect the
// harness could not observe.
func mutatingObserved(observed ObservedEffect) bool {
	return observed.Class == EffectClassWrite || observed.Class == EffectClassProcess ||
		observed.Observation == ObservationUnobservable
}

func controlEventsEqual(left, right ControlEvent) bool {
	return left.SchemaVersion == right.SchemaVersion &&
		left.Sequence == right.Sequence &&
		left.Kind == right.Kind &&
		left.Action == right.Action &&
		left.Lease == right.Lease &&
		left.Declared.Equal(right.Declared) &&
		left.Observed == right.Observed &&
		left.Obligation == right.Obligation &&
		left.Identity.Equal(right.Identity) &&
		left.Receipt == right.Receipt &&
		left.Claim == right.Claim
}

func copyControlEvent(event ControlEvent) ControlEvent {
	event.Declared = copyEffectDeclaration(event.Declared)
	return event
}

func copyEffectDeclaration(declaration EffectDeclaration) EffectDeclaration {
	declaration.Reads = append([]PathEffect(nil), declaration.Reads...)
	declaration.Writes = append([]PathEffect(nil), declaration.Writes...)
	declaration.Processes = append([]string(nil), declaration.Processes...)
	declaration.Network = append([]string(nil), declaration.Network...)
	declaration.Secrets = append([]string(nil), declaration.Secrets...)
	return declaration
}
