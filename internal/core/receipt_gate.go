package core

import (
	"fmt"
	"maps"
)

// The receipt gate is the multi-obligation sibling of the control monitor: it
// tracks a target, a receipt, and an accepted claim per obligation ID, with no
// journal sequencing. The control monitor governs one obligation at a time but
// adds ordering, approval, leases, and effects. Both are preregistered
// experiment treatments, so neither subsumes the other; what they genuinely
// share lives in receipt.go.

type ReceiptGateEventKind string

const (
	ReceiptGateSetTarget ReceiptGateEventKind = "set_target"
	ReceiptGateRecord    ReceiptGateEventKind = "record_receipt"
	ReceiptGateClaim     ReceiptGateEventKind = "claim_lifecycle"
)

// ReceiptGateEvent is a deliberately closed event vocabulary for the first
// receipt experiment. Persistence and journal ordering are deferred.
type ReceiptGateEvent struct {
	Kind         ReceiptGateEventKind
	ObligationID string
	Identity     ReceiptIdentity
	Receipt      VerificationReceipt
	Claim        LifecycleClaim
}

// ReceiptGateState keeps only the current identity, current receipt, and most
// recent accepted claim per obligation. It is an immutable snapshot: Apply does
// not mutate its input maps.
type ReceiptGateState struct {
	Targets  map[string]ReceiptIdentity
	Receipts map[string]VerificationReceipt
	Claims   map[string]LifecycleClaim
}

func NewReceiptGateState() ReceiptGateState {
	return ReceiptGateState{
		Targets:  make(map[string]ReceiptIdentity),
		Receipts: make(map[string]VerificationReceipt),
		Claims:   make(map[string]LifecycleClaim),
	}
}

func cloneReceiptGateState(state ReceiptGateState) ReceiptGateState {
	return ReceiptGateState{
		Targets:  cloneOrEmpty(state.Targets),
		Receipts: cloneOrEmpty(state.Receipts),
		Claims:   cloneOrEmpty(state.Claims),
	}
}

// cloneOrEmpty keeps a cloned state usable: maps.Clone returns nil for a nil
// map, and the reducer writes into every field. A zero ReceiptGateState is a
// legitimate input, so the nil case has to survive the copy.
func cloneOrEmpty[K comparable, V any](source map[K]V) map[K]V {
	if source == nil {
		return make(map[K]V)
	}
	return maps.Clone(source)
}

// ApplyReceiptGateEvent applies one closed receipt-gate transition. Invalid or
// stale evidence fails closed and returns the original state unchanged.
func ApplyReceiptGateEvent(state ReceiptGateState, event ReceiptGateEvent) (ReceiptGateState, *GateViolation) {
	next := cloneReceiptGateState(state)

	switch event.Kind {
	case ReceiptGateSetTarget:
		if event.ObligationID == "" || !event.Identity.Valid() {
			return state, gateViolation("receipt.target_invalid", "a target requires an obligation and complete digest identity")
		}
		if current, exists := state.Targets[event.ObligationID]; exists && current.Equal(event.Identity) {
			return state, nil
		}
		next.Targets[event.ObligationID] = event.Identity
		delete(next.Claims, event.ObligationID)
		return next, nil

	case ReceiptGateRecord:
		if !event.Receipt.Valid() {
			return state, gateViolation("receipt.invalid", "receipt body digest or identity is invalid")
		}
		// Receipts are keyed by the obligation sealed into the receipt, which is
		// the same namespace claims look up. An event that names a different
		// obligation than its receipt is a mismatch, not a re-keying.
		if event.ObligationID != "" && event.ObligationID != event.Receipt.Obligation {
			return state, gateViolation("receipt.obligation_mismatch", "event obligation differs from the obligation sealed into the receipt")
		}
		if existing, exists := state.Receipts[event.Receipt.Obligation]; exists && existing.BodyDigest == event.Receipt.BodyDigest {
			return state, nil
		}
		// A replacement receipt is new evidence; any claim accepted on the
		// strength of the previous receipt no longer has a basis.
		delete(next.Claims, event.Receipt.Obligation)
		next.Receipts[event.Receipt.Obligation] = event.Receipt
		return next, nil

	case ReceiptGateClaim:
		if event.ObligationID == "" || !validLifecycleClaim(event.Claim) {
			return state, gateViolation("receipt.claim_invalid", "claim requires a known lifecycle value and obligation")
		}
		target, exists := state.Targets[event.ObligationID]
		if !exists {
			return state, gateViolation("receipt.target_missing", "claim has no current obligation identity")
		}
		receipt, exists := state.Receipts[event.ObligationID]
		if !exists {
			return state, gateViolation("receipt.missing", "claim has no receipt")
		}
		if !receipt.Valid() {
			return state, gateViolation("receipt.invalid", "stored receipt body digest or identity is invalid")
		}
		if receipt.Outcome != ReceiptPassed {
			return state, gateViolation("receipt.failed", "claim requires a passing receipt")
		}
		if !receipt.Identity.Equal(target) {
			return state, gateViolation("receipt.stale", "receipt identity differs from current obligation identity")
		}
		next.Claims[event.ObligationID] = event.Claim
		return next, nil

	default:
		return state, gateViolation("receipt.event_unknown", fmt.Sprintf("unsupported event kind %q", event.Kind))
	}
}
