package core

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"strings"
)

// ReceiptIdentity captures every input whose change invalidates a verification
// result. Each value is a SHA-256 digest so the state machine can compare
// provenance without retaining source, command arguments, or lockfile contents.
type ReceiptIdentity struct {
	ScopeDigest         string
	SourceDigest        string
	VerifierDigest      string
	ArgumentsDigest     string
	ConfigurationDigest string
	RuntimeDigest       string
	LockfilesDigest     string
}

func (identity ReceiptIdentity) Equal(other ReceiptIdentity) bool {
	return identity == other
}

func (identity ReceiptIdentity) Valid() bool {
	return validReceiptDigest(identity.ScopeDigest) &&
		validReceiptDigest(identity.SourceDigest) &&
		validReceiptDigest(identity.VerifierDigest) &&
		validReceiptDigest(identity.ArgumentsDigest) &&
		validReceiptDigest(identity.ConfigurationDigest) &&
		validReceiptDigest(identity.RuntimeDigest) &&
		validReceiptDigest(identity.LockfilesDigest)
}

func (identity ReceiptIdentity) digest() string {
	return receiptDigest(
		"scope", identity.ScopeDigest,
		"source", identity.SourceDigest,
		"verifier", identity.VerifierDigest,
		"arguments", identity.ArgumentsDigest,
		"configuration", identity.ConfigurationDigest,
		"runtime", identity.RuntimeDigest,
		"lockfiles", identity.LockfilesDigest,
	)
}

type ReceiptOutcome string

const (
	ReceiptPassed ReceiptOutcome = "passed"
	ReceiptFailed ReceiptOutcome = "failed"
)

// VerificationReceipt is an immutable evidence envelope. BodyDigest is an
// unkeyed SHA-256 over the obligation, identity, and outcome: it detects
// corruption or partial mutation of a receipt after it was sealed, and nothing
// more. It is not a signature. Anyone can edit a receipt and reseal it, so a
// valid BodyDigest is no evidence that a verifier produced the outcome.
//
// Authenticity is a trust-boundary property that belongs to whoever constructs
// receipts: only a trusted verifier may call SealVerificationReceipt, and any
// receipt that crosses an untrusted boundary needs an authenticated format
// (keyed MAC or signature) that this reducer deliberately does not define.
type VerificationReceipt struct {
	ID         string
	Obligation string
	Identity   ReceiptIdentity
	Outcome    ReceiptOutcome
	BodyDigest string
}

// SealVerificationReceipt computes the integrity digest for a receipt body.
// Callers must treat it as a construction step inside the trusted verifier,
// not as an authenticity check; see VerificationReceipt.
func SealVerificationReceipt(receipt VerificationReceipt) VerificationReceipt {
	receipt.BodyDigest = receipt.bodyDigest()
	return receipt
}

func (receipt VerificationReceipt) Valid() bool {
	if receipt.ID == "" || receipt.Obligation == "" || !receipt.Identity.Valid() {
		return false
	}
	if receipt.Outcome != ReceiptPassed && receipt.Outcome != ReceiptFailed {
		return false
	}
	return receipt.BodyDigest == receipt.bodyDigest()
}

func (receipt VerificationReceipt) bodyDigest() string {
	return receiptDigest(
		"receipt", receipt.ID,
		"obligation", receipt.Obligation,
		"identity", receipt.Identity.digest(),
		"outcome", string(receipt.Outcome),
	)
}

type LifecycleClaim string

const (
	ClaimVerified     LifecycleClaim = "verified"
	ClaimDone         LifecycleClaim = "done"
	ClaimReadyToMerge LifecycleClaim = "ready_to_merge"
)

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

type ReceiptGateViolation struct {
	RuleID  string
	Message string
}

func (violation *ReceiptGateViolation) Error() string {
	return violation.RuleID + ": " + violation.Message
}

// ApplyReceiptGateEvent applies one closed receipt-gate transition. Invalid or
// stale evidence fails closed and returns the original state unchanged.
func ApplyReceiptGateEvent(state ReceiptGateState, event ReceiptGateEvent) (ReceiptGateState, *ReceiptGateViolation) {
	next := cloneReceiptGateState(state)

	switch event.Kind {
	case ReceiptGateSetTarget:
		if event.ObligationID == "" || !event.Identity.Valid() {
			return state, receiptGateViolation("receipt.target_invalid", "a target requires an obligation and complete digest identity")
		}
		if current, exists := next.Targets[event.ObligationID]; exists && current.Equal(event.Identity) {
			return next, nil
		}
		next.Targets[event.ObligationID] = event.Identity
		delete(next.Claims, event.ObligationID)
		return next, nil

	case ReceiptGateRecord:
		if !event.Receipt.Valid() {
			return state, receiptGateViolation("receipt.invalid", "receipt body digest or identity is invalid")
		}
		if existing, exists := next.Receipts[event.Receipt.Obligation]; exists && existing.BodyDigest == event.Receipt.BodyDigest {
			return next, nil
		}
		next.Receipts[event.Receipt.Obligation] = event.Receipt
		return next, nil

	case ReceiptGateClaim:
		if event.ObligationID == "" || !validLifecycleClaim(event.Claim) {
			return state, receiptGateViolation("receipt.claim_invalid", "claim requires a known lifecycle value and obligation")
		}
		target, exists := next.Targets[event.ObligationID]
		if !exists {
			return state, receiptGateViolation("receipt.target_missing", "claim has no current obligation identity")
		}
		receipt, exists := next.Receipts[event.ObligationID]
		if !exists {
			return state, receiptGateViolation("receipt.missing", "claim has no receipt")
		}
		if !receipt.Valid() {
			return state, receiptGateViolation("receipt.invalid", "stored receipt body digest or identity is invalid")
		}
		if receipt.Outcome != ReceiptPassed {
			return state, receiptGateViolation("receipt.failed", "claim requires a passing receipt")
		}
		if !receipt.Identity.Equal(target) {
			return state, receiptGateViolation("receipt.stale", "receipt identity differs from current obligation identity")
		}
		next.Claims[event.ObligationID] = event.Claim
		return next, nil

	default:
		return state, receiptGateViolation("receipt.event_unknown", fmt.Sprintf("unsupported event kind %q", event.Kind))
	}
}

func cloneReceiptGateState(state ReceiptGateState) ReceiptGateState {
	next := NewReceiptGateState()
	for key, value := range state.Targets {
		next.Targets[key] = value
	}
	for key, value := range state.Receipts {
		next.Receipts[key] = value
	}
	for key, value := range state.Claims {
		next.Claims[key] = value
	}
	return next
}

func validLifecycleClaim(claim LifecycleClaim) bool {
	return claim == ClaimVerified || claim == ClaimDone || claim == ClaimReadyToMerge
}

func receiptGateViolation(ruleID, message string) *ReceiptGateViolation {
	return &ReceiptGateViolation{RuleID: ruleID, Message: message}
}

func validReceiptDigest(value string) bool {
	const prefix = "sha256:"
	if !strings.HasPrefix(value, prefix) || len(value) != len(prefix)+sha256.Size*2 {
		return false
	}
	encoded := value[len(prefix):]
	if encoded != strings.ToLower(encoded) {
		return false
	}
	_, err := hex.DecodeString(encoded)
	return err == nil
}

func receiptDigest(parts ...string) string {
	hash := sha256.New()
	for _, part := range parts {
		var length [8]byte
		binary.BigEndian.PutUint64(length[:], uint64(len(part)))
		_, _ = hash.Write(length[:])
		_, _ = io.WriteString(hash, part)
	}
	return "sha256:" + hex.EncodeToString(hash.Sum(nil))
}
