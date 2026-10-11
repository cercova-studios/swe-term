package core

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"io"
	"strings"
)

// Evidence primitives shared by every gate in this package. They live here
// rather than inside one reducer because no reducer owns them: the receipt
// gate, the control monitor, and the V&V ladder all compare the same
// identities and the same sealed receipts.

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

func validLifecycleClaim(claim LifecycleClaim) bool {
	return claim == ClaimVerified || claim == ClaimDone || claim == ClaimReadyToMerge
}

// GateViolation is the refusal shape for the map-keyed gates (receipt and V&V).
// The rule ID names the invariant, not the reducer's implementation, so it is
// part of each gate's external contract. The control monitor reports refusals
// as a ControlDecision instead, because it must also accept an idempotent
// replay — a case that has no violation to return.
type GateViolation struct {
	RuleID  string
	Message string
}

func (violation *GateViolation) Error() string {
	return violation.RuleID + ": " + violation.Message
}

func gateViolation(ruleID, message string) *GateViolation {
	return &GateViolation{RuleID: ruleID, Message: message}
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
