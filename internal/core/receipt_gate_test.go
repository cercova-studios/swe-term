package core

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"reflect"
	"testing"
)

func TestReceiptGateControlTrace(t *testing.T) {
	baseline := testReceiptIdentity()
	receipt := testReceipt(baseline, ReceiptPassed)

	stale := receipt
	stale.Identity.SourceDigest = testDigest("source-changed")
	if !resultOnlyControlAllows(stale) {
		t.Fatal("result-only control must expose the stale-receipt false-promotion baseline")
	}

	tampered := receipt
	tampered.Identity.ConfigurationDigest = testDigest("configuration-tampered")
	if !resultOnlyControlAllows(tampered) {
		t.Fatal("result-only control must expose the tampered-receipt false-promotion baseline")
	}
}

func TestReceiptGateTraces(t *testing.T) {
	baseline := testReceiptIdentity()
	passed := testReceipt(baseline, ReceiptPassed)
	failed := testReceipt(baseline, ReceiptFailed)
	tampered := passed
	tampered.Identity.ConfigurationDigest = testDigest("configuration-tampered")

	tests := []struct {
		name      string
		events    []ReceiptGateEvent
		wantRule  string
		wantClaim LifecycleClaim
	}{
		{
			name: "fresh passing receipt permits lifecycle claim",
			events: []ReceiptGateEvent{
				setTarget("build", baseline),
				recordReceipt(passed),
				claimLifecycle("build", ClaimDone),
			},
			wantClaim: ClaimDone,
		},
		{
			name: "missing receipt fails closed",
			events: []ReceiptGateEvent{
				setTarget("build", baseline),
				claimLifecycle("build", ClaimDone),
			},
			wantRule: "receipt.missing",
		},
		{
			name: "failed receipt fails closed",
			events: []ReceiptGateEvent{
				setTarget("build", baseline),
				recordReceipt(failed),
				claimLifecycle("build", ClaimDone),
			},
			wantRule: "receipt.failed",
		},
		{
			name: "source change inside scope invalidates receipt",
			events: []ReceiptGateEvent{
				setTarget("build", baseline),
				recordReceipt(passed),
				setTarget("build", withSource(baseline, "source-changed")),
				claimLifecycle("build", ClaimDone),
			},
			wantRule: "receipt.stale",
		},
		{
			name: "verifier change invalidates receipt",
			events: []ReceiptGateEvent{
				setTarget("build", baseline),
				recordReceipt(passed),
				setTarget("build", withVerifier(baseline, "verifier-changed")),
				claimLifecycle("build", ClaimDone),
			},
			wantRule: "receipt.stale",
		},
		{
			name: "verifier arguments change invalidates receipt",
			events: []ReceiptGateEvent{
				setTarget("build", baseline),
				recordReceipt(passed),
				setTarget("build", withArguments(baseline, "arguments-changed")),
				claimLifecycle("build", ClaimDone),
			},
			wantRule: "receipt.stale",
		},
		{
			name: "configuration change invalidates receipt",
			events: []ReceiptGateEvent{
				setTarget("build", baseline),
				recordReceipt(passed),
				setTarget("build", withConfiguration(baseline, "configuration-changed")),
				claimLifecycle("build", ClaimDone),
			},
			wantRule: "receipt.stale",
		},
		{
			name: "runtime change invalidates receipt",
			events: []ReceiptGateEvent{
				setTarget("build", baseline),
				recordReceipt(passed),
				setTarget("build", withRuntime(baseline, "runtime-changed")),
				claimLifecycle("build", ClaimDone),
			},
			wantRule: "receipt.stale",
		},
		{
			name: "lockfile change invalidates receipt",
			events: []ReceiptGateEvent{
				setTarget("build", baseline),
				recordReceipt(passed),
				setTarget("build", withLockfiles(baseline, "lockfiles-changed")),
				claimLifecycle("build", ClaimDone),
			},
			wantRule: "receipt.stale",
		},
		{
			name: "tampered receipt body fails closed",
			events: []ReceiptGateEvent{
				setTarget("build", baseline),
				recordReceipt(tampered),
			},
			wantRule: "receipt.invalid",
		},
		{
			name: "unchanged scope preserves valid receipt",
			events: []ReceiptGateEvent{
				setTarget("build", baseline),
				recordReceipt(passed),
				setTarget("build", baseline),
				claimLifecycle("build", ClaimVerified),
			},
			wantClaim: ClaimVerified,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state := NewReceiptGateState()
			var gotRule string
			for _, event := range test.events {
				next, violation := ApplyReceiptGateEvent(state, event)
				if violation != nil {
					gotRule = violation.RuleID
					break
				}
				state = next
			}
			if gotRule != test.wantRule {
				t.Fatalf("rule = %q, want %q", gotRule, test.wantRule)
			}
			if test.wantClaim != "" && state.Claims["build"] != test.wantClaim {
				t.Fatalf("claim = %q, want %q", state.Claims["build"], test.wantClaim)
			}
		})
	}
}

func TestReceiptGateReplayIsByteEquivalent(t *testing.T) {
	identity := testReceiptIdentity()
	receipt := testReceipt(identity, ReceiptPassed)
	trace := []ReceiptGateEvent{
		setTarget("build", identity),
		recordReceipt(receipt),
		claimLifecycle("build", ClaimDone),
	}

	once := applyTrace(t, trace)
	replayed := applyTrace(t, append([]ReceiptGateEvent{setTarget("build", identity)}, trace...))

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

func TestReceiptGateDoesNotMutateInputState(t *testing.T) {
	identity := testReceiptIdentity()
	state := NewReceiptGateState()
	next, violation := ApplyReceiptGateEvent(state, setTarget("build", identity))
	if violation != nil {
		t.Fatal(violation)
	}
	if !reflect.DeepEqual(state, NewReceiptGateState()) {
		t.Fatalf("input state mutated: %#v", state)
	}
	if !next.Targets["build"].Equal(identity) {
		t.Fatal("next state is missing target")
	}
}

func applyTrace(t *testing.T, trace []ReceiptGateEvent) ReceiptGateState {
	t.Helper()
	state := NewReceiptGateState()
	for _, event := range trace {
		next, violation := ApplyReceiptGateEvent(state, event)
		if violation != nil {
			t.Fatal(violation)
		}
		state = next
	}
	return state
}

func setTarget(obligation string, identity ReceiptIdentity) ReceiptGateEvent {
	return ReceiptGateEvent{Kind: ReceiptGateSetTarget, ObligationID: obligation, Identity: identity}
}

func recordReceipt(receipt VerificationReceipt) ReceiptGateEvent {
	return ReceiptGateEvent{Kind: ReceiptGateRecord, Receipt: receipt}
}

func claimLifecycle(obligation string, claim LifecycleClaim) ReceiptGateEvent {
	return ReceiptGateEvent{Kind: ReceiptGateClaim, ObligationID: obligation, Claim: claim}
}

func testReceipt(identity ReceiptIdentity, outcome ReceiptOutcome) VerificationReceipt {
	return SealVerificationReceipt(VerificationReceipt{
		ID:         "receipt-build-1",
		Obligation: "build",
		Identity:   identity,
		Outcome:    outcome,
	})
}

func testReceiptIdentity() ReceiptIdentity {
	return ReceiptIdentity{
		ScopeDigest:         testDigest("scope"),
		SourceDigest:        testDigest("source"),
		VerifierDigest:      testDigest("verifier"),
		ArgumentsDigest:     testDigest("arguments"),
		ConfigurationDigest: testDigest("configuration"),
		RuntimeDigest:       testDigest("runtime"),
		LockfilesDigest:     testDigest("lockfiles"),
	}
}

func withSource(identity ReceiptIdentity, value string) ReceiptIdentity {
	identity.SourceDigest = testDigest(value)
	return identity
}

func withVerifier(identity ReceiptIdentity, value string) ReceiptIdentity {
	identity.VerifierDigest = testDigest(value)
	return identity
}

func withArguments(identity ReceiptIdentity, value string) ReceiptIdentity {
	identity.ArgumentsDigest = testDigest(value)
	return identity
}

func withConfiguration(identity ReceiptIdentity, value string) ReceiptIdentity {
	identity.ConfigurationDigest = testDigest(value)
	return identity
}

func withRuntime(identity ReceiptIdentity, value string) ReceiptIdentity {
	identity.RuntimeDigest = testDigest(value)
	return identity
}

func withLockfiles(identity ReceiptIdentity, value string) ReceiptIdentity {
	identity.LockfilesDigest = testDigest(value)
	return identity
}

func testDigest(value string) string {
	sum := sha256.Sum256([]byte(value))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func resultOnlyControlAllows(receipt VerificationReceipt) bool {
	return receipt.Outcome == ReceiptPassed
}
