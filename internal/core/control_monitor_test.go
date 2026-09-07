package core

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestControlMonitorLegalTrace(t *testing.T) {
	identity := testControlReceiptIdentity("a")
	receipt := testPassingReceipt(identity)
	events := []ControlEvent{
		testControlEvent(1, ControlSetReceiptTarget, func(event *ControlEvent) { event.Identity = identity }),
		testControlEvent(2, ControlApprovalGranted, func(event *ControlEvent) { event.Action = "edit" }),
		testControlEvent(3, ControlLeaseAcquired, func(event *ControlEvent) { event.Action, event.Lease = "edit", "lease-1" }),
		testControlEvent(4, ControlEffectsDeclared, func(event *ControlEvent) { event.Lease, event.Effects = "lease-1", []string{"workspace/main.go"} }),
		testControlEvent(5, ControlEffectObserved, func(event *ControlEvent) { event.Lease, event.Effect = "lease-1", "workspace/main.go" }),
		testControlEvent(6, ControlLeaseReleased, func(event *ControlEvent) { event.Lease = "lease-1" }),
		testControlEvent(7, ControlReceiptRecorded, func(event *ControlEvent) { event.Receipt = receipt }),
		testControlEvent(8, ControlLifecycleClaimed, func(event *ControlEvent) { event.Claim = ClaimVerified }),
	}

	state := applyAcceptedControlEvents(t, ControlMonitorState{}, events)
	if state.ActiveLease != "" || len(state.DeclaredEffects) != 0 {
		t.Fatalf("legal trace leaked mutation state: %#v", state)
	}
	if !state.CurrentReceipt.Valid() || !state.CurrentReceipt.Identity.Equal(identity) {
		t.Fatalf("legal trace did not retain current receipt: %#v", state.CurrentReceipt)
	}
}

func TestControlMonitorControlTrace(t *testing.T) {
	identity := testControlReceiptIdentity("a")
	staleReceipt := testPassingReceipt(identity)
	changedTarget := testControlReceiptIdentity("b")

	// The control models an outcome-only gate: a passed receipt is enough to
	// authorize the claim, even if the target changed afterwards. It is not a
	// production implementation; it makes the causal contrast explicit.
	if !controlPermitsLifecycle(staleReceipt) {
		t.Fatal("control should accept a passed receipt")
	}
	if staleReceipt.Identity.Equal(changedTarget) {
		t.Fatal("fixture must model a changed receipt target")
	}
}

func TestControlMonitorTreatmentTrace(t *testing.T) {
	identity := testControlReceiptIdentity("a")
	changedTarget := testControlReceiptIdentity("b")
	receipt := testPassingReceipt(identity)
	state := applyAcceptedControlEvents(t, ControlMonitorState{}, []ControlEvent{
		testControlEvent(1, ControlSetReceiptTarget, func(event *ControlEvent) { event.Identity = changedTarget }),
	})
	_, decision := ApplyControlEvent(state, testControlEvent(2, ControlReceiptRecorded, func(event *ControlEvent) {
		event.Receipt = receipt
	}))
	if decision.RuleID != ControlReceiptStale {
		t.Fatalf("treatment accepted stale receipt: %#v", decision)
	}
}

func TestControlMonitorRejectsIllegalTracesWithStableRules(t *testing.T) {
	identity := testControlReceiptIdentity("a")
	otherIdentity := testControlReceiptIdentity("b")
	receipt := testPassingReceipt(identity)

	tests := []struct {
		name     string
		prefix   []ControlEvent
		event    ControlEvent
		wantRule ControlRuleID
	}{
		{
			name:     "mutation requires matching approval",
			event:    testControlEvent(1, ControlLeaseAcquired, func(event *ControlEvent) { event.Action, event.Lease = "edit", "lease-1" }),
			wantRule: ControlApprovalRequired,
		},
		{
			name: "only one lease can be active",
			prefix: []ControlEvent{
				testControlEvent(1, ControlApprovalGranted, func(event *ControlEvent) { event.Action = "edit" }),
				testControlEvent(2, ControlLeaseAcquired, func(event *ControlEvent) { event.Action, event.Lease = "edit", "lease-1" }),
			},
			event:    testControlEvent(3, ControlLeaseAcquired, func(event *ControlEvent) { event.Action, event.Lease = "edit", "lease-2" }),
			wantRule: ControlLeaseConflict,
		},
		{
			name: "effect must be declared",
			prefix: []ControlEvent{
				testControlEvent(1, ControlApprovalGranted, func(event *ControlEvent) { event.Action = "edit" }),
				testControlEvent(2, ControlLeaseAcquired, func(event *ControlEvent) { event.Action, event.Lease = "edit", "lease-1" }),
				testControlEvent(3, ControlEffectsDeclared, func(event *ControlEvent) { event.Lease, event.Effects = "lease-1", []string{"workspace/main.go"} }),
			},
			event:    testControlEvent(4, ControlEffectObserved, func(event *ControlEvent) { event.Lease, event.Effect = "lease-1", "workspace/other.go" }),
			wantRule: ControlEffectUndeclared,
		},
		{
			name:     "lifecycle requires receipt",
			prefix:   []ControlEvent{testControlEvent(1, ControlSetReceiptTarget, func(event *ControlEvent) { event.Identity = identity })},
			event:    testControlEvent(2, ControlLifecycleClaimed, func(event *ControlEvent) { event.Claim = ClaimDone }),
			wantRule: ControlLifecycleReceiptRequired,
		},
		{
			name:     "stale receipt is rejected",
			prefix:   []ControlEvent{testControlEvent(1, ControlSetReceiptTarget, func(event *ControlEvent) { event.Identity = otherIdentity })},
			event:    testControlEvent(2, ControlReceiptRecorded, func(event *ControlEvent) { event.Receipt = receipt }),
			wantRule: ControlReceiptStale,
		},
		{
			name: "target change invalidates recorded receipt",
			prefix: []ControlEvent{
				testControlEvent(1, ControlSetReceiptTarget, func(event *ControlEvent) { event.Identity = identity }),
				testControlEvent(2, ControlReceiptRecorded, func(event *ControlEvent) { event.Receipt = receipt }),
				testControlEvent(3, ControlSetReceiptTarget, func(event *ControlEvent) { event.Identity = otherIdentity }),
			},
			event:    testControlEvent(4, ControlLifecycleClaimed, func(event *ControlEvent) { event.Claim = ClaimReadyToMerge }),
			wantRule: ControlLifecycleReceiptRequired,
		},
		{
			name:     "unsupported event fails closed",
			event:    testControlEvent(1, ControlEventKind("model_policy"), nil),
			wantRule: ControlEventSchemaUnsupported,
		},
		{
			name: "cancellation cannot become lifecycle success",
			prefix: []ControlEvent{
				testControlEvent(1, ControlSetReceiptTarget, func(event *ControlEvent) { event.Identity = identity }),
				testControlEvent(2, ControlReceiptRecorded, func(event *ControlEvent) { event.Receipt = receipt }),
				testControlEvent(3, ControlCancelled, nil),
			},
			event:    testControlEvent(4, ControlLifecycleClaimed, func(event *ControlEvent) { event.Claim = ClaimDone }),
			wantRule: ControlLifecycleCancelled,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state := applyAcceptedControlEvents(t, ControlMonitorState{}, test.prefix)
			before := state
			next, decision := ApplyControlEvent(state, test.event)
			if decision.RuleID != test.wantRule || decision.Accepted {
				t.Fatalf("decision = %#v, want rejected %q", decision, test.wantRule)
			}
			if !reflect.DeepEqual(next, before) {
				t.Fatalf("rejected event mutated state:\n got: %#v\nwant: %#v", next, before)
			}
		})
	}
}

func TestControlMonitorReplayDuplicateAndSequenceRules(t *testing.T) {
	first := testControlEvent(1, ControlApprovalGranted, func(event *ControlEvent) { event.Action = "edit" })
	state, decision := ApplyControlEvent(ControlMonitorState{}, first)
	if !decision.Accepted {
		t.Fatalf("first event rejected: %#v", decision)
	}

	replayed, decision := ApplyControlEvent(state, first)
	if !decision.Accepted || !reflect.DeepEqual(replayed, state) {
		t.Fatalf("immediate duplicate was not idempotent: state=%#v decision=%#v", replayed, decision)
	}

	altered := first
	altered.Action = "other"
	if _, decision := ApplyControlEvent(state, altered); decision.RuleID != ControlEventSequence {
		t.Fatalf("altered duplicate rule = %q, want %q", decision.RuleID, ControlEventSequence)
	}

	gap := testControlEvent(3, ControlCancelled, nil)
	if _, decision := ApplyControlEvent(state, gap); decision.RuleID != ControlEventSequence {
		t.Fatalf("gap rule = %q, want %q", decision.RuleID, ControlEventSequence)
	}
}

func TestControlMonitorPrefixReplayMatchesUninterruptedTrace(t *testing.T) {
	identity := testControlReceiptIdentity("a")
	events := []ControlEvent{
		testControlEvent(1, ControlSetReceiptTarget, func(event *ControlEvent) { event.Identity = identity }),
		testControlEvent(2, ControlApprovalGranted, func(event *ControlEvent) { event.Action = "edit" }),
		testControlEvent(3, ControlLeaseAcquired, func(event *ControlEvent) { event.Action, event.Lease = "edit", "lease-1" }),
		testControlEvent(4, ControlEffectsDeclared, func(event *ControlEvent) { event.Lease, event.Effects = "lease-1", []string{"workspace/main.go"} }),
		testControlEvent(5, ControlEffectObserved, func(event *ControlEvent) { event.Lease, event.Effect = "lease-1", "workspace/main.go" }),
		testControlEvent(6, ControlLeaseReleased, func(event *ControlEvent) { event.Lease = "lease-1" }),
	}

	uninterrupted := applyAcceptedControlEvents(t, ControlMonitorState{}, events)
	prefix := applyAcceptedControlEvents(t, ControlMonitorState{}, events[:3])
	replayed := applyAcceptedControlEvents(t, prefix, events[3:])
	if !reflect.DeepEqual(replayed, uninterrupted) {
		t.Fatalf("replay state differs:\n got: %#v\nwant: %#v", replayed, uninterrupted)
	}
}

func TestControlMonitorBoundsEffectDeclaration(t *testing.T) {
	state := applyAcceptedControlEvents(t, ControlMonitorState{}, []ControlEvent{
		testControlEvent(1, ControlApprovalGranted, func(event *ControlEvent) { event.Action = "edit" }),
		testControlEvent(2, ControlLeaseAcquired, func(event *ControlEvent) { event.Action, event.Lease = "edit", "lease-1" }),
	})

	effects := make([]string, maxDeclaredEffects+1)
	for index := range effects {
		effects[index] = fmt.Sprintf("workspace/file-%d", index)
	}
	_, decision := ApplyControlEvent(state, testControlEvent(3, ControlEffectsDeclared, func(event *ControlEvent) {
		event.Lease, event.Effects = "lease-1", effects
	}))
	if decision.RuleID != ControlEffectDeclarationInvalid {
		t.Fatalf("oversized declaration rule = %q, want %q", decision.RuleID, ControlEffectDeclarationInvalid)
	}
}

func applyAcceptedControlEvents(t *testing.T, state ControlMonitorState, events []ControlEvent) ControlMonitorState {
	t.Helper()
	for _, event := range events {
		var decision ControlDecision
		state, decision = ApplyControlEvent(state, event)
		if !decision.Accepted {
			t.Fatalf("event %d (%s) rejected: %#v", event.Sequence, event.Kind, decision)
		}
	}
	return state
}

func testControlEvent(sequence uint64, kind ControlEventKind, mutate func(*ControlEvent)) ControlEvent {
	event := ControlEvent{SchemaVersion: controlMonitorSchemaVersion, Sequence: sequence, Kind: kind}
	if mutate != nil {
		mutate(&event)
	}
	return event
}

func testControlReceiptIdentity(fill string) ReceiptIdentity {
	digest := "sha256:" + strings.Repeat(fill, 64)
	return ReceiptIdentity{
		ScopeDigest: digest, SourceDigest: digest, VerifierDigest: digest,
		ArgumentsDigest: digest, ConfigurationDigest: digest, RuntimeDigest: digest,
		LockfilesDigest: digest,
	}
}

func testPassingReceipt(identity ReceiptIdentity) VerificationReceipt {
	return SealVerificationReceipt(VerificationReceipt{
		ID: "receipt-1", Obligation: "tests", Identity: identity, Outcome: ReceiptPassed,
	})
}

func controlPermitsLifecycle(receipt VerificationReceipt) bool {
	return receipt.Valid() && receipt.Outcome == ReceiptPassed
}
