package core

import (
	"reflect"
	"testing"
)

// The idempotence promise at the top of ControlEvent is a property over all
// accepted events, not over the handful of traces the example corpus happens
// to contain. Stated as examples it was vacuous: every frozen fixture carries
// a nil declaration slice, so the nil-versus-empty distinction that broke
// replay was unreachable. Enumerating the shapes the field can take is
// enough to fail; a generator is not required for a two-valued axis.
func TestControlEventReplayIsIdempotentAcrossEffectsShapes(t *testing.T) {
	shapes := []struct {
		name     string
		declared EffectDeclaration
	}{
		{"zero", EffectDeclaration{}},
		{"empty", EffectDeclaration{Reads: []PathEffect{}, Observation: ObservationObserved}},
		{"one", EffectDeclaration{Writes: []PathEffect{{Path: "workspace/main.go"}}, Observation: ObservationObserved}},
		{"several", EffectDeclaration{
			Writes:      []PathEffect{{Path: "workspace/main.go"}},
			Network:     []string{"example.com"},
			Observation: ObservationObserved,
		}},
	}

	for _, shape := range shapes {
		t.Run(shape.name, func(t *testing.T) {
			event := ControlEvent{
				SchemaVersion: controlMonitorSchemaVersion,
				Sequence:      1,
				Kind:          ControlApprovalGranted,
				Action:        "deploy",
				Declared:      shape.declared,
			}

			applied, first := ApplyControlEvent(ControlMonitorState{}, event)
			if !first.Accepted {
				t.Fatalf("event rejected on first apply: %q", first.RuleID)
			}

			replayed, second := ApplyControlEvent(applied, event)
			if !second.Accepted {
				t.Fatalf("identical replay rejected with %q; replay must be idempotent", second.RuleID)
			}
			if !reflect.DeepEqual(replayed, applied) {
				t.Fatalf("replay changed state:\n got %+v\nwant %+v", replayed, applied)
			}
		})
	}
}
