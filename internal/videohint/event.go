package videohint

import (
	"time"

	"github.com/sosuke-ai/tomoe-pc/internal/meeting"
)

// EventStage identifies which step of Poll's per-tick pipeline an Event
// describes — the point of this type is to let a UI show *how* and
// *when* videohint is working, not just its end result.
type EventStage string

const (
	StageWindowNotFound EventStage = "window_not_found"
	StageCaptureFailed  EventStage = "capture_failed"
	StageFrameCaptured  EventStage = "frame_captured"
	StageNotACall       EventStage = "not_a_call"
	StageNoRule         EventStage = "no_rule"
	StageRingMatched    EventStage = "ring_matched"
	StageNoRingMatch    EventStage = "no_ring_match"
	StageAmbiguousRing  EventStage = "ambiguous_ring"
	StageNoLabelRegion  EventStage = "no_label_region"
	StageOCRHit         EventStage = "ocr_hit"
	StageOCRMiss        EventStage = "ocr_miss"
	StageEscalated      EventStage = "escalated"
)

// HintAttachMaxAge is the suggested maxAge for
// speaker.Tracker.SetHintForRecent when attaching a StageOCRHit Event's
// Name: generous enough to cover Poll's own polling interval plus
// capture/OCR latency, without reaching back so far it risks attributing
// a name to a different speaker who has since started talking.
const HintAttachMaxAge = 30 * time.Second

// Event describes one step of a single Poll tick. Poll emits exactly one
// Event per tick that finds a meeting window (fewer stages fire before an
// early continue, e.g. StageWindowNotFound is the only Event on a tick
// with no meeting window on screen).
type Event struct {
	Time     time.Time
	Platform meeting.Platform
	Stage    EventStage
	// Detail is a human-readable summary, e.g. "130x130 ring at
	// (1386,97), confidence 0.91" — meant to be shown directly in a UI.
	Detail string
	// Name is set only for StageOCRHit: the recognized name text. This
	// is the one Event field a caller needs to inspect programmatically
	// (to attach it via speaker.Tracker.SetHintForRecent) — everything
	// else is purely informational.
	Name string
	// Thumbnail is set only for StageOCRHit: a PNG-encoded crop of the
	// matched ring's own bounding box (the participant's video tile,
	// not just their name label) — see RingThumbnailPNG. Lets a viewer
	// sanity-check a recognized name against who was actually on
	// screen, not just trust the OCR text alone.
	Thumbnail []byte
}

// sendEvent delivers ev to events without blocking Poll's loop: if the
// consumer isn't keeping up (or events is nil, e.g. a caller that
// doesn't care about the activity trace), the event is dropped rather
// than stalling polling — same non-blocking-drop shape as
// live.Coordinator's segment channel.
func sendEvent(events chan<- Event, ev Event) {
	if events == nil {
		return
	}
	select {
	case events <- ev:
	default:
	}
}
