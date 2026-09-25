//go:build darwin

package videohint

import (
	"context"
	"fmt"
	"time"

	"github.com/sosuke-ai/tomoe-pc/internal/meeting"
	"github.com/sosuke-ai/tomoe-pc/internal/teamsvideo"
)

const (
	// maxSnapshotsPerPoll caps how many escalation snapshots one Poll
	// call writes, so a long idle meeting doesn't fill the disk.
	maxSnapshotsPerPoll = 5
	// minSnapshotInterval rate-limits captures within that cap.
	minSnapshotInterval = 60 * time.Second
)

// Poll periodically looks for a window FindMeetingWindow's heuristic
// matches as Teams (the only window-finder wired up so far —
// Zoom/Meet/Webex/Slack each need their own before this can capture
// anything for them; see docs/macos-support.md), captures a frame, and
// either produces a naming hint (logged for now — actually relabeling
// an internal/speaker.Tracker cluster from a hint is separate,
// not-yet-built follow-on work) or escalates it to the pending
// snapshot staging area (CaptureUnrecognizedUI /
// config.UnrecognizedUIPendingDir). Only PlatformTeams has a
// calibrated rule today (see rule.go); every other platform's rule
// table entry is still empty, so those always escalate — expected,
// not a bug.
//
// A capture+detect attempt runs on every ticker tick (every interval),
// AND immediately whenever trigger fires — internal/live's Coordinator
// signals trigger the moment it hears a monitor-source speaker with no
// video hint yet (see Coordinator.HintNeeded), so a still-unknown
// speaker gets an OCR attempt as soon as possible instead of waiting up
// to interval. trigger-driven attempts are debounced by
// triggerDebounce so a speaker who keeps talking without ever getting
// a hint can't trigger attempts faster than that; the ticker itself is
// never debounced. trigger may be nil if a caller doesn't want this
// (e.g. a bare interval-only poll). Both interval and triggerDebounce
// come from config.toml (MeetingConfig.VideoHintPollInterval/
// VideoHintTriggerDebounce) at the call site, read fresh each time a
// new meeting session starts — so, unlike speaker.Tracker's tuning,
// retuning these takes a fresh session rather than a live mid-session
// hot-reload, but still needs no rebuild or relaunch.
//
// Every step is reported on events (one Event per stage reached this
// tick, in order — see EventStage) so a caller can show not just
// videohint's end result but how and when it got there: whether a
// window was found this tick, what size frame it captured, whether the
// ring matched and where, whether OCR read anything. Sends are
// non-blocking (see sendEvent) — a slow or absent consumer never stalls
// polling. events may be nil if the caller doesn't want the trace.
//
// A StageOCRHit Event's Name field is the only part of this a caller
// needs to act on (e.g. via speaker.Tracker.SetHintForRecent) — this
// package deliberately has no internal/speaker dependency itself; that
// wiring is the caller's job (internal/daemon, internal/backend), since
// only they hold both the Tracker and this goroutine's lifecycle.
//
// Escalated snapshots are NOT the permanent library. FindMeetingWindow
// matches any non-trivial-titled Microsoft-Teams-owned window, which
// includes plain chat tabs and the post-meeting recording/playback
// page, not just an actual call — confirmed live during this feature's
// own testing, capturing a real private chat conversation and a
// recording page instead of a meeting. Rule.Chrome (see rule.go) now
// gates on real active-call chrome before Ring/Label are even
// attempted, which rules out both of those specific cases, but it's a
// narrower, differently-fallible check (a fixed icon match) than "is
// this really a call," so nothing captured here is treated as safe to
// keep or use for calibration until a human explicitly reviews and
// approves it via `tomoe videohint approve` — see ApproveSnapshot.
//
// Blocks until ctx is cancelled; meant to be run in its own goroutine,
// one per live meeting session, cancelled when that session stops
// (see internal/daemon and internal/backend's meeting start/stop).
func Poll(ctx context.Context, interval, triggerDebounce time.Duration, trigger <-chan struct{}, events chan<- Event) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	var lastCapture time.Time
	var lastAttempt time.Time
	captured := 0

	attempt := func(debounce bool) {
		if debounce && !lastAttempt.IsZero() && time.Since(lastAttempt) < triggerDebounce {
			return
		}
		lastAttempt = time.Now()
		pollOnce(events, &lastCapture, &captured)
	}

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			attempt(false)
		case <-trigger:
			attempt(true)
		}
	}
}

// pollOnce runs one capture+detect(+escalate) attempt — the body of a
// single Poll tick, factored out so both the scheduled ticker and an
// immediate trigger (see Poll) share exactly one implementation.
// lastCapture/captured are the same rate-limit/cap state Poll's loop
// carries across attempts, passed by pointer since both trigger- and
// ticker-driven attempts share it.
func pollOnce(events chan<- Event, lastCapture *time.Time, captured *int) {
	platform := meeting.PlatformTeams // only window-finder wired up so far

	windowID, err := teamsvideo.FindMeetingWindow()
	if err != nil {
		sendEvent(events, Event{Time: time.Now(), Platform: platform, Stage: StageWindowNotFound, Detail: "no meeting window found on screen"})
		return // no meeting window on screen right now -- normal, not an error
	}

	frame, err := teamsvideo.CaptureWindowRGB(windowID)
	if err != nil {
		sendEvent(events, Event{Time: time.Now(), Platform: platform, Stage: StageCaptureFailed, Detail: err.Error()})
		return
	}
	sendEvent(events, Event{Time: time.Now(), Platform: platform, Stage: StageFrameCaptured, Detail: fmt.Sprintf("captured %dx%d frame", frame.Width, frame.Height)})

	reason := "no rule configured for this platform"
	rule, ok := ruleFor(platform)
	if ok && rule.Chrome.configured() && !DetectCallChrome(frame.Pix, frame.Width, frame.Height, rule.Chrome) {
		// FindMeetingWindow's title heuristic alone can match a window
		// that's Teams-owned and non-trivially titled but isn't
		// actually a call (confirmed live: a chat conversation, a
		// post-meeting recording/playback page) -- neither has
		// anything worth calibrating a Ring/Label rule from, so this
		// is a hard stop, not an escalation: nothing captured here.
		sendEvent(events, Event{Time: time.Now(), Platform: platform, Stage: StageNotACall, Detail: "captured window has no active-call chrome (Leave button not found) -- likely not a live call"})
		return
	}
	if !ok {
		sendEvent(events, Event{Time: time.Now(), Platform: platform, Stage: StageNoRule, Detail: reason})
	} else if ring, found, ambiguous := DetectRing(frame.Pix, frame.Width, frame.Height, rule.Ring); ambiguous {
		// More than one plausible ring in the same frame -- observed
		// live: two people highlighted at once, and picking "the
		// best-scoring one" attributed a naming hint to the wrong
		// person. Skip attribution entirely for this tick rather than
		// guess; nothing to escalate either, since this isn't a rule
		// gap, it's a genuinely ambiguous moment that should resolve
		// itself once only one ring is lit.
		reason = "multiple rings found in the same frame -- skipping attribution"
		sendEvent(events, Event{Time: time.Now(), Platform: platform, Stage: StageAmbiguousRing, Detail: reason})
	} else if found {
		sendEvent(events, Event{Time: time.Now(), Platform: platform, Stage: StageRingMatched, Detail: fmt.Sprintf("ring at (%d,%d) %dx%d, confidence %.2f", ring.X, ring.Y, ring.Width, ring.Height, ring.Confidence)})

		if !rule.Label.configured() {
			reason = "ring found but no label region configured"
			sendEvent(events, Event{Time: time.Now(), Platform: platform, Stage: StageNoLabelRegion, Detail: reason})
		} else {
			name, ocrErr := RecognizeLabel(frame.Pix, frame.Width, frame.Height, *ring, rule.Label)
			if ocrErr == nil && name != "" {
				// Best-effort: a thumbnail failure shouldn't discard an
				// otherwise-good naming hint.
				thumb, thumbErr := RingThumbnailPNG(frame.Pix, frame.Width, frame.Height, *ring)
				if thumbErr != nil {
					thumb = nil
				}
				sendEvent(events, Event{Time: time.Now(), Platform: platform, Stage: StageOCRHit, Detail: fmt.Sprintf("OCR read %q from the label region", name), Name: name, Thumbnail: thumb})
				return // got a usable hint -- nothing to escalate this tick
			}
			reason = "ring found but OCR produced no text"
			detail := reason
			if ocrErr != nil {
				detail = fmt.Sprintf("%s: %v", reason, ocrErr)
			}
			sendEvent(events, Event{Time: time.Now(), Platform: platform, Stage: StageOCRMiss, Detail: detail})
		}
	} else {
		reason = "no ring match found"
		sendEvent(events, Event{Time: time.Now(), Platform: platform, Stage: StageNoRingMatch, Detail: reason})
	}

	if *captured >= maxSnapshotsPerPoll {
		return
	}
	if !lastCapture.IsZero() && time.Since(*lastCapture) < minSnapshotInterval {
		return
	}

	meta := SnapshotMeta{
		Platform: platform,
		// WindowTitle/WindowOwner are left blank: FindMeetingWindow
		// doesn't expose what it matched internally, and adding
		// that lookup is out of scope for this change (teamsvideo
		// itself doesn't need modifying otherwise).
		Width:     frame.Width,
		Height:    frame.Height,
		Timestamp: time.Now(),
		Reason:    reason,
	}
	if err := CaptureUnrecognizedUI(meta, frame.Pix, frame.Width, frame.Height); err != nil {
		fmt.Printf("videohint: failed to capture snapshot: %v\n", err)
		return
	}
	sendEvent(events, Event{Time: time.Now(), Platform: platform, Stage: StageEscalated, Detail: fmt.Sprintf("captured snapshot for review (%s)", reason)})
	*lastCapture = time.Now()
	*captured++
}
