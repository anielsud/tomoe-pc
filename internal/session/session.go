package session

import "time"

// Session represents a meeting transcription session.
type Session struct {
	ID        string    `json:"id"`
	Title     string    `json:"title"`
	Platform  string    `json:"platform,omitempty"` // "Teams", "Meet", "Zoom", etc.
	Language  string    `json:"language,omitempty"` // ISO 639-1 code: "en", "bn", etc.
	CreatedAt time.Time `json:"created_at"`
	EndedAt   time.Time `json:"ended_at,omitempty"`
	Duration  float64   `json:"duration"` // seconds
	Sources   []string  `json:"sources"`  // e.g., ["mic", "monitor"]
	Segments  []Segment `json:"segments"`
	AudioPath string    `json:"audio_path,omitempty"`
}

// Segment is a single transcribed utterance within a session.
type Segment struct {
	ID        string  `json:"id"`
	Speaker   string  `json:"speaker"`
	Text      string  `json:"text"`
	StartTime float64 `json:"start_time"`         // seconds from session start
	EndTime   float64 `json:"end_time"`           // seconds from session start
	Source    string  `json:"source"`             // "mic" or "monitor"
	Language  string  `json:"language,omitempty"` // ISO 639-1 code: "en", "bn", etc.
	// Status is "" (default, meaning final -- also what every segment
	// from before this field existed implicitly means), "live" (the
	// person is still talking; Text is pass 1's partial hypothesis and
	// will keep growing under the same ID), or "pending" (the utterance
	// is done, Text is pass 1's finished-but-unrefined text, and a
	// higher-quality re-decode is in flight). A later update carrying
	// the same ID and Status "" supersedes either. See internal/live's
	// two-pass pipeline.
	Status string `json:"status,omitempty"`
	// Decision is which rule inside speaker.Tracker.Assign produced
	// Speaker for this segment ("confident", "sticky", "short-segment",
	// "new-speaker" — see speaker.AssignDecision), or "" for the mic
	// source (always "You", never audio-clustered) or when no
	// clustering ran at all. Plain string rather than importing
	// speaker.AssignDecision here, since internal/session has no other
	// reason to depend on internal/speaker. Diagnostic only, purely
	// informational for a diagnostics view (see
	// docs/macos-video-hints.md) -- never read back to change
	// behavior, and safe for older sessions on disk to simply lack it.
	Decision string `json:"decision,omitempty"`
}
