package speaker

import (
	"testing"
	"time"
)

func TestTrackerAssignSameSpeaker(t *testing.T) {
	tracker := NewTracker(0.8)

	// Two very similar embeddings should be same speaker
	emb1 := []float32{1, 0, 0, 0, 0}
	emb2 := []float32{0.99, 0.01, 0, 0, 0}

	label1, _ := tracker.Assign(emb1, 2*time.Second)
	label2, _ := tracker.Assign(emb2, 2*time.Second)

	if label1 != "Person 1" {
		t.Errorf("first label = %q, want %q", label1, "Person 1")
	}
	if label2 != "Person 1" {
		t.Errorf("second label = %q, want %q (same speaker)", label2, "Person 1")
	}
	if tracker.NumSpeakers() != 1 {
		t.Errorf("NumSpeakers() = %d, want 1", tracker.NumSpeakers())
	}
}

func TestTrackerAssignDifferentSpeakers(t *testing.T) {
	tracker := NewTracker(0.8)

	// Orthogonal embeddings → different speakers
	emb1 := []float32{1, 0, 0, 0, 0}
	emb2 := []float32{0, 1, 0, 0, 0}

	label1, _ := tracker.Assign(emb1, 2*time.Second)
	label2, _ := tracker.Assign(emb2, 2*time.Second)

	if label1 != "Person 1" {
		t.Errorf("first label = %q, want %q", label1, "Person 1")
	}
	if label2 != "Person 2" {
		t.Errorf("second label = %q, want %q (different speaker)", label2, "Person 2")
	}
	if tracker.NumSpeakers() != 2 {
		t.Errorf("NumSpeakers() = %d, want 2", tracker.NumSpeakers())
	}
}

func TestTrackerThreshold(t *testing.T) {
	// With a very high threshold, similar vectors should still be
	// different speakers -- once the sticky-speaker grace window (see
	// stickyGraceWindow) has elapsed, so this isolates the pure
	// threshold check from that separate, intentional exception.
	tracker := NewTracker(0.999)
	clock := &fakeClock{t: time.Now()}
	tracker.nowFn = clock.now

	emb1 := []float32{1, 0, 0}
	emb2 := []float32{0.95, 0.3, 0}

	tracker.Assign(emb1, 2*time.Second)
	clock.advance(DefaultTuning().StickyGraceWindow + time.Second)
	label2, _ := tracker.Assign(emb2, 2*time.Second)

	if label2 != "Person 2" {
		t.Errorf("with high threshold, label = %q, want %q", label2, "Person 2")
	}
}

func TestTrackerReset(t *testing.T) {
	tracker := NewTracker(0.8)

	tracker.Assign([]float32{1, 0, 0}, 2*time.Second)
	tracker.Assign([]float32{0, 1, 0}, 2*time.Second)

	if tracker.NumSpeakers() != 2 {
		t.Fatalf("NumSpeakers() before reset = %d, want 2", tracker.NumSpeakers())
	}

	tracker.Reset()

	if tracker.NumSpeakers() != 0 {
		t.Errorf("NumSpeakers() after reset = %d, want 0", tracker.NumSpeakers())
	}

	// After reset, next embedding is Person 1 again
	label, _ := tracker.Assign([]float32{1, 0, 0}, 2*time.Second)
	if label != "Person 1" {
		t.Errorf("after reset, label = %q, want %q", label, "Person 1")
	}
}

func TestTrackerEmptyEmbedding(t *testing.T) {
	tracker := NewTracker(0.8)
	label, _ := tracker.Assign(nil, 2*time.Second)
	if label != "Unknown" {
		t.Errorf("empty embedding label = %q, want %q", label, "Unknown")
	}
}

func TestTrackerDefaultThreshold(t *testing.T) {
	// Invalid thresholds should use default
	tracker := NewTracker(0)
	if got := tracker.Tuning().Threshold; got != DefaultThreshold {
		t.Errorf("threshold = %v, want %v", got, DefaultThreshold)
	}

	tracker = NewTracker(-1)
	if got := tracker.Tuning().Threshold; got != DefaultThreshold {
		t.Errorf("threshold = %v, want %v", got, DefaultThreshold)
	}
}

func TestTrackerMultipleSpeakers(t *testing.T) {
	tracker := NewTracker(0.7)

	// Simulate 3 distinct speakers
	speakers := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
	}

	for i, emb := range speakers {
		label, _ := tracker.Assign(emb, 2*time.Second)
		want := "Person " + string(rune('1'+i))
		if label != want {
			t.Errorf("speaker %d label = %q, want %q", i, label, want)
		}
	}

	// Re-assign same embeddings — should match existing speakers
	for i, emb := range speakers {
		label, _ := tracker.Assign(emb, 2*time.Second)
		want := "Person " + string(rune('1'+i))
		if label != want {
			t.Errorf("re-assign speaker %d label = %q, want %q", i, label, want)
		}
	}

	if tracker.NumSpeakers() != 3 {
		t.Errorf("NumSpeakers() = %d, want 3", tracker.NumSpeakers())
	}
}

func TestTrackerSetHintForRecent(t *testing.T) {
	tracker := NewTracker(0.8)

	tracker.Assign([]float32{1, 0, 0}, 2*time.Second) // Person 1
	tracker.Assign([]float32{0, 1, 0}, 2*time.Second) // Person 2, most recently assigned

	if ok := tracker.SetHintForRecent("Nazanin", time.Minute); !ok {
		t.Fatal("SetHintForRecent() = false, want true")
	}

	// The hint should attach to Person 2 (most recently assigned), not Person 1.
	label1, _ := tracker.Assign([]float32{0.99, 0.01, 0}, 2*time.Second)
	if label1 != "Person 1" {
		t.Errorf("Person 1 label = %q, want unchanged %q", label1, "Person 1")
	}
	label2, _ := tracker.Assign([]float32{0, 0.99, 0.01}, 2*time.Second)
	if label2 != "Person 2 (Nazanin)" {
		t.Errorf("Person 2 label = %q, want %q", label2, "Person 2 (Nazanin)")
	}
}

func TestTrackerSetHintForRecent_KeepsFullNameOverLaterTruncation(t *testing.T) {
	tracker := NewTracker(0.8)
	tracker.Assign([]float32{0, 1, 0}, 2*time.Second) // Person 1

	if ok := tracker.SetHintForRecent("Nazanin Ramezani", time.Minute); !ok {
		t.Fatal("SetHintForRecent() = false, want true")
	}
	// Re-select Person 1 as most recently assigned, then attach a
	// truncated read of the same name (as a tile label might produce
	// on a later OCR pass) -- it must not clobber the full name.
	tracker.Assign([]float32{0, 0.99, 0.01}, 2*time.Second)
	if ok := tracker.SetHintForRecent("Nazanin Rame…", time.Minute); !ok {
		t.Fatal("SetHintForRecent() = false, want true")
	}

	label, _ := tracker.Assign([]float32{0, 0.98, 0.02}, 2*time.Second)
	if label != "Person 1 (Nazanin Ramezani)" {
		t.Errorf("label = %q, want full name kept, got truncated overwrite", label)
	}
}

func TestTrackerSetHintForRecent_UpgradesTruncatedNameToFuller(t *testing.T) {
	tracker := NewTracker(0.8)
	tracker.Assign([]float32{0, 1, 0}, 2*time.Second) // Person 1

	tracker.SetHintForRecent("Nazanin Rame…", time.Minute)
	tracker.Assign([]float32{0, 0.99, 0.01}, 2*time.Second)
	if ok := tracker.SetHintForRecent("Nazanin Ramezani", time.Minute); !ok {
		t.Fatal("SetHintForRecent() = false, want true")
	}

	label, _ := tracker.Assign([]float32{0, 0.98, 0.02}, 2*time.Second)
	if label != "Person 1 (Nazanin Ramezani)" {
		t.Errorf("label = %q, want the fuller name to replace the truncated one", label)
	}
}

func TestTrackerSetHintForRecent_UnrelatedNameOverwrites(t *testing.T) {
	tracker := NewTracker(0.8)
	tracker.Assign([]float32{0, 1, 0}, 2*time.Second) // Person 1

	tracker.SetHintForRecent("Nazanin Ramezani", time.Minute)
	tracker.Assign([]float32{0, 0.99, 0.01}, 2*time.Second)
	if ok := tracker.SetHintForRecent("Colin Whittingham", time.Minute); !ok {
		t.Fatal("SetHintForRecent() = false, want true")
	}

	label, _ := tracker.Assign([]float32{0, 0.98, 0.02}, 2*time.Second)
	if label != "Person 1 (Colin Whittingham)" {
		t.Errorf("label = %q, want an unrelated name to overwrite normally", label)
	}
}

func TestTrackerSetHintForRecent_TooOld(t *testing.T) {
	tracker := NewTracker(0.8)
	tracker.Assign([]float32{1, 0, 0}, 2*time.Second)

	if ok := tracker.SetHintForRecent("Someone", -time.Second); ok {
		t.Error("SetHintForRecent() with a negative maxAge = true, want false")
	}
}

func TestTrackerSetHintForRecent_NoSpeakersYet(t *testing.T) {
	tracker := NewTracker(0.8)
	if ok := tracker.SetHintForRecent("Someone", time.Minute); ok {
		t.Error("SetHintForRecent() before any Assign = true, want false")
	}
}

func TestTrackerResetClearsHints(t *testing.T) {
	tracker := NewTracker(0.8)
	tracker.Assign([]float32{1, 0, 0}, 2*time.Second)
	tracker.SetHintForRecent("Nazanin", time.Minute)
	tracker.Reset()

	label, _ := tracker.Assign([]float32{1, 0, 0}, 2*time.Second)
	if label != "Person 1" {
		t.Errorf("after reset, label = %q, want plain %q (hint should be cleared)", label, "Person 1")
	}
}

// fakeClock lets tests deterministically control Tracker.nowFn without
// sleeping — see stickyGraceWindow's doc comment for why Assign needs
// to be time-aware at all.
type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time          { return c.t }
func (c *fakeClock) advance(d time.Duration) { c.t = c.t.Add(d) }

func TestTrackerStickySpeaker_AcceptsNearMissFromRecentSpeaker(t *testing.T) {
	tracker := NewTracker(0.8)
	clock := &fakeClock{t: time.Now()}
	tracker.nowFn = clock.now

	label1, _ := tracker.Assign([]float32{1, 0, 0, 0}, 2*time.Second)
	if label1 != "Person 1" {
		t.Fatalf("first label = %q, want %q", label1, "Person 1")
	}

	// A noisier embedding from the same speaker's next short sentence:
	// similarity to Person 1 falls just under the strict 0.8 threshold,
	// but within the sticky margin (0.15), and it's well within the
	// grace window.
	clock.advance(500 * time.Millisecond)
	near := []float32{0.7, 0.7141428, 0, 0} // cosine sim to {1,0,0,0} ~= 0.70
	label2, _ := tracker.Assign(near, 2*time.Second)
	if label2 != "Person 1" {
		t.Errorf("near-miss recent-speaker label = %q, want %q (sticky match)", label2, "Person 1")
	}
	if tracker.NumSpeakers() != 1 {
		t.Errorf("NumSpeakers() = %d, want 1 (sticky match should not create a new speaker)", tracker.NumSpeakers())
	}
}

func TestTrackerStickySpeaker_DoesNotApplyPastGraceWindow(t *testing.T) {
	tracker := NewTracker(0.8)
	clock := &fakeClock{t: time.Now()}
	tracker.nowFn = clock.now

	tracker.Assign([]float32{1, 0, 0, 0}, 2*time.Second) // Person 1

	clock.advance(DefaultTuning().StickyGraceWindow + time.Second)
	near := []float32{0.7, 0.7141428, 0, 0}
	label2, _ := tracker.Assign(near, 2*time.Second)
	if label2 != "Person 2" {
		t.Errorf("near-miss outside grace window label = %q, want %q (new speaker)", label2, "Person 2")
	}
}

func TestTrackerStickySpeaker_DoesNotApplyToADifferentBestMatch(t *testing.T) {
	tracker := NewTracker(0.8)
	clock := &fakeClock{t: time.Now()}
	tracker.nowFn = clock.now

	tracker.Assign([]float32{1, 0, 0, 0}, 2*time.Second) // Person 1
	tracker.Assign([]float32{0, 0, 0, 1}, 2*time.Second) // Person 2, now the most recently assigned

	// A genuinely poor match for everyone (best match is Person 1, but
	// nowhere near even the sticky margin) must still become a new
	// speaker, regardless of timing.
	clock.advance(100 * time.Millisecond)
	label, _ := tracker.Assign([]float32{0.3, 0.3, 0.3, 0.3}, 2*time.Second)
	if label != "Person 3" {
		t.Errorf("poor match label = %q, want %q (new speaker)", label, "Person 3")
	}
}

func TestTrackerStickySpeaker_DoesNotPolluteCentroid(t *testing.T) {
	tracker := NewTracker(0.8)
	clock := &fakeClock{t: time.Now()}
	tracker.nowFn = clock.now

	tracker.Assign([]float32{1, 0, 0, 0}, 2*time.Second) // Person 1's centroid: {1,0,0,0}

	clock.advance(time.Second)
	tracker.Assign([]float32{0.7, 0.7141428, 0, 0}, 2*time.Second) // sticky match, must not update the centroid

	// A fresh, perfectly-orthogonal embedding should still be judged
	// against Person 1's ORIGINAL centroid, not one dragged toward the
	// noisier sticky-matched sample.
	clock.advance(time.Second)
	label, _ := tracker.Assign([]float32{0, 1, 0, 0}, 2*time.Second)
	if label == "Person 1" {
		t.Errorf("orthogonal embedding matched Person 1 — centroid was polluted by the sticky match")
	}
}

func TestTrackerShortSegment_DefaultsToLastSpeakerEvenOnPoorMatch(t *testing.T) {
	tracker := NewTracker(0.8)
	clock := &fakeClock{t: time.Now()}
	tracker.nowFn = clock.now

	tracker.Assign([]float32{1, 0, 0, 0}, 2*time.Second) // Person 1

	// A short interjection ("Um") whose embedding doesn't even clear the
	// sticky margin against Person 1 -- without the short-segment
	// fallback this would become a brand-new speaker, reproducing the
	// real-world failure this fix targets (a dozen-plus new "Person N"
	// labels for one continuous speaker's filler words).
	clock.advance(2 * time.Second)
	label, _ := tracker.Assign([]float32{0, 0, 1, 0}, 300*time.Millisecond)
	if label != "Person 1" {
		t.Errorf("short poor-match segment label = %q, want %q (default to last speaker)", label, "Person 1")
	}
	if tracker.NumSpeakers() != 1 {
		t.Errorf("NumSpeakers() = %d, want 1 (short segment must not spawn a new speaker)", tracker.NumSpeakers())
	}
}

func TestTrackerShortSegment_DoesNotApplyPastGraceWindow(t *testing.T) {
	tracker := NewTracker(0.8)
	clock := &fakeClock{t: time.Now()}
	tracker.nowFn = clock.now

	tracker.Assign([]float32{1, 0, 0, 0}, 2*time.Second) // Person 1

	clock.advance(DefaultTuning().ShortSegmentGraceWindow + time.Second)
	label, _ := tracker.Assign([]float32{0, 0, 1, 0}, 300*time.Millisecond)
	if label != "Person 2" {
		t.Errorf("short segment far past the grace window label = %q, want %q (new speaker)", label, "Person 2")
	}
}

func TestTrackerShortSegment_NoPriorAssignmentStillCreatesNewSpeaker(t *testing.T) {
	tracker := NewTracker(0.8)

	// The very first segment ever seen: there's no "last speaker" to
	// default to, so a short/noisy embedding must still found a real
	// speaker rather than being silently dropped.
	label, _ := tracker.Assign([]float32{1, 0, 0, 0}, 300*time.Millisecond)
	if label != "Person 1" {
		t.Errorf("first-ever short segment label = %q, want %q", label, "Person 1")
	}
	if tracker.NumSpeakers() != 1 {
		t.Errorf("NumSpeakers() = %d, want 1", tracker.NumSpeakers())
	}
}

func TestTrackerShortSegment_DoesNotPolluteCentroid(t *testing.T) {
	tracker := NewTracker(0.8)
	clock := &fakeClock{t: time.Now()}
	tracker.nowFn = clock.now

	tracker.Assign([]float32{1, 0, 0, 0}, 2*time.Second) // Person 1's centroid: {1,0,0,0}

	clock.advance(time.Second)
	tracker.Assign([]float32{0, 0, 1, 0}, 300*time.Millisecond) // short-segment fallback, must not update the centroid

	// A fresh, perfectly-orthogonal embedding should still be judged
	// against Person 1's ORIGINAL centroid, not one dragged toward the
	// noisy short segment.
	clock.advance(time.Second)
	label, _ := tracker.Assign([]float32{0, 1, 0, 0}, 2*time.Second)
	if label == "Person 1" {
		t.Errorf("orthogonal embedding matched Person 1 — centroid was polluted by the short-segment fallback")
	}
}

func TestTrackerShortSegment_PrefersABetterMatchingOtherSpeaker(t *testing.T) {
	tracker := NewTracker(0.8)
	clock := &fakeClock{t: time.Now()}
	tracker.nowFn = clock.now

	// Two already-distinguished speakers, established with confident
	// matches so each has a real centroid of their own.
	tracker.Assign([]float32{1, 0, 0, 0}, 2*time.Second) // Person 1 ("Kevin")
	tracker.Assign([]float32{1, 0, 0, 0}, 2*time.Second) // reinforce Person 1's centroid
	clock.advance(time.Second)
	tracker.Assign([]float32{0, 0, 1, 0}, 2*time.Second) // Person 2 ("Shaf")
	tracker.Assign([]float32{0, 0, 1, 0}, 2*time.Second) // reinforce Person 2's centroid

	// Kevin speaks again, briefly, immediately followed by an
	// even briefer reply from Shaf -- a real quick back-and-forth.
	// Without the fix, Shaf's short reply would default to "whoever
	// spoke last" (Kevin) purely because it's short and recent, even
	// though it's clearly a much better match for Shaf's own centroid.
	clock.advance(time.Second)
	tracker.Assign([]float32{0.99, 0, 0.01, 0}, 2*time.Second) // Kevin again (long enough)

	clock.advance(time.Second)
	label, _ := tracker.Assign([]float32{0, 0, 0.99, 0.01}, 300*time.Millisecond) // Shaf's brief reply
	if label != "Person 2" {
		t.Errorf("short reply label = %q, want %q (Shaf's own centroid, not blindly Kevin's)", label, "Person 2")
	}
}

func TestTrackerShortSegment_StillDefaultsToLastSpeakerWithoutABetterMatch(t *testing.T) {
	tracker := NewTracker(0.8)
	clock := &fakeClock{t: time.Now()}
	tracker.nowFn = clock.now

	tracker.Assign([]float32{1, 0, 0, 0}, 2*time.Second) // Person 1

	// A short, noisy segment with no strong match anywhere -- the
	// original "default to last speaker" behavior must still apply;
	// the fix only redirects when some OTHER cluster is clearly the
	// better match, not whenever bestIdx happens to differ at all.
	clock.advance(time.Second)
	label, _ := tracker.Assign([]float32{0, 1, 0, 0}, 300*time.Millisecond)
	if label != "Person 1" {
		t.Errorf("label = %q, want %q (no better match exists, keep the default)", label, "Person 1")
	}
	if tracker.NumSpeakers() != 1 {
		t.Errorf("NumSpeakers() = %d, want 1", tracker.NumSpeakers())
	}
}

func TestTrackerShortSegment_AtMinDurationBehavesAsNormal(t *testing.T) {
	tracker := NewTracker(0.8)
	clock := &fakeClock{t: time.Now()}
	tracker.nowFn = clock.now

	tracker.Assign([]float32{1, 0, 0, 0}, 2*time.Second) // Person 1

	// Exactly at minAssignDuration (not below it) — the fallback must
	// NOT apply; a poor match at this duration is a real new speaker.
	clock.advance(time.Second)
	label, _ := tracker.Assign([]float32{0, 0, 1, 0}, DefaultTuning().MinAssignDuration)
	if label != "Person 2" {
		t.Errorf("poor match exactly at minAssignDuration label = %q, want %q (new speaker)", label, "Person 2")
	}
}

func TestTrackerSetHintForRecent_MergesClustersOnMatchingHint(t *testing.T) {
	tracker := NewTracker(0.8)
	clock := &fakeClock{t: time.Now()}
	tracker.nowFn = clock.now

	tracker.Assign([]float32{1, 0, 0, 0}, 2*time.Second) // Person 1
	tracker.SetHintForRecent("Christian Stanton", time.Minute)

	clock.advance(time.Second)
	tracker.Assign([]float32{0, 1, 0, 0}, 2*time.Second) // Person 2, a distinct-sounding embedding
	if tracker.NumSpeakers() != 2 {
		t.Fatalf("NumSpeakers() = %d, want 2 before the matching hint lands", tracker.NumSpeakers())
	}

	// A video hint resolves Person 2 to the SAME real name already
	// attached to Person 1 -- independent evidence they're one person,
	// even though their embeddings didn't cluster together.
	clock.advance(time.Second)
	if ok := tracker.SetHintForRecent("Christian Stanton", time.Minute); !ok {
		t.Fatal("SetHintForRecent() = false, want true")
	}

	if got := tracker.NumSpeakers(); got != 1 {
		t.Errorf("NumSpeakers() = %d, want 1 after merge", got)
	}

	// A fresh embedding matching either original centroid should now
	// resolve to the SAME merged identity.
	clock.advance(time.Second)
	label, _ := tracker.Assign([]float32{0, 0.99, 0.01, 0}, 2*time.Second)
	if label != "Person 1 (Christian Stanton)" {
		t.Errorf("label after merge = %q, want %q", label, "Person 1 (Christian Stanton)")
	}
}

func TestTrackerSetHintForRecent_MergeKeepsMoreCompleteName(t *testing.T) {
	tracker := NewTracker(0.8)
	clock := &fakeClock{t: time.Now()}
	tracker.nowFn = clock.now

	tracker.Assign([]float32{1, 0, 0, 0}, 2*time.Second) // Person 1
	tracker.SetHintForRecent("Christian Rame…", time.Minute)

	clock.advance(time.Second)
	tracker.Assign([]float32{0, 1, 0, 0}, 2*time.Second) // Person 2

	clock.advance(time.Second)
	tracker.SetHintForRecent("Christian Ramezani", time.Minute)

	clock.advance(time.Second)
	label, _ := tracker.Assign([]float32{1, 0, 0, 0}, 2*time.Second)
	if label != "Person 1 (Christian Ramezani)" {
		t.Errorf("label after merge = %q, want the fuller name kept", label)
	}
}

func TestTrackerSetHintForRecent_DoesNotMergeUnrelatedNames(t *testing.T) {
	tracker := NewTracker(0.8)
	clock := &fakeClock{t: time.Now()}
	tracker.nowFn = clock.now

	tracker.Assign([]float32{1, 0, 0, 0}, 2*time.Second) // Person 1
	tracker.SetHintForRecent("Christian Stanton", time.Minute)

	clock.advance(time.Second)
	tracker.Assign([]float32{0, 1, 0, 0}, 2*time.Second) // Person 2

	clock.advance(time.Second)
	tracker.SetHintForRecent("Jessica Dannemann", time.Minute)

	if got := tracker.NumSpeakers(); got != 2 {
		t.Errorf("NumSpeakers() = %d, want 2 (unrelated names must not merge)", got)
	}
}

func TestTrackerSetHintForRecent_DoesNotMergeOnAmbiguousShortFragment(t *testing.T) {
	tracker := NewTracker(0.8)
	clock := &fakeClock{t: time.Now()}
	tracker.nowFn = clock.now

	tracker.Assign([]float32{1, 0, 0, 0}, 2*time.Second) // Person 1
	tracker.SetHintForRecent("Ann", time.Minute)

	clock.advance(time.Second)
	tracker.Assign([]float32{0, 1, 0, 0}, 2*time.Second) // Person 2

	clock.advance(time.Second)
	// "Ann" is too short/ambiguous a fragment to safely merge on, even
	// though it IS technically a prefix of "Annika".
	tracker.SetHintForRecent("Annika", time.Minute)

	if got := tracker.NumSpeakers(); got != 2 {
		t.Errorf("NumSpeakers() = %d, want 2 (short ambiguous fragment must not trigger a merge)", got)
	}
}

func TestTrackerSetHintForRecent_MergeDoesNotRenumberUnrelatedClusters(t *testing.T) {
	tracker := NewTracker(0.8)
	clock := &fakeClock{t: time.Now()}
	tracker.nowFn = clock.now

	tracker.Assign([]float32{1, 0, 0, 0}, 2*time.Second) // Person 1
	tracker.SetHintForRecent("Christian Stanton", time.Minute)

	clock.advance(time.Second)
	tracker.Assign([]float32{0, 1, 0, 0}, 2*time.Second) // Person 2

	clock.advance(time.Second)
	tracker.Assign([]float32{0, 0, 1, 0}, 2*time.Second) // Person 3, unrelated to the merge below

	clock.advance(time.Second)
	tracker.Assign([]float32{0, 1, 0, 0}, 2*time.Second)       // re-select Person 2 as most recent
	tracker.SetHintForRecent("Christian Stanton", time.Minute) // merges Person 2 into Person 1

	clock.advance(time.Second)
	label, _ := tracker.Assign([]float32{0, 0, 0.99, 0.01}, 2*time.Second)
	if label != "Person 3" {
		t.Errorf("label = %q, want %q (merging 1&2 must not renumber Person 3)", label, "Person 3")
	}
}

func TestTrackerAssign_NeedsHintUntilOneAttaches(t *testing.T) {
	tracker := NewTracker(0.8)

	_, needsHint := tracker.Assign([]float32{1, 0, 0}, 2*time.Second)
	if !needsHint {
		t.Error("brand new speaker: needsHint = false, want true")
	}

	tracker.SetHintForRecent("Nazanin", time.Minute)

	_, needsHint = tracker.Assign([]float32{0.99, 0.01, 0}, 2*time.Second)
	if needsHint {
		t.Error("speaker with an attached hint: needsHint = true, want false")
	}
}

func TestTrackerAssign_EmptyEmbeddingNeverNeedsHint(t *testing.T) {
	tracker := NewTracker(0.8)
	_, needsHint := tracker.Assign(nil, 2*time.Second)
	if needsHint {
		t.Error("empty embedding: needsHint = true, want false (no real speaker to hint)")
	}
}

func TestIsTruncationOf(t *testing.T) {
	cases := []struct {
		short, long string
		want        bool
	}{
		{"Nazanin Rame…", "Nazanin Ramezani", true},
		{"nazanin rame", "Nazanin Ramezani", true}, // case-insensitive
		{"Nazanin Ramezani", "Nazanin Ramezani", true},
		{"Nazanin Ramezani", "Nazanin Rame…", false}, // short is actually longer
		{"Colin", "Nazanin Ramezani", false},         // not a prefix at all
		{"", "Nazanin Ramezani", false},
		{"Nazanin", "", false},
	}
	for _, c := range cases {
		if got := isTruncationOf(c.short, c.long); got != c.want {
			t.Errorf("isTruncationOf(%q, %q) = %v, want %v", c.short, c.long, got, c.want)
		}
	}
}
