package speaker

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

// DefaultThreshold is the default cosine similarity threshold for
// same-speaker assignment. Lowered from an original 0.65 to 0.55 after
// live diagnostic logging against a real multi-participant call: across
// ~20 real assignments, NOT ONE ever reached 0.65 (every continuation
// relied on the sticky/short-segment fallbacks below), while
// likely-same-person returns clustered around 0.59-0.68 and
// likely-different-people clustered around 0.20-0.50 -- a real,
// separable gap, just centered lower than the old threshold assumed.
// 0.55 sits in that gap: comfortably above the "different person"
// range, low enough to catch same-person returns on their own merits
// instead of needing sticky/short-segment to rescue every one of them.
const DefaultThreshold = 0.55

// Tuning holds every clustering knob found to need real-world
// retuning rather than a fixed-forever constant -- exposed as a
// struct (not consts) specifically so a live Tracker's behavior can be
// adjusted via SetTuning (e.g. from a config-file hot-reload) without
// a rebuild or even a relaunch. See DefaultTuning for the reasoning
// behind each default value.
type Tuning struct {
	// Threshold is the cosine similarity a fresh embedding needs
	// against an existing centroid for a confident match (see
	// DefaultThreshold).
	Threshold float64
	// StickyGraceWindow and StickyThresholdMargin implement a "sticky
	// speaker" continuity heuristic: VAD splits one person's
	// continuous turn into several short segments whenever they pause
	// for a beat between sentences, and a short segment's embedding
	// is measurably noisier than a longer one's. If the best-matching
	// centroid is also whoever was just assigned, and not long ago, a
	// near-miss on similarity is far more likely to be "same person,
	// noisier embedding" than "a different person who happens to
	// sound similar," so it's accepted as the same speaker. The
	// centroid itself is NOT updated on a sticky-only match (see
	// Assign) -- only a full, confident match gets folded into the
	// running average, so a fragmented sentence can't drag a good
	// centroid toward a bad one.
	StickyGraceWindow     time.Duration
	StickyThresholdMargin float64
	// MinAssignDuration and ShortSegmentGraceWindow handle a segment
	// too short to trust a fresh identity decision from at all: a
	// filler word ("Um", "Uh") or any sub-second interjection produces
	// an embedding dominated by near-silence/breath noise, and can
	// end up with low similarity to EVERY existing centroid --
	// including its real speaker's -- failing even the sticky check
	// above. Observed live: a dozen-plus brand-new "Person N" labels
	// inside one minute, nearly all one-word interjections from what
	// was actually a single ongoing speaker. Too little signal to
	// know who this is should default to "whoever was just speaking",
	// not "a new person". ShortSegmentGraceWindow is deliberately
	// wider than StickyGraceWindow: a short interjection can trail the
	// last confidently-assigned speech by more than a few seconds and
	// still obviously belong to the same person (observed gaps up to
	// ~12s in a real transcript).
	MinAssignDuration       time.Duration
	ShortSegmentGraceWindow time.Duration
}

// DefaultTuning returns the tuning this package ships with.
func DefaultTuning() Tuning {
	return Tuning{
		Threshold:               DefaultThreshold,
		StickyGraceWindow:       3 * time.Second,
		StickyThresholdMargin:   0.15,
		MinAssignDuration:       700 * time.Millisecond,
		ShortSegmentGraceWindow: 15 * time.Second,
	}
}

// TuningFromSeconds builds a Tuning from plain float64 seconds --
// internal/config's MeetingConfig stores these as seconds (TOML has no
// duration type), so this is the one place that conversion happens,
// shared by internal/backend and internal/daemon rather than
// duplicated at each call site. Zero/invalid values are passed through
// as-is; SetTuning (the only place this ever feeds into) already falls
// back to DefaultTuning's value for anything zero/invalid.
func TuningFromSeconds(threshold, stickyGraceWindowSec, stickyThresholdMargin, minAssignDurationSec, shortSegmentGraceWindowSec float64) Tuning {
	return Tuning{
		Threshold:               threshold,
		StickyGraceWindow:       time.Duration(stickyGraceWindowSec * float64(time.Second)),
		StickyThresholdMargin:   stickyThresholdMargin,
		MinAssignDuration:       time.Duration(minAssignDurationSec * float64(time.Second)),
		ShortSegmentGraceWindow: time.Duration(shortSegmentGraceWindowSec * float64(time.Second)),
	}
}

// Tracker performs online speaker clustering using cosine similarity of embeddings.
// Speakers are labeled "Person 1", "Person 2", etc. — optionally suffixed
// with a real name in parens (e.g. "Person 2 (Nazanin Rame...)") once a
// hint attaches that cluster via SetHintForRecent; see its doc comment.
type Tracker struct {
	mu        sync.Mutex
	tuning    Tuning
	centroids [][]float32 // one centroid per known speaker
	counts    []int       // number of embeddings merged into each centroid
	hints     []string    // one optional name-hint per speaker, parallel to centroids

	// lastAssignedIdx/At track the most recent successful Assign, so a
	// video hint (which has no direct link to a cluster ID — it only
	// knows "this name is active right now") can be attributed to
	// "whoever was probably just speaking" via SetHintForRecent, and so
	// Assign itself can apply the sticky-speaker grace window above.
	lastAssignedIdx int
	lastAssignedAt  time.Time

	// aliasOf redirects a merged-away cluster index to the canonical
	// index it was merged into (see mergeInto/canonical). A
	// merged-away slot's centroid/count/hint entries are left in
	// place rather than deleted -- deleting would shift every later
	// index and silently change what "Person N" means for indices
	// created after it.
	aliasOf map[int]int

	// nowFn is time.Now by default; overridable in tests so the
	// sticky-speaker grace window is deterministically testable
	// without sleeping.
	nowFn func() time.Time
}

// NewTracker creates a Tracker with the given cosine similarity
// threshold and DefaultTuning for everything else. Embeddings with
// similarity >= threshold to a centroid are assigned to that speaker.
// Use SetTuning afterward for full control (e.g. from a config
// hot-reload) over the sticky/short-segment knobs too.
func NewTracker(threshold float64) *Tracker {
	tuning := DefaultTuning()
	if threshold > 0 && threshold <= 1 {
		tuning.Threshold = threshold
	}
	return &Tracker{
		tuning: tuning,
		nowFn:  time.Now,
	}
}

// SetTuning replaces every clustering knob at once, taking effect
// immediately for the next Assign call. Safe to call concurrently with
// Assign/SetHintForRecent from any goroutine (e.g. a config-file
// watcher on its own timer) -- this is the whole point: retuning
// clustering behavior live, without a rebuild or even a relaunch.
// Fields that are zero/invalid fall back to DefaultTuning's value
// rather than disabling that check entirely, since a zero
// StickyGraceWindow etc. reads as "not configured" rather than "off".
func (t *Tracker) SetTuning(tuning Tuning) {
	def := DefaultTuning()
	if tuning.Threshold <= 0 || tuning.Threshold > 1 {
		tuning.Threshold = def.Threshold
	}
	if tuning.StickyGraceWindow <= 0 {
		tuning.StickyGraceWindow = def.StickyGraceWindow
	}
	if tuning.StickyThresholdMargin <= 0 {
		tuning.StickyThresholdMargin = def.StickyThresholdMargin
	}
	if tuning.MinAssignDuration <= 0 {
		tuning.MinAssignDuration = def.MinAssignDuration
	}
	if tuning.ShortSegmentGraceWindow <= 0 {
		tuning.ShortSegmentGraceWindow = def.ShortSegmentGraceWindow
	}

	t.mu.Lock()
	t.tuning = tuning
	t.mu.Unlock()
}

// Tuning returns the tracker's current tuning (e.g. for logging what
// values a decision was actually made under).
func (t *Tracker) Tuning() Tuning {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.tuning
}

// Assign assigns an embedding to a speaker, creating a new speaker if no
// match is found. duration is how much audio the embedding was computed
// from (see minAssignDuration's doc comment — too little audio never
// spawns a new speaker or breaks continuity, regardless of similarity).
// Returns a label like "Person 1", or "Person 1 (Name)" if a video hint
// has already been attached to that speaker via SetHintForRecent, plus
// needsHint: true if this speaker still has no hint attached, so a
// caller (internal/live's Coordinator) can signal that a video-hint
// check is worth doing right away rather than waiting for
// videohint.Poll's next scheduled tick — see Coordinator.HintNeeded.
func (t *Tracker) Assign(embedding []float32, duration time.Duration) (label string, needsHint bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if len(embedding) == 0 {
		return "Unknown", false
	}

	// Find the best matching centroid
	bestIdx := -1
	bestSim := 0.0

	for i, centroid := range t.centroids {
		sim := CosineSimilarity(embedding, centroid)
		if sim > bestSim {
			bestSim = sim
			bestIdx = i
		}
	}
	if bestIdx >= 0 {
		bestIdx = t.canonical(bestIdx)
	}

	now := t.nowFn()
	sinceLast := time.Duration(-1)
	if !t.lastAssignedAt.IsZero() {
		sinceLast = now.Sub(t.lastAssignedAt)
	}

	if bestIdx >= 0 && bestSim >= t.tuning.Threshold {
		// Confident match: fold it into the running average.
		t.updateCentroid(bestIdx, embedding)
		t.lastAssignedIdx = bestIdx
		t.lastAssignedAt = now
		debugLogAssign("confident", bestIdx, bestSim, duration, sinceLast, t.tuning.Threshold)
		return t.label(bestIdx), t.hints[bestIdx] == ""
	}

	sticky := bestIdx >= 0 && bestIdx == t.lastAssignedIdx &&
		!t.lastAssignedAt.IsZero() && now.Sub(t.lastAssignedAt) <= t.tuning.StickyGraceWindow &&
		bestSim >= t.tuning.Threshold-t.tuning.StickyThresholdMargin

	if sticky {
		// Near-miss on similarity, but this is whoever was just
		// speaking, moments ago -- treat it as the same speaker
		// without folding it into the centroid (see doc comment above
		// StickyGraceWindow for why not).
		t.lastAssignedAt = now
		debugLogAssign("sticky", bestIdx, bestSim, duration, sinceLast, t.tuning.Threshold)
		return t.label(bestIdx), t.hints[bestIdx] == ""
	}

	if duration < t.tuning.MinAssignDuration && len(t.centroids) > 0 &&
		!t.lastAssignedAt.IsZero() && now.Sub(t.lastAssignedAt) <= t.tuning.ShortSegmentGraceWindow {
		// Not enough audio to trust either a confident match or even
		// the sticky check above -- default to whoever was just
		// speaking rather than spawning a new person from noise. Same
		// "don't pollute a good centroid with a bad signal" principle
		// as the sticky match: the centroid isn't touched. lastAssignedAt
		// IS refreshed, so a run of short interjections keeps renewing
		// its own grace window instead of expiring mid-run.
		idx := t.lastAssignedIdx
		t.lastAssignedAt = now
		debugLogAssign("short-segment", idx, bestSim, duration, sinceLast, t.tuning.Threshold)
		return t.label(idx), t.hints[idx] == ""
	}

	// New speaker
	newCentroid := make([]float32, len(embedding))
	copy(newCentroid, embedding)
	t.centroids = append(t.centroids, newCentroid)
	t.counts = append(t.counts, 1)
	t.hints = append(t.hints, "")
	idx := len(t.centroids) - 1
	t.lastAssignedIdx = idx
	t.lastAssignedAt = now
	debugLogAssign("new-speaker", idx, bestSim, duration, sinceLast, t.tuning.Threshold)
	return t.label(idx), true
}

// debugLogAssign is a TEMPORARY diagnostic: prints the real numbers
// behind every clustering decision (which similarity/duration/timing
// values actually occur against real captured audio) so
// threshold/margin/window constants above can be tuned from real data
// instead of guesses. Remove once real-world values have been
// gathered and the constants above are retuned against them.
func debugLogAssign(decision string, idx int, bestSim float64, duration, sinceLast time.Duration, threshold float64) {
	fmt.Printf("speaker: assign decision=%-13s idx=%d bestSim=%.3f threshold=%.3f dur=%v sinceLast=%v\n",
		decision, idx, bestSim, threshold, duration, sinceLast)
}

// label builds the display label for speaker idx: "Person N", or
// "Person N (Name)" if a hint has been attached.
func (t *Tracker) label(idx int) string {
	idx = t.canonical(idx)
	base := fmt.Sprintf("Person %d", idx+1)
	if idx < len(t.hints) && t.hints[idx] != "" {
		return fmt.Sprintf("%s (%s)", base, t.hints[idx])
	}
	return base
}

// canonical follows the alias chain (see mergeInto) to the still-active
// index a merged-away index now resolves to. Returns idx unchanged if
// it was never merged away.
func (t *Tracker) canonical(idx int) int {
	for {
		next, aliased := t.aliasOf[idx]
		if !aliased {
			return idx
		}
		idx = next
	}
}

// mergeInto merges fromIdx's cluster into toIdx's: folds fromIdx's
// accumulated centroid into toIdx's running average (weighted by how
// many embeddings each has seen so far), and aliases fromIdx to toIdx
// so every future Assign/label call against either resolves to the
// same identity. Neither slot is deleted from centroids/counts/hints
// -- doing so would shift every later index and silently change what
// "Person N" means for indices created after it. A no-op if both
// canonicalize to the same cluster already.
func (t *Tracker) mergeInto(fromIdx, toIdx int) {
	fromIdx = t.canonical(fromIdx)
	toIdx = t.canonical(toIdx)
	if fromIdx == toIdx {
		return
	}

	totalCount := t.counts[fromIdx] + t.counts[toIdx]
	if totalCount > 0 {
		wFrom, wTo := float32(t.counts[fromIdx]), float32(t.counts[toIdx])
		for i := range t.centroids[toIdx] {
			t.centroids[toIdx][i] = (t.centroids[toIdx][i]*wTo + t.centroids[fromIdx][i]*wFrom) / float32(totalCount)
		}
		t.counts[toIdx] = totalCount
	}

	if t.aliasOf == nil {
		t.aliasOf = make(map[int]int)
	}
	t.aliasOf[fromIdx] = toIdx
	if t.lastAssignedIdx == fromIdx {
		t.lastAssignedIdx = toIdx
	}
}

// SetHintForRecent attaches name as a hint to whichever speaker was most
// recently assigned an embedding, provided that assignment happened
// within maxAge. This is how a video hint — which only knows "this name
// is active right now," not which cluster ID it corresponds to — gets
// attributed to a specific speaker: "whoever the audio pipeline most
// recently heard" is the best available proxy for "whoever the ring is
// currently around," since the video hint and the audio pipeline are
// both keyed to the same monitor-source (other participants') audio.
// Reports whether it found a recent-enough assignment to attach to.
func (t *Tracker) SetHintForRecent(name string, maxAge time.Duration) bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	if len(t.centroids) == 0 || t.lastAssignedAt.IsZero() || t.nowFn().Sub(t.lastAssignedAt) > maxAge {
		return false
	}

	idx := t.lastAssignedIdx // Assign only ever stores a canonical index here.

	// If a DIFFERENT already-canonical cluster already carries a
	// confidently-matching name, that's independent visual evidence
	// idx and that cluster are the same real person -- merge them
	// instead of labeling idx as a second, separate "Person N" for
	// someone the audio-only clustering already has a name for. This
	// is exactly the kind of audio-clustering mistake video hints can
	// correct: two "Person N"s that a video hint resolves to the same
	// name almost certainly ARE the same person, regardless of why the
	// embeddings didn't cluster together.
	for other := range t.centroids {
		if other == idx || t.canonical(other) != other {
			continue // not idx, and not a real still-active cluster
		}
		if existing := t.hints[other]; existing != "" && sameIdentity(name, existing) {
			t.mergeInto(idx, other)
			if isTruncationOf(existing, name) {
				// The new read is the more complete name -- keep it.
				t.hints[other] = name
			}
			return true
		}
	}

	if existing := t.hints[idx]; existing != "" && isTruncationOf(name, existing) {
		// A meeting app's active-speaker tile can truncate a long name
		// (e.g. "Nazanin Rame…") on one read and show it in full on
		// another. Without this check, whichever OCR read happens to
		// land last wins unconditionally -- a later truncated read
		// would silently clobber an already-attached, more complete
		// name. Keep the better one; still report success since a
		// recent-enough assignment to attach to was found either way.
		return true
	}
	t.hints[idx] = name
	return true
}

// isTruncationOf reports whether short reads like a truncated prefix
// of long: long, case-insensitively and after trimming a trailing
// ellipsis/whitespace from either side, starts with short, and isn't
// itself shorter than short. This only ever protects a more-or-
// equally-complete existing hint from being overwritten by a shorter
// one -- it never blocks a genuine correction to a longer/different
// name (long being shorter than short already fails the check).
func isTruncationOf(short, long string) bool {
	short = strings.TrimSpace(strings.TrimRight(strings.TrimSpace(short), "…"))
	long = strings.TrimSpace(strings.TrimRight(strings.TrimSpace(long), "…"))
	if short == "" || long == "" || len(short) > len(long) {
		return false
	}
	return strings.HasPrefix(strings.ToLower(long), strings.ToLower(short))
}

// normalizeName lowercases and trims whitespace/a trailing ellipsis,
// the same normalization isTruncationOf applies before comparing.
func normalizeName(s string) string {
	return strings.ToLower(strings.TrimSpace(strings.TrimRight(strings.TrimSpace(s), "…")))
}

// sameIdentity reports whether a and b look like the same real name --
// stricter than isTruncationOf alone, since this gates an irreversible
// cluster merge rather than just picking which hint text to keep.
// Requires an exact match after normalizing, or a truncation
// relationship (either direction) where the shorter side is at least
// 4 characters -- long enough to rule out merging on a trivial,
// ambiguous fragment (a bare first initial, "Mr", etc.).
func sameIdentity(a, b string) bool {
	na, nb := normalizeName(a), normalizeName(b)
	if na == "" || nb == "" {
		return false
	}
	if na == nb {
		return true
	}
	shorter, longer := na, nb
	if len(longer) < len(shorter) {
		shorter, longer = longer, shorter
	}
	return len(shorter) >= 4 && strings.HasPrefix(longer, shorter)
}

// Reset clears all speaker centroids and hints.
func (t *Tracker) Reset() {
	t.mu.Lock()
	t.centroids = nil
	t.counts = nil
	t.hints = nil
	t.aliasOf = nil
	t.lastAssignedIdx = 0
	t.lastAssignedAt = time.Time{}
	t.mu.Unlock()
}

// NumSpeakers returns the number of distinct identified speakers --
// clusters merged together by mergeInto (see SetHintForRecent) count
// once, not once per merged-away slot.
func (t *Tracker) NumSpeakers() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	n := 0
	for i := range t.centroids {
		if t.canonical(i) == i {
			n++
		}
	}
	return n
}

// updateCentroid updates a centroid with a new embedding using running average.
func (t *Tracker) updateCentroid(idx int, embedding []float32) {
	count := float32(t.counts[idx])
	newCount := count + 1

	for i := range t.centroids[idx] {
		t.centroids[idx][i] = (t.centroids[idx][i]*count + embedding[i]) / newCount
	}
	t.counts[idx] = int(newCount)
}
