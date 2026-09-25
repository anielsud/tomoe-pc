package live

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sosuke-ai/tomoe-pc/internal/audio"
	"github.com/sosuke-ai/tomoe-pc/internal/session"
	"github.com/sosuke-ai/tomoe-pc/internal/speaker"
	"github.com/sosuke-ai/tomoe-pc/internal/transcribe"
)

// SourceType identifies an audio source.
type SourceType string

const (
	SourceMic     SourceType = "mic"
	SourceMonitor SourceType = "monitor"
)

// Config configures the live transcription coordinator.
type Config struct {
	// MicCapturer is the microphone stream capturer (optional, nil to skip mic).
	MicCapturer *audio.StreamCapturer
	// MonitorCapturer is the monitor stream capturer (optional, nil to skip monitor).
	MonitorCapturer *audio.StreamCapturer
	// Engine is the transcription engine (required).
	Engine transcribe.Engine
	// Embedder is the speaker embedding extractor (optional, nil to skip speaker ID for monitor).
	Embedder *speaker.Embedder
	// Tracker is the speaker clustering tracker (optional, nil to skip speaker ID for monitor).
	Tracker *speaker.Tracker
	// VADPath is the path to the Silero VAD model file.
	VADPath string
	// SegmentBufferSize is the channel buffer for output segments.
	SegmentBufferSize int
	// StreamingEngine enables two-pass transcription (optional, nil to
	// keep today's single-pass behavior): pass 1 is StreamingEngine,
	// decoded incrementally so text appears as it's spoken rather than
	// only once a whole VAD segment completes; pass 2 re-decodes the
	// same completed segment through Engine (offline, allowed to be
	// slower/better) and supersedes pass 1's text once ready. See
	// pipeline.go's drainVAD.
	StreamingEngine transcribe.StreamingEngine
	// SkipMonitorDiarization disables speaker embedding/clustering for
	// the monitor source, labeling every monitor-source segment
	// "System Audio" instead of attempting "Person N" identification.
	// Set this when the monitor source is a whole-system audio tap
	// (macOS's "Everything" source-picker option) rather than one
	// specific app's audio: a system-wide tap can mix multiple
	// unrelated audio streams together (notifications, music, several
	// apps at once), so per-embedding speaker clustering isn't
	// meaningful there the way it is for one app's own audio. Defaults
	// to false (diarize), unchanged from before this field existed.
	SkipMonitorDiarization bool
}

// Stats holds runtime statistics about the coordinator.
type Stats struct {
	MicSegments     int
	MonitorSegments int
	Duration        time.Duration
}

// Coordinator manages one or two live transcription pipelines (mic + monitor).
type Coordinator struct {
	cfg          Config
	segmentCh    chan session.Segment
	activityCh   chan struct{} // signalled when VAD detects ongoing speech
	// hintNeededCh is signalled when a monitor-source speaker with no
	// video hint yet is heard. The bool is a priority flag: true means
	// "bypass videohint.Poll's normal trigger debounce" (used for a
	// brand-new speaker, where a fast double-shot attempt matters more
	// than the usual rate limit), false is the ordinary debounced case.
	hintNeededCh chan bool
	startTime    time.Time

	// segmentUpdateCh carries revisions to a segment already sent on
	// segmentCh (same ID) -- pass 2's refined text superseding pass 1's.
	// Only used when cfg.StreamingEngine is set.
	segmentUpdateCh chan session.Segment
	// refineCh queues pass-2 refinement jobs from drainVAD to
	// refineWorker. Deliberately drained to completion on shutdown (see
	// Start's closer goroutine) rather than abandoned on ctx.Done(), so
	// a segment queued for refinement right as the session stops still
	// gets its update rather than staying "pending" forever.
	refineCh chan refinementJob
	refineWG sync.WaitGroup

	cancel context.CancelFunc
	wg     sync.WaitGroup

	micCount     atomic.Int64
	monitorCount atomic.Int64

	// transcribeMu serializes transcription calls (sherpa-onnx thread safety).
	transcribeMu sync.Mutex

	// segIDCounter generates unique segment IDs.
	segIDCounter atomic.Int64
}

// New creates a new Coordinator with the given configuration.
func New(cfg Config) *Coordinator {
	bufSize := cfg.SegmentBufferSize
	if bufSize <= 0 {
		bufSize = 64
	}
	return &Coordinator{
		cfg:             cfg,
		segmentCh:       make(chan session.Segment, bufSize),
		segmentUpdateCh: make(chan session.Segment, bufSize),
		activityCh:      make(chan struct{}, 1),
		hintNeededCh:    make(chan bool, 1),
		refineCh:        make(chan refinementJob, bufSize),
	}
}

// Start begins processing audio from all configured sources.
func (c *Coordinator) Start(ctx context.Context) error {
	ctx, c.cancel = context.WithCancel(ctx)
	c.startTime = time.Now()

	hasMic := c.cfg.MicCapturer != nil
	hasMonitor := c.cfg.MonitorCapturer != nil

	if !hasMic && !hasMonitor {
		return fmt.Errorf("at least one audio source is required")
	}

	if hasMic {
		if err := c.cfg.MicCapturer.Start(); err != nil {
			return fmt.Errorf("starting mic capturer: %w", err)
		}
		c.wg.Add(1)
		go c.processPipeline(ctx, c.cfg.MicCapturer, SourceMic)
	}

	if hasMonitor {
		if err := c.cfg.MonitorCapturer.Start(); err != nil {
			if hasMic {
				_ = c.cfg.MicCapturer.Stop()
			}
			return fmt.Errorf("starting monitor capturer: %w", err)
		}
		c.wg.Add(1)
		go c.processPipeline(ctx, c.cfg.MonitorCapturer, SourceMonitor)
	}

	c.refineWG.Add(1)
	go c.refineWorker()

	// Closer goroutine: waits for all pipelines to finish (including
	// their final VAD flush, which may still enqueue a last refinement
	// job or two), only THEN closes refineCh -- so refineWorker keeps
	// draining right up until every queued job has actually run, never
	// abandoning a "pending" segment mid-refinement -- and only after
	// refineWorker itself has finished does it close the two segment
	// channels, since a late refinement update would otherwise arrive
	// after SegmentUpdates() looks closed to a consumer.
	go func() {
		c.wg.Wait()
		close(c.refineCh)
		c.refineWG.Wait()
		close(c.segmentCh)
		close(c.segmentUpdateCh)
	}()

	return nil
}

// Segments returns the channel that receives newly-transcribed segments.
func (c *Coordinator) Segments() <-chan session.Segment {
	return c.segmentCh
}

// SegmentUpdates returns the channel that receives revisions to a
// segment already delivered on Segments() (same ID): pass 2's refined
// text superseding pass 1's, once ready. Only fires when the Config
// this Coordinator was created with set StreamingEngine.
func (c *Coordinator) SegmentUpdates() <-chan session.Segment {
	return c.segmentUpdateCh
}

// Activity returns a channel signalled when VAD detects ongoing speech.
// Use this to reset silence timers — speech is in progress even though
// no completed segment has been emitted yet.
func (c *Coordinator) Activity() <-chan struct{} {
	return c.activityCh
}

// HintNeeded returns a channel signalled whenever a monitor-source
// speaker with no video hint attached yet is heard. Wire this into
// videohint.Poll's trigger parameter so a still-unlabeled speaker gets
// an immediate ring/OCR attempt instead of waiting for Poll's next
// scheduled tick — see internal/speaker.Tracker.Assign's needsHint
// return value, which is what actually decides when this fires. The
// bool is a priority flag (true = bypass Poll's normal trigger
// debounce) -- see signalHintNeeded.
func (c *Coordinator) HintNeeded() <-chan bool {
	return c.hintNeededCh
}

// signalHintNeeded sends on hintNeededCh without blocking -- if the
// single-slot buffer is already full (Poll hasn't drained the
// previous signal yet), this drops rather than stalling the
// transcription pipeline, same as every other non-blocking channel
// send in this package.
func (c *Coordinator) signalHintNeeded(priority bool) {
	select {
	case c.hintNeededCh <- priority:
	default:
	}
}

// Stop stops all pipelines and waits for them to finish.
// Closes the underlying audio capturers to release device handles.
func (c *Coordinator) Stop() {
	if c.cancel != nil {
		c.cancel()
	}
	if c.cfg.MicCapturer != nil {
		_ = c.cfg.MicCapturer.Stop()
	}
	if c.cfg.MonitorCapturer != nil {
		_ = c.cfg.MonitorCapturer.Stop()
	}
	c.wg.Wait()

	// Close capturers to release underlying audio devices.
	// Must happen after wg.Wait() so pipelines are done reading.
	if c.cfg.MicCapturer != nil {
		c.cfg.MicCapturer.Close()
	}
	if c.cfg.MonitorCapturer != nil {
		c.cfg.MonitorCapturer.Close()
	}
}

// Stats returns runtime statistics.
func (c *Coordinator) Stats() Stats {
	return Stats{
		MicSegments:     int(c.micCount.Load()),
		MonitorSegments: int(c.monitorCount.Load()),
		Duration:        time.Since(c.startTime),
	}
}

// AudioSamples returns accumulated audio from all sources (concatenated).
// For single-source sessions only. Use MicSamples/MonitorSamples for per-track access.
func (c *Coordinator) AudioSamples() []float32 {
	var all []float32
	if c.cfg.MicCapturer != nil {
		all = append(all, c.cfg.MicCapturer.AllSamples()...)
	}
	if c.cfg.MonitorCapturer != nil {
		all = append(all, c.cfg.MonitorCapturer.AllSamples()...)
	}
	return all
}

// MicSamples returns accumulated audio from the mic source, or nil if no mic.
func (c *Coordinator) MicSamples() []float32 {
	if c.cfg.MicCapturer == nil {
		return nil
	}
	return c.cfg.MicCapturer.AllSamples()
}

// MonitorSamples returns accumulated audio from the monitor source, or nil if no monitor.
func (c *Coordinator) MonitorSamples() []float32 {
	if c.cfg.MonitorCapturer == nil {
		return nil
	}
	return c.cfg.MonitorCapturer.AllSamples()
}

// IsDualSource returns true if both mic and monitor sources are configured.
func (c *Coordinator) IsDualSource() bool {
	return c.cfg.MicCapturer != nil && c.cfg.MonitorCapturer != nil
}

func (c *Coordinator) nextSegID() string {
	id := c.segIDCounter.Add(1)
	return fmt.Sprintf("seg-%d", id)
}

func (c *Coordinator) elapsed() float64 {
	return time.Since(c.startTime).Seconds()
}
