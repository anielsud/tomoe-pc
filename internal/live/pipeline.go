package live

import (
	"context"
	"fmt"
	"strings"
	"time"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

	"github.com/sosuke-ai/tomoe-pc/internal/audio"
	"github.com/sosuke-ai/tomoe-pc/internal/session"
	"github.com/sosuke-ai/tomoe-pc/internal/sigfix"
	"github.com/sosuke-ai/tomoe-pc/internal/transcribe"
)

const (
	vadSampleRate = 16000
	vadWindowSize = 512

	// minLiveAudioSamples gates the first live-partial emission (see
	// liveState/emitLivePartial): a speaker assignment needs enough
	// accumulated speech for a stable embedding, so pass 1's partial
	// text is buffered silently until this much speech has accumulated,
	// then shown (and grows from there on every subsequent partial).
	// Half a second is imperceptible as a delay but avoids computing a
	// speaker off a handful of frames.
	minLiveAudioSamples = vadSampleRate / 2
)

// refinementJob is one pass-2 request: re-decode a completed segment's
// audio through the (offline, higher-quality) Engine and supersede the
// pass-1 text already emitted for it.
type refinementJob struct {
	id        string
	samples   []float32
	speaker   string
	startTime float64
	endTime   float64
	source    SourceType
	pass1Text string // fallback if refinement produces nothing usable
}

// liveState tracks pass 1's in-progress utterance for one pipeline: the
// growing partial text, whether (and what) a live segment has already
// been shown for it, and the audio accumulated so far for an early —
// but, once decided, stable — speaker assignment. Zero value is "no
// utterance in progress."
type liveState struct {
	partial   string
	id        string
	speaker   string
	startTime float64
	audio     []float32
}

func (ls *liveState) reset() {
	*ls = liveState{}
}

// processPipeline runs a single source pipeline: reads windows → VAD → transcribe → emit segments.
func (c *Coordinator) processPipeline(ctx context.Context, sc *audio.StreamCapturer, source SourceType) {
	defer c.wg.Done()

	// Create a VAD instance for this source
	vadConfig := &sherpa.VadModelConfig{
		SileroVad: sherpa.SileroVadModelConfig{
			Model:              c.cfg.VADPath,
			Threshold:          0.5,
			MinSilenceDuration: 0.5,
			MinSpeechDuration:  0.25,
			WindowSize:         vadWindowSize,
			MaxSpeechDuration:  30.0,
		},
		SampleRate: vadSampleRate,
		NumThreads: 1,
		Provider:   "cpu",
	}

	vad := sherpa.NewVoiceActivityDetector(vadConfig, 60.0)
	if vad == nil {
		return
	}
	defer sherpa.DeleteVoiceActivityDetector(vad)
	sigfix.AfterSherpa()

	// Pass 1 (optional): a streaming session for this source, giving
	// incremental partial text as audio arrives rather than only once a
	// whole VAD segment completes. Nil (falls back to today's
	// single-pass, synchronous-decode-on-completion behavior) if the
	// realtime model isn't configured/available.
	var streamSess transcribe.StreamingSession
	if c.cfg.StreamingEngine != nil {
		var err error
		streamSess, err = c.cfg.StreamingEngine.NewSession()
		if err != nil {
			fmt.Printf("live: failed to start streaming session for %s (falling back to non-realtime): %v\n", source, err)
			streamSess = nil
		} else {
			defer streamSess.Close()
		}
	}
	var live liveState

	windows := sc.Windows()
	for {
		select {
		case <-ctx.Done():
			// Flush VAD and process remaining segments
			vad.Flush()
			c.drainVAD(vad, source, streamSess, &live)
			return

		case window, ok := <-windows:
			if !ok {
				// Channel closed — capturer stopped
				vad.Flush()
				c.drainVAD(vad, source, streamSess, &live)
				return
			}

			// Feed window to VAD (must be exactly windowSize)
			if len(window) == vadWindowSize {
				vad.AcceptWaveform(window)
				isSpeech := vad.IsSpeech()

				if streamSess != nil {
					if isSpeech {
						// Only accumulate speech, not silence -- keeps
						// the eventual speaker embedding clean.
						live.audio = append(live.audio, window...)
					}
					text, err := streamSess.Feed(window)
					if err == nil && text != live.partial {
						live.partial = text
						if text != "" {
							c.emitLivePartial(source, &live, text)
						}
					}
				}

				// Signal activity when VAD detects ongoing speech
				if isSpeech {
					select {
					case c.activityCh <- struct{}{}:
					default:
					}
				}
			}

			// Process any completed speech segments
			c.drainVAD(vad, source, streamSess, &live)
		}
	}
}

// emitLivePartial publishes pass 1's growing text for the utterance in
// progress: the first call (once enough audio has accumulated for a
// speaker assignment — see minLiveAudioSamples) creates a new "live"
// segment; every call after that updates the same segment ID in place.
// "live" (not "pending") signals to consumers that this text may still
// change because the person is still talking, not just because pass 2
// hasn't run yet — see drainVAD, which is what actually transitions a
// segment to "pending" once the utterance itself is done.
func (c *Coordinator) emitLivePartial(source SourceType, live *liveState, text string) {
	if live.id == "" {
		if len(live.audio) < minLiveAudioSamples {
			return
		}
		live.id = c.nextSegID()
		live.speaker = c.assignSpeaker(source, live.audio)
		live.startTime = c.elapsed()

		seg := session.Segment{
			ID:        live.id,
			Speaker:   live.speaker,
			Text:      text,
			StartTime: live.startTime,
			EndTime:   c.elapsed(),
			Source:    string(source),
			Language:  "en", // the streaming engine is English-only today
			Status:    "live",
		}
		select {
		case c.segmentCh <- seg:
		default:
		}
		return
	}

	seg := session.Segment{
		ID:        live.id,
		Speaker:   live.speaker,
		Text:      text,
		StartTime: live.startTime,
		EndTime:   c.elapsed(),
		Source:    string(source),
		Language:  "en",
		Status:    "live",
	}
	select {
	case c.segmentUpdateCh <- seg:
	default:
	}
}

// drainVAD transcribes all completed speech segments from the VAD.
// streamSess/live are pass 1's streaming state (see processPipeline)
// -- nil/unused when two-pass transcription isn't configured, in which
// case this behaves exactly as before: one synchronous decode per
// completed segment, emitted as final immediately.
func (c *Coordinator) drainVAD(vad *sherpa.VoiceActivityDetector, source SourceType, streamSess transcribe.StreamingSession, live *liveState) {
	for !vad.IsEmpty() {
		segment := vad.Front()
		vad.Pop()

		if len(segment.Samples) == 0 {
			continue
		}

		// Apply DSP pipeline
		samples := audio.ProcessPipeline(segment.Samples, vadSampleRate, -40)
		duration := float64(len(samples)) / vadSampleRate
		endTime := c.elapsed()
		startTime := endTime - duration

		if streamSess != nil {
			text := strings.TrimSpace(live.partial)
			streamSess.Reset()

			if text == "" {
				live.reset()
				continue
			}

			// Reuse the speaker already assigned when the live partial
			// first appeared, if there was one — recomputing from this
			// segment's full audio would risk a different "Person N"
			// for the very utterance that was already shown under the
			// first one, and would double-count this utterance into
			// the tracker's centroid. An utterance that finished before
			// accumulating minLiveAudioSamples never got a live partial
			// at all, so falls back to computing it fresh here, exactly
			// as before live partials existed.
			id := live.id
			spk := live.speaker
			wasLive := id != ""
			if !wasLive {
				id = c.nextSegID()
				spk = c.assignSpeaker(source, samples)
			}

			seg := session.Segment{
				ID:        id,
				Speaker:   spk,
				Text:      text,
				StartTime: startTime,
				EndTime:   endTime,
				Source:    string(source),
				Language:  "en",
				Status:    "pending",
			}
			if wasLive {
				select {
				case c.segmentUpdateCh <- seg:
				default:
				}
			} else {
				select {
				case c.segmentCh <- seg:
				default:
				}
			}

			select {
			case c.refineCh <- refinementJob{
				id: id, samples: samples, speaker: spk,
				startTime: startTime, endTime: endTime, source: source,
				pass1Text: text,
			}:
			default:
				// Refinement queue is backed up -- pass 1's text stands
				// as final rather than blocking the live pipeline.
			}

			live.reset()
		} else {
			// Single-pass (no streaming engine configured): unchanged
			// from before this feature existed.
			spk := c.assignSpeaker(source, samples)

			c.transcribeMu.Lock()
			result, err := c.cfg.Engine.TranscribeDirect(samples)
			c.transcribeMu.Unlock()

			if err != nil || result == nil || strings.TrimSpace(result.Text) == "" {
				continue
			}

			seg := session.Segment{
				ID:        c.nextSegID(),
				Speaker:   spk,
				Text:      strings.TrimSpace(result.Text),
				StartTime: startTime,
				EndTime:   endTime,
				Source:    string(source),
				Language:  result.Language,
			}
			select {
			case c.segmentCh <- seg:
			default:
			}
		}

		// Update counters
		if source == SourceMic {
			c.micCount.Add(1)
		} else {
			c.monitorCount.Add(1)
		}
	}
}

// refineWorker drains refinement jobs (pass 2): re-decode a segment's
// audio through the offline Engine (allowed to be slower/better than
// pass 1's streaming decode) and supersede its text via segmentUpdateCh.
// Runs until refineCh is closed (see Start) rather than on ctx.Done(),
// so a job queued right at shutdown still gets processed.
func (c *Coordinator) refineWorker() {
	defer c.refineWG.Done()

	for job := range c.refineCh {
		c.transcribeMu.Lock()
		result, err := c.cfg.Engine.TranscribeDirect(job.samples)
		c.transcribeMu.Unlock()

		text := job.pass1Text
		lang := "en"
		if err == nil && result != nil && strings.TrimSpace(result.Text) != "" {
			text = strings.TrimSpace(result.Text)
			if result.Language != "" {
				lang = result.Language
			}
		}
		// Either way, Status becomes "" (final): refinement failing just
		// means pass 1's text is what stands, not that the segment stays
		// marked "pending" forever.
		seg := session.Segment{
			ID:        job.id,
			Speaker:   job.speaker,
			Text:      text,
			StartTime: job.startTime,
			EndTime:   job.endTime,
			Source:    string(job.source),
			Language:  lang,
		}
		select {
		case c.segmentUpdateCh <- seg:
		default:
		}
	}
}

// assignSpeaker determines the speaker label for a segment.
func (c *Coordinator) assignSpeaker(source SourceType, samples []float32) string {
	if source == SourceMic {
		return "You"
	}

	if c.cfg.SkipMonitorDiarization {
		return "System Audio"
	}

	// For monitor source, try speaker embedding + clustering
	if c.cfg.Embedder != nil && c.cfg.Tracker != nil {
		embedding, err := c.cfg.Embedder.Extract(samples)
		if err == nil && len(embedding) > 0 {
			duration := time.Duration(float64(len(samples)) / vadSampleRate * float64(time.Second))
			label, needsHint := c.cfg.Tracker.Assign(embedding, duration)
			if needsHint {
				// Non-blocking: a video-hint check is worth doing right
				// away rather than waiting for videohint.Poll's next
				// scheduled tick, but this pipeline must never stall
				// waiting for a slow/absent consumer.
				select {
				case c.hintNeededCh <- struct{}{}:
				default:
				}
			}
			return label
		}
	}

	return "Other"
}
