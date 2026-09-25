package daemon

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/google/uuid"

	"github.com/sosuke-ai/tomoe-pc/internal/audio"
	"github.com/sosuke-ai/tomoe-pc/internal/config"
	"github.com/sosuke-ai/tomoe-pc/internal/hotkey"
	"github.com/sosuke-ai/tomoe-pc/internal/live"
	"github.com/sosuke-ai/tomoe-pc/internal/meeting"
	"github.com/sosuke-ai/tomoe-pc/internal/meetingaudio"
	"github.com/sosuke-ai/tomoe-pc/internal/models"
	"github.com/sosuke-ai/tomoe-pc/internal/platform"
	"github.com/sosuke-ai/tomoe-pc/internal/session"
	"github.com/sosuke-ai/tomoe-pc/internal/speaker"
	"github.com/sosuke-ai/tomoe-pc/internal/transcribe"
	"github.com/sosuke-ai/tomoe-pc/internal/videohint"
)

// Daemon orchestrates the hotkey → capture → transcribe → clipboard pipeline,
// and optionally supports meeting recording with live transcription.
type Daemon struct {
	cfg     *config.Config
	engines *transcribe.EngineSet
	svc     *platform.Services

	// Meeting mode dependencies (optional — nil disables meeting mode)
	meetingHotkey hotkey.Listener
	embedder      *speaker.Embedder
	tracker       *speaker.Tracker
	store         *session.Store
	modelStatus   *models.Status
	detector      *meeting.Detector
	// streamingEngine powers live transcription's realtime ("pass 1")
	// pass; nil if the English streaming model isn't downloaded, in
	// which case sessions fall back to today's single-pass behavior.
	streamingEngine transcribe.StreamingEngine

	// Background save pipeline. Each meeting stop enqueues; one worker
	// drains the queue serially so concurrent diarization can't corrupt
	// sherpa-onnx state. Buffer keeps stop→start non-blocking under burst.
	saveQueue chan *meetingState
	saveWG    sync.WaitGroup

	tray *daemonTray
}

const saveQueueDepth = 16

// MeetingOpts holds optional dependencies for meeting recording mode.
type MeetingOpts struct {
	MeetingHotkey   hotkey.Listener
	Embedder        *speaker.Embedder
	Tracker         *speaker.Tracker
	Store           *session.Store
	ModelStatus     *models.Status
	Detector        *meeting.Detector
	StreamingEngine transcribe.StreamingEngine
}

// New creates a Daemon with the given dependencies.
func New(cfg *config.Config, engines *transcribe.EngineSet, svc *platform.Services, opts *MeetingOpts) *Daemon {
	d := &Daemon{
		cfg:     cfg,
		engines: engines,
		svc:     svc,
	}
	if opts != nil {
		d.meetingHotkey = opts.MeetingHotkey
		d.embedder = opts.Embedder
		d.tracker = opts.Tracker
		d.store = opts.Store
		d.modelStatus = opts.ModelStatus
		d.detector = opts.Detector
		d.streamingEngine = opts.StreamingEngine
	}
	return d
}

// Run starts the daemon main loop. Blocks until SIGTERM/SIGINT or context cancel.
func (d *Daemon) Run(ctx context.Context) error {
	// Write PID file
	if err := WritePID(); err != nil {
		return fmt.Errorf("writing PID file: %w", err)
	}
	defer RemovePID()

	// Start the background save worker. Drains pending saves on shutdown
	// so we never lose a session that was queued before the daemon exited.
	d.saveQueue = make(chan *meetingState, saveQueueDepth)
	d.saveWG.Add(1)
	go d.saveWorker()
	defer func() {
		close(d.saveQueue)
		d.saveWG.Wait()
	}()

	// Start system tray — use config for language list (not engine availability)
	defaultLang := "en"
	var languages []string
	if d.cfg.Multilingual.Enabled && len(d.cfg.Multilingual.Languages) > 0 {
		languages = d.cfg.Multilingual.Languages
		if d.cfg.Multilingual.DefaultLang != "" {
			defaultLang = d.cfg.Multilingual.DefaultLang
		}
	}
	// runWithTray owns the system tray for the process lifetime. On Linux
	// this just starts systray in a goroutine and calls body immediately
	// (unchanged from before); on darwin, Cocoa requires systray's native
	// loop to run on the actual OS main thread, so runWithTray calls
	// systray.Run itself (blocking) and runs body in a goroutine from
	// inside onReady instead. See run_linux.go / run_darwin.go.
	return runWithTray(languages, defaultLang, func(tray *daemonTray) error {
		d.tray = tray
		defer d.tray.Close()

		// Register dictation hotkey
		if err := d.svc.Hotkey.Register(); err != nil {
			return fmt.Errorf("registering dictation hotkey: %w", err)
		}

		// Register meeting hotkey (optional)
		var meetingCh <-chan struct{}
		if d.meetingHotkey != nil {
			if err := d.meetingHotkey.Register(); err != nil {
				fmt.Fprintf(os.Stderr, "Warning: could not register meeting hotkey: %v\n", err)
			} else {
				meetingCh = d.meetingHotkey.Keydown()
			}
		}

		// Start meeting auto-detector (optional)
		var detectCh <-chan meeting.MeetingEvent
		if d.detector != nil {
			if err := d.detector.Start(ctx); err != nil {
				fmt.Fprintf(os.Stderr, "Warning: meeting auto-detect unavailable: %v\n", err)
			} else {
				detectCh = d.detector.Events()
				defer d.detector.Stop()
			}
		}

		// Signal handling
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)
		defer signal.Stop(sigCh)

		bindings := d.cfg.Hotkey.Binding + " for dictation"
		if meetingCh != nil {
			meetingBinding := d.cfg.Hotkey.MeetingBinding
			if meetingBinding == "" {
				meetingBinding = "Super+Shift+X"
			}
			bindings += ", " + meetingBinding + " for meeting"
		}
		_ = d.svc.Notifier.Send("Tomoe", fmt.Sprintf("Ready — %s", bindings))
		fmt.Printf("Tomoe daemon started. Hotkeys: %s\n", bindings)

		// State
		var dictState *streamingDictation
		var meetState *meetingState

		// Auto-stop channel — signalled by streaming dictation on silence timeout
		autoStopCh := make(chan struct{}, 1)

		// toggleDictation handles start/stop dictation from any trigger (hotkey or tray).
		toggleDictation := func(lang string) {
			if meetState != nil {
				return // ignore dictation while meeting active
			}
			if dictState == nil {
				ds, err := d.startStreamingDictation(ctx, autoStopCh, lang)
				if err != nil {
					_ = d.svc.Notifier.Send("Tomoe", fmt.Sprintf("Dictation failed: %v", err))
					fmt.Printf("Error starting dictation: %v\n", err)
					return
				}
				dictState = ds
				d.tray.SetDictating()
				_ = d.svc.Notifier.Send("Tomoe", fmt.Sprintf("Dictating (%s)...", lang))
				fmt.Printf("Streaming dictation started (%s)...\n", lang)
			} else {
				d.stopDictation(dictState)
				dictState = nil
			}
		}

		// toggleMeeting handles start/stop meeting from any trigger (hotkey or tray).
		toggleMeeting := func(lang string) {
			if dictState != nil {
				return // ignore meeting while dictating
			}
			if meetState == nil {
				ms, err := d.startMeetingWithPlatform(ctx, "", lang)
				if err != nil {
					_ = d.svc.Notifier.Send("Tomoe", fmt.Sprintf("Meeting start failed: %v", err))
					fmt.Printf("Error starting meeting: %v\n", err)
					return
				}
				meetState = ms
				d.tray.SetMeetingRecording()
				_ = d.svc.Notifier.Send("Tomoe", fmt.Sprintf("Meeting recording started (%s)", lang))
				fmt.Printf("Meeting recording started (%s)...\n", lang)
			} else {
				d.stopMeeting(meetState)
				meetState = nil
			}
		}

		for {
			select {
			case <-ctx.Done():
				fmt.Println("Shutting down...")
				d.cleanupAll(dictState, meetState)
				return nil

			case sig := <-sigCh:
				fmt.Printf("Received %s, shutting down...\n", sig)
				d.cleanupAll(dictState, meetState)
				return nil

			case <-d.tray.quitCh:
				fmt.Println("Quit from tray, shutting down...")
				d.cleanupAll(dictState, meetState)
				return nil

			case <-d.svc.Hotkey.Keydown():
				toggleDictation(defaultLang)

			case lang := <-d.tray.dictationCh:
				toggleDictation(lang)

			case <-autoStopCh:
				if dictState != nil {
					timeout := d.silenceTimeout()
					fmt.Printf("Auto-stopping dictation after %.0fs of silence\n", timeout.Seconds())
					_ = d.svc.Notifier.Send("Tomoe", "Dictation auto-stopped (silence)")
					d.stopDictation(dictState)
					dictState = nil
				}

			case <-meetingCh:
				toggleMeeting(defaultLang)

			case lang := <-d.tray.meetingCh:
				toggleMeeting(lang)

			case evt := <-detectCh:
				switch evt.Type {
				case meeting.MeetingStarted:
					if dictState != nil || meetState != nil {
						break // ignore if already recording
					}
					platform := string(evt.Platform)
					ms, err := d.startMeetingWithPlatform(ctx, platform, defaultLang)
					if err != nil {
						_ = d.svc.Notifier.Send("Tomoe", fmt.Sprintf("Auto-detect meeting start failed: %v", err))
						fmt.Printf("Error auto-starting meeting: %v\n", err)
						break
					}
					meetState = ms
					d.tray.SetMeetingRecording()
					msg := fmt.Sprintf("%s meeting detected — recording started", platform)
					_ = d.svc.Notifier.Send("Tomoe", msg)
					fmt.Println(msg)
				case meeting.MeetingStopped:
					if meetState != nil {
						d.stopMeeting(meetState)
						meetState = nil
					}
				}
			}
		}
	})
}

func (d *Daemon) cleanupAll(dict *streamingDictation, meet *meetingState) {
	if dict != nil {
		d.stopDictation(dict)
	}
	if meet != nil {
		d.stopMeeting(meet)
	}
}

// --- Streaming Dictation ---

type streamingDictation struct {
	coordinator *live.Coordinator
	streamer    *live.DictationStreamer
	cancel      context.CancelFunc
}

func (d *Daemon) silenceTimeout() time.Duration {
	t := d.cfg.Output.SilenceTimeout
	if t <= 0 {
		t = 5.0
	}
	return time.Duration(t * float64(time.Second))
}

func (d *Daemon) startStreamingDictation(ctx context.Context, autoStopCh chan struct{}, lang string) (*streamingDictation, error) {
	var vadPath string
	if d.modelStatus != nil {
		vadPath = d.modelStatus.VADPath
	}

	micDevice := d.cfg.Audio.Device
	if micDevice == "" {
		micDevice = "default"
	}
	micCapturer, err := audio.NewCapturer(micDevice, audio.Input)
	if err != nil {
		return nil, fmt.Errorf("creating mic capturer: %w", err)
	}

	cfg := live.Config{
		Engine:            d.engines.Get(lang),
		MicCapturer:       audio.NewStreamCapturer(micCapturer, audio.DefaultWindowSize, 128),
		VADPath:           vadPath,
		SegmentBufferSize: 32,
	}

	dictCtx, cancel := context.WithCancel(ctx)
	coordinator := live.New(cfg)
	if err := coordinator.Start(dictCtx); err != nil {
		cfg.MicCapturer.Close()
		cancel()
		return nil, fmt.Errorf("starting coordinator: %w", err)
	}

	// Re-grab hotkeys — audio device init can interfere with X11 key grabs
	hotkey.ReGrabAll()

	streamer := live.NewDictationStreamer(coordinator, live.DictationConfig{
		Clipboard:      d.svc.Clipboard,
		UseClipboard:   d.cfg.Output.Clipboard,
		AutoPaste:      d.cfg.Output.AutoPaste,
		SilenceTimeout: d.silenceTimeout(),
		OnAutoStop: func() {
			select {
			case autoStopCh <- struct{}{}:
			default:
			}
		},
	})

	return &streamingDictation{
		coordinator: coordinator,
		streamer:    streamer,
		cancel:      cancel,
	}, nil
}

func (d *Daemon) stopDictation(ds *streamingDictation) {
	ds.coordinator.Stop()
	ds.cancel()
	<-ds.streamer.Done()
	d.tray.SetIdle()
	// Re-grab hotkeys — audio/tray operations can interfere with X11 grabs
	hotkey.ReGrabAll()
	fmt.Println("Dictation stopped.")
}

// --- Meeting Recording ---

type meetingState struct {
	coordinator     *live.Coordinator
	session         *session.Session
	done            chan struct{}
	videoHintCancel context.CancelFunc
}

func (d *Daemon) startMeetingWithPlatform(ctx context.Context, platform string, lang string) (*meetingState, error) {
	var vadPath string
	if d.modelStatus != nil {
		vadPath = d.modelStatus.VADPath
	}

	cfg := live.Config{
		Engine:            d.engines.Get(lang),
		Embedder:          d.embedder,
		Tracker:           d.tracker,
		VADPath:           vadPath,
		SegmentBufferSize: 64,
	}
	// Realtime pass is English-only (see internal/transcribe's
	// StreamingEngine); other languages keep today's single-pass path.
	if lang == "en" {
		cfg.StreamingEngine = d.streamingEngine
	}

	// Set up mic capturer
	micDevice := d.cfg.Audio.Device
	if micDevice == "" {
		micDevice = "default"
	}
	micCapturer, err := audio.NewCapturer(micDevice, audio.Input)
	if err != nil {
		return nil, fmt.Errorf("creating mic capturer: %w", err)
	}
	cfg.MicCapturer = audio.NewStreamCapturer(micCapturer, audio.DefaultWindowSize, 128)

	// Set up monitor capturer (optional) — the second audio source: a
	// PulseAudio monitor device on Linux (config.toml's monitor_device,
	// falling back to the system default if empty), or on macOS a
	// specific app's PID or "everything" (see internal/meetingaudio) --
	// the CLI daemon has no interactive picker, so this is
	// config.toml-driven; leaving monitor_device empty means "no
	// monitor" on macOS specifically (a PID wouldn't survive an app
	// restart anyway), unlike Linux's "use the default device" meaning
	// for the same empty value.
	monCapturer, err := meetingaudio.NewMonitorSource(d.cfg.Meeting.MonitorDevice)
	if err != nil {
		cfg.MicCapturer.Close()
		return nil, fmt.Errorf("creating monitor capturer: %w", err)
	}
	if monCapturer != nil {
		cfg.MonitorCapturer = monCapturer
		cfg.SkipMonitorDiarization = d.cfg.Meeting.MonitorDevice == "everything"
	}

	// Reset speaker tracker
	if d.tracker != nil {
		d.tracker.Reset()
	}

	coordinator := live.New(cfg)
	if err := coordinator.Start(ctx); err != nil {
		cfg.MicCapturer.Close()
		if cfg.MonitorCapturer != nil {
			cfg.MonitorCapturer.Close()
		}
		return nil, fmt.Errorf("starting coordinator: %w", err)
	}

	// Re-grab hotkeys — audio device init can interfere with X11 key grabs
	hotkey.ReGrabAll()

	// Create session
	var sources []string
	sources = append(sources, "mic")
	if cfg.MonitorCapturer != nil {
		sources = append(sources, "monitor")
	}

	title := fmt.Sprintf("Meeting %s", time.Now().Format("2006-01-02 15:04"))
	if platform != "" {
		title = fmt.Sprintf("%s Meeting %s", platform, time.Now().Format("2006-01-02 15:04"))
	}

	sess := &session.Session{
		ID:        uuid.New().String(),
		Title:     title,
		Platform:  platform,
		CreatedAt: time.Now(),
		Sources:   sources,
	}

	// Drain segments (and, for two-pass sessions, their refinements) in
	// their own goroutines, closing done only once both have finished —
	// same reasoning as internal/live's own closer goroutine: a
	// refinement queued right at shutdown should still get applied
	// before this daemon considers the meeting fully stopped.
	done := make(chan struct{})
	var mu sync.Mutex
	var drainWG sync.WaitGroup
	drainWG.Add(2)
	go func() {
		defer drainWG.Done()
		for seg := range coordinator.Segments() {
			mu.Lock()
			sess.Segments = append(sess.Segments, seg)
			mu.Unlock()
			if seg.Language != "" {
				fmt.Printf("[%s] [%s] %s: %s\n", formatTimestamp(seg.StartTime), seg.Language, seg.Speaker, seg.Text)
			} else {
				fmt.Printf("[%s] %s: %s\n", formatTimestamp(seg.StartTime), seg.Speaker, seg.Text)
			}
		}
	}()
	go func() {
		defer drainWG.Done()
		for seg := range coordinator.SegmentUpdates() {
			mu.Lock()
			for i := range sess.Segments {
				if sess.Segments[i].ID == seg.ID {
					sess.Segments[i] = seg
					break
				}
			}
			mu.Unlock()
			fmt.Printf("[%s] %s: %s (refined)\n", formatTimestamp(seg.StartTime), seg.Speaker, seg.Text)
		}
	}()
	go func() {
		drainWG.Wait()
		close(done)
	}()

	// Screen-based speaker-label hints (macOS only; no-op on Linux —
	// see internal/videohint). Its own context, not the daemon-lifetime
	// ctx passed in above, since it must stop when this meeting does,
	// not when the daemon exits — same reasoning as streamingDictation's
	// cancel field elsewhere in this file.
	videoHintCtx, videoHintCancel := context.WithCancel(ctx)
	videoHintEvents := make(chan videohint.Event, 32)
	pollInterval, triggerDebounce := d.cfg.Meeting.VideoHintTiming()
	go videohint.Poll(videoHintCtx, pollInterval, triggerDebounce, coordinator.HintNeeded(), videoHintEvents)
	go func() {
		for {
			select {
			case <-videoHintCtx.Done():
				return
			case ev, ok := <-videoHintEvents:
				if !ok {
					return
				}
				fmt.Printf("[videohint] %s: %s\n", ev.Stage, ev.Detail)
				if ev.Stage == videohint.StageOCRHit && ev.Name != "" && d.tracker != nil {
					if !d.tracker.SetHintForRecent(ev.Name, videohint.HintAttachMaxAge) {
						fmt.Printf("[videohint] hint %q had no recent enough speaker to attach to\n", ev.Name)
					}
				}
			}
		}
	}()

	return &meetingState{
		coordinator:     coordinator,
		session:         sess,
		done:            done,
		videoHintCancel: videoHintCancel,
	}, nil
}

func (d *Daemon) stopMeeting(ms *meetingState) {
	ms.videoHintCancel()
	ms.coordinator.Stop()
	<-ms.done

	// Update tray immediately so the user sees feedback
	d.tray.SetIdle()
	// Re-grab hotkeys — audio/tray operations can interfere with X11 grabs
	hotkey.ReGrabAll()

	// Finalize session timestamps synchronously (lightweight)
	duration := time.Since(ms.session.CreatedAt)
	ms.session.EndedAt = time.Now()
	ms.session.Duration = duration.Seconds()
	segCount := len(ms.session.Segments)

	msg := fmt.Sprintf("Meeting stopped — %d segments, %s (saving...)", segCount, duration.Round(time.Second))
	_ = d.svc.Notifier.Send("Tomoe", msg)
	fmt.Println(msg)

	// Hand off to the serial save worker. The worker runs audio encode +
	// diarization + persistence one meeting at a time, so concurrent stops
	// can't race on sherpa-onnx state. The buffered channel keeps the
	// foreground non-blocking under typical bursts.
	d.saveQueue <- ms
}

// saveWorker drains queued meetings and persists each one serially.
// Must be the only goroutine calling saveMeetingAsync.
func (d *Daemon) saveWorker() {
	defer d.saveWG.Done()
	for ms := range d.saveQueue {
		d.saveMeetingAsync(ms)
	}
}

// saveMeetingAsync encodes audio, persists the session, then runs
// diarization as a refinement pass. Saving before diarization ensures
// the session is recoverable even if diarization or the daemon crashes.
func (d *Daemon) saveMeetingAsync(ms *meetingState) {
	if d.store == nil {
		return
	}

	// Save audio as M4A
	var tracks [][]float32
	if ms.coordinator.IsDualSource() {
		mic := ms.coordinator.MicSamples()
		mon := ms.coordinator.MonitorSamples()
		if len(mic) > 0 && len(mon) > 0 {
			tracks = [][]float32{mic, mon}
		}
	} else {
		samples := ms.coordinator.AudioSamples()
		if len(samples) > 0 {
			tracks = [][]float32{samples}
		}
	}
	if len(tracks) > 0 {
		audioPath := filepath.Join(config.SessionDir(), ms.session.ID, "audio.m4a")
		if err := session.SaveAudioM4A(tracks, 16000, audioPath); err == nil {
			ms.session.AudioPath = audioPath
		} else {
			fmt.Printf("Error saving audio: %v\n", err)
		}
	}

	// Persist the session before diarization. Speaker labels are still the
	// live tracker output; diarization below will refine and re-save them.
	if err := d.store.Save(ms.session); err != nil {
		fmt.Printf("Error saving session: %v\n", err)
	}

	// Refinement: neural diarization in an isolated subprocess so a
	// sherpa-onnx Process() crash never reaches the daemon. The subprocess
	// retries GPU → GPU → CPU; on success it overwrites the session.json
	// on disk with refined speaker labels.
	if d.modelStatus != nil && d.modelStatus.DiarizationReady() {
		if err := session.RunDiarizeWithRetry(ms.session.ID, nil); err != nil {
			fmt.Printf("Warning: diarization failed (session saved without refinement): %v\n", err)
		} else if reloaded, err := d.store.Load(ms.session.ID); err == nil {
			ms.session = reloaded
		}
	}

	msg := fmt.Sprintf("Meeting saved — %s", ms.session.Title)
	_ = d.svc.Notifier.Send("Tomoe", msg)
	fmt.Println(msg)
}

func formatTimestamp(seconds float64) string {
	d := time.Duration(seconds * float64(time.Second))
	m := int(d.Minutes())
	s := int(d.Seconds()) % 60
	return fmt.Sprintf("%02d:%02d", m, s)
}
