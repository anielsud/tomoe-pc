package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/spf13/cobra"

	"github.com/sosuke-ai/tomoe-pc/internal/audio"
	"github.com/sosuke-ai/tomoe-pc/internal/config"
	"github.com/sosuke-ai/tomoe-pc/internal/daemon"
	"github.com/sosuke-ai/tomoe-pc/internal/gpu"
	"github.com/sosuke-ai/tomoe-pc/internal/hotkey"
	"github.com/sosuke-ai/tomoe-pc/internal/meeting"
	"github.com/sosuke-ai/tomoe-pc/internal/models"
	"github.com/sosuke-ai/tomoe-pc/internal/platform"
	"github.com/sosuke-ai/tomoe-pc/internal/session"
	"github.com/sosuke-ai/tomoe-pc/internal/speaker"
	"github.com/sosuke-ai/tomoe-pc/internal/transcribe"
	"github.com/sosuke-ai/tomoe-pc/internal/videohint"
)

func main() {
	// Re-exec with LD_LIBRARY_PATH if GPU libraries are installed.
	// Must happen before any cgo/sherpa-onnx code loads.
	config.EnsureGPULibs()

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

var rootCmd = &cobra.Command{
	Use:   "tomoe",
	Short: "Local-first speech-to-text for Linux",
	Long:  "Tomoe captures microphone audio, transcribes locally using Parakeet TDT via sherpa-onnx, and delivers text to clipboard.",
	RunE:  runStart,
}

func init() {
	rootCmd.AddCommand(startCmd)
	rootCmd.AddCommand(initCmd)
	rootCmd.AddCommand(statusCmd)
	rootCmd.AddCommand(configCmd)
	rootCmd.AddCommand(modelCmd)
	rootCmd.AddCommand(sessionCmd)
	rootCmd.AddCommand(transcribeCmd)
	rootCmd.AddCommand(devicesCmd)
	rootCmd.AddCommand(stopCmd)
	rootCmd.AddCommand(versionCmd)
	rootCmd.AddCommand(videohintCmd)
}

// startCmd is an alias for the root command.
var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Start the daemon (alias for running tomoe with no arguments)",
	RunE:  runStart,
}

func runStart(cmd *cobra.Command, args []string) error {
	if !config.Exists() {
		fmt.Println("First run detected, running auto-init...")
		if err := runInit(cmd, args); err != nil {
			return err
		}
	}

	// Check if already running
	if daemon.IsRunning() {
		return fmt.Errorf("daemon already running (PID %d)", daemon.ReadPID())
	}

	// Load config
	cfg, err := config.Load(config.Path())
	if err != nil {
		return fmt.Errorf("loading config: %w", err)
	}

	// Check models
	mgr := models.NewManager(cfg.Transcription.ModelPath)
	status := mgr.Check()
	if !status.Ready() {
		return fmt.Errorf("models not downloaded (run 'tomoe init' or 'tomoe model download')")
	}

	// Create transcription engines (multilingual if configured)
	engines, err := transcribe.NewEngineSetFromConfig(transcribe.Config{
		EncoderPath:    status.EncoderPath,
		DecoderPath:    status.DecoderPath,
		JoinerPath:     status.JoinerPath,
		TokensPath:     status.TokensPath,
		VADPath:        status.VADPath,
		UseGPU:         cfg.Transcription.GPUEnabled,
		DecodingMethod: cfg.Transcription.DecodingMethod,
		MaxActivePaths: cfg.Transcription.MaxActivePaths,
		HotwordsFile:   cfg.Transcription.HotwordsFile,
		HotwordsScore:  cfg.Transcription.HotwordsScore,
	}, status, &cfg.Multilingual)
	if err != nil {
		return fmt.Errorf("creating transcription engine: %w", err)
	}
	defer engines.Close()

	// Create platform services
	svc, err := platform.New(cfg)
	if err != nil {
		return fmt.Errorf("initializing platform services: %w", err)
	}
	defer svc.Close()

	// Set up meeting mode dependencies (optional — non-fatal if any fail)
	opts := &daemon.MeetingOpts{
		ModelStatus: status,
		Store:       session.NewStore(config.SessionDir()),
	}

	// Create meeting hotkey
	meetingBinding := cfg.Hotkey.MeetingBinding
	if meetingBinding == "" {
		meetingBinding = "Super+Shift+X"
	}
	if mhk, err := hotkey.NewListener(meetingBinding); err == nil {
		opts.MeetingHotkey = mhk
	} else {
		fmt.Fprintf(os.Stderr, "Warning: meeting hotkey %q: %v\n", meetingBinding, err)
	}

	// Create speaker embedder (optional — only if model available)
	if status.SpeakerEmbeddingReady {
		if emb, err := speaker.NewEmbedder(status.SpeakerEmbeddingPath); err == nil {
			opts.Embedder = emb
			threshold := speaker.DefaultThreshold
			if cfg.Meeting.SpeakerThreshold > 0 {
				threshold = cfg.Meeting.SpeakerThreshold
			}
			tracker := speaker.NewTracker(threshold)
			tracker.SetTuning(speaker.TuningFromSeconds(
				cfg.Meeting.SpeakerThreshold,
				cfg.Meeting.StickyGraceWindow,
				cfg.Meeting.StickyThresholdMargin,
				cfg.Meeting.MinAssignDuration,
				cfg.Meeting.ShortSegmentGraceWindow,
			))
			opts.Tracker = tracker
			defer emb.Close()

			// Watch config.toml so clustering tuning can be retuned
			// live -- no rebuild, no relaunch. See MeetingConfig's doc
			// comment for why this exists.
			stopConfigWatch := config.Watch(config.Path(), 2*time.Second, func(newCfg *config.Config) {
				tracker.SetTuning(speaker.TuningFromSeconds(
					newCfg.Meeting.SpeakerThreshold,
					newCfg.Meeting.StickyGraceWindow,
					newCfg.Meeting.StickyThresholdMargin,
					newCfg.Meeting.MinAssignDuration,
					newCfg.Meeting.ShortSegmentGraceWindow,
				))
				fmt.Printf("config: reloaded speaker clustering tuning: %+v\n", tracker.Tuning())
			})
			defer stopConfigWatch()
		}
	}

	// Create meeting auto-detector (optional)
	if cfg.Meeting.AutoDetect {
		opts.Detector = meeting.NewDetector()
	}

	// Create the realtime streaming engine (optional — only if the
	// English streaming model is downloaded; see internal/live's
	// two-pass pipeline). Sessions fall back to single-pass without it.
	if status.EnglishStreamingReady {
		streamingEngine, err := transcribe.NewStreamingEngine(transcribe.StreamingConfig{
			EncoderPath: status.EnglishStreamingEncoderPath,
			DecoderPath: status.EnglishStreamingDecoderPath,
			JoinerPath:  status.EnglishStreamingJoinerPath,
			TokensPath:  status.EnglishStreamingTokensPath,
		})
		if err == nil {
			opts.StreamingEngine = streamingEngine
			defer streamingEngine.Close()
		} else {
			fmt.Fprintf(os.Stderr, "Warning: failed to load English streaming model: %v (live transcription will use single-pass mode)\n", err)
		}
	}

	// Run daemon
	d := daemon.New(cfg, engines, svc, opts)
	return d.Run(context.Background())
}

var stopCmd = &cobra.Command{
	Use:   "stop",
	Short: "Stop the running daemon",
	RunE: func(cmd *cobra.Command, args []string) error {
		if !daemon.IsRunning() {
			fmt.Println("Daemon is not running.")
			return nil
		}
		if err := daemon.StopRemote(); err != nil {
			return fmt.Errorf("stopping daemon: %w", err)
		}
		fmt.Println("Stop signal sent to daemon.")
		return nil
	},
}

var initCmd = &cobra.Command{
	Use:   "init",
	Short: "Detect system, generate config, and download model",
	RunE:  runInit,
}

func runInit(cmd *cobra.Command, args []string) error {
	fmt.Println("=== Tomoe Auto-Init ===")
	fmt.Println()

	// Detect GPU
	fmt.Println("Detecting GPU...")
	gpuInfo := gpu.Detect()
	fmt.Println(gpuInfo)
	fmt.Println()

	// Detect display server
	displayServer := os.Getenv("XDG_SESSION_TYPE")
	if displayServer == "" {
		displayServer = "unknown"
	}
	fmt.Printf("Display server: %s\n", displayServer)
	fmt.Println()

	// Generate config
	cfg := config.DefaultConfig()
	cfg.Transcription.GPUEnabled = gpuInfo.Available && gpuInfo.Sufficient

	cfgPath := config.Path()
	if err := config.Save(cfg, cfgPath); err != nil {
		return fmt.Errorf("saving config: %w", err)
	}
	fmt.Printf("Config written to: %s\n", cfgPath)
	fmt.Println()

	// Download models
	mgr := models.NewManager(cfg.Transcription.ModelPath)
	if err := mgr.Download(false); err != nil {
		return fmt.Errorf("downloading models: %w", err)
	}
	fmt.Println()

	// Summary
	modelStatus := mgr.Check()
	fmt.Println(modelStatus)
	fmt.Println()

	fmt.Println("=== Init complete ===")
	return nil
}

var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show daemon status, system info, model info, and GPU detection",
	RunE: func(cmd *cobra.Command, args []string) error {
		fmt.Println("=== Tomoe Status ===")
		fmt.Println()

		// GPU
		gpuInfo := gpu.Detect()
		fmt.Println(gpuInfo)
		fmt.Println()

		// Config
		cfgPath := config.Path()
		if config.Exists() {
			cfg, err := config.Load(cfgPath)
			if err != nil {
				return fmt.Errorf("loading config: %w", err)
			}
			fmt.Printf("Config: %s\n", cfgPath)
			fmt.Printf("  Hotkey:      %s\n", cfg.Hotkey.Binding)
			fmt.Printf("  Audio:       %s\n", cfg.Audio.Device)
			fmt.Printf("  GPU enabled: %v\n", cfg.Transcription.GPUEnabled)
			fmt.Printf("  Model path:  %s\n", cfg.Transcription.ModelPath)
			fmt.Printf("  Clipboard:   %v\n", cfg.Output.Clipboard)
			fmt.Printf("  Auto-paste:  %v\n", cfg.Output.AutoPaste)
			if cfg.Multilingual.Enabled {
				fmt.Printf("  Multilingual: enabled (languages: %v, default: %s)\n",
					cfg.Multilingual.Languages, cfg.Multilingual.DefaultLang)
			}
			if cfg.Transcription.DecodingMethod == "modified_beam_search" {
				fmt.Printf("  Hotwords:    %s (score: %.1f)\n",
					cfg.Transcription.HotwordsFile, cfg.Transcription.HotwordsScore)
			}
		} else {
			fmt.Printf("Config: not found (run 'tomoe init')\n")
		}
		fmt.Println()

		// Models
		mgr := models.NewManager(config.ModelDir())
		modelStatus := mgr.Check()
		fmt.Println(modelStatus)
		fmt.Println()

		// Audio devices
		devices, err := audio.ListDevices()
		if err != nil {
			fmt.Printf("Audio: error listing devices (%v)\n", err)
		} else if len(devices) == 0 {
			fmt.Println("Audio: no capture devices found")
		} else {
			fmt.Printf("Audio: %d capture device(s)\n", len(devices))
			for _, d := range devices {
				def := ""
				if d.IsDefault {
					def = " *"
				}
				fmt.Printf("  - %s%s\n", d.Name, def)
			}
		}
		fmt.Println()

		// Display server
		ds := os.Getenv("XDG_SESSION_TYPE")
		if ds == "" {
			ds = "unknown"
		}
		fmt.Printf("Display: %s\n", ds)
		fmt.Println()

		// Daemon
		if daemon.IsRunning() {
			fmt.Printf("Daemon: running (PID %d)\n", daemon.ReadPID())
		} else {
			fmt.Println("Daemon: not running")
		}

		return nil
	},
}

var modelCmd = &cobra.Command{
	Use:   "model",
	Short: "Manage transcription models",
}

func init() {
	modelCmd.AddCommand(modelDownloadCmd)
	modelCmd.AddCommand(modelStatusCmd)
}

var modelDownloadCmd = &cobra.Command{
	Use:   "download",
	Short: "Download or re-download models (multilingual models auto-download when enabled in config)",
	RunE: func(cmd *cobra.Command, args []string) error {
		force, _ := cmd.Flags().GetBool("force")
		multilingual, _ := cmd.Flags().GetBool("multilingual")
		mgr := models.NewManager(config.ModelDir())

		if err := mgr.Download(force); err != nil {
			return err
		}

		// Download multilingual models if enabled in config or explicitly requested
		if !multilingual && config.Exists() {
			cfg, err := config.Load(config.Path())
			if err == nil && cfg.Multilingual.Enabled {
				multilingual = true
			}
		}

		if multilingual {
			fmt.Println("\nDownloading multilingual models...")
			if err := mgr.DownloadMultilingual(force); err != nil {
				return err
			}
		}

		return nil
	},
}

func init() {
	modelDownloadCmd.Flags().Bool("force", false, "Force re-download even if models exist")
	modelDownloadCmd.Flags().Bool("multilingual", false, "Download language identification and Bengali models (auto-detected from config)")
}

var modelStatusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show downloaded model info and integrity check",
	RunE: func(cmd *cobra.Command, args []string) error {
		mgr := models.NewManager(config.ModelDir())
		status := mgr.Check()
		fmt.Println(status)
		return nil
	},
}

var transcribeCmd = &cobra.Command{
	Use:   "transcribe <file>",
	Short: "Transcribe an audio file (WAV, FLAC, OGG, MP3) with speaker identification",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		filePath := args[0]

		if !transcribe.IsSupportedFormat(filePath) {
			return fmt.Errorf("unsupported audio format: %s (supported: .wav, .flac, .ogg, .mp3, .m4a)", filePath)
		}

		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			return fmt.Errorf("file not found: %s", filePath)
		}

		// Load config or use defaults
		var cfg *config.Config
		if config.Exists() {
			var err error
			cfg, err = config.Load(config.Path())
			if err != nil {
				return fmt.Errorf("loading config: %w", err)
			}
		} else {
			cfg = config.DefaultConfig()
		}

		// Check model status
		mgr := models.NewManager(cfg.Transcription.ModelPath)
		status := mgr.Check()
		if !status.Ready() {
			return fmt.Errorf("models not downloaded (run 'tomoe init' or 'tomoe model download')")
		}

		gpuInfo := gpu.Detect()
		useGPU := gpuInfo.Available && gpuInfo.Sufficient

		// Create transcription engines (multilingual if configured)
		engines, err := transcribe.NewEngineSetFromConfig(transcribe.Config{
			EncoderPath:    status.EncoderPath,
			DecoderPath:    status.DecoderPath,
			JoinerPath:     status.JoinerPath,
			TokensPath:     status.TokensPath,
			VADPath:        status.VADPath,
			UseGPU:         useGPU,
			DecodingMethod: cfg.Transcription.DecodingMethod,
			MaxActivePaths: cfg.Transcription.MaxActivePaths,
			HotwordsFile:   cfg.Transcription.HotwordsFile,
			HotwordsScore:  cfg.Transcription.HotwordsScore,
		}, status, &cfg.Multilingual)
		if err != nil {
			return fmt.Errorf("creating transcription engine: %w", err)
		}
		defer engines.Close()

		engine := engines.Default()

		// If diarization models are available, transcribe with speaker labels
		if status.DiarizationReady() {
			return transcribeWithSpeakers(engine, filePath, status, useGPU)
		}

		// Fallback: plain transcription without speaker identification
		result, err := engine.TranscribeFile(filePath)
		if err != nil {
			return fmt.Errorf("transcription failed: %w", err)
		}

		if result.Text == "" {
			fmt.Println("(no speech detected)")
			return nil
		}

		fmt.Println(result.Text)

		if lang := result.Language; lang != "" {
			fmt.Fprintf(os.Stderr, "Language: %s | Duration: %.1fs\n", lang, result.Duration)
		} else {
			fmt.Fprintf(os.Stderr, "Duration: %.1fs\n", result.Duration)
		}

		return nil
	},
}

// transcribeWithSpeakers runs diarization then transcribes each speaker segment.
func transcribeWithSpeakers(engine transcribe.Engine, filePath string, status *models.Status, useGPU bool) error {
	// Decode audio to float32 for diarization
	samples, err := session.DecodeToFloat32(filePath)
	if err != nil {
		return fmt.Errorf("decoding audio: %w", err)
	}

	duration := float64(len(samples)) / 16000.0
	fmt.Fprintf(os.Stderr, "Identifying speakers in %.1fs of audio...\n", duration)

	// Run diarization
	diarSegments, speakerMap, err := session.Diarize(samples, session.DiarizeConfig{
		SegmentationModelPath: status.SpeakerSegmentationPath,
		EmbeddingModelPath:    status.SpeakerEmbeddingPath,
		Threshold:             1.1,
		MergeThreshold:        0.55,
		UseGPU:                useGPU,
	})
	if err != nil {
		return fmt.Errorf("diarization failed: %w", err)
	}

	if len(diarSegments) == 0 {
		fmt.Println("(no speech detected)")
		return nil
	}

	fmt.Fprintf(os.Stderr, "Found %d speakers, transcribing...\n", len(speakerMap))

	// Transcribe each diarization segment
	for _, ds := range diarSegments {
		startIdx := int(ds.Start * 16000)
		endIdx := int(ds.End * 16000)
		if startIdx < 0 {
			startIdx = 0
		}
		if endIdx > len(samples) {
			endIdx = len(samples)
		}
		if startIdx >= endIdx {
			continue
		}

		segSamples := samples[startIdx:endIdx]
		result, err := engine.TranscribeDirect(segSamples)
		if err != nil || result.Text == "" {
			continue
		}

		label := speakerMap[ds.Speaker]
		fmt.Printf("[%s] %s: %s\n", formatTimestamp(ds.Start), label, result.Text)
	}

	fmt.Fprintf(os.Stderr, "Duration: %.1fs\n", duration)
	return nil
}

func formatTimestamp(seconds float64) string {
	m := int(seconds) / 60
	s := int(seconds) % 60
	return fmt.Sprintf("%02d:%02d", m, s)
}

var devicesCmd = &cobra.Command{
	Use:   "devices",
	Short: "List audio input devices",
	RunE: func(cmd *cobra.Command, args []string) error {
		devices, err := audio.ListDevices()
		if err != nil {
			return fmt.Errorf("listing audio devices: %w", err)
		}

		if len(devices) == 0 {
			fmt.Println("No audio capture devices found.")
			return nil
		}

		for _, d := range devices {
			def := ""
			if d.IsDefault {
				def = " (default)"
			}
			fmt.Printf("  %s%s\n", d.Name, def)
		}
		return nil
	},
}

// Version is set at build time via -ldflags.
var Version = "dev"

var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print the version",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Printf("tomoe %s\n", Version)
	},
}

var sessionCmd = &cobra.Command{
	Use:   "session",
	Short: "Manage recorded sessions",
}

func init() {
	sessionRetranscribeCmd.Flags().BoolP("verbose", "v", false, "Print processing details")
	sessionCmd.AddCommand(sessionRetranscribeCmd)
	sessionCmd.AddCommand(sessionListCmd)
}

var sessionRetranscribeCmd = &cobra.Command{
	Use:   "re-transcribe <session-id>",
	Short: "Re-process a session's audio: re-transcribe and identify speakers",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		mgr := models.NewManager(config.ModelDir())
		status := mgr.Check()
		if !status.Ready() {
			return fmt.Errorf("transcription models not downloaded (run 'tomoe model download')")
		}
		if !status.DiarizationReady() {
			return fmt.Errorf("diarization models not downloaded (run 'tomoe model download')")
		}

		verbose, _ := cmd.Flags().GetBool("verbose")
		gpuInfo := gpu.Detect()
		useGPU := gpuInfo.Available && gpuInfo.Sufficient

		if verbose {
			fmt.Println(gpuInfo)
		}

		store := session.NewStore(config.SessionDir())
		sess, err := store.Load(args[0])
		if err != nil {
			return fmt.Errorf("loading session: %w", err)
		}

		if sess.AudioPath == "" {
			return fmt.Errorf("session %q has no recorded audio", sess.Title)
		}

		// Step 1: Diarize to identify speakers
		fmt.Printf("Identifying speakers in %q...\n", sess.Title)
		count, err := session.ReidentifyByDiarization(sess, session.DiarizeConfig{
			SegmentationModelPath: status.SpeakerSegmentationPath,
			EmbeddingModelPath:    status.SpeakerEmbeddingPath,
			Threshold:             1.1,
			MergeThreshold:        0.55,
			UseGPU:                useGPU,
			Verbose:               verbose,
		})
		if err != nil {
			return fmt.Errorf("diarization failed: %w", err)
		}

		if err := store.Save(sess); err != nil {
			return fmt.Errorf("saving session: %w", err)
		}

		fmt.Printf("Done: identified speakers for %d segments.\n", count)
		return nil
	},
}

var sessionListCmd = &cobra.Command{
	Use:   "list",
	Short: "List all saved sessions",
	RunE: func(cmd *cobra.Command, args []string) error {
		store := session.NewStore(config.SessionDir())
		sessions, err := store.List()
		if err != nil {
			return fmt.Errorf("listing sessions: %w", err)
		}
		if len(sessions) == 0 {
			fmt.Println("No sessions found.")
			return nil
		}

		for _, sess := range sessions {
			fmt.Printf("  %s  %s  (%d segments, %.0fs)\n",
				sess.ID, sess.Title, len(sess.Segments), sess.Duration)
		}
		return nil
	},
}

var configCmd = &cobra.Command{
	Use:   "config",
	Short: "Print current configuration",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfgPath := config.Path()
		if !config.Exists() {
			fmt.Println("No config file found. Run 'tomoe init' to generate one.")
			return nil
		}

		data, err := os.ReadFile(cfgPath)
		if err != nil {
			return fmt.Errorf("reading config: %w", err)
		}

		fmt.Print(string(data))
		return nil
	},
}

var videohintCmd = &cobra.Command{
	Use:   "videohint",
	Short: "Review meeting-window screenshots escalated for manual labeling-rule review (macOS)",
	Long: "During macOS meeting mode, internal/videohint captures a screenshot whenever it sees a\n" +
		"window it doesn't have a labeling rule for yet, staging it for review rather than using\n" +
		"it automatically -- the matched window can be the wrong thing entirely (e.g. a chat tab,\n" +
		"not an actual call). Nothing here is treated as safe to keep or use for calibration until\n" +
		"you explicitly approve it.",
}

func init() {
	videohintCmd.AddCommand(videohintListCmd)
	videohintCmd.AddCommand(videohintApproveCmd)
	videohintCmd.AddCommand(videohintDiscardCmd)
}

var videohintListCmd = &cobra.Command{
	Use:   "list",
	Short: "List pending snapshots awaiting review",
	RunE: func(cmd *cobra.Command, args []string) error {
		pending, err := videohint.ListPendingUnrecognizedUIs()
		if err != nil {
			return err
		}
		if len(pending) == 0 {
			fmt.Println("No pending snapshots.")
			return nil
		}
		for _, p := range pending {
			fmt.Printf("%s  %-8s %dx%d  %s  %s\n",
				p.Meta.Timestamp.Format("2006-01-02 15:04:05"), p.Meta.Platform,
				p.Meta.Width, p.Meta.Height, p.ID, p.Meta.Reason)
		}
		return nil
	},
}

var videohintApproveCmd = &cobra.Command{
	Use:   "approve <id>",
	Short: "Move a pending snapshot into the permanent unrecognized-UI library for future rule-writing",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		if err := videohint.ApproveUnrecognizedUI(args[0]); err != nil {
			return err
		}
		fmt.Printf("Approved %s\n", args[0])
		return nil
	},
}

var videohintDiscardCmd = &cobra.Command{
	Use:   "discard <id>",
	Short: "Permanently delete a pending snapshot without approving it",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		if err := videohint.DiscardUnrecognizedUI(args[0]); err != nil {
			return err
		}
		fmt.Printf("Discarded %s\n", args[0])
		return nil
	},
}
