package config

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/pelletier/go-toml/v2"
)

// Config is the top-level configuration, mapped to ~/.config/tomoe/config.toml.
type Config struct {
	Hotkey        HotkeyConfig        `toml:"hotkey"`
	Audio         AudioConfig         `toml:"audio"`
	Transcription TranscriptionConfig `toml:"transcription"`
	Output        OutputConfig        `toml:"output"`
	Meeting       MeetingConfig       `toml:"meeting"`
	Multilingual  MultilingualConfig  `toml:"multilingual"`
}

// HotkeyConfig holds global hotkey settings.
type HotkeyConfig struct {
	Binding        string `toml:"binding"`
	MeetingBinding string `toml:"meeting_binding"`
}

// AudioConfig holds audio capture settings.
type AudioConfig struct {
	Device string `toml:"device"`
}

// TranscriptionConfig holds transcription engine settings.
type TranscriptionConfig struct {
	GPUEnabled     bool    `toml:"gpu_enabled"`
	ModelPath      string  `toml:"model_path"`
	HotwordsFile   string  `toml:"hotwords_file"`
	HotwordsScore  float32 `toml:"hotwords_score"`
	DecodingMethod string  `toml:"decoding_method"` // "greedy_search" or "modified_beam_search"
	MaxActivePaths int     `toml:"max_active_paths"`
}

// OutputConfig holds output behavior settings.
type OutputConfig struct {
	AutoPaste      bool    `toml:"auto_paste"`
	Clipboard      bool    `toml:"clipboard"`
	SilenceTimeout float64 `toml:"silence_timeout"` // auto-stop dictation after N seconds of silence (0=disabled)
}

// MultilingualConfig holds multilingual transcription settings.
type MultilingualConfig struct {
	Enabled     bool     `toml:"enabled"`
	Languages   []string `toml:"languages"`    // e.g. ["en", "bn"]
	DefaultLang string   `toml:"default_lang"` // fallback language: "en"
}

// MeetingConfig holds Phase 2 meeting transcription settings.
//
// The speaker-clustering and video-hint-timing fields below (from
// SpeakerThreshold down) are hot-reloaded: internal/backend and
// internal/daemon both watch config.toml's mtime and re-apply these
// live via speaker.Tracker.SetTuning, so they can be retuned without a
// rebuild or even a relaunch — added after a real live-tuning session
// (see speaker.DefaultThreshold's doc comment) needed several
// rebuild+relaunch cycles just to test one constant change at a time.
type MeetingConfig struct {
	DefaultSources     string  `toml:"default_sources"`      // "mic", "monitor", "both"
	MonitorDevice      string  `toml:"monitor_device"`       // monitor source device name
	SpeakerThreshold   float64 `toml:"speaker_threshold"`    // cosine similarity threshold for a confident speaker match
	MaxSpeechDuration  float64 `toml:"max_speech_duration"`  // seconds
	MinSilenceDuration float64 `toml:"min_silence_duration"` // seconds
	AutoSave           bool    `toml:"auto_save"`            // save session on stop
	AutoDetect         bool    `toml:"auto_detect"`          // auto-detect meetings via PulseAudio

	// StickyGraceWindow/StickyThresholdMargin/MinAssignDuration/
	// ShortSegmentGraceWindow mirror speaker.Tuning's fields exactly
	// (seconds here instead of time.Duration, since TOML has no
	// duration type) — see speaker.Tuning's doc comment for what each
	// one does.
	StickyGraceWindow       float64 `toml:"sticky_grace_window"`
	StickyThresholdMargin   float64 `toml:"sticky_threshold_margin"`
	MinAssignDuration       float64 `toml:"min_assign_duration"`
	ShortSegmentGraceWindow float64 `toml:"short_segment_grace_window"`

	// VideoHintPollInterval/VideoHintTriggerDebounce mirror
	// videohint.Poll's interval/trigger-debounce parameters (seconds).
	VideoHintPollInterval    float64 `toml:"video_hint_poll_interval"`
	VideoHintTriggerDebounce float64 `toml:"video_hint_trigger_debounce"`
}

// VideoHintTiming converts VideoHintPollInterval/VideoHintTriggerDebounce
// into time.Duration, falling back to DefaultConfig's values for
// anything zero/invalid — the one place this seconds-to-Duration
// conversion happens, shared by internal/backend and internal/daemon
// rather than duplicated at each videohint.Poll call site.
func (m MeetingConfig) VideoHintTiming() (pollInterval, triggerDebounce time.Duration) {
	def := DefaultConfig().Meeting
	interval := m.VideoHintPollInterval
	if interval <= 0 {
		interval = def.VideoHintPollInterval
	}
	debounce := m.VideoHintTriggerDebounce
	if debounce <= 0 {
		debounce = def.VideoHintTriggerDebounce
	}
	return time.Duration(interval * float64(time.Second)), time.Duration(debounce * float64(time.Second))
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() *Config {
	return &Config{
		Hotkey: HotkeyConfig{
			Binding:        "Super+Shift+S",
			MeetingBinding: "Super+Shift+X",
		},
		Audio: AudioConfig{
			Device: "default",
		},
		Transcription: TranscriptionConfig{
			GPUEnabled:     false,
			ModelPath:      ModelDir(),
			DecodingMethod: "greedy_search",
			HotwordsScore:  1.5,
			MaxActivePaths: 4,
		},
		Output: OutputConfig{
			AutoPaste:      true,
			Clipboard:      true,
			SilenceTimeout: 5.0,
		},
		Multilingual: MultilingualConfig{
			Enabled:     false,
			Languages:   []string{"en"},
			DefaultLang: "en",
		},
		Meeting: MeetingConfig{
			DefaultSources:     "both",
			SpeakerThreshold:   0.55,
			MaxSpeechDuration:  30.0,
			MinSilenceDuration: 0.5,
			AutoSave:           true,
			AutoDetect:         true,

			StickyGraceWindow:       3.0,
			StickyThresholdMargin:   0.15,
			MinAssignDuration:       0.7,
			ShortSegmentGraceWindow: 15.0,

			VideoHintPollInterval:    5.0,
			VideoHintTriggerDebounce: 1.0,
		},
	}
}

// Path returns the default config file path (~/.config/tomoe/config.toml).
// Respects $XDG_CONFIG_HOME if set.
func Path() string {
	dir := os.Getenv("XDG_CONFIG_HOME")
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			home = os.Getenv("HOME")
		}
		dir = filepath.Join(home, ".config")
	}
	return filepath.Join(dir, "tomoe", "config.toml")
}

// ModelDir returns the default model storage directory (~/.local/share/tomoe/models/).
// Respects $XDG_DATA_HOME if set.
func ModelDir() string {
	dir := os.Getenv("XDG_DATA_HOME")
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			home = os.Getenv("HOME")
		}
		dir = filepath.Join(home, ".local", "share")
	}
	return filepath.Join(dir, "tomoe", "models")
}

// SessionDir returns the session storage directory (~/.local/share/tomoe/sessions/).
func SessionDir() string {
	return filepath.Join(DataDir(), "sessions")
}

// UnrecognizedUIPendingDir returns the staging directory
// (~/.local/share/tomoe/unrecognized-uis-pending/) internal/videohint
// writes escalation snapshots to — a captured frame + metadata per
// meeting-app UI its rule table doesn't recognize yet. This is
// deliberately NOT the permanent library: a captured window can be the
// wrong thing entirely (e.g. a chat tab, not an actual call), so
// nothing here is treated as safe to keep or use for calibration until
// a human reviews and approves it — see UnrecognizedUIApprovedDir and
// `tomoe videohint`.
func UnrecognizedUIPendingDir() string {
	return filepath.Join(DataDir(), "unrecognized-uis-pending")
}

// UnrecognizedUIApprovedDir returns the permanent escalation snapshot
// library (~/.local/share/tomoe/unrecognized-uis-approved/) — where a
// snapshot lands only once explicitly approved via `tomoe videohint
// approve`, at which point it's fair game for writing a real rule from.
func UnrecognizedUIApprovedDir() string {
	return filepath.Join(DataDir(), "unrecognized-uis-approved")
}

// DataDir returns the base data directory (~/.local/share/tomoe/).
func DataDir() string {
	dir := os.Getenv("XDG_DATA_HOME")
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			home = os.Getenv("HOME")
		}
		dir = filepath.Join(home, ".local", "share")
	}
	return filepath.Join(dir, "tomoe")
}

// LibDir returns the directory for additional shared libraries (~/.local/share/tomoe/lib/).
// Used for GPU provider .so files downloaded by `make install-gpu`.
func LibDir() string {
	return filepath.Join(DataDir(), "lib")
}

// Exists reports whether the config file exists at the default path.
func Exists() bool {
	_, err := os.Stat(Path())
	return err == nil
}

// Load reads and parses the config file at the given path.
// Starts from DefaultConfig so fields absent from the file retain their defaults.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config: %w", err)
	}

	cfg := DefaultConfig()
	if err := toml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parsing config: %w", err)
	}

	return cfg, nil
}

// Watch polls path's mtime every interval and calls onChange with a
// freshly reloaded Config whenever it changes -- the mechanism behind
// retuning live speaker-clustering/video-hint-timing behavior without
// a rebuild or even a relaunch (see MeetingConfig's doc comment).
// onChange is never called concurrently with itself, and never for
// the file's state as of when Watch was called (only for a real
// change made afterward). A reload that fails to parse is logged and
// skipped, leaving whatever's currently applied in place rather than
// falling back to defaults. Returns a stop func; safe to call more
// than once.
func Watch(path string, interval time.Duration, onChange func(*Config)) (stop func()) {
	var lastMod time.Time
	if info, err := os.Stat(path); err == nil {
		lastMod = info.ModTime()
	}

	done := make(chan struct{})
	var stopOnce sync.Once

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				info, err := os.Stat(path)
				if err != nil || !info.ModTime().After(lastMod) {
					continue
				}
				lastMod = info.ModTime()
				cfg, err := Load(path)
				if err != nil {
					fmt.Printf("config: reload of %s failed, keeping previous values: %v\n", path, err)
					continue
				}
				onChange(cfg)
			}
		}
	}()

	return func() { stopOnce.Do(func() { close(done) }) }
}

// Save writes the config to the given path, creating parent directories as needed.
func Save(cfg *Config, path string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("creating config directory: %w", err)
	}

	header := fmt.Sprintf("# Generated by tomoe auto-init on %s\n\n",
		time.Now().Format(time.RFC3339))

	data, err := toml.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("marshaling config: %w", err)
	}

	content := []byte(header)
	content = append(content, data...)

	if err := os.WriteFile(path, content, 0o644); err != nil {
		return fmt.Errorf("writing config: %w", err)
	}

	return nil
}
