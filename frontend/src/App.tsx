import { useState, useEffect } from 'react'
import TranscriptPane from './components/TranscriptPane'
import SourceSelector from './components/SourceSelector'
import SessionControls from './components/SessionControls'
import SessionList from './components/SessionList'
import SettingsPanel from './components/SettingsPanel'
import StatusBar from './components/StatusBar'
import ExportDialog from './components/ExportDialog'
import VideoHintActivity from './components/VideoHintActivity'
import VideoHintReview from './components/VideoHintReview'
import DiagnosticsPane from './components/DiagnosticsPane'
import { useTranscript } from './hooks/useTranscript'
import { useSession } from './hooks/useSession'
import { DeviceInfo, Session, AudioSourceView } from './types'

type View = 'live' | 'sessions' | 'settings' | 'videohints' | 'diagnostics';

function App() {
  const [view, setView] = useState<View>('live');
  const [micDevice, setMicDevice] = useState('default');
  const [monitorDevice, setMonitorDevice] = useState('');
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [monitors, setMonitors] = useState<DeviceInfo[]>([]);
  const [audioSources, setAudioSources] = useState<AudioSourceView[]>([]);
  const [languages, setLanguages] = useState<string[]>(['en']);
  const [selectedLang, setSelectedLang] = useState('en');
  const [systemAudioMode, setSystemAudioMode] = useState<'manual' | 'auto'>('manual');
  const [exportSessionId, setExportSessionId] = useState<string | null>(null);
  const { segments, clear: clearTranscript } = useTranscript();
  const { isRecording, startTime, reset: resetSession } = useSession();

  useEffect(() => {
    loadDevices();
    loadLanguages();

    // Handle in-window keyboard shortcut for meeting toggle.
    // Super+Shift+M (metaKey) or Ctrl+Shift+M (fallback for Linux/WebKitGTK
    // where Super key may not set metaKey).
    function handleKeyDown(e: KeyboardEvent) {
      if (e.shiftKey && e.key === 'M' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        if (isRecording) {
          handleStop();
        } else {
          handleStart();
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isRecording, micDevice, monitorDevice]);

  // Keep the macOS audio-source picker current while it's actually
  // being looked at: refreshing during an active recording would just
  // change options out from under a disabled, already-committed
  // picker for no benefit.
  useEffect(() => {
    if (systemAudioMode !== 'auto' || isRecording) return;
    const interval = setInterval(loadAudioSources, 3000);
    return () => clearInterval(interval);
  }, [systemAudioMode, isRecording]);

  async function loadDevices() {
    try {
      if (window.go?.backend?.App) {
        const devs = await window.go.backend.App.ListAudioDevices();
        setDevices(devs || []);
        const mons = await window.go.backend.App.ListMonitorSources();
        setMonitors(mons || []);
        // Auto-select default monitor source (for speaker/system audio capture)
        if (mons && mons.length > 0 && !monitorDevice) {
          const def = mons.find((m: DeviceInfo) => m.IsDefault);
          setMonitorDevice(def ? def.Name : mons[0].Name);
        }
        try {
          const mode = await window.go.backend.App.SystemAudioMode();
          if (mode === 'auto' || mode === 'manual') {
            setSystemAudioMode(mode);
            if (mode === 'auto') await loadAudioSources();
          }
        } catch {
          // Method may not exist in older builds — default to 'manual'
        }
      }
    } catch (e) {
      console.error('Failed to load devices:', e);
    }
  }

  // macOS's audio-source list changes as apps start/stop playing audio
  // (unlike Linux's fairly static PulseAudio device list), so this is
  // called on load and polled periodically while it's actually visible
  // and selectable — see the effect below.
  async function loadAudioSources() {
    try {
      if (!window.go?.backend?.App?.ListAudioSources) return;
      const sources = await window.go.backend.App.ListAudioSources();
      setAudioSources(sources || []);
      // Default to "Everything" (matches the old auto-detect's
      // always-on-if-available behavior) rather than leaving the
      // picker on "No System Audio" until the user notices it.
      setMonitorDevice(prev => prev || 'everything');
    } catch (e) {
      console.error('Failed to load audio sources:', e);
    }
  }

  async function loadLanguages() {
    try {
      if (!window.go?.backend?.App) return;

      // Try dedicated language API first
      try {
        const langs = await window.go.backend.App.GetAvailableLanguages();
        if (langs && langs.length > 0) {
          setLanguages(langs);
          const defLang = await window.go.backend.App.GetDefaultLanguage();
          if (defLang) setSelectedLang(defLang);
          return;
        }
      } catch {
        // Method may not exist in older builds — fall through to config
      }

      // Fallback: read from config
      const cfg = await window.go.backend.App.GetConfig();
      if (cfg?.Multilingual?.Enabled && cfg.Multilingual.Languages?.length > 1) {
        setLanguages(cfg.Multilingual.Languages);
        if (cfg.Multilingual.DefaultLang) {
          setSelectedLang(cfg.Multilingual.DefaultLang);
        }
      }
    } catch (e) {
      console.error('Failed to load languages:', e);
    }
  }

  async function handleStart() {
    try {
      clearTranscript();
      await window.go.backend.App.StartSession(micDevice, monitorDevice, selectedLang, "");
    } catch (e) {
      console.error('Failed to start session:', e);
    }
  }

  async function handleStop(): Promise<Session | null> {
    try {
      const sess = await window.go.backend.App.StopSession();
      resetSession();
      return sess;
    } catch (e) {
      console.error('Failed to stop session:', e);
      return null;
    }
  }

  return (
    <div className="app">
      <div className="toolbar">
        <SourceSelector
          devices={devices}
          monitors={monitors}
          micDevice={micDevice}
          monitorDevice={monitorDevice}
          onMicChange={setMicDevice}
          onMonitorChange={setMonitorDevice}
          disabled={isRecording}
          systemAudioMode={systemAudioMode}
          audioSources={audioSources}
        />
        {languages.length > 1 && (
          <select
            className="lang-selector"
            value={selectedLang}
            onChange={e => setSelectedLang(e.target.value)}
            disabled={isRecording}
          >
            {languages.map(l => (
              <option key={l} value={l}>{l.toUpperCase()}</option>
            ))}
          </select>
        )}
        <div className="spacer" />
        <SessionControls
          isRecording={isRecording}
          segmentCount={segments.length}
          onStart={handleStart}
          onStop={handleStop}
        />
        <button
          className="btn-icon"
          title="Past Sessions"
          onClick={() => setView(view === 'sessions' ? 'live' : 'sessions')}
        >
          &#x1F4CB;
        </button>
        {systemAudioMode === 'auto' && (
          <button
            className="btn-icon"
            title="Pending Screenshots (macOS meeting-label review)"
            onClick={() => setView(view === 'videohints' ? 'live' : 'videohints')}
          >
            &#x1F4F7;
          </button>
        )}
        <button
          className="btn-icon"
          title="Diagnostics (live speaker-labeling internals)"
          onClick={() => setView(view === 'diagnostics' ? 'live' : 'diagnostics')}
        >
          &#x1F50D;
        </button>
        <button
          className="btn-icon"
          title="Settings"
          onClick={() => setView(view === 'settings' ? 'live' : 'settings')}
        >
          &#x2699;
        </button>
      </div>

      {view === 'live' && (
        <>
          {systemAudioMode === 'auto' && <VideoHintActivity />}
          <TranscriptPane segments={segments} isRecording={isRecording} />
        </>
      )}

      {view === 'sessions' && (
        <SessionList onExport={setExportSessionId} />
      )}

      {view === 'settings' && (
        <SettingsPanel />
      )}

      {view === 'videohints' && (
        <VideoHintReview />
      )}

      {view === 'diagnostics' && (
        <DiagnosticsPane />
      )}

      <StatusBar
        isRecording={isRecording}
        startTime={startTime}
        segmentCount={segments.length}
      />

      {exportSessionId && (
        <ExportDialog
          sessionId={exportSessionId}
          onClose={() => setExportSessionId(null)}
        />
      )}
    </div>
  )
}

export default App
