import { useEffect, useRef, useState } from 'react';
import { EventsOn } from '../../wailsjs/runtime/runtime';
import { Segment } from '../types';

const MAX_SEGMENTS = 300;

// speaker.Tracker.label() embeds a resolved video-hint name in parens
// directly in the Speaker string, e.g. "Person 1 (Kevin)" — parsed
// back out here rather than re-deriving it from anywhere else, so this
// view never needs its own copy of that logic.
function parseSpeaker(speaker: string): { base: string; hint: string | null } {
  const m = speaker.match(/^(.*?)(?:\s\(([^)]+)\))?$/);
  if (!m) return { base: speaker, hint: null };
  return { base: m[1], hint: m[2] || null };
}

// decisionTag turns a raw AssignDecision + "does this speaker have a
// video hint yet" into the "OCR" / "OCR+Centroid match" tag: a hint
// alone means some earlier segment's confident/sticky/etc. match
// carried a name over; a *confident* decision on THIS segment means
// the audio side independently re-confirmed the same identity this
// instant, hence the stronger "+Centroid match" tag.
function decisionTag(decision: string | undefined, hasHint: boolean): string {
  if (!hasHint) return '';
  return decision === 'confident' ? 'OCR+Centroid match' : 'OCR';
}

function formatSpeakerLabel(seg: Segment): string {
  const { base, hint } = parseSpeaker(seg.speaker);
  if (!hint) return base;
  const tag = decisionTag(seg.decision, true);
  return `${hint} [${base}${tag ? ', ' + tag : ''}]`;
}

function formatTimestamp(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const pad = (n: number) => String(n).padStart(2, '0');
  return `[${pad(h)}:${pad(m)}:${pad(sec)}]`;
}

function statusLabel(status: string | undefined): string {
  switch (status) {
    case 'live':
      return 'listening…';
    case 'pending':
      return 'refining…';
    default:
      return 'final';
  }
}

// DiagnosticsPane is a real-time view into HOW each transcript line
// got its speaker label — decision kind (confident/sticky/short-
// segment/new-speaker), whether a video hint has attached, and
// utterance status (live/pending/final) — using the exact same
// transcript:segment(:update) events TranscriptPane consumes, so
// there's no separate backend event stream to keep in sync. Kept as
// its own view rather than folded into TranscriptPane so the normal
// transcript stays exactly as clean as it looks today; this is a
// tuning/debugging tool, not something an end user needs to see.
//
// Known limitation, not fixed here: a video hint only labels segments
// emitted AFTER it lands (see speaker.Tracker's doc comments) —
// already-sent segments for that speaker are never retroactively
// relabeled. So a line's speaker tag can jump straight from plain
// "Person N" to "Name [Person N, OCR]" on a LATER line, rather than
// the same line visibly upgrading through every stage.
export default function DiagnosticsPane() {
  const [segments, setSegments] = useState<Segment[]>([]);
  const logRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    const cancelNew = EventsOn('transcript:segment', (seg: Segment) => {
      setSegments(prev => [...prev, seg].slice(-MAX_SEGMENTS));
    });
    const cancelUpdate = EventsOn('transcript:segment:update', (seg: Segment) => {
      setSegments(prev => prev.map(s => (s.id === seg.id ? seg : s)));
    });
    return () => {
      cancelNew();
      cancelUpdate();
    };
  }, []);

  useEffect(() => {
    if (autoScroll) {
      logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
    }
  }, [segments, autoScroll]);

  return (
    <div className="diagnostics-pane">
      <div className="panel-header">
        <h2>Diagnostics</h2>
        <span className="session-count">{segments.length} segments</span>
      </div>
      <div
        className="diagnostics-log"
        ref={logRef}
        onScroll={e => {
          const el = e.currentTarget;
          setAutoScroll(el.scrollHeight - el.scrollTop - el.clientHeight < 40);
        }}
      >
        {segments.length === 0 ? (
          <div className="empty-state" style={{ height: 200 }}>
            No segments yet — start a meeting session to see live speaker-assignment diagnostics.
          </div>
        ) : (
          segments.map(seg => (
            <div key={seg.id} className={`diagnostics-row status-${seg.status || 'final'}`}>
              <span className="diagnostics-time">{formatTimestamp(seg.start_time)}</span>
              <span className="diagnostics-speaker">{formatSpeakerLabel(seg)}:</span>
              <span className={`diagnostics-text ${seg.status === 'live' ? 'diagnostics-text-live' : ''}`}>
                {seg.text}
              </span>
              <span className="diagnostics-status-badge">{statusLabel(seg.status)}</span>
              {seg.decision && <span className="diagnostics-decision-badge">{seg.decision}</span>}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
