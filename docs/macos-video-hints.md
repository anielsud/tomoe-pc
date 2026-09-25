# macOS video-hint speaker labeling

Covers `internal/videohint`: reading a meeting app's on-screen
active-speaker UI (a colored ring around whoever's talking, plus their
name label) and turning it into a naming hint for
`internal/speaker`'s audio-only clustering. See
[`macos-support.md`](macos-support.md) for overall status/roadmap and
[`macos-audio-capture.md`](macos-audio-capture.md) for the window
discovery/capture this builds on.

## Concept

`internal/speaker`'s embedding + clustering pipeline runs unchanged
from Linux — cluster audio, label clusters "Person N". This package
adds a second, independent signal: whenever a confident visual hint
lands, *label* the currently-active cluster with a real name
(`"Person N (Name)"`), carrying that label forward for the cluster's
later turns even without a fresh hint. A cluster that never gets a
hint stays "Person N" — the same as Linux does today, not a
regression. Only turns transcribed *after* a hint lands show the name;
earlier turns from the same speaker correctly keep the plain label
they had at the time.

```text
Teams window → capture frame → call-chrome gate → ring detection → OCR label
                                      │                  │             │
                                 not a call?         no match?    empty/error?
                                      │                  │             │
                                      └──────────────────┴─────────────┘
                                                          │
                                              escalate to review queue
                                          (only if chrome gate passed —
                                           see "Escalation library" below)
```

## Ring detection (`ring.go`)

Pure Go, no cgo: color-threshold every pixel against a target RGB
within a tolerance, connected-components label the result (4-connectivity
flood fill), then for each component check it's plausibly ring-shaped
(pixel count well below its bounding box's full area — a filled blob
wouldn't be) and within a min/max area-fraction of the whole frame.
Unit-tested against synthetic frames (`ring_test.go`).

**Teams calibration** (from a real, live, multi-participant call,
explicit authorization obtained first): active-speaker ring is
RGB(129,136,243), a hollow rounded-square border. `ColorTolerance: 25`
is deliberately generous to survive lighting/monitor variation.

**Simultaneous rings, fixed.** Teams can highlight more than one
recent speaker at once — observed live: two people both had a ring at
the same moment, and the original "just return the single
best-scoring match" behavior confidently attributed the hint to
whichever one scored higher, silently mislabeling the other.
`DetectRing` now returns three values (`match, found, ambiguous`):
finding *more than one* plausible candidate sets `ambiguous` and
returns no match at all, rather than guessing between them. The
poller reports this as its own `StageAmbiguousRing` event (distinct
from "no ring found") and still escalates the frame — a real "two
people highlighted" moment is exactly the kind of case worth keeping
in the snapshot library, even though nothing here is a rule gap to
fix.

## Label geometry (`label.go`, `rule.go`'s `LabelRegion`)

**Model the label's position/size in absolute pixels relative to the
ring's bottom edge, not as a fraction of the ring's own size.** This
was the second attempt, not the first — worth keeping the story
because the wrong mental model is a natural one to reach for:

The original calibration modeled the label as a fraction of the ring's
bounding box (bottom 27% of ring height, full ring width), eyeballed
against one gallery-tile screenshot. It happened to look right for
that one tile size and was wrong everywhere else. Measuring two real
frames at very different scales — a full-screen 1-on-1 tile
(~1794x1026px ring) and a ~440x245px gallery tile — showed label
position/size differing by more than 3x in ring-relative *fraction*
terms, but matching within a few pixels in *absolute* offset from the
ring's bottom edge and absolute label height. That's exactly what
fixed-size font rendering predicts (Teams renders the name label at a
constant on-screen font size regardless of tile size) and a
fraction-of-ring model structurally can't express. Current values:
`BottomOffset: 52`, `Height: 40`, `MaxWidth: 300` (clamped to the
ring's own width if narrower, so a small tile's crop doesn't spill into
a neighboring tile). Re-running OCR against every real escalated frame
that had a ring match: 0/7 succeeded under the fractional model, 7/7
succeed under the fixed-pixel model.

**Porting lesson:** when calibrating any "where does app X draw UI
element Y" rule from a single screenshot, get at least two real
examples at very different absolute scales before trusting a
ring/tile-*relative* model — fixed-font-size UI chrome won't scale
proportionally, and one example can't distinguish "scales with the
container" from "constant size, just measured at one particular
container size."

## OCR (`ocr_darwin.go`, `ocr_bridge_darwin.m`)

Vision.framework text recognition (`VNRecognizeTextRequest`), the same
cgo/Objective-C bridge shape as `internal/guestaudio`. Two real bugs
found wiring this up:

1. **Autoreleased objects assigned into a `__block` variable without
   retaining.** This file builds without ARC (manual retain/release,
   matching the rest of this project's ObjC bridges); Vision hands the
   completion handler's block autoreleased objects, and assigning one
   straight into a `__block` var let it get freed before the outer
   function read it back — segfaulting on every call. Fixed by
   explicitly retaining in the block and releasing before return.
2. **Holding a Go slice's backing array by raw pointer across the cgo
   boundary isn't safe** once anything (like a completion handler) can
   outlive the call that took the pointer. Switched from
   `CGDataProviderCreateWithData` (wraps the caller's pointer directly)
   to `CGDataProviderCreateWithCFData` over an immediately-copied
   `NSData`.

**Noise stripping.** Observed live: a real name OCR'd as "Devin
Dobrowolski Priv" — "Priv" (a truncated "Privacy") came from Teams'
background-blur/privacy indicator overlapping the label crop, not from
the name. `cleanOCRName` (`label.go`) strips a single trailing word
from the OCR result if it matches (or is a partial-word prefix of) a
small known-noise list (`privacy`, `muted`, `mute`, `recording`,
`live`) — matched only as the *last* word, since a real name is never
expected to end with one of these, and only ever strips one word, so
a genuinely two-word name is never touched. This is a text-level
patch, not a geometry recalibration — the actual overlay's on-screen
position hasn't been measured, so if a *different* trailing artifact
shows up it won't yet be caught; extend the list rather than assume
this is exhaustive.

## Call-chrome gate (`chrome.go`)

**`FindMeetingWindow`'s title heuristic isn't enough to confirm a
window is actually showing a live call.** Two real, distinct
wrong-window captures made it through live: a 1:1 chat conversation,
and the post-meeting recording/playback page — both Teams-owned, both
non-trivially titled, neither an actual call. One of them, worse than
just wasting a capture, produced a *plausible-looking wrong OCR read*:
Vision happily read the page's own window-title text off screen and
returned it as if it were a recognized speaker's name.

Fixed with `DetectCallChrome`: look for the "Leave call" hang-up
icon's small red glyph within a fixed-position search window
(**absolute pixels from the frame's top-right corner, not a fraction
of frame size** — same reasoning as label geometry: this is native
toolbar chrome, and Teams renders it at a constant pixel size/position
regardless of window size). Calibrated against the real escalation
library: exactly 110 matching pixels in every one of 15 real call
frames checked (across four different window sizes), 0 in all 7
non-call frames. Wired into `Poll` as a hard gate before Ring/Label are
even attempted — a rejected frame emits `StageNotACall` and is never
written to the escalation library at all.

**Matched by hue+saturation, not exact RGB** — a deliberate bet that a
semantic "danger/leave" accent color keeps its hue across light/dark
theme even if its brightness shifts, since theme changes mostly affect
background/surface luminance rather than accent hues in most native UI
toolkits. Untested against an actual light-mode capture — none exists
in this library yet.

**Calibration gotcha found live:** a first pass searched too tall a Y
window (20-95px from the top) and picked up a second, unrelated
reddish element (a notification-badge-shaped area at y~20-37) inside
the *same* chat-window false positive at a pixel count close enough to
the real icon's (120 vs. 110) to nearly defeat the gate. Tightening the
search window to y∈[44,63] (matching the real icon's actual measured
position) separated the two completely. Porting lesson: a
"pixel count within a search box" gate is only as good as how tightly
the box excludes *other* things that share the target color — measure
the false positive's exact location before assuming a wider net is
safer.

## Escalation library (`snapshot.go`)

A captured frame + metadata that didn't produce a confident hint lands
in a **staging** directory
(`config.UnrecognizedUIPendingDir`), never the permanent library
(`config.UnrecognizedUIApprovedDir`) — promotion requires a human
explicitly approving it (`tomoe videohint {list,approve,discard}`, or
the GUI's pending-screenshots panel). This exists specifically because
a captured window can be the wrong thing entirely (see the call-chrome
gate above) — nothing here is safe to calibrate a rule from, or even
safe to have captured at all, until reviewed. The call-chrome gate
(added later) prevents most wrong-window captures from reaching this
queue at all; the staging/approval step remains as defense for
whatever it doesn't catch.

Rate-limited (skip if <60s since the last capture in the same `Poll`
call) and capped (stop after 5 snapshots per call) so a long idle
meeting can't fill the disk. Retroactively applying the label-geometry
and chrome-gate fixes to a real ~22-snapshot library collected across
one day's live meetings: escalation-worthy snapshots dropped to 7, and
every one of those 7 is a genuine "ring wasn't visually present at that
instant" miss, not a bug.

## Speaker-tracker wiring (`internal/speaker.Tracker.SetHintForRecent`)

A video hint only knows "this name is active right now," not which
audio cluster ID it belongs to — it's attributed to whichever cluster
the audio pipeline most recently assigned an embedding to (both
signals are keyed to the same monitor-source audio). Baked in at
`Tracker.Assign` time, so this is a point-in-time relabel, not
retroactive.

**Sticky-speaker continuity fix:** live multi-participant testing found
one person's continuous turn fragmenting into a fresh "Person N" per
sentence — each single-sentence VAD segment produced a noisier
embedding than a longer utterance, occasionally missing
`speaker.Tracker`'s similarity threshold. Fixed with a sticky-speaker
heuristic in `internal/speaker` — see `speaker.Tuning`'s doc comment
for the full mechanism, current thresholds, and the later real-call
diagnostic session that retuned `DefaultThreshold` itself (0.65 → 0.55)
from real similarity scores rather than a guess. Every one of these
constants is now hot-reloadable from `config.toml` (`MeetingConfig`,
`config.Watch`) — no rebuild or relaunch needed to retune it again.

**Cluster merging on a matching video hint.** A video hint is
independent evidence of identity, separate from audio similarity —
if it resolves *two different* "Person N" clusters to the same real
name, that's a strong signal they're actually one person whose
embeddings simply never clustered together (exactly the kind of
mistake real-call diagnostics found the audio side making).
`SetHintForRecent` now merges in that case (`speaker.Tracker.mergeInto`):
folds the newer cluster's centroid into the existing one's running
average and aliases it, so every future match against either resolves
to the same identity — without ever renumbering an unrelated
"Person N" (see `Tracker.canonical`). Gated by `sameIdentity`, stricter
than the truncation check used elsewhere: requires an exact match, or
a truncation relationship with the shorter name at least 4 characters,
so a bare ambiguous fragment (a first initial, "Mr") can never trigger
a merge.

## Polling (`poller_darwin.go`)

Runs on a fixed ticker *and* an immediate trigger:
`live.Coordinator.HintNeeded()` signals the moment a monitor-source
speaker with no hint yet is heard (`speaker.Tracker.Assign`'s
`needsHint` return value), debounced separately from the ticker (now a
`Poll` parameter, `triggerDebounce`, default 1s) so a still-talking
unlabeled speaker can't hammer ScreenCaptureKit + Vision faster than
that. Both the ticker interval (default 5s, down from an original 10s)
and the debounce come from `config.toml`
(`MeetingConfig.VideoHintPollInterval`/`VideoHintTriggerDebounce`),
read fresh each time a meeting session starts — unlike
`speaker.Tracker`'s tuning, these aren't hot-reloaded *mid-session*
(the ticker's already running by the time a session starts), but still
need no rebuild or relaunch: a new session picks up an edited config
immediately.

Every stage reached each tick is reported on an `events` channel
(window found/not, frame captured, not-a-call, ring matched/not, OCR
hit/miss, escalated) — non-blocking, so a slow/absent consumer never
stalls polling. `internal/daemon` logs every stage; `internal/backend`
additionally buffers a short ring (`GetVideoHintActivity`) and forwards
each one live via a `"videohint:activity"` Wails event to a frontend
ticker.

## Still open

- **Window-finding only works for Teams.** Zoom/Meet/Webex/Slack each
  need their own window-finder (native owner-name match for
  Zoom/Slack; browser owner-name + title match for Meet/Webex, reusing
  `internal/meeting/platform.go`'s existing pattern) before video hints
  produce anything for them.
- **Orphaned hints.** A hint with no recent-enough speaker to attach to
  (`SetHintForRecent` returns `false`) is logged but otherwise silently
  dropped, not retried.
- **Truncated names.** Teams' active-speaker tile often truncates the
  name (e.g. "Nazanin Rame…") — a fine hint, not necessarily the
  participant's actual full name. Investigated: macOS's own local data
  sources (Outlook's mail/calendar cache, macOS Contacts, macOS
  Calendar) were all empty/proprietary dead ends on the one real
  machine checked; Teams' own local IndexedDB cache has a real but
  narrow mri→displayName mapping (from the @-mention feature — only
  ~18% coverage against real meeting participants in one measured
  session) not accepted as a real solve. Remaining candidates: reading
  the roster/participants panel (opportunistic OCR, needs live
  calibration), or Microsoft Graph API (`/me/people`) — genuine
  complete coverage, but a fundamentally bigger feature (network +
  OAuth, not a passive local read).
- **Ambiguous-ring escalations still need a human look.** Nothing
  automatically distinguishes "two people genuinely both just spoke"
  from a false-positive second ring (e.g. noise coincidentally passing
  the color/shape filters) — both currently escalate the same way.
