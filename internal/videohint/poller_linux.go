package videohint

import (
	"context"
	"time"
)

// Poll is a no-op on Linux: there's no screen-based video-hint signal
// there (no internal/teamsvideo equivalent), and callers already treat
// meeting mode as fully working without one. Kept so
// internal/daemon/internal/backend's call sites don't need runtime.GOOS
// checks — matches internal/meetingaudio's cross-platform seam. trigger
// and events are accepted (and ignored) for signature parity with
// poller_darwin.go; nothing is ever sent on or read from them here.
func Poll(ctx context.Context, interval, triggerDebounce time.Duration, trigger <-chan bool, events chan<- Event) {
	<-ctx.Done()
}
