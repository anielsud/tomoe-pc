package videohint

import "testing"

// drawTestFrame builds a packed RGB buffer (width*height*3 bytes)
// filled with bgColor, then draws a hollow rectangular border of
// ringColor and thickness at the given bounds.
func drawTestFrame(width, height int, bgColor [3]uint8, x, y, w, h, thickness int, ringColor [3]uint8) []byte {
	pix := make([]byte, width*height*3)
	for i := 0; i < width*height; i++ {
		pix[i*3], pix[i*3+1], pix[i*3+2] = bgColor[0], bgColor[1], bgColor[2]
	}

	setPixel := func(px, py int) {
		if px < 0 || px >= width || py < 0 || py >= height {
			return
		}
		idx := (py*width + px) * 3
		pix[idx], pix[idx+1], pix[idx+2] = ringColor[0], ringColor[1], ringColor[2]
	}

	for t := 0; t < thickness; t++ {
		for px := x; px < x+w; px++ {
			setPixel(px, y+t)
			setPixel(px, y+h-1-t)
		}
		for py := y; py < y+h; py++ {
			setPixel(x+t, py)
			setPixel(x+w-1-t, py)
		}
	}
	return pix
}

func TestDetectRing_FindsHollowBorder(t *testing.T) {
	const width, height = 200, 150
	ringColor := [3]uint8{100, 50, 200}
	bgColor := [3]uint8{20, 20, 20}

	pix := drawTestFrame(width, height, bgColor, 40, 30, 80, 60, 4, ringColor)

	cfg := RingConfig{
		TargetColor:     ringColor,
		ColorTolerance:  10,
		MinAreaFraction: 0.001,
		MaxAreaFraction: 0.5,
	}

	match, ok, _ := DetectRing(pix, width, height, cfg)
	if !ok {
		t.Fatal("DetectRing() found no match, want a match")
	}

	// Allow a little slack: the border's bounding box should roughly
	// match the drawn rectangle (40,30)-(120,90).
	if match.X < 38 || match.X > 42 || match.Y < 28 || match.Y > 32 {
		t.Errorf("match origin = (%d,%d), want near (40,30)", match.X, match.Y)
	}
	if match.Width < 75 || match.Width > 85 || match.Height < 55 || match.Height > 65 {
		t.Errorf("match size = %dx%d, want near 80x60", match.Width, match.Height)
	}
	if match.Confidence < hollownessFloor {
		t.Errorf("Confidence = %v, want >= %v", match.Confidence, hollownessFloor)
	}
}

func TestDetectRing_UnconfiguredReturnsNoMatch(t *testing.T) {
	pix := make([]byte, 100*100*3)
	var cfg RingConfig // zero value
	match, ok, _ := DetectRing(pix, 100, 100, cfg)
	if ok || match != nil {
		t.Errorf("DetectRing() with unconfigured RingConfig = (%v, %v), want (nil, false)", match, ok)
	}
}

func TestDetectRing_RejectsSolidBlob(t *testing.T) {
	const width, height = 100, 100
	blobColor := [3]uint8{200, 200, 200}
	bgColor := [3]uint8{0, 0, 0}

	pix := make([]byte, width*height*3)
	for i := 0; i < width*height; i++ {
		pix[i*3], pix[i*3+1], pix[i*3+2] = bgColor[0], bgColor[1], bgColor[2]
	}
	// Fill a solid (non-hollow) rectangle.
	for py := 20; py < 60; py++ {
		for px := 20; px < 60; px++ {
			idx := (py*width + px) * 3
			pix[idx], pix[idx+1], pix[idx+2] = blobColor[0], blobColor[1], blobColor[2]
		}
	}

	cfg := RingConfig{
		TargetColor:     blobColor,
		ColorTolerance:  10,
		MinAreaFraction: 0.001,
		MaxAreaFraction: 0.5,
	}

	if match, ok, _ := DetectRing(pix, width, height, cfg); ok {
		t.Errorf("DetectRing() matched a solid blob: %+v, want no match", match)
	}
}

func TestDetectRing_RejectsOutOfSizeRange(t *testing.T) {
	const width, height = 200, 150
	ringColor := [3]uint8{100, 50, 200}
	bgColor := [3]uint8{20, 20, 20}

	// A tiny ring, smaller than MinAreaFraction allows.
	pix := drawTestFrame(width, height, bgColor, 5, 5, 6, 6, 1, ringColor)

	cfg := RingConfig{
		TargetColor:     ringColor,
		ColorTolerance:  10,
		MinAreaFraction: 0.05, // requires at least 5% of the frame
		MaxAreaFraction: 0.5,
	}

	if match, ok, _ := DetectRing(pix, width, height, cfg); ok {
		t.Errorf("DetectRing() matched an undersized ring: %+v, want no match (below MinAreaFraction)", match)
	}
}

func TestDetectRing_AmbiguousWithTwoPlausibleRings(t *testing.T) {
	const width, height = 200, 150
	ringColor := [3]uint8{100, 50, 200}
	bgColor := [3]uint8{20, 20, 20}

	// Two separate, real speakers highlighted in the same frame --
	// draw a second border directly onto the frame drawTestFrame
	// already produced, rather than starting from a blank buffer
	// (which would just erase the first ring).
	pix := drawTestFrame(width, height, bgColor, 20, 20, 50, 40, 4, ringColor)
	setPixel := func(px, py int) {
		idx := (py*width + px) * 3
		pix[idx], pix[idx+1], pix[idx+2] = ringColor[0], ringColor[1], ringColor[2]
	}
	for t := 0; t < 4; t++ {
		for px := 120; px < 170; px++ {
			setPixel(px, 90+t)
			setPixel(px, 129-t)
		}
		for py := 90; py < 130; py++ {
			setPixel(120+t, py)
			setPixel(169-t, py)
		}
	}

	cfg := RingConfig{
		TargetColor:     ringColor,
		ColorTolerance:  10,
		MinAreaFraction: 0.001,
		MaxAreaFraction: 0.5,
	}

	match, found, ambiguous := DetectRing(pix, width, height, cfg)
	if !ambiguous {
		t.Fatal("DetectRing() ambiguous = false, want true (two plausible rings in the same frame)")
	}
	if found || match != nil {
		t.Errorf("DetectRing() = (%+v, %v) on an ambiguous frame, want (nil, false)", match, found)
	}
}

func TestDetectRing_NoMatchingColor(t *testing.T) {
	const width, height = 100, 100
	pix := make([]byte, width*height*3)
	for i := 0; i < width*height; i++ {
		pix[i*3], pix[i*3+1], pix[i*3+2] = 10, 10, 10 // uniform dark background
	}

	cfg := RingConfig{
		TargetColor:     [3]uint8{255, 0, 0},
		ColorTolerance:  10,
		MinAreaFraction: 0.001,
		MaxAreaFraction: 0.5,
	}

	if match, ok, _ := DetectRing(pix, width, height, cfg); ok {
		t.Errorf("DetectRing() matched with no ring-colored pixels present: %+v", match)
	}
}
