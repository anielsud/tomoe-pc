package videohint

import "testing"

func TestLabelRect(t *testing.T) {
	ring := RingMatch{X: 1386, Y: 97, Width: 130, Height: 130}
	label := LabelRegion{BottomOffset: 52, Height: 40, MaxWidth: 300}

	x, y, w, h := LabelRect(ring, label)
	// MaxWidth (300) is wider than the ring (130), so the crop clamps
	// to the ring's own width rather than spilling past it.
	if x != ring.X || w != ring.Width {
		t.Errorf("LabelRect() x,w = %d,%d, want %d,%d (clamped to ring width)", x, w, ring.X, ring.Width)
	}
	if y != ring.Y+ring.Height-52 {
		t.Errorf("LabelRect() y = %d, want %d", y, ring.Y+ring.Height-52)
	}
	if h != 40 {
		t.Errorf("LabelRect() h = %d, want 40", h)
	}
}

func TestLabelRect_NarrowerThanRing(t *testing.T) {
	ring := RingMatch{X: 5, Y: 97, Width: 1794, Height: 1026}
	label := LabelRegion{BottomOffset: 52, Height: 40, MaxWidth: 300}

	x, y, w, h := LabelRect(ring, label)
	if x != ring.X || w != 300 {
		t.Errorf("LabelRect() x,w = %d,%d, want %d,%d (MaxWidth, not clamped)", x, w, ring.X, 300)
	}
	if y != ring.Y+ring.Height-52 || h != 40 {
		t.Errorf("LabelRect() y,h = %d,%d, want %d,%d", y, h, ring.Y+ring.Height-52, 40)
	}
}

func TestCropRGB(t *testing.T) {
	const width, height = 10, 10
	pix := make([]byte, width*height*3)
	for i := 0; i < width*height; i++ {
		pix[i*3] = byte(i) // encode pixel index into red channel
	}

	crop, cw, ch, err := cropRGB(pix, width, height, 2, 3, 4, 2)
	if err != nil {
		t.Fatalf("cropRGB() error = %v", err)
	}
	if cw != 4 || ch != 2 {
		t.Fatalf("cropRGB() size = %dx%d, want 4x2", cw, ch)
	}
	// First cropped pixel should be the source pixel at (2,3): index 3*10+2=32.
	if crop[0] != 32 {
		t.Errorf("crop[0] (red channel) = %d, want 32", crop[0])
	}
	// Second row's first pixel: source (2,4): index 4*10+2=42.
	if crop[1*4*3] != 42 {
		t.Errorf("crop row1 first pixel red channel = %d, want 42", crop[1*4*3])
	}
}

func TestCropRGB_ClampsToFrameBounds(t *testing.T) {
	const width, height = 10, 10
	pix := make([]byte, width*height*3)

	// Rectangle runs past the right and bottom edges.
	crop, cw, ch, err := cropRGB(pix, width, height, 8, 8, 5, 5)
	if err != nil {
		t.Fatalf("cropRGB() error = %v", err)
	}
	if cw != 2 || ch != 2 {
		t.Errorf("cropRGB() clamped size = %dx%d, want 2x2", cw, ch)
	}
	if len(crop) != cw*ch*3 {
		t.Errorf("len(crop) = %d, want %d", len(crop), cw*ch*3)
	}
}

func TestCropRGB_EmptyAfterClamping(t *testing.T) {
	const width, height = 10, 10
	pix := make([]byte, width*height*3)

	if _, _, _, err := cropRGB(pix, width, height, 20, 20, 5, 5); err == nil {
		t.Error("cropRGB() with an entirely out-of-bounds rectangle: want error, got nil")
	}
}

func TestCleanOCRName(t *testing.T) {
	cases := []struct {
		raw, want string
	}{
		{"Devin Dobrowolski Priv", "Devin Dobrowolski"},
		{"Devin Dobrowolski Privacy", "Devin Dobrowolski"},
		{"Nazanin Ramezani Muted", "Nazanin Ramezani"},
		{"Nazanin Ramezani Recording", "Nazanin Ramezani"},
		{"Christian Stanton", "Christian Stanton"}, // no noise word, unchanged
		{"Privacy", "Privacy"},                     // single word alone is never stripped
		{"  Devin Dobrowolski Priv  ", "Devin Dobrowolski"},
	}
	for _, c := range cases {
		if got := cleanOCRName(c.raw); got != c.want {
			t.Errorf("cleanOCRName(%q) = %q, want %q", c.raw, got, c.want)
		}
	}
}
