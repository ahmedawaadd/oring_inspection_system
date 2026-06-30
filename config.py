"""Configuration constants and colour palette for the O-ring inspection tool."""

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PREVIEW_RESOLUTION = (1280, 960)
BLUR_KERNEL_SIZE   = (5, 5)

DEFAULT_NOISE_THRESHOLD = 30
DEFAULT_DIFF_THRESHOLD  = 50   # stored ×10; divide by 10 for actual value (0–50.0)

REFERENCE_PATHS = ["reference_1.jpg", "reference_2.jpg"]
ROI_PATHS       = ["roi_1.npy",       "roi_2.npy"]
WINDOW_NAME     = "O-ring Inspection"
LOGS_DIR        = "inspections"

THUMB_W, THUMB_H = 110, 75   # reference image thumbnail size in bottom bar

# Barcode scanner (read directly from its input device via evdev)
SCANNER_DEVICE    = None       # explicit /dev/input/eventX path, or None to auto-detect
SCANNER_NAME_HINT = "scanner"  # prefer a device whose name contains this (case-insensitive)
GRAB_SCANNER      = True        # take exclusive access so scans don't leak to other windows

# ---------------------------------------------------------------------------
# Colours (BGR)
# ---------------------------------------------------------------------------
# OpenCV uses BGR order (Blue, Green, Red) instead of the usual RGB,
# so the numbers below look backwards compared to what you might expect.
GREEN       = ( 60, 200,  60)
RED         = ( 50,  50, 220)
WHITE       = (240, 240, 240)
YELLOW      = ( 30, 200, 255)
GRAY        = (130, 130, 130)
SLOT_COLORS = {1: (200, 140,   0), 2: (0, 160, 240)}
