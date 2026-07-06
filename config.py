"""Configuration for the O-ring inspection system.

Every tunable value lives here so behaviour can be adjusted without
touching application logic.
"""

# Camera preview size, also used for the display window
PREVIEW_RESOLUTION = (1280, 960)

# Gaussian blur kernel applied before image comparison
BLUR_KERNEL_SIZE = (5, 5)

# Trackbar defaults. The diff threshold is stored x10 so the integer
# slider can represent 0.0 to 50.0 in steps of 0.1.
DEFAULT_NOISE_THRESHOLD = 30
DEFAULT_DIFF_THRESHOLD = 50

# Trackbar labels, defined once because they are needed both to create
# the sliders and to read them back every frame
NOISE_TRACKBAR = "Noise filter  (0-100)"
DIFF_TRACKBAR = "Diff threshold x10 (0-500)"

# On-disk persistence, indexed by slot (slot 1 uses index 0)
REFERENCE_PATHS = ["reference_1.jpg", "reference_2.jpg"]
ROI_PATHS = ["roi_1.npy", "roi_2.npy"]
LOGS_DIR = "inspections"

WINDOW_NAME = "O-ring Inspection"

# Reference thumbnail size shown in the bottom status bar
THUMB_W, THUMB_H = 110, 75

# Barcode scanner, read directly from its input device via evdev
SCANNER_DEVICE = None          # explicit /dev/input/eventX path, or None to auto-detect
SCANNER_NAME_HINT = "scanner"  # prefer a device whose name contains this (case-insensitive)
GRAB_SCANNER = True            # take exclusive access so scans don't leak to other windows

# Colours in BGR order because OpenCV uses BGR, not RGB
GREEN = (60, 200, 60)
RED = (50, 50, 220)
WHITE = (240, 240, 240)
YELLOW = (30, 200, 255)
GRAY = (130, 130, 130)
SLOT_COLORS = {1: (200, 140, 0), 2: (0, 160, 240)}
