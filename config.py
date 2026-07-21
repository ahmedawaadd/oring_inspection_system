"""
config.py

Configuration for the O-ring inspection system.

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
SCANNER_DEVICE = None             # explicit /dev/input/eventX path, or None to look up by name
# Known barcode scanners, matched by device name (case-insensitive substring).
# Searched in order at startup; the first one connected is used. Add new
# models here so operators don't have to select their scanner manually.
SCANNER_NAMES = [
    "Honeywell 1950g",
    "Hand Held Products IT4600",
]
GRAB_SCANNER = True               # take exclusive access so scans don't leak to other windows

# Barcodes are a fixed length; the entry popup auto-commits as soon as this
# many characters have been scanned or typed, so no ENTER press is needed.
BARCODE_LENGTH = 7

# How long the scan buffer must be quiet before it is read. A scanner
# "types" its whole burst with only a few ms between characters, so this
# much silence means the scan is over; reading sooner could take the
# first BARCODE_LENGTH characters of a longer code and leave its tail
# behind as a phantom barcode for the next part.
SCANNER_SETTLE_SECONDS = 0.1

# Colours in BGR order because OpenCV uses BGR, not RGB
GREEN = (60, 200, 60)
RED = (50, 50, 220)
WHITE = (240, 240, 240)
YELLOW = (30, 200, 255)
GRAY = (130, 130, 130)
SLOT_COLORS = {1: (200, 140, 0), 2: (0, 160, 240)}
