#!/usr/bin/env python3
"""O-ring inspection script for Raspberry Pi 5 + Pi HQ Camera (IMX477)."""

import os
import sys
import time

import cv2
import numpy as np
from picamera2 import Picamera2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PREVIEW_RESOLUTION = (1280, 960)  # live preview size (lower = faster refresh)
CAPTURE_RESOLUTION = (4056, 3040) # IMX477 full res used for reference + inspect
BLUR_KERNEL_SIZE   = (5, 5)       # Gaussian blur kernel (must be odd x odd)
DIFF_THRESHOLD     = 10.0         # Mean pixel diff below this → PASS

REFERENCE_PATH = "reference.jpg"
WINDOW_NAME    = "O-ring Inspection"
# ---------------------------------------------------------------------------

# Colours (BGR)
GREEN  = (0, 220, 0)
RED    = (0, 0, 220)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
YELLOW = (0, 200, 220)


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)


def compare(ref_proc, sample_proc):
    diff = cv2.absdiff(ref_proc, sample_proc)
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)
    mean_diff = float(np.mean(thresh))
    return mean_diff < DIFF_THRESHOLD, mean_diff


def capture_still(cam):
    """Switch to full-res still, capture, return BGR image, resume preview."""
    cam.stop()
    still_cfg = cam.create_still_configuration(
        main={"size": CAPTURE_RESOLUTION, "format": "RGB888"}
    )
    cam.configure(still_cfg)
    cam.start()
    time.sleep(0.5)  # brief settle after mode switch
    frame = cam.capture_array()
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Switch back to preview mode
    cam.stop()
    preview_cfg = cam.create_video_configuration(
        main={"size": PREVIEW_RESOLUTION, "format": "RGB888"}
    )
    cam.configure(preview_cfg)
    cam.start()
    return bgr


def draw_overlay(frame, ref_loaded, last_result):
    h, w = frame.shape[:2]
    bar_h = 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    controls = "  R: Save reference    SPACE: Inspect    Q: Quit"
    cv2.putText(frame, controls, (10, h - bar_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

    # Reference status
    ref_text  = "Reference: LOADED" if ref_loaded else "Reference: NONE"
    ref_color = GREEN if ref_loaded else YELLOW
    cv2.putText(frame, ref_text, (10, h - bar_h + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, ref_color, 1, cv2.LINE_AA)

    # Last result
    if last_result:
        result_str, diff_val = last_result
        color = GREEN if result_str == "PASS" else RED
        label = f"Last: {result_str}  (diff={diff_val:.2f})"
        tw, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0], None
        cv2.putText(frame, label, (w - 320, h - bar_h + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    return frame


def flash_result(frame, passed):
    """Draw a full-frame PASS/FAIL banner briefly."""
    color = GREEN if passed else RED
    label = "PASS" if passed else "FAIL"
    h, w = frame.shape[:2]
    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (w, h), color, 30)
    cv2.putText(banner, label, (w // 2 - 120, h // 2 + 40),
                cv2.FONT_HERSHEY_DUPLEX, 5, color, 10, cv2.LINE_AA)
    cv2.imshow(WINDOW_NAME, banner)
    cv2.waitKey(1200)


def main():
    cam = Picamera2()
    preview_cfg = cam.create_video_configuration(
        main={"size": PREVIEW_RESOLUTION, "format": "RGB888"}
    )
    cam.configure(preview_cfg)
    cam.start()
    time.sleep(2)  # warm-up

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *PREVIEW_RESOLUTION)

    # Load existing reference if present
    ref_proc   = None
    ref_loaded = False
    if os.path.exists(REFERENCE_PATH):
        ref_image  = cv2.imread(REFERENCE_PATH)
        ref_proc   = preprocess(ref_image)
        ref_loaded = True
        print(f"Loaded existing reference from {REFERENCE_PATH}")

    last_result = None  # (result_str, diff_val)

    print("Live preview open.")
    print("  R     → save reference image")
    print("  SPACE → inspect against reference")
    print("  Q     → quit")

    try:
        while True:
            frame = cam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            display = draw_overlay(frame.copy(), ref_loaded, last_result)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('r'):
                print("Capturing reference…")
                still = capture_still(cam)
                cv2.imwrite(REFERENCE_PATH, still)
                ref_proc   = preprocess(still)
                ref_loaded = True
                last_result = None
                print(f"Reference saved to {REFERENCE_PATH}")

            elif key == ord(' '):
                if not ref_loaded:
                    print("No reference loaded — press R first.")
                    continue
                print("Inspecting…")
                still        = capture_still(cam)
                sample_proc  = preprocess(still)
                passed, diff = compare(ref_proc, sample_proc)
                result_str   = "PASS" if passed else "FAIL"
                last_result  = (result_str, diff)
                print(f"{result_str}  (mean_diff={diff:.2f}, threshold={DIFF_THRESHOLD})")
                flash_result(display, passed)

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
