#!/usr/bin/env python3
"""O-ring inspection script for Raspberry Pi 5 + Pi HQ Camera (IMX477)."""

import os
import time

import cv2
import numpy as np
from picamera2 import Picamera2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PREVIEW_RESOLUTION = (1280, 960)   # live preview size (lower = faster refresh)
CAPTURE_RESOLUTION = (4056, 3040)  # IMX477 full res used for reference + inspect
BLUR_KERNEL_SIZE   = (5, 5)        # Gaussian blur kernel (must be odd x odd)

# Starting values for the interactive sliders
DEFAULT_NOISE_THRESHOLD = 30   # per-pixel diff below this is ignored (0–100)
DEFAULT_DIFF_THRESHOLD  = 50   # mean of surviving pixels × 10 → 5.0 (0–500 → 0.0–50.0)

REFERENCE_PATHS = ["reference_1.jpg", "reference_2.jpg"]
WINDOW_NAME     = "O-ring Inspection"
# ---------------------------------------------------------------------------

# Colours (BGR)
GREEN  = (0, 220, 0)
RED    = (0, 0, 220)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
YELLOW = (0, 200, 220)
CYAN   = (220, 220, 0)


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)


def compare(ref_proc, sample_proc, noise_thresh, diff_thresh):
    diff = cv2.absdiff(ref_proc, sample_proc)
    _, thresh = cv2.threshold(diff, noise_thresh, 255, cv2.THRESH_BINARY)
    mean_diff = float(np.mean(thresh))
    return mean_diff < diff_thresh, mean_diff


def compare_against_all(refs, sample_proc, noise_thresh, diff_thresh):
    """PASS if sample matches any loaded reference. Returns (passed, best_diff, matched_slot)."""
    best_diff  = None
    best_slot  = None
    for slot, ref_proc in refs.items():
        passed, diff_val = compare(ref_proc, sample_proc, noise_thresh, diff_thresh)
        if best_diff is None or diff_val < best_diff:
            best_diff = diff_val
            best_slot = slot
        if passed:
            return True, diff_val, slot
    return False, best_diff, best_slot


def capture_still(cam):
    """Switch to full-res still, capture, return BGR image, resume preview."""
    cam.stop()
    cam.configure(cam.create_still_configuration(
        main={"size": CAPTURE_RESOLUTION, "format": "RGB888"}
    ))
    cam.start()
    time.sleep(0.5)
    bgr = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)

    cam.stop()
    cam.configure(cam.create_video_configuration(
        main={"size": PREVIEW_RESOLUTION, "format": "RGB888"}
    ))
    cam.start()
    return bgr


def draw_overlay(frame, refs, live_result):
    h, w = frame.shape[:2]
    bar_h = 80

    roi = frame[h - bar_h:h, :]
    roi[:] = (roi * 0.4).astype(np.uint8)

    controls = "1: Save Ref 1    2: Save Ref 2    SPACE: Inspect    Q: Quit"
    cv2.putText(frame, controls, (10, h - bar_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1, cv2.LINE_AA)

    # Reference slot status
    for slot in [1, 2]:
        loaded = slot in refs
        text   = f"Ref {slot}: {'LOADED' if loaded else 'NONE'}"
        color  = GREEN if loaded else YELLOW
        x      = 10 if slot == 1 else 220
        cv2.putText(frame, text, (x, h - bar_h + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    if live_result:
        result_str, diff_val, matched_slot = live_result
        color = GREEN if result_str == "PASS" else RED
        match_label = f"vs Ref {matched_slot}" if matched_slot else ""
        label = f"{result_str}  diff={diff_val:.1f}  {match_label}"
        cv2.putText(frame, label, (10, h - bar_h + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return frame


def flash_result(frame, passed):
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
    cam.configure(cam.create_video_configuration(
        main={"size": PREVIEW_RESOLUTION, "format": "RGB888"}
    ))
    cam.start()
    time.sleep(2)  # warm-up

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *PREVIEW_RESOLUTION)

    # Sliders — DIFF_THRESHOLD stored ×10 so trackbar stays integer
    cv2.createTrackbar("Noise filter  (0-100)", WINDOW_NAME,
                       DEFAULT_NOISE_THRESHOLD, 100, lambda _: None)
    cv2.createTrackbar("Diff threshold x10 (0-500)", WINDOW_NAME,
                       DEFAULT_DIFF_THRESHOLD, 500, lambda _: None)

    # Load any existing reference images on startup
    refs = {}  # {1: preprocessed_array, 2: preprocessed_array}
    for slot, path in enumerate(REFERENCE_PATHS, start=1):
        if os.path.exists(path):
            refs[slot] = preprocess(cv2.imread(path))
            print(f"Loaded existing reference {slot} from {path}")

    sample_proc = None  # last captured inspect frame, kept for live slider feedback
    live_result = None  # (result_str, diff_val, matched_slot)

    print("Live preview open.")
    print("  1     → save Reference 1")
    print("  2     → save Reference 2")
    print("  SPACE → inspect (PASS if matches either reference)")
    print("  Q     → quit")
    print("  Drag sliders to tune sensitivity live after an inspect capture.")

    try:
        while True:
            noise_thresh = cv2.getTrackbarPos("Noise filter  (0-100)", WINDOW_NAME)
            diff_thresh  = cv2.getTrackbarPos("Diff threshold x10 (0-500)", WINDOW_NAME) / 10.0

            # Recompute live whenever sliders change (no re-capture needed)
            if refs and sample_proc is not None:
                passed, diff_val, slot = compare_against_all(
                    refs, sample_proc, noise_thresh, diff_thresh
                )
                live_result = ("PASS" if passed else "FAIL", diff_val, slot)

            frame   = cam.capture_array()
            frame   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            display = draw_overlay(frame.copy(), refs, live_result)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key in (ord('1'), ord('2')):
                slot = int(chr(key))
                path = REFERENCE_PATHS[slot - 1]
                print(f"Capturing reference {slot}…")
                still      = capture_still(cam)
                cv2.imwrite(path, still)
                refs[slot] = preprocess(still)
                sample_proc = None
                live_result = None
                print(f"Reference {slot} saved to {path}")

            elif key == ord(' '):
                if not refs:
                    print("No references loaded — press 1 or 2 first.")
                    continue
                print("Inspecting…")
                still       = capture_still(cam)
                sample_proc = preprocess(still)
                passed, diff_val, slot = compare_against_all(
                    refs, sample_proc, noise_thresh, diff_thresh
                )
                live_result = ("PASS" if passed else "FAIL", diff_val, slot)
                print(f"{live_result[0]}  (diff={diff_val:.1f}  best_ref={slot}  "
                      f"noise={noise_thresh}  threshold={diff_thresh:.1f})")
                flash_result(display, passed)

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
