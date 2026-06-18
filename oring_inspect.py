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

DEFAULT_NOISE_THRESHOLD = 30   # per-pixel diff below this is ignored (0–100)
DEFAULT_DIFF_THRESHOLD  = 50   # stored ×10; divide by 10 for actual value (0–50.0)

REFERENCE_PATHS = ["reference_1.jpg", "reference_2.jpg"]
ROI_PATHS       = ["roi_1.npy", "roi_2.npy"]   # persist crop coords across runs
WINDOW_NAME     = "O-ring Inspection"
# ---------------------------------------------------------------------------

SCALE_X = CAPTURE_RESOLUTION[0] / PREVIEW_RESOLUTION[0]
SCALE_Y = CAPTURE_RESOLUTION[1] / PREVIEW_RESOLUTION[1]

# Colours (BGR)
GREEN  = (0, 220, 0)
RED    = (0, 0, 220)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
YELLOW = (0, 200, 220)
CYAN   = (220, 220, 0)
SLOT_COLORS = {1: (255, 180, 0), 2: (0, 180, 255)}  # blue-ish / orange-ish


# ---------------------------------------------------------------------------
# Mouse callback state (mutated in-place so the callback can share it)
# ---------------------------------------------------------------------------
mouse = {
    "active_slot": None,   # 1 or 2 while user is drawing
    "drawing": False,
    "pt1": (0, 0),
    "pt2": (0, 0),
    "roi_ready": False,    # set True on mouse-up so main loop can capture
}


def on_mouse(event, x, y, flags, param):
    if mouse["active_slot"] is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse["drawing"]   = True
        mouse["roi_ready"] = False
        mouse["pt1"]       = (x, y)
        mouse["pt2"]       = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and mouse["drawing"]:
        mouse["pt2"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and mouse["drawing"]:
        mouse["drawing"]   = False
        mouse["pt2"]       = (x, y)
        mouse["roi_ready"] = True


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def normalise_rect(pt1, pt2):
    """Return (x1, y1, x2, y2) with top-left / bottom-right guaranteed."""
    x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
    x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
    return x1, y1, x2, y2


def scale_roi(roi, scale_x=SCALE_X, scale_y=SCALE_Y):
    x1, y1, x2, y2 = roi
    return (int(x1 * scale_x), int(y1 * scale_y),
            int(x2 * scale_x), int(y2 * scale_y))


def crop(image, roi):
    x1, y1, x2, y2 = roi
    return image[y1:y2, x1:x2]


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)


def compare(ref_proc, sample_proc, noise_thresh, diff_thresh):
    # Resize sample to match reference in case of sub-pixel coord differences
    if ref_proc.shape != sample_proc.shape:
        sample_proc = cv2.resize(sample_proc, (ref_proc.shape[1], ref_proc.shape[0]))
    diff = cv2.absdiff(ref_proc, sample_proc)
    _, thresh = cv2.threshold(diff, noise_thresh, 255, cv2.THRESH_BINARY)
    mean_diff = float(np.mean(thresh))
    return mean_diff < diff_thresh, mean_diff


def capture_still(cam):
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


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def draw_overlay(frame, rois, refs, live_results):
    h, w = frame.shape[:2]

    # Draw saved ROI boxes on live feed
    for slot, roi in rois.items():
        color = SLOT_COLORS[slot]
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Ref {slot}", (x1 + 4, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # While user is actively drawing, show the live rubber-band box
    if mouse["active_slot"] is not None and (mouse["drawing"] or not mouse["roi_ready"]):
        slot  = mouse["active_slot"]
        color = SLOT_COLORS[slot]
        cv2.rectangle(frame, mouse["pt1"], mouse["pt2"], color, 2)

    # Bottom status bar
    bar_h = 90
    roi_region = frame[h - bar_h:h, :]
    roi_region[:] = (roi_region * 0.35).astype(np.uint8)

    if mouse["active_slot"] is not None:
        hint = f"Drawing Ref {mouse['active_slot']} — click and drag, release to confirm"
        cv2.putText(frame, hint, (10, h - bar_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1, cv2.LINE_AA)
    else:
        controls = "1/2: Draw reference region    SPACE: Inspect    Q: Quit"
        cv2.putText(frame, controls, (10, h - bar_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

    # Per-slot status
    for slot in [1, 2]:
        has_roi = slot in rois
        has_ref = slot in refs
        if has_roi and has_ref:
            status = "READY"
            color  = GREEN
        elif has_roi:
            status = "ROI set, no ref"
            color  = YELLOW
        else:
            status = "NONE"
            color  = YELLOW
        label = f"Ref {slot}: {status}"
        x = 10 if slot == 1 else 300
        cv2.putText(frame, label, (x, h - bar_h + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    # Per-slot inspection results
    if live_results:
        for slot, (passed, diff_val) in live_results.items():
            color  = GREEN if passed else RED
            result = "PASS" if passed else "FAIL"
            label  = f"Ref {slot}: {result}  diff={diff_val:.1f}"
            x = 10 if slot == 1 else 300
            cv2.putText(frame, label, (x, h - bar_h + 74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    return frame


def flash_result(frame, passed, per_slot):
    h, w = frame.shape[:2]
    banner = frame.copy()
    overall_color = GREEN if passed else RED
    overall_label = "PASS" if passed else "FAIL"
    cv2.rectangle(banner, (0, 0), (w, h), overall_color, 30)
    cv2.putText(banner, overall_label, (w // 2 - 120, h // 2),
                cv2.FONT_HERSHEY_DUPLEX, 5, overall_color, 10, cv2.LINE_AA)

    # Per-slot breakdown below
    y = h // 2 + 80
    for slot, (slot_passed, diff_val) in sorted(per_slot.items()):
        color = GREEN if slot_passed else RED
        label = f"Ref {slot}: {'PASS' if slot_passed else 'FAIL'}  diff={diff_val:.1f}"
        cv2.putText(banner, label, (w // 2 - 160, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        y += 50

    cv2.imshow(WINDOW_NAME, banner)
    cv2.waitKey(1800)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": PREVIEW_RESOLUTION, "format": "RGB888"}
    ))
    cam.start()
    time.sleep(2)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *PREVIEW_RESOLUTION)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    cv2.createTrackbar("Noise filter  (0-100)", WINDOW_NAME,
                       DEFAULT_NOISE_THRESHOLD, 100, lambda _: None)
    cv2.createTrackbar("Diff threshold x10 (0-500)", WINDOW_NAME,
                       DEFAULT_DIFF_THRESHOLD, 500, lambda _: None)

    # Load persisted ROIs and reference crops
    rois = {}   # {slot: (x1,y1,x2,y2)} in preview coords
    refs = {}   # {slot: preprocessed crop array}

    for slot in [1, 2]:
        roi_path = ROI_PATHS[slot - 1]
        ref_path = REFERENCE_PATHS[slot - 1]
        if os.path.exists(roi_path):
            rois[slot] = tuple(np.load(roi_path).tolist())
            print(f"Loaded ROI {slot} from {roi_path}")
        if os.path.exists(ref_path):
            refs[slot] = preprocess(cv2.imread(ref_path))
            print(f"Loaded reference {slot} from {ref_path}")

    sample_crops = {}  # {slot: preprocessed crop} from last inspect
    live_results = {}  # {slot: (passed, diff_val)}

    print("Live preview open.")
    print("  1 / 2 → click and drag to define reference region for O-ring 1 / 2")
    print("  SPACE → inspect both regions independently")
    print("  Q     → quit")

    try:
        while True:
            noise_thresh = cv2.getTrackbarPos("Noise filter  (0-100)", WINDOW_NAME)
            diff_thresh  = cv2.getTrackbarPos("Diff threshold x10 (0-500)", WINDOW_NAME) / 10.0

            # ROI was just drawn — capture still and save reference crop
            if mouse["roi_ready"] and mouse["active_slot"] is not None:
                slot = mouse["active_slot"]
                roi_preview = normalise_rect(mouse["pt1"], mouse["pt2"])

                # Ignore tiny accidental clicks
                if (roi_preview[2] - roi_preview[0]) > 10 and (roi_preview[3] - roi_preview[1]) > 10:
                    print(f"Capturing reference {slot}…")
                    still     = capture_still(cam)
                    roi_full  = scale_roi(roi_preview)
                    ref_crop  = crop(still, roi_full)
                    cv2.imwrite(REFERENCE_PATHS[slot - 1], ref_crop)
                    np.save(ROI_PATHS[slot - 1], np.array(roi_preview))
                    rois[slot]        = roi_preview
                    refs[slot]        = preprocess(ref_crop)
                    sample_crops.pop(slot, None)
                    live_results.pop(slot, None)
                    print(f"Reference {slot} saved  ROI={roi_preview}")

                mouse["roi_ready"]   = False
                mouse["active_slot"] = None

            # Recompute live results as sliders change
            for slot, ref_proc in refs.items():
                if slot in sample_crops:
                    passed, diff_val = compare(ref_proc, sample_crops[slot],
                                               noise_thresh, diff_thresh)
                    live_results[slot] = (passed, diff_val)

            frame   = cam.capture_array()
            frame   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            display = draw_overlay(frame.copy(), rois, refs, live_results)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key in (ord('1'), ord('2')):
                slot = int(chr(key))
                print(f"Draw the region for O-ring {slot} — click and drag on the preview.")
                mouse["active_slot"] = slot
                mouse["drawing"]     = False
                mouse["roi_ready"]   = False

            elif key == ord(' '):
                active_refs = {s: r for s, r in refs.items() if s in rois}
                if not active_refs:
                    print("No references set — press 1 or 2 to draw a region first.")
                    continue

                print("Inspecting…")
                still = capture_still(cam)

                per_slot = {}
                for slot, ref_proc in active_refs.items():
                    roi_full     = scale_roi(rois[slot])
                    sample_crop  = preprocess(crop(still, roi_full))
                    sample_crops[slot] = sample_crop
                    passed, diff_val   = compare(ref_proc, sample_crop,
                                                 noise_thresh, diff_thresh)
                    per_slot[slot]     = (passed, diff_val)
                    live_results[slot] = (passed, diff_val)

                overall_passed = all(p for p, _ in per_slot.values())
                for slot, (passed, diff_val) in sorted(per_slot.items()):
                    print(f"  Ref {slot}: {'PASS' if passed else 'FAIL'}  diff={diff_val:.1f}")
                print(f"Overall: {'PASS' if overall_passed else 'FAIL'}"
                      f"  (noise={noise_thresh}  threshold={diff_thresh:.1f})")
                flash_result(display, overall_passed, per_slot)

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
