#!/usr/bin/env python3
"""O-ring inspection script for Raspberry Pi 5 + Pi HQ Camera (IMX477)."""

import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
from picamera2 import Picamera2

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

# ---------------------------------------------------------------------------
# Mouse state
# ---------------------------------------------------------------------------
# OpenCV's mouse callback is a separate function that can't return values to
# the main loop directly. This shared dictionary is the workaround, both
# the callback and the main loop read and write the same object, so changes
# in one are immediately visible in the other.
mouse = {
    "active_slot": None,  # which slot (1 or 2) the user is drawing for right now
    "drawing":     False, # true while the left mouse button is held down
    "pt1":         (0, 0),
    "pt2":         (0, 0),
    "roi_ready":   False, # flipped to True on mouse-up so the main loop knows to act
}


def on_mouse(event, x, y, flags, param):
    # Ignore mouse events if no slot is active (user hasn't pressed 1 or 2 yet)
    if mouse["active_slot"] is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse.update(drawing=True, roi_ready=False, pt1=(x, y), pt2=(x, y))
    elif event == cv2.EVENT_MOUSEMOVE and mouse["drawing"]:
        # Keep updating pt2 as the mouse moves
        mouse["pt2"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and mouse["drawing"]:
        # signal the main loop on mouse release to capture the reference
        mouse.update(drawing=False, pt2=(x, y), roi_ready=True)



# Core helpers

def normalise_rect(pt1, pt2):
    """Return (x1, y1, x2, y2) with top-left / bottom-right guaranteed.
    Needed because the user might drag in any direction."""
    x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
    x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
    return x1, y1, x2, y2


def crop(image, roi):
    # NumPy slicing: image[rows, cols] 
    x1, y1, x2, y2 = roi
    return image[y1:y2, x1:x2]


def preprocess(image):
    # Convert to greyscale, colour isn't needed for comparison
    # Blur slightly to smooth out camera sensor noise so
    # two photos of the same thing don't look different just from sensor noise.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)


def compare(ref_proc, sample_proc, noise_thresh, diff_thresh):
    # Resize the sample to match the reference if they're slightly different
    # sizes (can happen due to sub-pixel differences in ROI coordinates)
    if ref_proc.shape != sample_proc.shape:
        sample_proc = cv2.resize(sample_proc, (ref_proc.shape[1], ref_proc.shape[0]))

    # Subtract the two images pixel by pixel (identical pixels cancel to zero
    # differences show up as bright spots
    diff = cv2.absdiff(ref_proc, sample_proc)

    # Zero out any differences smaller than noise_thresh 
    _, thresh = cv2.threshold(diff, noise_thresh, 255, cv2.THRESH_BINARY)

    # Average the remaining differences. If that average exceeds diff_thresh,
    # there's enough difference to call it a FAIL.
    mean_diff = float(np.mean(thresh))
    return mean_diff < diff_thresh, mean_diff


def capture_still(cam):
    # The camera outputs RGB but OpenCV works in BGR, so convert on the way in
    return cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)


def load_thumb(path):
    """Load a reference image from disk and shrink it to thumbnail size."""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    # INTER_AREA is the best interpolation method for shrinking images
    return cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)



# UI helpers

def _dark_panel(frame, x1, y1, x2, y2, alpha=0.82):
    """Darken a rectangular region of the frame to create a UI bar.
    alpha controls how dark: 0.82 means 82% dark colour, 18% original image."""
    region = frame[y1:y2, x1:x2]
    dark   = np.full_like(region, (20, 20, 20))
    cv2.addWeighted(dark, alpha, region, 1.0 - alpha, 0, region)
    frame[y1:y2, x1:x2] = region


# Overlay drawing

def draw_overlay(frame, rois, refs, live_results, thumbs, barcode,
                 noise_thresh, diff_thresh):
    h, w = frame.shape[:2]

    # ROI boxes on live feed 
    # Draw a coloured rectangle for each saved region of interest so the
    # operator can see where the system is looking
    for slot, roi in rois.items():
        color = SLOT_COLORS[slot]
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Ref {slot}", (x1 + 4, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)

    # Show the live rubber-band box while the user is actively drawing
    if mouse["active_slot"] is not None and (mouse["drawing"] or not mouse["roi_ready"]):
        cv2.rectangle(frame, mouse["pt1"], mouse["pt2"],
                      SLOT_COLORS[mouse["active_slot"]], 2)

    # Top bar (50 px) 
    _dark_panel(frame, 0, 0, w, 50)

    bc_text = f"BARCODE  #{barcode}" if barcode is not None else "BARCODE:  (press B)"
    cv2.putText(frame, bc_text, (16, 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, YELLOW, 2, cv2.LINE_AA)

    # Measure the threshold text width first so we can right-align it
    thr_text = f"Noise {noise_thresh}   Threshold {diff_thresh:.1f}"
    (tw, _), _ = cv2.getTextSize(thr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.putText(frame, thr_text, (w - tw - 16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, GRAY, 1, cv2.LINE_AA)

    # Bottom bar (130 px) 
    bar_y = h - 130
    _dark_panel(frame, 0, bar_y, w, h)

    # Show a drawing hint while the user is defining an ROI, otherwise show
    # the normal keyboard controls
    if mouse["active_slot"] is not None:
        hint = f"Drawing Ref {mouse['active_slot']}   click and drag, release to confirm"
        cv2.putText(frame, hint, (16, bar_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, YELLOW, 1, cv2.LINE_AA)
    else:
        ctrl = "1 / 2: Draw reference     SPACE: Inspect     B: Set barcode     Q: Quit"
        cv2.putText(frame, ctrl, (16, bar_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, GRAY, 1, cv2.LINE_AA)

    # Slot panels: one for each reference (thumbnail on the left, status on the right)
    for idx, slot in enumerate([1, 2]):
        sc = SLOT_COLORS[slot]
        px = 16 + idx * 380  # space the two panels across the bar
        ty = bar_y + 38

        # Show the reference thumbnail, or a grey placeholder if none is saved yet
        thumb = thumbs.get(slot)
        if thumb is not None:
            frame[ty:ty + THUMB_H, px:px + THUMB_W] = thumb
            cv2.rectangle(frame, (px, ty), (px + THUMB_W, ty + THUMB_H), sc, 1)
        else:
            cv2.rectangle(frame, (px, ty), (px + THUMB_W, ty + THUMB_H), (50, 50, 50), -1)
            cv2.rectangle(frame, (px, ty), (px + THUMB_W, ty + THUMB_H), sc, 1)
            cv2.putText(frame, "NO REF",
                        (px + 20, ty + THUMB_H // 2 + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1, cv2.LINE_AA)

        # Status label beside the thumbnail
        lx = px + THUMB_W + 12
        ly = ty + 14

        has_roi = slot in rois
        has_ref = slot in refs
        if has_roi and has_ref:
            status_txt, s_color = "READY",  GREEN   # good to go
        elif has_roi:
            status_txt, s_color = "ROI SET", YELLOW  # region drawn but no reference captured yet
        else:
            status_txt, s_color = "NO ROI",  GRAY    # not set up at all

        cv2.putText(frame, f"REF {slot}", (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, sc, 1, cv2.LINE_AA)
        cv2.putText(frame, status_txt, (lx, ly + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, s_color, 1, cv2.LINE_AA)

        # Show the last inspection result for this slot if one exists
        if slot in live_results:
            passed, diff_val = live_results[slot]
            r_color = GREEN if passed else RED
            cv2.putText(frame, "PASS" if passed else "FAIL",
                        (lx, ly + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, r_color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"diff {diff_val:.1f}",
                        (lx, ly + 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1, cv2.LINE_AA)

    return frame


def draw_barcode_popup(frame, text, error):
    """Draw an in-window barcode entry dialog over the live camera frame.
    The actual keyboard input is handled in the main loop  this function
    just renders whatever text has been typed so far."""
    h, w = frame.shape[:2]

    # Dim the camera feed behind the dialog so it doesn't compete visually
    cv2.convertScaleAbs(frame, frame, alpha=0.45)

    # Centre the dialog box on screen
    dw, dh = 480, 220
    dx = (w - dw) // 2
    dy = (h - dh) // 2

    cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), (30, 30, 30), -1)
    cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), (90, 90, 90), 2)

    cv2.putText(frame, "Enter Barcode Number", (dx + 20, dy + 42),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, WHITE, 2, cv2.LINE_AA)
    cv2.line(frame, (dx + 16, dy + 52), (dx + dw - 16, dy + 52),
             (70, 70, 70), 1)

    ix1, iy1, ix2, iy2 = dx + 16, dy + 64, dx + dw - 16, dy + 130
    cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (50, 50, 50), -1)
    cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (100, 100, 100), 1)
    cv2.putText(frame, text + "|", (ix1 + 12, iy2 - 14),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, YELLOW, 2, cv2.LINE_AA)

    cv2.putText(frame, "ENTER to confirm    ESC to cancel    (scan or type)",
                (dx + 16, dy + 158),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, GRAY, 1, cv2.LINE_AA)

    # Only shown if the operator typed something invalid and pressed ENTER
    if error:
        cv2.putText(frame, error, (dx + 16, dy + 192),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, RED, 1, cv2.LINE_AA)

    return frame


def flash_result(frame, passed, per_slot):
    # Work on a copy so we don't permanently draw over the display frame
    h, w = frame.shape[:2]
    banner = frame.copy()
    color  = GREEN if passed else RED
    label  = "PASS" if passed else "FAIL"

    cv2.rectangle(banner, (0, 0), (w, h), color, 30)

    # Measure the text width so we can centre it precisely
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = 6.0
    thick = 14
    (lw, _), _ = cv2.getTextSize(label, font, scale, thick)
    cv2.putText(banner, label, ((w - lw) // 2, h // 2 - 20),
                font, scale, color, thick, cv2.LINE_AA)

    # Per-slot breakdown below the main verdict, also centred
    y = h // 2 + 72
    for slot, (slot_passed, diff_val) in sorted(per_slot.items()):
        sc   = GREEN if slot_passed else RED
        txt  = f"Ref {slot}:  {'PASS' if slot_passed else 'FAIL'}   diff {diff_val:.1f}"
        (sw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)
        cv2.putText(banner, txt, ((w - sw) // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, sc, 2, cv2.LINE_AA)
        y += 48

    cv2.imshow(WINDOW_NAME, banner)
    cv2.waitKey(1800)  # freeze the banner on screen for 1.8 seconds



# Logging

def save_inspection(barcode, frame, per_slot, overall_passed):
    # Each barcode gets its own folder so results are easy to browse by part
    folder = os.path.join(LOGS_DIR, str(barcode))
    os.makedirs(folder, exist_ok=True)

    ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    verdict = "PASS" if overall_passed else "FAIL"
    color   = GREEN if overall_passed else RED

    # Stamp the barcode and verdict directly onto the saved image so the
    # result is readable without opening the CSV
    out = frame.copy()
    cv2.putText(out, f"#{barcode}  {verdict}", (10, 80),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3, cv2.LINE_AA)

    img_path = os.path.join(folder, f"{ts}_{verdict}.jpg")
    cv2.imwrite(img_path, out)

    # Append one row to the CSV for this barcode, writing the header first
    # if this is the first inspection for this barcode
    log_path     = os.path.join(folder, "log.csv")
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "barcode", "overall",
                             "slot1", "slot1_diff", "slot2", "slot2_diff"])
        row = [ts, barcode, verdict]
        for slot in [1, 2]:
            if slot in per_slot:
                p, d = per_slot[slot]
                row += ["PASS" if p else "FAIL", f"{d:.1f}"]
            else:
                row += ["N/A", "N/A"]
        writer.writerow(row)

    print(f"Saved → {img_path}")


# Main

def main():
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": PREVIEW_RESOLUTION, "format": "RGB888"}
    ))
    cam.start()
    time.sleep(2)  # give the camera sensor time to settle before capturing

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *PREVIEW_RESOLUTION)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # These sliders appear above the window. The trackbar values are read
    # inside the loop every frame, so moving them takes effect immediately.
    cv2.createTrackbar("Noise filter  (0-100)", WINDOW_NAME,
                       DEFAULT_NOISE_THRESHOLD, 100, lambda _: None)
    cv2.createTrackbar("Diff threshold x10 (0-500)", WINDOW_NAME,
                       DEFAULT_DIFF_THRESHOLD, 500, lambda _: None)

    rois   = {}  # {slot: (x1, y1, x2, y2)} the region boxes in camera coordinates
    refs   = {}  # {slot: preprocessed numpy array} the reference images to compare against
    thumbs = {}  # {slot: small BGR image} thumbnails shown in the status bar

    # Load anything saved from a previous session so the operator doesn't
    # have to redraw regions and recapture references every time
    for slot in [1, 2]:
        roi_path = ROI_PATHS[slot - 1]
        ref_path = REFERENCE_PATHS[slot - 1]
        if os.path.exists(roi_path):
            rois[slot] = tuple(np.load(roi_path).tolist())
        if os.path.exists(ref_path):
            refs[slot]   = preprocess(cv2.imread(ref_path))
            thumbs[slot] = load_thumb(ref_path)

    sample_crops    = {}  # {slot: preprocessed crop} from the most recent inspection
    live_results    = {}  # {slot: (passed, diff_val)} shown in the status bar
    current_barcode = None

    # Show the barcode popup straight away, operator must set a barcode before
    # anything else, and this avoids blocking on a terminal input() call
    popup = {"active": True, "text": "", "error": ""}

    try:
        while True:
            # Read slider values fresh every frame so changes take effect immediately.
            noise_thresh = cv2.getTrackbarPos("Noise filter  (0-100)", WINDOW_NAME)
            diff_thresh  = cv2.getTrackbarPos("Diff threshold x10 (0-500)", WINDOW_NAME) / 10.0

            # Handle a completed ROI draw 
            # roi_ready is set by the mouse callback when the user releases
            # the mouse button. We act on it here in the main loop.
            if mouse["roi_ready"] and mouse["active_slot"] is not None:
                slot        = mouse["active_slot"]
                roi_preview = normalise_rect(mouse["pt1"], mouse["pt2"])

                # Ignore accidental tiny clicks (less than 10px in either direction)
                if (roi_preview[2] - roi_preview[0]) > 10 and \
                   (roi_preview[3] - roi_preview[1]) > 10:
                    still    = capture_still(cam)
                    ref_crop = crop(still, roi_preview)
                    cv2.imwrite(REFERENCE_PATHS[slot - 1], ref_crop)   # persist to disk
                    np.save(ROI_PATHS[slot - 1], np.array(roi_preview)) # persist ROI coords
                    rois[slot]   = roi_preview
                    refs[slot]   = preprocess(ref_crop)
                    # Build thumbnail directly from the in-memory crop. no need
                    # to read the file we just saved back off disk
                    thumbs[slot] = cv2.resize(ref_crop, (THUMB_W, THUMB_H),
                                              interpolation=cv2.INTER_AREA)
                    # Clear any stale inspection data for this slot
                    sample_crops.pop(slot, None)
                    live_results.pop(slot, None)
                    print(f"Reference {slot} saved  ROI={roi_preview}")

                mouse["roi_ready"]   = False
                mouse["active_slot"] = None

            # Live threshold recomputation 
            # If the operator moves a slider, re-run the comparison on the
            # last captured sample so the result updates without pressing SPACE
            for slot, ref_proc in refs.items():
                if slot in sample_crops:
                    passed, diff_val   = compare(ref_proc, sample_crops[slot],
                                                 noise_thresh, diff_thresh)
                    live_results[slot] = (passed, diff_val)

            # Build and show the display frame 
            raw     = cam.capture_array()
            frame   = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            # Pass a copy so draw_overlay can draw on it freely without
            # affecting the raw frame (we still need the clean version for inspection)
            display = draw_overlay(frame.copy(), rois, refs, live_results, thumbs,
                                   current_barcode, noise_thresh, diff_thresh)

            if popup["active"]:
                display = draw_barcode_popup(display, popup["text"], popup["error"])

            cv2.imshow(WINDOW_NAME, display)

            # waitKey(1) waits 1ms for a keypress. The & 0xFF masks the result
            # to 8 bits, which is needed on some platforms for correct key codes.
            key = cv2.waitKey(1) & 0xFF

            # Popup mode key handling 
            # While the popup is open, all keypresses go to the barcode input.
            # The 'continue' at the end jumps back to the top of the loop,
            # skipping the normal key handling below.
            if popup["active"]:
                if key == 27:   # ESC
                    # Only allow closing the popup if a barcode is already set
                    if current_barcode is not None:
                        popup.update(active=False, text="", error="")
                elif key in (13, 10):   # ENTER (13) or numpad ENTER (10)
                    t = popup["text"]
                    if len(t) > 0:
                        current_barcode = t
                        popup.update(active=False, text="", error="")
                        print(f"Barcode set to {current_barcode}")
                    else:
                        popup["error"] = "Barcode cannot be empty"
                elif key == 8:  # BACKSPACE
                    popup["text"]  = popup["text"][:-1]
                    popup["error"] = ""
                elif (48 <= key <= 57 or 65 <= key <= 90 or 97 <= key <= 122) \
                        and len(popup["text"]) < 20:   # digits, A–Z, a–z
                    popup["text"] += chr(key)
                    popup["error"] = ""
                continue  # don't fall through to normal key handling below

            # Normal mode key handling 
            if key == ord('q'):
                break

            elif key == ord('b'):
                popup.update(active=True, text="", error="")

            elif key in (ord('1'), ord('2')):
                slot = int(chr(key))
                mouse["active_slot"] = slot
                mouse["drawing"]     = False
                mouse["roi_ready"]   = False

            elif key == ord(' '):
                # Only inspect slots that have both a region and a reference
                # (possible to have an ROI saved without a reference if the
                # reference file was deleted between sessions)
                active_refs = {s: r for s, r in refs.items() if s in rois}
                if not active_refs:
                    print("No references set. press 1 or 2 to draw a region first.")
                    continue
                if current_barcode is None:
                    popup.update(active=True, text="",
                                 error="Set a barcode before inspecting")
                    continue

                print("Inspecting…")
                still    = capture_still(cam)
                per_slot = {}
                for slot, ref_proc in active_refs.items():
                    sample_crop        = preprocess(crop(still, rois[slot]))
                    sample_crops[slot] = sample_crop  # keep for live slider recomputation
                    passed, diff_val   = compare(ref_proc, sample_crop,
                                                 noise_thresh, diff_thresh)
                    per_slot[slot]     = (passed, diff_val)
                    live_results[slot] = (passed, diff_val)

                overall_passed = all(p for p, _ in per_slot.values())
                for slot, (passed, diff_val) in sorted(per_slot.items()):
                    print(f"  Ref {slot}: {'PASS' if passed else 'FAIL'}  diff={diff_val:.1f}")
                print(f"Overall: {'PASS' if overall_passed else 'FAIL'}"
                      f"  (noise={noise_thresh}  threshold={diff_thresh:.1f})")

                # Rebuild display from the inspection still now that live_results
                # is updated. Without this, the saved image would show the
                # previous inspection's results in the status bar. the 'display'
                # variable above was constructed before the comparison ran.
                display = draw_overlay(still.copy(), rois, refs, live_results, thumbs,
                                       current_barcode, noise_thresh, diff_thresh)
                save_inspection(current_barcode, display, per_slot, overall_passed)
                flash_result(display, overall_passed, per_slot)

    finally:
        # Always clean up the camera and windows, even if an exception is thrown
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
