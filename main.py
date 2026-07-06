#!/usr/bin/env python3
"""O-ring inspection for Raspberry Pi 5 + Pi HQ Camera (IMX477).

Entry point and main loop only. The rest of the system lives in:
  config.py   tunable values, paths, and colours
  camera.py   Picamera2 setup and capture
  vision.py   image preprocessing and comparison
  scanner.py  barcode scanner input via evdev
  ui.py       OpenCV drawing and mouse input
  storage.py  reference persistence and inspection logs
"""

import queue

import cv2

import storage
import ui
from camera import capture_still, create_camera
from config import (DEFAULT_DIFF_THRESHOLD, DEFAULT_NOISE_THRESHOLD,
                    DIFF_TRACKBAR, GRAB_SCANNER, NOISE_TRACKBAR,
                    PREVIEW_RESOLUTION, SCANNER_DEVICE, SCANNER_NAME_HINT,
                    WINDOW_NAME)
from scanner import BarcodeScanner
from vision import compare, crop, make_thumb, normalise_rect, preprocess


def setup_window():  # pragma: no cover, requires a display and OpenCV highgui
    """Create the display window with mouse callback and tuning sliders.
    Slider values are read back every frame, so moving them takes effect
    immediately."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *PREVIEW_RESOLUTION)
    cv2.setMouseCallback(WINDOW_NAME, ui.on_mouse)
    cv2.createTrackbar(NOISE_TRACKBAR, WINDOW_NAME,
                       DEFAULT_NOISE_THRESHOLD, 100, lambda _: None)
    cv2.createTrackbar(DIFF_TRACKBAR, WINDOW_NAME,
                       DEFAULT_DIFF_THRESHOLD, 500, lambda _: None)


def handle_completed_roi(cam, rois, refs, thumbs, sample_crops, live_results):
    """Capture and persist a new reference after the user finishes drawing.
    The mouse callback sets roi_ready on mouse-up; the actual capture
    happens here in the main loop."""
    slot = ui.mouse["active_slot"]
    roi = normalise_rect(ui.mouse["pt1"], ui.mouse["pt2"])

    # Ignore accidental tiny clicks (under 10 px in either direction)
    if (roi[2] - roi[0]) > 10 and (roi[3] - roi[1]) > 10:
        still = capture_still(cam)
        ref_crop = crop(still, roi)
        storage.save_reference(slot, ref_crop, roi)
        rois[slot] = roi
        refs[slot] = preprocess(ref_crop)
        # Build the thumbnail from the in-memory crop, no need to read
        # the file just written back off disk
        thumbs[slot] = make_thumb(ref_crop)
        # Clear stale inspection data for this slot
        sample_crops.pop(slot, None)
        live_results.pop(slot, None)
        print(f"Reference {slot} saved  ROI={roi}")

    ui.mouse["roi_ready"] = False
    ui.mouse["active_slot"] = None


def handle_popup_key(key, popup, scanner, barcode):
    """Handle one keypress while the barcode popup is open. The scanner is
    read separately via evdev, so this only covers manual typing plus ESC
    to cancel. Human typing is slow enough for one key per frame.
    Returns the (possibly updated) current barcode."""
    if key == 27:  # ESC
        # Only allow closing the popup if a barcode is already set
        if barcode is not None:
            popup.update(active=False, text="", error="")
    elif key in (13, 10):  # ENTER (13) or numpad ENTER (10)
        # Commit manually typed text, or a scan whose scanner didn't
        # append its own ENTER terminator
        text = popup["text"] or scanner.take_buffer()
        if text:
            barcode = text
            popup.update(active=False, text="", error="")
            print(f"Barcode set to {barcode}")
        else:
            popup["error"] = "Barcode cannot be empty"
    elif key == 8:  # BACKSPACE
        popup["text"] = popup["text"][:-1]
        popup["error"] = ""
    elif (48 <= key <= 57 or 65 <= key <= 90 or 97 <= key <= 122) \
            and len(popup["text"]) < 20:  # digits, A-Z, a-z
        popup["text"] += chr(key)
        popup["error"] = ""
    return barcode


def run_inspection(cam, active_refs, rois, sample_crops, live_results,
                   noise_thresh, diff_thresh):
    """Capture a still and compare every active slot against its reference.
    Returns (still, per_slot results, overall pass)."""
    print("Inspecting...")
    still = capture_still(cam)
    per_slot = {}
    for slot, ref_proc in active_refs.items():
        sample_crop = preprocess(crop(still, rois[slot]))
        sample_crops[slot] = sample_crop  # keep for live slider recomputation
        passed, diff_val = compare(ref_proc, sample_crop, noise_thresh, diff_thresh)
        per_slot[slot] = (passed, diff_val)
        live_results[slot] = (passed, diff_val)

    overall_passed = all(p for p, _ in per_slot.values())
    for slot, (passed, diff_val) in sorted(per_slot.items()):
        print(f"  Ref {slot}: {'PASS' if passed else 'FAIL'}  diff={diff_val:.1f}")
    print(f"Overall: {'PASS' if overall_passed else 'FAIL'}"
          f"  (noise={noise_thresh}  threshold={diff_thresh:.1f})")
    return still, per_slot, overall_passed


def main():  # pragma: no cover, drives real camera and GUI; logic lives in the tested helpers
    cam = create_camera()
    setup_window()

    # Load anything saved by a previous session
    rois, refs, thumbs = storage.load_references()

    sample_crops = {}      # {slot: preprocessed crop} from the most recent inspection
    live_results = {}      # {slot: (passed, diff_val)} shown in the status bar
    current_barcode = None

    # Show the barcode popup straight away: the operator must set a
    # barcode before anything else, and this avoids blocking on a
    # terminal input() call
    popup = {"active": True, "text": "", "error": ""}

    # Falls back to manual typing if no scanner is present
    scanner = BarcodeScanner(device_path=SCANNER_DEVICE,
                             name_hint=SCANNER_NAME_HINT, grab=GRAB_SCANNER)

    try:
        while True:
            # Read slider values fresh every frame so changes apply immediately
            noise_thresh = cv2.getTrackbarPos(NOISE_TRACKBAR, WINDOW_NAME)
            diff_thresh = cv2.getTrackbarPos(DIFF_TRACKBAR, WINDOW_NAME) / 10.0

            # Apply any completed scan. Scanning always sets the barcode
            # and closes the popup, whether or not it was open
            try:
                while True:
                    current_barcode = scanner.results.get_nowait()
                    popup.update(active=False, text="", error="")
                    print(f"Barcode scanned: {current_barcode}")
            except queue.Empty:
                pass

            # Act on a finished ROI draw (mouse released)
            if ui.mouse["roi_ready"] and ui.mouse["active_slot"] is not None:
                handle_completed_roi(cam, rois, refs, thumbs,
                                     sample_crops, live_results)

            # If the operator moves a slider, re-run the comparison on the
            # last captured sample so results update without pressing SPACE
            for slot, ref_proc in refs.items():
                if slot in sample_crops:
                    live_results[slot] = compare(ref_proc, sample_crops[slot],
                                                 noise_thresh, diff_thresh)

            # Build and show the display frame. Overlay draws on a copy so
            # the clean frame stays available for inspection
            frame = capture_still(cam)
            display = ui.draw_overlay(frame.copy(), rois, refs, live_results,
                                      thumbs, current_barcode,
                                      noise_thresh, diff_thresh)

            if popup["active"]:
                # Show the live scan being assembled, or the typed text
                shown = scanner.snapshot() or popup["text"]
                display = ui.draw_barcode_popup(display, shown, popup["error"])

            cv2.imshow(WINDOW_NAME, display)

            # waitKey(1) waits 1 ms for a keypress. The & 0xFF masks to
            # 8 bits, needed on some platforms for correct key codes
            key = cv2.waitKey(1) & 0xFF

            if popup["active"]:
                current_barcode = handle_popup_key(key, popup, scanner,
                                                   current_barcode)
                continue  # don't fall through to normal key handling

            if key == ord('q'):
                break

            elif key == ord('b'):
                popup.update(active=True, text="", error="")

            elif key in (ord('1'), ord('2')):
                # Arm ROI drawing for the chosen slot
                ui.mouse["active_slot"] = int(chr(key))
                ui.mouse["drawing"] = False
                ui.mouse["roi_ready"] = False

            elif key == ord(' '):
                # Only inspect slots with both a region and a reference
                # (an ROI can exist without a reference if the reference
                # file was deleted between sessions)
                active_refs = {s: r for s, r in refs.items() if s in rois}
                if not active_refs:
                    print("No references set. Press 1 or 2 to draw a region first.")
                    continue
                if current_barcode is None:
                    popup.update(active=True, text="",
                                 error="Set a barcode before inspecting")
                    continue

                still, per_slot, overall_passed = run_inspection(
                    cam, active_refs, rois, sample_crops, live_results,
                    noise_thresh, diff_thresh)

                # Rebuild the display from the inspection still now that
                # live_results is updated, otherwise the saved image would
                # show the previous inspection's results in the status bar
                display = ui.draw_overlay(still.copy(), rois, refs,
                                          live_results, thumbs,
                                          current_barcode,
                                          noise_thresh, diff_thresh)
                storage.save_inspection(current_barcode, display,
                                        per_slot, overall_passed)
                ui.flash_result(display, overall_passed, per_slot)

    finally:
        # Always release the camera, scanner, and windows, even on exception
        scanner.close()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
