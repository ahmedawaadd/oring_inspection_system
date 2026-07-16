#!/usr/bin/env python3
"""
main.py

O-ring inspection for Raspberry Pi 5 + Pi HQ Camera (IMX477).

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
from config import (BARCODE_LENGTH, DEFAULT_DIFF_THRESHOLD,
                    DEFAULT_NOISE_THRESHOLD, DIFF_TRACKBAR, GRAB_SCANNER,
                    NOISE_TRACKBAR, PREVIEW_RESOLUTION, SCANNER_DEVICE,
                    SCANNER_NAME, SCANNER_SETTLE_SECONDS, WINDOW_NAME)
from scanner import BarcodeScanner
from vision import compare, crop, make_thumb, normalise_rect, preprocess


def setup_window():  # pragma: no cover, requires a display and OpenCV highgui
    """Create the display window with mouse callback and tuning sliders.
    Slider values are read back every frame, so moving them takes effect
    immediately."""
    # WINDOW_GUI_NORMAL disables Qt's expanded GUI (status bar, toolbar,
    # pixel picker). The pixel picker repaints on every mouse-move over
    # the image, which tanks the framerate on the Pi while hovering
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
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


def open_popup(popup, scanner, error=""):
    """Open the barcode popup, discarding anything the scanner collected
    while it was closed. Ignoring the scanner between popups is not the
    same as emptying it: without this flush, a stray scan made while no
    barcode was being asked for would be committed the instant the popup
    reopens, logging the next part under the wrong barcode."""
    scanner.flush()
    popup.update(active=True, text="", error=error)


def _commit_barcode(popup, text):
    """Accept `text` as the current barcode: close the popup and clear its
    state. Returns the barcode so callers can update their own variable."""
    popup.update(active=False, text="", error="")
    print(f"Barcode set to {text}")
    return text


def _accept_barcode(popup, text, barcode):
    """Commit `text` if it is exactly BARCODE_LENGTH characters, otherwise
    reject it with a visible error. Quietly accepting a truncated or
    over-long code would put wrong data in the inspection log, which is
    far worse than making the operator rescan. Returns the (possibly
    updated) current barcode."""
    if len(text) == BARCODE_LENGTH:
        return _commit_barcode(popup, text)
    popup["error"] = f"Barcode must be {BARCODE_LENGTH} characters"
    return barcode


def handle_popup_key(key, popup, scanner, barcode):
    """Handle one keypress while the barcode popup is open. The scanner is
    read separately via evdev, so this only covers manual typing plus ESC
    to cancel. Human typing is slow enough for one key per frame.

    Typing (or scanning) a full BARCODE_LENGTH code auto-commits it, so no
    ENTER press is needed; ENTER still works as a manual fallback for a
    scanner that doesn't append its own terminator. Anything that isn't
    exactly BARCODE_LENGTH characters is rejected, not committed.
    Returns the (possibly updated) current barcode."""
    if key == 27:  # ESC
        # Only allow closing the popup if a barcode is already set
        if barcode is not None:
            popup.update(active=False, text="", error="")
    elif key in (13, 10):  # ENTER (13) or numpad ENTER (10)
        # Commit manually typed text, or a scan whose scanner didn't
        # append its own ENTER terminator. Rejected text stays in the
        # field so the operator can finish typing it.
        text = popup["text"] or scanner.take_buffer()
        if text:
            barcode = _accept_barcode(popup, text, barcode)
        else:
            popup["error"] = "Barcode cannot be empty"
    elif key == 8:  # BACKSPACE
        popup["text"] = popup["text"][:-1]
        popup["error"] = ""
    elif (48 <= key <= 57 or 65 <= key <= 90 or 97 <= key <= 122) \
            and len(popup["text"]) < BARCODE_LENGTH:  # digits, A-Z, a-z
        popup["text"] += chr(key)
        popup["error"] = ""
        # Auto-commit as soon as the fixed-length barcode is complete
        if len(popup["text"]) >= BARCODE_LENGTH:
            barcode = _accept_barcode(popup, popup["text"], barcode)
    return barcode


def poll_scanner(scanner, popup, barcode):
    """Accept barcode-scanner input, but only while the popup is open, so a
    new barcode is taken only when the system is actually asking for one.
    A scan is committed either when the scanner appends its own ENTER (it
    arrives on the results queue) or, for scanners without a terminator,
    once the assembled buffer reaches BARCODE_LENGTH *and* the burst has
    settled: taking the buffer mid-scan would split one long scan into a
    barcode now and a stray tail later. Every code is length-checked
    before it is accepted. Returns the (possibly updated) barcode."""
    if not popup["active"]:
        return barcode
    # Completed scans (scanner appended ENTER). Stop once a commit closes
    # the popup; anything left over is discarded by the next open_popup
    try:
        while popup["active"]:
            barcode = _accept_barcode(popup, scanner.results.get_nowait(),
                                      barcode)
    except queue.Empty:
        pass
    # Partial scan with no terminator: long enough and finished arriving
    if popup["active"] and len(scanner.snapshot()) >= BARCODE_LENGTH \
            and scanner.settled():
        barcode = _accept_barcode(popup, scanner.take_buffer(), barcode)
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

    popup = {"active": False, "text": "", "error": ""}

    # Falls back to manual typing if no scanner is present
    scanner = BarcodeScanner(device_path=SCANNER_DEVICE,
                             name=SCANNER_NAME, grab=GRAB_SCANNER,
                             settle=SCANNER_SETTLE_SECONDS)

    # Show the barcode popup straight away: the operator must set a
    # barcode before anything else, and this avoids blocking on a
    # terminal input() call
    open_popup(popup, scanner)

    try:
        while True:
            # Read slider values fresh every frame so changes apply immediately
            noise_thresh = cv2.getTrackbarPos(NOISE_TRACKBAR, WINDOW_NAME)
            diff_thresh = cv2.getTrackbarPos(DIFF_TRACKBAR, WINDOW_NAME) / 10.0

            # Take scanner input only while the popup is asking for a
            # barcode, so a new part is accepted only between inspections
            current_barcode = poll_scanner(scanner, popup, current_barcode)

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
                    open_popup(popup, scanner,
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

                # A pass advances the production line: clear the barcode and
                # prompt for the next part. A fail keeps the same barcode so
                # the operator re-inspects until it passes.
                if overall_passed:
                    print(f"{current_barcode} passed. Scan the next barcode.")
                    current_barcode = None
                    open_popup(popup, scanner)

    finally:
        # Always release the camera, scanner, and windows, even on exception
        scanner.close()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
