#!/usr/bin/env python3
"""O-ring inspection script for Raspberry Pi 5 + Pi HQ Camera (IMX477).

This file is just the entry point now. The real work lives in the modules
it pulls in: Vision for image maths, Inspection for the pass/fail logic,
Storage for the disk, UI for the on-screen drawing and BarcodeScanner for
the scanner. main() wires them together and runs the event loop.
"""

import queue
import time

import cv2
from picamera2 import Picamera2

from Settings import (
    PREVIEW_RESOLUTION, DEFAULT_NOISE_THRESHOLD, DEFAULT_DIFF_THRESHOLD,
    WINDOW_NAME, SCANNER_DEVICE, SCANNER_NAME_HINT, GRAB_SCANNER,
)
from Vision import normalise_rect, crop, capture_still
from Inspection import run_inspection, recompute_live_results
from BarcodeScanner import BarcodeScanner
from Storage import load_saved, save_reference, save_inspection
from UI import mouse, on_mouse, draw_overlay, draw_barcode_popup, flash_result


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

    # Load anything saved from a previous session (regions, references, thumbs)
    rois, refs, thumbs = load_saved()

    sample_crops    = {}  # {slot: preprocessed crop} from the most recent inspection
    live_results    = {}  # {slot: (passed, diff_val)} shown in the status bar
    current_barcode = None

    # Show the barcode popup straight away, operator must set a barcode before
    # anything else, and this avoids blocking on a terminal input() call
    popup = {"active": True, "text": "", "error": ""}

    # Connect the barcode scanner (reads its input device directly so fast
    # scans aren't dropped by the GUI). Falls back to manual typing if absent.
    scanner = BarcodeScanner(device_path=SCANNER_DEVICE,
                             name_hint=SCANNER_NAME_HINT, grab=GRAB_SCANNER)

    try:
        while True:
            # Read slider values fresh every frame so changes take effect immediately.
            noise_thresh = cv2.getTrackbarPos("Noise filter  (0-100)", WINDOW_NAME)
            diff_thresh  = cv2.getTrackbarPos("Diff threshold x10 (0-500)", WINDOW_NAME) / 10.0

            # Apply any completed barcode scan. Scanning always sets the barcode
            # and closes the popup, whether or not it was open.
            try:
                while True:
                    current_barcode = scanner.results.get_nowait()
                    popup.update(active=False, text="", error="")
                    print(f"Barcode scanned: {current_barcode}")
            except queue.Empty:
                pass

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
                    # Persist to disk and get back the preprocessed ref + thumbnail
                    refs[slot], thumbs[slot] = save_reference(slot, ref_crop, roi_preview)
                    rois[slot] = roi_preview
                    # Clear any stale inspection data for this slot
                    sample_crops.pop(slot, None)
                    live_results.pop(slot, None)
                    print(f"Reference {slot} saved  ROI={roi_preview}")

                mouse["roi_ready"]   = False
                mouse["active_slot"] = None

            # If the operator moves a slider, re-run the comparison on the
            # last captured sample so the result updates without pressing SPACE
            live_results.update(
                recompute_live_results(refs, sample_crops, noise_thresh, diff_thresh)
            )

            # Build and show the display frame. The live frame is never reused
            # afterwards, so draw_overlay can draw straight onto it.
            display = draw_overlay(capture_still(cam), rois, refs, live_results,
                                   thumbs, current_barcode, noise_thresh, diff_thresh)

            if popup["active"]:
                # Show the live scan being assembled, or the manually typed text
                shown = scanner.snapshot() or popup["text"]
                display = draw_barcode_popup(display, shown, popup["error"])

            cv2.imshow(WINDOW_NAME, display)

            # waitKey(1) waits 1ms for a keypress. The & 0xFF masks the result
            # to 8 bits, which is needed on some platforms for correct key codes.
            key = cv2.waitKey(1) & 0xFF

            # Popup mode key handling
            # The scanner is read separately (via evdev above), so here we only
            # handle the rare case of someone typing a barcode by hand, plus
            # ESC to cancel. Human typing is slow enough for one key per frame.
            if popup["active"]:
                if key == 27:   # ESC
                    # Only allow closing the popup if a barcode is already set
                    if current_barcode is not None:
                        popup.update(active=False, text="", error="")
                elif key in (13, 10):   # ENTER (13) or numpad ENTER (10)
                    # Commit either manually typed text, or a scan whose scanner
                    # didn't append its own ENTER terminator
                    t = popup["text"] or scanner.take_buffer()
                    if t:
                        current_barcode = t
                        popup.update(active=False, text="", error="")
                        print(f"Barcode set to {current_barcode}")
                    else:
                        popup["error"] = "Barcode cannot be empty"
                elif key == 8:  # BACKSPACE
                    popup["text"]  = popup["text"][:-1]
                    popup["error"] = ""
                elif (48 <= key <= 57 or 65 <= key <= 90 or 97 <= key <= 122) \
                        and len(popup["text"]) < 20:   # digits, A-Z, a-z
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

                print("Inspecting...")
                still    = capture_still(cam)
                per_slot, crops = run_inspection(still, active_refs, rois,
                                                 noise_thresh, diff_thresh)
                sample_crops.update(crops)   # keep for live slider recomputation
                live_results.update(per_slot)

                overall_passed = all(p for p, _ in per_slot.values())
                for slot, (passed, diff_val) in sorted(per_slot.items()):
                    print(f"  Ref {slot}: {'PASS' if passed else 'FAIL'}  diff={diff_val:.1f}")
                print(f"Overall: {'PASS' if overall_passed else 'FAIL'}"
                      f"  (noise={noise_thresh}  threshold={diff_thresh:.1f})")

                # Rebuild display from the inspection still now that live_results
                # is updated. Without this, the saved image would show the
                # previous inspection's results in the status bar.
                display = draw_overlay(still, rois, refs, live_results, thumbs,
                                       current_barcode, noise_thresh, diff_thresh)
                save_inspection(current_barcode, display, per_slot, overall_passed)
                flash_result(display, overall_passed, per_slot)

    finally:
        # Always clean up the camera, scanner, and windows, even on exception
        scanner.close()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
