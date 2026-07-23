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

import hmac
import queue

import cv2

import storage
import ui
from camera import capture_still, create_camera
from config import (BARCODE_LENGTH, DIFF_TRACKBAR, ENGINEER_LOGIN_KEY,
                    ENGINEER_LOGOUT_KEY, ENGINEER_PASSWORD,
                    ENGINEER_SCAN_KEY, ENGINEER_USERNAME, GRAB_SCANNER,
                    LOGIN_FIELD_MAX_LENGTH, NOISE_TRACKBAR,
                    PREVIEW_RESOLUTION, SCANNER_DEVICE, SCANNER_NAMES,
                    SCANNER_SETTLE_SECONDS, WINDOW_NAME)
from scanner import BarcodeScanner
from vision import compare, crop, make_thumb, normalise_rect, preprocess


def setup_window(noise_thresh, diff_thresh):  # pragma: no cover, requires a display and OpenCV highgui
    """Create the display window with mouse callback and tuning sliders.
    Their saved positions are visible to everyone, but only authenticated
    Engineer mode is allowed to adopt changes."""
    # WINDOW_GUI_NORMAL disables Qt's expanded GUI (status bar, toolbar,
    # pixel picker). The pixel picker repaints on every mouse-move over
    # the image, which tanks the framerate on the Pi while hovering
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *PREVIEW_RESOLUTION)
    cv2.setMouseCallback(WINDOW_NAME, ui.on_mouse)
    cv2.createTrackbar(NOISE_TRACKBAR, WINDOW_NAME,
                       noise_thresh, 100, lambda _: None)
    cv2.createTrackbar(DIFF_TRACKBAR, WINDOW_NAME,
                       int(round(diff_thresh * 10)), 500, lambda _: None)


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
    popup.update(active=True, text="", error=error,
                 inspection_requested=False)


def _commit_barcode(popup, text):
    """Accept `text` as the current barcode: close the popup and clear its
    entry state. The inspection flag lets scanner and manual entry share
    one trigger without either input path touching camera hardware.
    Returns the barcode so callers can update their own variable."""
    popup.update(active=False, text="", error="",
                 inspection_requested=True)
    print(f"Barcode set to {text}")
    return text


def _accept_barcode(popup, text, barcode):
    """Commit `text` if it is exactly BARCODE_LENGTH characters, otherwise
    reject it with a visible error. Quietly accepting a truncated or
    over-long code would put wrong data in the inspection log, which is
    far worse than making the operator rescan. Returns the (possibly
    updated) current barcode."""
    if len(text) != BARCODE_LENGTH:
        popup["error"] = f"Barcode must be {BARCODE_LENGTH} characters"
        return barcode
    # A failed part keeps its barcode. Requiring that same code for the
    # next attempt prevents a different part from silently replacing it
    # before the failure has been cleared.
    if barcode is not None and text != barcode:
        popup["error"] = f"Re-scan {barcode} to re-inspect"
        return barcode
    return _commit_barcode(popup, text)


def open_engineer_login(login, scanner):
    """Open a clean credential dialog and discard scanner input. Barcode
    events are not credentials, and retaining a scan made during login
    could trigger an inspection as soon as the dialog closes."""
    scanner.flush()
    login.update(active=True, field="username", username="",
                 password="", error="")


def handle_engineer_login_key(key, login):
    """Handle one credential-dialog key. Returns True only when the
    configured engineer credentials were accepted."""
    field = login["field"]
    if key == 27:  # ESC cancels without changing the current mode
        login.update(active=False, username="", password="", error="")
    elif key == ENGINEER_LOGIN_KEY:  # TAB moves between the two fields
        login["field"] = "password" if field == "username" else "username"
        login["error"] = ""
    elif key in (13, 10):  # ENTER advances or submits
        if field == "username":
            login["field"] = "password"
        elif (hmac.compare_digest(login["username"], ENGINEER_USERNAME)
              and hmac.compare_digest(login["password"], ENGINEER_PASSWORD)):
            login.update(active=False, username="", password="", error="")
            return True
        else:
            # Keep the error generic so the dialog does not reveal which
            # half of the credential pair was correct.
            login.update(field="password", password="",
                         error="Invalid username or password")
    elif key == 8:  # BACKSPACE
        login[field] = login[field][:-1]
        login["error"] = ""
    elif 32 <= key <= 126 and len(login[field]) < LOGIN_FIELD_MAX_LENGTH:
        login[field] += chr(key)
        login["error"] = ""
    return False


def disarm_roi():
    """Cancel reference drawing whenever privilege changes so a rectangle
    armed in Engineer mode cannot be released later in Operator mode."""
    ui.mouse.update(active_slot=None, drawing=False, roi_ready=False)


def arm_reference(slot, engineer_mode):
    """Arm a reference slot only for an authenticated engineer. Keeping
    this permission check beside the state mutation prevents UI changes
    from accidentally exposing calibration to Operator mode."""
    if not engineer_mode:
        return False
    ui.mouse.update(active_slot=slot, drawing=False, roi_ready=False)
    return True


def handle_pending_roi(engineer_mode, cam, rois, refs, thumbs,
                       sample_crops, live_results):
    """Complete an armed reference only while Engineer mode is still active.
    Authorization is checked again on mouse-up because login state can
    change between arming a slot and releasing the button."""
    if not ui.mouse["roi_ready"] or ui.mouse["active_slot"] is None:
        return False
    if not engineer_mode:
        disarm_roi()
        return False
    handle_completed_roi(cam, rois, refs, thumbs,
                         sample_crops, live_results)
    return True


def sync_thresholds(engineer_mode, noise_thresh, diff_thresh):
    """Read and persist sliders only for an authenticated engineer.
    Operator mode restores the saved positions before inspection, because
    OpenCV trackbars cannot be disabled and must never become an authority."""
    if not engineer_mode:
        diff_position = int(round(diff_thresh * 10))
        if cv2.getTrackbarPos(NOISE_TRACKBAR, WINDOW_NAME) != noise_thresh:
            cv2.setTrackbarPos(NOISE_TRACKBAR, WINDOW_NAME, noise_thresh)
        if cv2.getTrackbarPos(DIFF_TRACKBAR, WINDOW_NAME) != diff_position:
            cv2.setTrackbarPos(
                DIFF_TRACKBAR, WINDOW_NAME, diff_position)
        return noise_thresh, diff_thresh

    new_noise = cv2.getTrackbarPos(NOISE_TRACKBAR, WINDOW_NAME)
    new_diff = cv2.getTrackbarPos(DIFF_TRACKBAR, WINDOW_NAME) / 10.0
    if (new_noise, new_diff) != (noise_thresh, diff_thresh):
        storage.save_thresholds(new_noise, new_diff)
    return new_noise, new_diff


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
    noise_thresh, diff_thresh = storage.load_thresholds()
    setup_window(noise_thresh, diff_thresh)

    # Load anything saved by a previous session
    rois, refs, thumbs = storage.load_references()

    sample_crops = {}      # {slot: preprocessed crop} from the most recent inspection
    live_results = {}      # {slot: (passed, diff_val)} shown in the status bar
    current_barcode = None
    engineer_mode = False  # authentication is intentionally never persisted

    popup = {"active": False, "text": "", "error": "",
             "inspection_requested": False}
    login = {"active": False, "field": "username", "username": "",
             "password": "", "error": ""}

    # Falls back to manual typing if no scanner is present
    scanner = BarcodeScanner(device_path=SCANNER_DEVICE,
                             names=SCANNER_NAMES, grab=GRAB_SCANNER,
                             settle=SCANNER_SETTLE_SECONDS)

    # Operator mode is the safe startup default. If calibration has not
    # been completed, direct the user to an engineer instead of accepting
    # a barcode that cannot be inspected.
    active_refs = {s: r for s, r in refs.items() if s in rois}
    startup_error = "" if active_refs else "Engineer setup required - press TAB"
    open_popup(popup, scanner, error=startup_error)

    try:
        while True:
            # Trackbars remain visible in OpenCV, but only Engineer mode is
            # allowed to turn their positions into inspection settings.
            noise_thresh, diff_thresh = sync_thresholds(
                engineer_mode, noise_thresh, diff_thresh)

            # Take scanner input only while the popup is asking for a
            # barcode. Login suspends scanner consumption, and an Operator
            # cannot start an inspection until at least one reference exists.
            active_refs = {s: r for s, r in refs.items() if s in rois}
            if not login["active"] and active_refs:
                current_barcode = poll_scanner(
                    scanner, popup, current_barcode)

            # Check authorization again at capture time. This blocks a
            # rectangle armed before logout from replacing a reference.
            handle_pending_roi(
                engineer_mode, cam, rois, refs, thumbs,
                sample_crops, live_results)

            active_refs = {s: r for s, r in refs.items() if s in rois}
            if popup["inspection_requested"] and active_refs:
                # Consume this request before capture so one scan can never
                # produce duplicate log entries if later work is slow.
                popup["inspection_requested"] = False
                still, per_slot, overall_passed = run_inspection(
                    cam, active_refs, rois, sample_crops, live_results,
                    noise_thresh, diff_thresh)

                # Draw from the inspection still after live_results updates,
                # otherwise the saved image would show the previous verdict.
                display = ui.draw_overlay(still.copy(), rois, refs,
                                          live_results, thumbs,
                                          current_barcode,
                                          noise_thresh, diff_thresh,
                                          engineer_mode)
                storage.save_inspection(current_barcode, display,
                                        per_slot, overall_passed,
                                        noise_thresh, diff_thresh)
                ui.flash_result(display, overall_passed, per_slot)

                if overall_passed:
                    print(f"{current_barcode} passed. Scan the next barcode.")
                    current_barcode = None
                    if engineer_mode:
                        # Return to the unobstructed calibration view.
                        popup.update(active=False, text="", error="",
                                     inspection_requested=False)
                    else:
                        open_popup(popup, scanner)
                else:
                    # The same barcode is required before another attempt,
                    # so a failed part cannot be silently replaced.
                    open_popup(
                        popup, scanner,
                        error=f"FAIL - re-scan {current_barcode} to re-inspect")
                continue

            # Engineers get immediate feedback from the last sample while
            # calibrating; Operator values are fixed by sync_thresholds.
            for slot, ref_proc in refs.items():
                if slot in sample_crops:
                    live_results[slot] = compare(ref_proc, sample_crops[slot],
                                                 noise_thresh, diff_thresh)

            # Build and show the display frame. Overlay draws on a copy so
            # the clean frame stays available for inspection
            frame = capture_still(cam)
            display = ui.draw_overlay(frame.copy(), rois, refs, live_results,
                                      thumbs, current_barcode,
                                      noise_thresh, diff_thresh,
                                      engineer_mode)

            if popup["active"] and not login["active"]:
                # Show the live scan being assembled, or the typed text
                shown = scanner.snapshot() or popup["text"]
                display = ui.draw_barcode_popup(display, shown, popup["error"])

            if login["active"]:
                display = ui.draw_engineer_login(display, login)

            cv2.imshow(WINDOW_NAME, display)

            # waitKey(1) waits 1 ms for a keypress. The & 0xFF masks to
            # 8 bits, needed on some platforms for correct key codes
            key = cv2.waitKey(1) & 0xFF

            if login["active"]:
                authenticated = handle_engineer_login_key(key, login)
                if not login["active"]:
                    # Discard anything scanned while credentials were being
                    # entered before barcode polling resumes.
                    scanner.flush()
                if authenticated:
                    engineer_mode = True
                    disarm_roi()
                    popup.update(active=False, text="", error="",
                                 inspection_requested=False)
                    print("Production Engineer mode enabled")
                continue

            # TAB is checked before barcode typing because the barcode popup
            # is normally active throughout Operator mode.
            if not engineer_mode and key == ENGINEER_LOGIN_KEY:
                open_engineer_login(login, scanner)
                continue

            if engineer_mode and key in (
                    ord(ENGINEER_LOGOUT_KEY.lower()),
                    ord(ENGINEER_LOGOUT_KEY.upper())):
                engineer_mode = False
                disarm_roi()
                retry_error = (
                    f"Re-scan {current_barcode} to re-inspect"
                    if current_barcode is not None else
                    ("" if active_refs else
                     "Engineer setup required - press TAB")
                )
                open_popup(popup, scanner, error=retry_error)
                print("Operator mode enabled")
                continue

            if engineer_mode and not popup["active"] and key in (
                    ord(ENGINEER_SCAN_KEY.lower()),
                    ord(ENGINEER_SCAN_KEY.upper())):
                error = "" if active_refs else "Set a reference before testing"
                open_popup(popup, scanner, error=error)
                continue

            if engineer_mode and key in (ord('1'), ord('2')):
                # Reference keys take priority over a barcode popup so the
                # engineer can always enter calibration. Scanner input is
                # flushed because that popup is being intentionally closed.
                popup.update(active=False, text="", error="",
                             inspection_requested=False)
                scanner.flush()
                arm_reference(int(chr(key)), engineer_mode)
                continue

            if popup["active"]:
                if active_refs:
                    current_barcode = handle_popup_key(
                        key, popup, scanner, current_barcode)
                else:
                    popup["error"] = "Engineer setup required - press TAB"
                continue  # don't fall through to normal key handling

            if key == ord('q'):
                break

    finally:
        # Always release the camera, scanner, and windows, even on exception
        scanner.close()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
