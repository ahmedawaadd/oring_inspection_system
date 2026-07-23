"""
ui.py

All OpenCV drawing and mouse input: overlay, popup, result banner."""

import cv2
import numpy as np

from config import (GRAY, GREEN, RED, SLOT_COLORS, THUMB_H, THUMB_W,
                    WHITE, WINDOW_NAME, YELLOW)

# OpenCV's mouse callback can't return values to the main loop directly.
# This shared dictionary is the workaround: both the callback and the
# main loop read and write the same object, so changes in one are
# immediately visible in the other.
mouse = {
    "active_slot": None,  # which slot (1 or 2) the user is drawing for right now
    "drawing": False,     # True while the left mouse button is held down
    "pt1": (0, 0),
    "pt2": (0, 0),
    "roi_ready": False,   # flipped to True on mouse-up so the main loop knows to act
}


def on_mouse(event, x, y, flags, param):
    """Track a click-and-drag rectangle for the active slot."""
    # Ignore mouse events if no slot is active (user hasn't pressed 1 or 2)
    if mouse["active_slot"] is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse.update(drawing=True, roi_ready=False, pt1=(x, y), pt2=(x, y))
    elif event == cv2.EVENT_MOUSEMOVE and mouse["drawing"]:
        mouse["pt2"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and mouse["drawing"]:
        # Signal the main loop on mouse release to capture the reference
        mouse.update(drawing=False, pt2=(x, y), roi_ready=True)


def _dark_panel(frame, x1, y1, x2, y2, alpha=0.82):
    """Darken a rectangular region of the frame to create a UI bar.
    alpha 0.82 means 82% dark colour, 18% original image."""
    region = frame[y1:y2, x1:x2]
    dark = np.full_like(region, (20, 20, 20))
    cv2.addWeighted(dark, alpha, region, 1.0 - alpha, 0, region)
    frame[y1:y2, x1:x2] = region


def draw_overlay(frame, rois, refs, live_results, thumbs, barcode,
                 noise_thresh, diff_thresh):
    """Draw ROI boxes, the top info bar, and the bottom status bar."""
    h, w = frame.shape[:2]

    # ROI box for each saved region so the operator can see where the
    # system is looking
    for slot, roi in rois.items():
        color = SLOT_COLORS[slot]
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Ref {slot}", (x1 + 4, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)

    # Live rubber-band box while the user is actively drawing
    if mouse["active_slot"] is not None and (mouse["drawing"] or not mouse["roi_ready"]):
        cv2.rectangle(frame, mouse["pt1"], mouse["pt2"],
                      SLOT_COLORS[mouse["active_slot"]], 2)

    # Top bar (50 px): barcode on the left, thresholds on the right
    _dark_panel(frame, 0, 0, w, 50)

    bc_text = f"BARCODE  #{barcode}" if barcode is not None else "BARCODE:  (scan part)"
    cv2.putText(frame, bc_text, (16, 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, YELLOW, 2, cv2.LINE_AA)

    # Measure the threshold text width first so it can be right-aligned
    thr_text = f"Noise {noise_thresh}   Threshold {diff_thresh:.1f}"
    (tw, _), _ = cv2.getTextSize(thr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.putText(frame, thr_text, (w - tw - 16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, GRAY, 1, cv2.LINE_AA)

    # Bottom bar (130 px): controls hint plus one panel per slot
    bar_y = h - 130
    _dark_panel(frame, 0, bar_y, w, h)

    # Drawing hint while defining an ROI, otherwise the keyboard controls
    if mouse["active_slot"] is not None:
        hint = f"Drawing Ref {mouse['active_slot']}   click and drag, release to confirm"
        cv2.putText(frame, hint, (16, bar_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, YELLOW, 1, cv2.LINE_AA)
    else:
        ctrl = "1 / 2: Draw reference     SCAN: Inspect     Q: Quit"
        cv2.putText(frame, ctrl, (16, bar_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, GRAY, 1, cv2.LINE_AA)

    # Slot panels: thumbnail on the left, status on the right
    for idx, slot in enumerate([1, 2]):
        sc = SLOT_COLORS[slot]
        px = 16 + idx * 380  # space the two panels across the bar
        ty = bar_y + 38

        # Reference thumbnail, or a grey placeholder if none is saved yet
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
            status_txt, s_color = "READY", GREEN     # good to go
        elif has_roi:
            status_txt, s_color = "ROI SET", YELLOW  # region drawn but no reference yet
        else:
            status_txt, s_color = "NO ROI", GRAY     # not set up at all

        cv2.putText(frame, f"REF {slot}", (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, sc, 1, cv2.LINE_AA)
        cv2.putText(frame, status_txt, (lx, ly + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, s_color, 1, cv2.LINE_AA)

        # Last inspection result for this slot if one exists
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
    """Draw the in-window barcode entry dialog over the live frame.
    Keyboard input is handled in the main loop; this only renders
    whatever text has been typed so far."""
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

    cv2.putText(frame, "Scan or type the barcode - submits automatically",
                (dx + 16, dy + 158),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, GRAY, 1, cv2.LINE_AA)

    # Only shown if the operator typed something invalid and pressed ENTER
    if error:
        cv2.putText(frame, error, (dx + 16, dy + 192),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, RED, 1, cv2.LINE_AA)

    return frame


def flash_result(frame, passed, per_slot):
    """Show a full-screen PASS/FAIL banner for 1.8 seconds."""
    # Work on a copy so the display frame isn't permanently drawn over
    h, w = frame.shape[:2]
    banner = frame.copy()
    color = GREEN if passed else RED
    label = "PASS" if passed else "FAIL"

    cv2.rectangle(banner, (0, 0), (w, h), color, 30)

    # Measure the text width so it can be centred precisely
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 6.0
    thick = 14
    (lw, _), _ = cv2.getTextSize(label, font, scale, thick)
    cv2.putText(banner, label, ((w - lw) // 2, h // 2 - 20),
                font, scale, color, thick, cv2.LINE_AA)

    # Per-slot breakdown below the main verdict, also centred
    y = h // 2 + 72
    for slot, (slot_passed, diff_val) in sorted(per_slot.items()):
        sc = GREEN if slot_passed else RED
        txt = f"Ref {slot}:  {'PASS' if slot_passed else 'FAIL'}   diff {diff_val:.1f}"
        (sw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)
        cv2.putText(banner, txt, ((w - sw) // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, sc, 2, cv2.LINE_AA)
        y += 48

    cv2.imshow(WINDOW_NAME, banner)
    cv2.waitKey(1800)
