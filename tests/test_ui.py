"""
test_ui.py

Tests for ui.py. The mouse state machine is tested precisely; the
drawing functions are tested for shape, side effects, and measurable
pixel changes, since pixel-perfect assertions on rendered text are
brittle across OpenCV versions."""

import cv2
import numpy as np

import config
import ui


# Mouse state machine

def test_mouse_ignored_when_no_active_slot():
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, 5, 6, 0, None)
    assert not ui.mouse["drawing"]
    assert not ui.mouse["roi_ready"]


def test_full_drag_sequence_sets_roi_ready():
    ui.mouse["active_slot"] = 1
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, 5, 6, 0, None)
    assert ui.mouse["drawing"] and ui.mouse["pt1"] == (5, 6)
    ui.on_mouse(cv2.EVENT_MOUSEMOVE, 50, 60, 0, None)
    assert ui.mouse["pt2"] == (50, 60)
    ui.on_mouse(cv2.EVENT_LBUTTONUP, 55, 66, 0, None)
    assert ui.mouse["roi_ready"]
    assert not ui.mouse["drawing"]
    assert ui.mouse["pt2"] == (55, 66)


def test_move_without_button_down_is_ignored():
    ui.mouse["active_slot"] = 1
    ui.on_mouse(cv2.EVENT_MOUSEMOVE, 50, 60, 0, None)
    assert ui.mouse["pt2"] == (0, 0)


def test_button_up_without_drag_is_ignored():
    # A stray mouse-up (button pressed before the slot was armed) must
    # not trigger a reference capture
    ui.mouse["active_slot"] = 1
    ui.on_mouse(cv2.EVENT_LBUTTONUP, 50, 60, 0, None)
    assert not ui.mouse["roi_ready"]


def test_new_press_resets_roi_ready():
    ui.mouse["active_slot"] = 1
    ui.mouse["roi_ready"] = True
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, 1, 2, 0, None)
    assert not ui.mouse["roi_ready"]
    assert ui.mouse["pt1"] == (1, 2) and ui.mouse["pt2"] == (1, 2)


# Panel helper

def test_dark_panel_darkens_region(frame):
    frame[:] = 200
    ui._dark_panel(frame, 0, 0, 100, 50)
    assert frame[25, 50].mean() < 200   # inside the panel: darkened
    assert frame[200, 200].mean() == 200  # outside: untouched


# Overlay rendering

def _overlay(frame, **overrides):
    kwargs = dict(rois={}, refs={}, live_results={}, thumbs={},
                  barcode=None, noise_thresh=30, diff_thresh=5.0)
    kwargs.update(overrides)
    return ui.draw_overlay(frame, **kwargs)


def test_draw_overlay_preserves_shape(frame):
    assert _overlay(frame.copy()).shape == frame.shape


def test_draw_overlay_modifies_frame(frame):
    out = _overlay(frame.copy(), barcode="ABC123")
    assert not np.array_equal(out, frame)


def test_draw_overlay_full_state(frame):
    # Exercise every branch at once: ROIs, references, results, thumbs
    thumb = np.zeros((config.THUMB_H, config.THUMB_W, 3), dtype=np.uint8)
    out = _overlay(frame.copy(),
                   rois={1: (100, 100, 300, 300), 2: (400, 100, 600, 300)},
                   refs={1: np.zeros((200, 200), dtype=np.uint8)},
                   live_results={1: (True, 2.0), 2: (False, 9.9)},
                   thumbs={1: thumb},
                   barcode="ABC123")
    assert out.shape == frame.shape


def test_engineer_mode_overlay_differs_from_operator(frame):
    operator = _overlay(frame.copy(), engineer_mode=False)
    engineer = _overlay(frame.copy(), engineer_mode=True)
    assert not np.array_equal(operator, engineer)


def test_threshold_values_are_hidden_from_operator(frame, monkeypatch):
    rendered = []
    original = cv2.putText

    def capture_text(image, text, *args, **kwargs):
        rendered.append(text)
        return original(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_text)
    _overlay(frame.copy(), engineer_mode=False,
             noise_thresh=37, diff_thresh=6.4)
    assert not any(text.startswith("Noise ") for text in rendered)

    rendered.clear()
    _overlay(frame.copy(), engineer_mode=True,
             noise_thresh=37, diff_thresh=6.4)
    assert "Noise 37   Threshold 6.4" in rendered


def test_draw_overlay_shows_rubber_band_while_drawing(frame):
    ui.mouse.update(active_slot=1, drawing=True, pt1=(100, 100), pt2=(300, 300))
    with_band = _overlay(frame.copy())
    ui.mouse.update(active_slot=None, drawing=False)
    without_band = _overlay(frame.copy())
    assert not np.array_equal(with_band, without_band)


# Barcode popup

def test_popup_dims_background(frame):
    frame[:] = 200
    out = ui.draw_barcode_popup(frame.copy(), "AB12", "")
    assert out[0, 0].mean() < 200  # corner pixel outside the dialog is dimmed


def test_popup_with_error_differs_from_without(frame):
    frame[:] = 200
    clean = ui.draw_barcode_popup(frame.copy(), "AB12", "")
    with_error = ui.draw_barcode_popup(frame.copy(), "AB12", "Barcode cannot be empty")
    assert not np.array_equal(clean, with_error)


# Production Engineer login

def test_engineer_login_dims_background(frame):
    frame[:] = 200
    login = {"field": "username", "username": "", "password": "",
             "error": ""}
    out = ui.draw_engineer_login(frame.copy(), login)
    assert out[0, 0].mean() < 200


def test_engineer_login_masks_password(frame, monkeypatch):
    rendered = []
    original = cv2.putText

    def capture_text(image, text, *args, **kwargs):
        rendered.append(text)
        return original(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_text)
    login = {"field": "password", "username": "prod",
             "password": "secret", "error": "Invalid username or password"}
    ui.draw_engineer_login(frame.copy(), login)

    assert "secret" not in rendered
    assert "*" * len("secret") + "|" in rendered


# Result banner

def test_flash_result_renders_banner(frame, monkeypatch):
    # imshow/waitKey need a display; capture the banner instead
    shown = {}
    monkeypatch.setattr(cv2, "imshow", lambda name, img: shown.update(img=img))
    monkeypatch.setattr(cv2, "waitKey", lambda ms: shown.update(waited=ms))

    ui.flash_result(frame.copy(), True, {1: (True, 2.0), 2: (True, 1.5)})
    assert shown["img"].shape == frame.shape
    assert not np.array_equal(shown["img"], frame)  # banner actually drawn
    assert shown["waited"] == 1800  # the 1.8 s freeze is part of the contract


def test_flash_result_does_not_mutate_input(frame, monkeypatch):
    monkeypatch.setattr(cv2, "imshow", lambda name, img: None)
    monkeypatch.setattr(cv2, "waitKey", lambda ms: None)
    original = frame.copy()
    ui.flash_result(frame, False, {1: (False, 9.9)})
    np.testing.assert_array_equal(frame, original)
