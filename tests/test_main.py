"""
test_main.py

Tests for main.py helpers plus an end-to-end workflow test that
drives ROI capture, inspection, and logging against a fake camera."""

import os
import queue

import cv2
import numpy as np
import pytest

import config
import main
import storage
import ui
from conftest import FakeCamera


class NoBufferScanner:
    """Scanner double for popup tests: an empty hardware buffer, so only
    typed input is in play."""

    def take_buffer(self):
        return ""


class BufferedScanner:
    """Scanner double holding a partial scan, for the manual-ENTER path."""

    def __init__(self, buffered):
        self._buffered = buffered

    def take_buffer(self):
        buf, self._buffered = self._buffered, ""
        return buf


class PollScanner:
    """Scanner double for poll_scanner: a results queue for completed scans
    plus a snapshot/take_buffer pair for an un-terminated partial scan.
    settled defaults to True; pass False to simulate a scan mid-burst."""

    def __init__(self, completed=(), partial="", settled=True):
        self.results = queue.Queue()
        for code in completed:
            self.results.put(code)
        self._partial = partial
        self._settled = settled

    def snapshot(self):
        return self._partial

    def settled(self):
        return self._settled

    def take_buffer(self):
        buf, self._partial = self._partial, ""
        return buf

    def flush(self):
        self._partial = ""
        while not self.results.empty():
            self.results.get_nowait()


@pytest.fixture
def popup():
    return {"active": True, "text": "", "error": ""}


@pytest.fixture
def login():
    return {"active": True, "field": "username", "username": "",
            "password": "", "error": ""}


ENTER, ESC, BACKSPACE = 13, 27, 8


# Popup key handling

def test_typing_appends_characters(popup):
    s = NoBufferScanner()
    bc = main.handle_popup_key(ord('A'), popup, s, None)
    bc = main.handle_popup_key(ord('7'), popup, s, bc)
    assert popup["text"] == "A7"
    assert bc is None  # nothing committed yet


def test_typing_auto_commits_at_barcode_length(popup):
    # A full-length barcode submits and requests inspection on its own
    s = NoBufferScanner()
    bc = None
    for ch in "ABC1234":  # BARCODE_LENGTH == 7 characters
        bc = main.handle_popup_key(ord(ch), popup, s, bc)
    assert bc == "ABC1234"
    assert not popup["active"]
    assert popup["text"] == ""
    assert popup["inspection_requested"]


def test_typing_does_not_commit_before_barcode_length(popup):
    s = NoBufferScanner()
    bc = None
    for ch in "ABC12":  # one short of the length
        bc = main.handle_popup_key(ord(ch), popup, s, bc)
    assert bc is None
    assert popup["active"]
    assert popup["text"] == "ABC12"


def test_typing_cannot_exceed_barcode_length(popup):
    # The length-limited field never accepts more than a full barcode
    s = NoBufferScanner()
    for ch in "ABCDEFG":  # exactly the length, commits and clears
        main.handle_popup_key(ord(ch), popup, s, None)
    assert config.BARCODE_LENGTH == 7
    assert not popup["active"]


def test_non_alphanumeric_keys_ignored(popup):
    s = NoBufferScanner()
    for k in (ord('-'), ord(' '), ord('/'), 255):  # 255 = no key pressed
        main.handle_popup_key(k, popup, s, None)
    assert popup["text"] == ""


def test_backspace_removes_last_char_and_clears_error(popup):
    s = NoBufferScanner()
    main.handle_popup_key(ord('A'), popup, s, None)
    main.handle_popup_key(ord('B'), popup, s, None)
    popup["error"] = "stale"
    main.handle_popup_key(BACKSPACE, popup, s, None)
    assert popup["text"] == "A"
    assert popup["error"] == ""


def test_enter_rejects_short_typed_text(popup):
    # A truncated code in the log is worse than a rejected one: ENTER on
    # anything but a full-length barcode must refuse, and keep the typed
    # text so the operator can finish it
    s = NoBufferScanner()
    for ch in "ABC":
        main.handle_popup_key(ord(ch), popup, s, None)
    bc = main.handle_popup_key(ENTER, popup, s, None)
    assert bc is None
    assert popup["active"]
    assert popup["error"] == "Barcode must be 7 characters"
    assert popup["text"] == "ABC"


def test_numpad_enter_also_commits(popup):
    bc = main.handle_popup_key(10, popup, BufferedScanner("SCAN999"), None)
    assert bc == "SCAN999"
    assert not popup["active"]


def test_enter_falls_back_to_scanner_buffer(popup):
    # Covers scanners not configured to append their own ENTER: the
    # operator scans, then presses ENTER manually
    bc = main.handle_popup_key(ENTER, popup, BufferedScanner("SCAN999"), None)
    assert bc == "SCAN999"
    assert not popup["active"]


def test_enter_on_empty_shows_error_and_stays_open(popup):
    bc = main.handle_popup_key(ENTER, popup, NoBufferScanner(), None)
    assert bc is None
    assert popup["active"]
    assert popup["error"] == "Barcode cannot be empty"


def test_esc_blocked_until_barcode_set(popup):
    # The operator must not be able to dismiss the popup and inspect
    # without a barcode
    main.handle_popup_key(ESC, popup, NoBufferScanner(), None)
    assert popup["active"]


def test_esc_closes_when_barcode_already_set(popup):
    bc = main.handle_popup_key(ESC, popup, NoBufferScanner(), "OLD1")
    assert bc == "OLD1"  # existing barcode kept
    assert not popup["active"]


# Scanner polling (only accepted while the popup is open)

def test_poll_scanner_commits_completed_scan(popup):
    # A completed scan closes the popup and becomes an inspection request
    s = PollScanner(completed=["ABC1234"])
    bc = main.poll_scanner(s, popup, None)
    assert bc == "ABC1234"
    assert not popup["active"]
    assert popup["inspection_requested"]


def test_poll_scanner_auto_commits_partial_at_length(popup):
    # A scanner with no ENTER terminator: the buffer reached the barcode
    # length, so it commits without waiting
    s = PollScanner(partial="ABC1234")
    bc = main.poll_scanner(s, popup, None)
    assert bc == "ABC1234"
    assert not popup["active"]
    assert s.snapshot() == ""  # buffer was consumed


def test_poll_scanner_ignores_partial_below_length(popup):
    s = PollScanner(partial="ABC12")
    bc = main.poll_scanner(s, popup, None)
    assert bc is None
    assert popup["active"]


def test_poll_scanner_ignored_while_popup_closed():
    # Between the accepted barcode and the next pass the popup is closed;
    # scans must not change the current barcode
    popup = {"active": False, "text": "", "error": ""}
    s = PollScanner(completed=["NEW5678"], partial="XYZ9999")
    bc = main.poll_scanner(s, popup, "OLD1234")
    assert bc == "OLD1234"
    assert not popup["active"]


def test_poll_scanner_waits_for_scan_to_settle(popup):
    # A full-length buffer that is still growing is a scan in flight;
    # taking it now would split one long scan into two barcodes
    s = PollScanner(partial="ABC1234", settled=False)
    bc = main.poll_scanner(s, popup, None)
    assert bc is None
    assert popup["active"]
    assert s.snapshot() == "ABC1234"  # left intact for the next poll


def test_poll_scanner_rejects_overlength_buffer(popup):
    # A settled buffer longer than BARCODE_LENGTH is one scan of the
    # wrong kind of code, not a barcode plus spare characters
    s = PollScanner(partial="ABCDEFGHIJ")
    bc = main.poll_scanner(s, popup, None)
    assert bc is None
    assert popup["active"]
    assert popup["error"] == "Barcode must be 7 characters"


def test_poll_scanner_rejects_wrong_length_completed_scan(popup):
    # Regression: the tail of a split scan ("HIJ" of "ABCDEFGHIJ") used
    # to be committed as the next part's barcode
    s = PollScanner(completed=["HIJ"])
    bc = main.poll_scanner(s, popup, None)
    assert bc is None
    assert popup["active"]
    assert popup["error"] == "Barcode must be 7 characters"


def test_poll_scanner_rejects_different_barcode_after_failure(popup):
    # A failed part owns the station until its same barcode is rescanned;
    # otherwise a new part could silently advance past an unresolved fail.
    s = PollScanner(completed=["NEW5678"])
    bc = main.poll_scanner(s, popup, "OLD1234")
    assert bc == "OLD1234"
    assert popup["active"]
    assert not popup.get("inspection_requested", False)
    assert popup["error"] == "Re-scan OLD1234 to re-inspect"


def test_poll_scanner_accepts_same_barcode_for_reinspection(popup):
    s = PollScanner(completed=["OLD1234"])
    bc = main.poll_scanner(s, popup, "OLD1234")
    assert bc == "OLD1234"
    assert not popup["active"]
    assert popup["inspection_requested"]


# Popup opening (must discard scanner input collected while closed)

def test_open_popup_discards_scans_made_while_closed():
    # Regression: a scan made while no popup was open used to sit in the
    # queue and get committed, sight unseen, the moment the popup reopened
    s = PollScanner(completed=["STRAY77"], partial="ABC1234")
    popup = {"active": False, "text": "", "error": ""}

    main.open_popup(popup, s)
    bc = main.poll_scanner(s, popup, None)

    assert bc is None
    assert popup["active"]  # still waiting for a real scan


def test_open_popup_resets_text_and_sets_error():
    popup = {"active": False, "text": "stale", "error": "",
             "inspection_requested": True}
    main.open_popup(popup, PollScanner(), error="Set a barcode before inspecting")
    assert popup == {"active": True, "text": "",
                     "error": "Set a barcode before inspecting",
                     "inspection_requested": False}


# Production Engineer authentication and permissions

def test_open_engineer_login_flushes_scanner_and_resets_fields():
    scanner = PollScanner(completed=["STRAY77"], partial="ABC1234")
    login = {"active": False, "field": "password", "username": "old",
             "password": "secret", "error": "stale"}

    main.open_engineer_login(login, scanner)

    assert login == {"active": True, "field": "username", "username": "",
                     "password": "", "error": ""}
    assert scanner.snapshot() == ""
    assert scanner.results.empty()


def test_engineer_login_shortcut_is_separate_from_tab_navigation():
    assert main.is_engineer_login_key(ord('/'))
    assert not main.is_engineer_login_key(config.LOGIN_FIELD_SWITCH_KEY)


def test_engineer_login_accepts_configured_credentials(login, monkeypatch):
    monkeypatch.setattr(main, "ENGINEER_USERNAME", "prod")
    monkeypatch.setattr(main, "ENGINEER_PASSWORD", "safe-pass")

    for ch in "prod":
        assert not main.handle_engineer_login_key(ord(ch), login)
    assert not main.handle_engineer_login_key(ENTER, login)
    for ch in "safe-pass":
        assert not main.handle_engineer_login_key(ord(ch), login)

    assert main.handle_engineer_login_key(ENTER, login)
    assert not login["active"]
    assert login["username"] == ""
    assert login["password"] == ""


def test_engineer_login_rejects_bad_credentials(login, monkeypatch):
    monkeypatch.setattr(main, "ENGINEER_USERNAME", "prod")
    monkeypatch.setattr(main, "ENGINEER_PASSWORD", "safe-pass")
    login.update(field="password", username="prod", password="wrong")

    assert not main.handle_engineer_login_key(ENTER, login)
    assert login["active"]
    assert login["field"] == "password"
    assert login["password"] == ""
    assert login["error"] == "Invalid username or password"


def test_engineer_login_tab_backspace_and_escape(login):
    main.handle_engineer_login_key(ord('A'), login)
    main.handle_engineer_login_key(BACKSPACE, login)
    assert login["username"] == ""

    main.handle_engineer_login_key(config.LOGIN_FIELD_SWITCH_KEY, login)
    assert login["field"] == "password"
    main.handle_engineer_login_key(ESC, login)
    assert not login["active"]


def test_engineer_login_limits_field_length(login):
    for _ in range(config.LOGIN_FIELD_MAX_LENGTH + 5):
        main.handle_engineer_login_key(ord('A'), login)
    assert len(login["username"]) == config.LOGIN_FIELD_MAX_LENGTH


def test_reference_arming_requires_engineer_mode():
    assert not main.arm_reference(1, engineer_mode=False)
    assert ui.mouse["active_slot"] is None

    assert main.arm_reference(2, engineer_mode=True)
    assert ui.mouse["active_slot"] == 2


def test_disarm_roi_clears_privileged_mouse_state():
    ui.mouse.update(active_slot=1, drawing=True, roi_ready=True)
    main.disarm_roi()
    assert ui.mouse["active_slot"] is None
    assert not ui.mouse["drawing"]
    assert not ui.mouse["roi_ready"]


def test_pending_reference_is_ignored_without_mouse_release():
    assert not main.handle_pending_roi(
        True, FakeCamera([]), {}, {}, {}, {}, {})


def test_operator_cannot_complete_prearmed_reference(workdir):
    ui.mouse.update(active_slot=1, drawing=False, roi_ready=True,
                    pt1=(100, 100), pt2=(300, 300))

    completed = main.handle_pending_roi(
        False, FakeCamera([]), {}, {}, {}, {}, {})

    assert not completed
    assert ui.mouse["active_slot"] is None
    assert not os.path.exists(config.REFERENCE_PATHS[0])


def test_engineer_can_complete_pending_reference(workdir, rng):
    still = rng.integers(0, 255, (960, 1280, 3), dtype=np.uint8)
    rois, refs, thumbs = {}, {}, {}
    ui.mouse.update(active_slot=1, drawing=False, roi_ready=True,
                    pt1=(100, 100), pt2=(300, 300))

    completed = main.handle_pending_roi(
        True, FakeCamera([still]), rois, refs, thumbs, {}, {})

    assert completed
    assert rois[1] == (100, 100, 300, 300)


def test_operator_threshold_sync_has_no_calibration_controls(monkeypatch):
    monkeypatch.setattr(
        cv2, "getTrackbarPos",
        lambda *args: pytest.fail("Operator mode must not read hidden sliders"))
    monkeypatch.setattr(
        cv2, "setTrackbarPos",
        lambda *args: pytest.fail("Operator mode must not expose sliders"))

    thresholds = main.sync_thresholds(False, 31, 4.2)

    assert thresholds == (31, 4.2)


def test_engineer_threshold_sync_persists_changes(monkeypatch):
    positions = {
        config.NOISE_TRACKBAR: 37,
        config.DIFF_TRACKBAR: 64,
    }
    saved = []
    windows = []

    def get_position(name, window):
        windows.append(window)
        return positions[name]

    monkeypatch.setattr(cv2, "getTrackbarPos", get_position)
    monkeypatch.setattr(
        storage, "save_thresholds",
        lambda noise, diff: saved.append((noise, diff)))

    thresholds = main.sync_thresholds(True, 30, 5.0)

    assert thresholds == (37, 6.4)
    assert saved == [(37, 6.4)]
    assert windows == [config.CALIBRATION_WINDOW_NAME] * 2


def test_unchanged_engineer_thresholds_are_not_rewritten(monkeypatch):
    positions = {
        config.NOISE_TRACKBAR: 30,
        config.DIFF_TRACKBAR: 50,
    }
    monkeypatch.setattr(
        cv2, "getTrackbarPos", lambda name, window: positions[name])
    monkeypatch.setattr(
        storage, "save_thresholds",
        lambda *args: pytest.fail("Unchanged settings should not be saved"))

    assert main.sync_thresholds(True, 30, 5.0) == (30, 5.0)


# ROI capture

def _drag(slot, pt1, pt2):
    ui.mouse.update(active_slot=slot, drawing=False, roi_ready=True,
                    pt1=pt1, pt2=pt2)


def test_completed_roi_saves_reference(workdir, rng):
    still = rng.integers(0, 255, (960, 1280, 3), dtype=np.uint8)
    cam = FakeCamera([still])
    rois, refs, thumbs = {}, {}, {}
    _drag(1, (100, 100), (300, 300))

    main.handle_completed_roi(cam, rois, refs, thumbs, {}, {})

    assert rois[1] == (100, 100, 300, 300)
    assert refs[1].shape == (200, 200)  # preprocessed crop of the drag area
    assert thumbs[1].shape == (config.THUMB_H, config.THUMB_W, 3)
    assert os.path.exists(config.REFERENCE_PATHS[0])
    assert os.path.exists(config.ROI_PATHS[0])
    # Mouse state must be fully reset for the next interaction
    assert ui.mouse["active_slot"] is None
    assert not ui.mouse["roi_ready"]


def test_completed_roi_clears_stale_results(workdir, rng):
    # Redrawing a reference invalidates the previous inspection for
    # that slot; keeping it would show a verdict against the old reference
    still = rng.integers(0, 255, (960, 1280, 3), dtype=np.uint8)
    sample_crops = {1: np.zeros((10, 10), dtype=np.uint8)}
    live_results = {1: (True, 0.0)}
    _drag(1, (100, 100), (300, 300))

    main.handle_completed_roi(FakeCamera([still]), {}, {}, {},
                              sample_crops, live_results)

    assert 1 not in sample_crops
    assert 1 not in live_results


def test_tiny_drag_rejected_but_state_reset(workdir):
    # Accidental clicks under 10 px must not overwrite a reference,
    # and the mouse must still disarm
    cam = FakeCamera([])  # capture would raise StopIteration if reached
    rois = {}
    _drag(1, (100, 100), (105, 300))

    main.handle_completed_roi(cam, rois, {}, {}, {}, {})

    assert rois == {}
    assert not os.path.exists(config.REFERENCE_PATHS[0])
    assert ui.mouse["active_slot"] is None


# Inspection

def test_run_inspection_pass_and_fail(workdir, rng):
    still = rng.integers(0, 255, (960, 1280, 3), dtype=np.uint8)
    rois = {1: (100, 100, 300, 300), 2: (400, 100, 600, 300)}
    from vision import crop, preprocess
    matching_ref = preprocess(crop(still, rois[1]))       # slot 1: identical
    wrong_ref = 255 - preprocess(crop(still, rois[2]))    # slot 2: inverted
    sample_crops, live_results = {}, {}

    _, per_slot, overall = main.run_inspection(
        FakeCamera([still]), {1: matching_ref, 2: wrong_ref}, rois,
        sample_crops, live_results, noise_thresh=30, diff_thresh=5.0)

    assert per_slot[1][0] is True
    assert per_slot[2][0] is False
    assert overall is False  # overall verdict is AND of all slots
    assert set(sample_crops) == {1, 2}  # kept for live slider recomputation
    assert live_results == per_slot


def test_run_inspection_all_pass(workdir, rng):
    still = rng.integers(0, 255, (960, 1280, 3), dtype=np.uint8)
    rois = {1: (100, 100, 300, 300)}
    from vision import crop, preprocess
    ref = preprocess(crop(still, rois[1]))

    _, per_slot, overall = main.run_inspection(
        FakeCamera([still]), {1: ref}, rois, {}, {}, 30, 5.0)

    assert overall is True
    assert per_slot[1][1] == 0.0


# End-to-end processing (no GUI): draw ROI, inspect, verify the log

def test_full_workflow_reference_then_inspection(workdir, rng):
    part_a = rng.integers(0, 255, (960, 1280, 3), dtype=np.uint8)
    part_b = rng.integers(0, 255, (960, 1280, 3), dtype=np.uint8)

    # Session 1: operator draws a reference around part A
    rois, refs, thumbs = storage.load_references()
    assert rois == {}
    _drag(1, (100, 100), (300, 300))
    main.handle_completed_roi(FakeCamera([part_a]), rois, refs, thumbs, {}, {})

    # Session 2: state reloads from disk, as on app restart
    rois2, refs2, _ = storage.load_references()
    assert rois2 == rois

    # Inspect part A again (pass), then part B (fail)
    for still, expect_pass in [(part_a, True), (part_b, False)]:
        _, per_slot, overall = main.run_inspection(
            FakeCamera([still]), refs2, rois2, {}, {}, 30, 5.0)
        assert overall is expect_pass
        storage.save_inspection("LOT42", cv2.cvtColor(still, cv2.COLOR_RGB2BGR),
                                per_slot, overall, 30, 5.0)

    # Both inspections landed in the barcode's log
    log = os.path.join(config.LOGS_DIR, "LOT42", "log.csv")
    with open(log) as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 3
    assert ",PASS," in lines[1]
    assert ",FAIL," in lines[2]
