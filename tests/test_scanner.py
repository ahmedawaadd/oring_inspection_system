"""
test_scanner.py

Tests for scanner.py. The evdev protocol is faked with small doubles
so key assembly, shift handling, and name lookup are tested without
hardware or real threads."""

import queue
import sys
import threading
import types
from collections import namedtuple

import scanner

FakeEvent = namedtuple("FakeEvent", "type code value")


class FakeEcodes:
    """Minimal evdev.ecodes double covering every code the app uses."""
    EV_KEY = 1
    KEY_ENTER = 28
    KEY_KPENTER = 96
    KEY_LEFTSHIFT = 42
    KEY_RIGHTSHIFT = 54
    KEY_MINUS = 12
    KEY_DOT = 52


for _i, _c in enumerate("0123456789"):
    setattr(FakeEcodes, f"KEY_{_c}", 100 + _i)
for _i, _c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    setattr(FakeEcodes, f"KEY_{_c}", 200 + _i)


class FakeDevice:
    """InputDevice double: replays a fixed event list, tracks lifecycle."""

    def __init__(self, path="/dev/input/event0", name="Test Scanner",
                 events=(), keys=None):
        self.path = path
        self.name = name
        self._events = list(events)
        self._keys = keys if keys is not None else [FakeEcodes.KEY_ENTER,
                                                    FakeEcodes.KEY_A]
        self.grabbed = False
        self.closed = False

    def capabilities(self):
        return {FakeEcodes.EV_KEY: self._keys}

    def read_loop(self):
        return iter(self._events)

    def grab(self):
        self.grabbed = True

    def ungrab(self):
        self.grabbed = False

    def close(self):
        self.closed = True


def key(code, value=1):
    return FakeEvent(FakeEcodes.EV_KEY, code, value)


def run_scanner(events):
    """Build a scanner around a fake device and run _run synchronously.
    Bypasses __init__ so no real evdev or thread is involved."""
    s = scanner.BarcodeScanner.__new__(scanner.BarcodeScanner)
    s.results = queue.Queue()
    s._buffer = ""
    s._lock = threading.Lock()
    s._stop = threading.Event()
    s._ecodes = FakeEcodes
    s._keymap = scanner._build_keymap(FakeEcodes)
    s.device = FakeDevice(events=events)
    s._run()
    return s


# Keymap

def test_keymap_covers_digits_letters_and_symbols():
    m = scanner._build_keymap(FakeEcodes)
    assert m[FakeEcodes.KEY_7] == "7"
    assert m[FakeEcodes.KEY_Q] == "q"
    assert m[FakeEcodes.KEY_MINUS] == "-"
    assert m[FakeEcodes.KEY_DOT] == "."
    assert len(m) == 38  # 10 digits + 26 letters + 2 symbols


# Event assembly (_run)

def test_enter_flushes_assembled_barcode():
    s = run_scanner([key(FakeEcodes.KEY_A), key(FakeEcodes.KEY_B),
                     key(FakeEcodes.KEY_1), key(FakeEcodes.KEY_ENTER)])
    assert s.results.get_nowait() == "ab1"


def test_kp_enter_also_terminates():
    s = run_scanner([key(FakeEcodes.KEY_9), key(FakeEcodes.KEY_KPENTER)])
    assert s.results.get_nowait() == "9"


def test_shift_uppercases_letters():
    s = run_scanner([
        key(FakeEcodes.KEY_LEFTSHIFT, 1),   # shift down
        key(FakeEcodes.KEY_A),
        key(FakeEcodes.KEY_LEFTSHIFT, 0),   # shift up
        key(FakeEcodes.KEY_B),
        key(FakeEcodes.KEY_ENTER),
    ])
    assert s.results.get_nowait() == "Ab"


def test_key_release_events_are_ignored():
    # Each physical press produces down and up events; counting both
    # would double every character
    s = run_scanner([key(FakeEcodes.KEY_A, 1), key(FakeEcodes.KEY_A, 0),
                     key(FakeEcodes.KEY_ENTER)])
    assert s.results.get_nowait() == "a"


def test_empty_scan_is_not_queued():
    s = run_scanner([key(FakeEcodes.KEY_ENTER)])
    assert s.results.empty()


def test_non_key_events_are_ignored():
    s = run_scanner([FakeEvent(99, FakeEcodes.KEY_A, 1),
                     key(FakeEcodes.KEY_ENTER)])
    assert s.results.empty()


def test_multiple_scans_queue_in_order():
    s = run_scanner([key(FakeEcodes.KEY_1), key(FakeEcodes.KEY_ENTER),
                     key(FakeEcodes.KEY_2), key(FakeEcodes.KEY_ENTER)])
    assert s.results.get_nowait() == "1"
    assert s.results.get_nowait() == "2"


# Buffer access

def test_take_buffer_returns_and_clears():
    s = run_scanner([key(FakeEcodes.KEY_A)])  # no ENTER, stays buffered
    assert s.take_buffer() == "a"
    assert s.take_buffer() == ""


def test_snapshot_does_not_clear():
    s = run_scanner([key(FakeEcodes.KEY_A)])
    assert s.snapshot() == "a"
    assert s.snapshot() == "a"


# Device lookup by name

def test_find_by_name_selects_the_named_scanner():
    devices = {
        "/dev/input/event0": FakeDevice("/dev/input/event0", "AT Keyboard"),
        "/dev/input/event1": FakeDevice("/dev/input/event1", "Honeywell 1950g"),
    }
    found = scanner.BarcodeScanner._find_by_name(
        devices.__getitem__, devices.keys, "Honeywell 1950g")
    assert found == "/dev/input/event1"


def test_find_by_name_is_case_insensitive():
    devices = {"/dev/input/event0": FakeDevice("/dev/input/event0", "Honeywell 1950g Keyboard")}
    found = scanner.BarcodeScanner._find_by_name(
        devices.__getitem__, devices.keys, "honeywell 1950g")
    assert found == "/dev/input/event0"


def test_find_by_name_ignores_keyboard_and_mouse():
    # Only the scanner may be selected; a keyboard or mouse must never be
    devices = {
        "/dev/input/event0": FakeDevice("/dev/input/event0", "AT Translated Set 2 keyboard"),
        "/dev/input/event1": FakeDevice("/dev/input/event1", "Logitech USB Mouse"),
    }
    found = scanner.BarcodeScanner._find_by_name(
        devices.__getitem__, devices.keys, "Honeywell 1950g")
    assert found is None


def test_find_by_name_returns_none_when_absent():
    found = scanner.BarcodeScanner._find_by_name(
        lambda: [], lambda: [], "Honeywell 1950g")
    assert found is None


def test_find_by_name_skips_unopenable_devices():
    def raise_oserror(path):
        raise OSError("permission denied")
    found = scanner.BarcodeScanner._find_by_name(
        raise_oserror, lambda: ["/dev/input/event0"], "Honeywell 1950g")
    assert found is None


# Construction and lifecycle

def test_missing_evdev_degrades_gracefully(monkeypatch):
    # sys.modules[name] = None makes "from evdev import ..." raise
    # ImportError, simulating an uninstalled package
    monkeypatch.setitem(sys.modules, "evdev", None)
    s = scanner.BarcodeScanner()
    assert s.device is None
    assert s.snapshot() == ""
    s.close()  # must not raise


def test_full_construction_with_fake_evdev(monkeypatch):
    # End to end: look up by name, grab, background thread, one scan
    device = FakeDevice(name="Honeywell 1950g",
                        events=[key(FakeEcodes.KEY_A), key(FakeEcodes.KEY_ENTER)])
    fake_evdev = types.ModuleType("evdev")
    fake_evdev.InputDevice = lambda path: device
    fake_evdev.ecodes = FakeEcodes
    fake_evdev.list_devices = lambda: [device.path]
    monkeypatch.setitem(sys.modules, "evdev", fake_evdev)

    s = scanner.BarcodeScanner(name="Honeywell 1950g", grab=True)
    assert s.name == "Honeywell 1950g"
    assert device.grabbed
    assert s.results.get(timeout=2) == "a"
    s.close()
    assert device.closed


def test_no_devices_found_leaves_scanner_disabled(monkeypatch):
    fake_evdev = types.ModuleType("evdev")
    fake_evdev.InputDevice = FakeDevice
    fake_evdev.ecodes = FakeEcodes
    fake_evdev.list_devices = lambda: []
    monkeypatch.setitem(sys.modules, "evdev", fake_evdev)

    s = scanner.BarcodeScanner()
    assert s.device is None


def test_stop_flag_halts_read_loop():
    # close() sets the stop event; the reader must exit even with events
    # still pending, otherwise shutdown hangs on a chatty device
    s = scanner.BarcodeScanner.__new__(scanner.BarcodeScanner)
    s.results = queue.Queue()
    s._buffer = ""
    s._lock = threading.Lock()
    s._stop = threading.Event()
    s._stop.set()
    s._ecodes = FakeEcodes
    s._keymap = scanner._build_keymap(FakeEcodes)
    s.device = FakeDevice(events=[key(FakeEcodes.KEY_A), key(FakeEcodes.KEY_ENTER)])
    s._run()
    assert s.results.empty()


def test_device_unplugged_mid_read_is_survived():
    # A yanked USB cable raises OSError inside read_loop; the thread
    # must swallow it instead of crashing with a traceback
    class UnpluggedDevice(FakeDevice):
        def read_loop(self):
            yield key(FakeEcodes.KEY_A)
            raise OSError("device unplugged")

    s = scanner.BarcodeScanner.__new__(scanner.BarcodeScanner)
    s.results = queue.Queue()
    s._buffer = ""
    s._lock = threading.Lock()
    s._stop = threading.Event()
    s._ecodes = FakeEcodes
    s._keymap = scanner._build_keymap(FakeEcodes)
    s.device = UnpluggedDevice()
    s._run()  # must not raise
    assert s.snapshot() == "a"  # the character before the unplug survived


def test_close_survives_ungrab_failure():
    class StubbornDevice(FakeDevice):
        def ungrab(self):
            raise OSError("already ungrabbed")

    s = scanner.BarcodeScanner.__new__(scanner.BarcodeScanner)
    s._stop = threading.Event()
    s.device = StubbornDevice()
    s.close()  # must not raise
    assert s.device.closed


def test_unopenable_device_leaves_scanner_disabled(monkeypatch):
    def raise_oserror(path):
        raise OSError("busy")
    fake_evdev = types.ModuleType("evdev")
    fake_evdev.InputDevice = raise_oserror
    fake_evdev.ecodes = FakeEcodes
    fake_evdev.list_devices = lambda: []
    monkeypatch.setitem(sys.modules, "evdev", fake_evdev)

    s = scanner.BarcodeScanner(device_path="/dev/input/event9")
    assert s.device is None
