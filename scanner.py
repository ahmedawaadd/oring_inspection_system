"""USB barcode scanner support, read directly from its input device via evdev.

A USB barcode scanner presents itself to the OS as a keyboard. OpenCV's
window can't keep up with how fast it "types" — cv2.waitKey only remembers
the last key between calls, so most characters of a fast scan get dropped.
To capture scans reliably we read the scanner's key events straight from
its input device with evdev, in a background thread, and assemble the
characters ourselves until the scanner sends ENTER.
"""

import queue
import threading


def _build_keymap(ecodes):
    """Map evdev key codes to characters (letters, digits, and a few symbols)."""
    m = {}
    for c in "0123456789":
        m[getattr(ecodes, f"KEY_{c}")] = c
    for c in "abcdefghijklmnopqrstuvwxyz":
        m[getattr(ecodes, f"KEY_{c.upper()}")] = c
    m[ecodes.KEY_MINUS] = "-"
    m[ecodes.KEY_DOT]   = "."
    return m


class BarcodeScanner:
    """Reads a barcode scanner via evdev in a background thread. Completed
    scans (terminated by ENTER) are pushed onto a queue for the main loop to
    drain. If evdev isn't installed or no scanner is found this does nothing
    and manual keyboard entry still works."""

    def __init__(self, device_path=None, name_hint="scanner", grab=True):
        self.results = queue.Queue()
        self._buffer = ""
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self.device  = None
        self.name    = None

        try:
            from evdev import InputDevice, ecodes, list_devices
        except ImportError:
            print("evdev not installed — scanner disabled, type the barcode manually.")
            return

        self._ecodes = ecodes
        self._keymap = _build_keymap(ecodes)

        if device_path is None:
            device_path = self._autodetect(InputDevice, ecodes, list_devices, name_hint)
        if device_path is None:
            print("No barcode scanner found — type the barcode manually.")
            return

        try:
            self.device = InputDevice(device_path)
            if grab:
                self.device.grab()   # exclusive access so scans don't leak elsewhere
            self.name = self.device.name
            print(f"Barcode scanner connected: {self.name} ({device_path})")
        except OSError as e:
            print(f"Could not open barcode scanner at {device_path}: {e}")
            self.device = None
            return

        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _autodetect(InputDevice, ecodes, list_devices, name_hint):
        # A scanner looks like a keyboard: it can produce ENTER and letter keys
        candidates = []
        for path in list_devices():
            try:
                dev = InputDevice(path)
            except OSError:
                continue
            keys = dev.capabilities().get(ecodes.EV_KEY, [])
            if ecodes.KEY_ENTER in keys and ecodes.KEY_A in keys:
                candidates.append(dev)
        # Prefer a device whose name hints it's a scanner (so we don't grab a
        # regular keyboard if one is also attached)
        for dev in candidates:
            if name_hint and name_hint.lower() in dev.name.lower():
                return dev.path
        return candidates[0].path if candidates else None

    def _run(self):
        ecodes = self._ecodes
        shift_keys = {ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT}
        shift = False
        try:
            for event in self.device.read_loop():
                if self._stop.is_set():
                    break
                if event.type != ecodes.EV_KEY:
                    continue
                if event.code in shift_keys:
                    shift = event.value in (1, 2)   # 1=down, 2=autorepeat
                    continue
                if event.value != 1:                # only act on key-down
                    continue
                if event.code in (ecodes.KEY_ENTER, ecodes.KEY_KPENTER):
                    # Scanner finished — hand the assembled barcode to the main loop
                    with self._lock:
                        code, self._buffer = self._buffer, ""
                    if code:
                        self.results.put(code)
                else:
                    ch = self._keymap.get(event.code)
                    if ch is not None:
                        with self._lock:
                            self._buffer += ch.upper() if shift else ch
        except OSError:
            pass   # device unplugged or closed

    def snapshot(self):
        """Return the partially-typed scan currently being assembled (for display)."""
        with self._lock:
            return self._buffer

    def take_buffer(self):
        """Return and clear the partial scan. Used when the operator presses
        ENTER manually because their scanner isn't configured to append one."""
        with self._lock:
            buf, self._buffer = self._buffer, ""
        return buf

    def close(self):
        self._stop.set()
        if self.device is not None:
            try:
                self.device.ungrab()
            except OSError:
                pass
            self.device.close()
