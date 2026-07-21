"""
scanner.py

Barcode scanner input via evdev.

A USB barcode scanner presents itself to the OS as a keyboard, but it
"types" faster than an OpenCV window can keep up with: cv2.waitKey only
remembers the last key between calls, so most characters of a fast scan
get dropped. To capture scans reliably we read the scanner's key events
straight from its input device with evdev, in a background thread, and
assemble the characters ourselves until the scanner sends ENTER.
"""

import queue
import threading
import time


def _build_keymap(ecodes):
    """Map evdev key codes to characters (letters, digits, a few symbols)."""
    m = {}
    for c in "0123456789":
        m[getattr(ecodes, f"KEY_{c}")] = c
    for c in "abcdefghijklmnopqrstuvwxyz":
        m[getattr(ecodes, f"KEY_{c.upper()}")] = c
    m[ecodes.KEY_MINUS] = "-"
    m[ecodes.KEY_DOT] = "."
    return m


class BarcodeScanner:
    """Reads a barcode scanner via evdev in a background thread.

    Completed scans (terminated by ENTER) are pushed onto self.results
    for the main loop to drain. If evdev isn't installed or no scanner is
    found this does nothing and manual keyboard entry still works.
    """

    def __init__(self, device_path=None, names=("Honeywell 1950g",), grab=True,
                 settle=0.1):
        self.results = queue.Queue()
        self._buffer = ""
        self._last_key = 0.0   # monotonic time of the last buffered character
        self._settle = settle  # quiet period before the buffer may be read
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.device = None
        self.name = None

        try:
            from evdev import InputDevice, ecodes, list_devices
        except ImportError:
            print("evdev not installed, scanner disabled. Type the barcode manually.")
            return

        self._ecodes = ecodes
        self._keymap = _build_keymap(ecodes)

        # A single name is accepted too, so old callers keep working.
        if isinstance(names, str):
            names = [names]

        if device_path is None:
            device_path = self._find_by_name(InputDevice, list_devices, names)
        if device_path is None:
            wanted = ", ".join(names)
            print(f"No known barcode scanner ({wanted}) found. Type the barcode manually.")
            return

        try:
            self.device = InputDevice(device_path)
            if grab:
                # Exclusive access so scans don't leak to other windows
                self.device.grab()
            self.name = self.device.name
            print(f"Barcode scanner connected: {self.name} ({device_path})")
        except OSError as e:
            print(f"Could not open barcode scanner at {device_path}: {e}")
            self.device = None
            return

        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _find_by_name(InputDevice, list_devices, names):
        """Return the path of the first input device matching one of `names`.

        The scanner enumerates as a keyboard, so it is indistinguishable
        from the operator's real keyboard or mouse by capabilities alone.
        We connect to it by name only (e.g. "Honeywell 1950g"), matched
        case-insensitively as a substring, so a keyboard or mouse is never
        grabbed. `names` is a list of known scanners in priority order: the
        earlier a name appears, the more it is preferred, so if two known
        scanners are plugged in the first listed one wins regardless of
        device enumeration order. If no known scanner is present we return
        None and the app falls back to manual entry."""
        if isinstance(names, str):
            names = [names]
        for name in names:
            for path in list_devices():
                try:
                    dev = InputDevice(path)
                except OSError:
                    continue
                if name.lower() in dev.name.lower():
                    return dev.path
        return None

    def _run(self):
        """Background thread: assemble key events into barcode strings."""
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
                    shift = event.value in (1, 2)  # 1=down, 2=autorepeat
                    continue
                if event.value != 1:  # only act on key-down
                    continue
                if event.code in (ecodes.KEY_ENTER, ecodes.KEY_KPENTER):
                    # Scan complete, hand the assembled barcode to the main loop
                    with self._lock:
                        code, self._buffer = self._buffer, ""
                    if code:
                        self.results.put(code)
                else:
                    ch = self._keymap.get(event.code)
                    if ch is not None:
                        with self._lock:
                            self._buffer += ch.upper() if shift else ch
                            self._last_key = time.monotonic()
        except OSError:
            pass  # device unplugged or closed

    def snapshot(self):
        """Return the partially typed scan currently being assembled,
        used to show live scan progress in the popup."""
        with self._lock:
            return self._buffer

    def take_buffer(self):
        """Return and clear the partial scan. Used when the operator
        presses ENTER manually because their scanner isn't configured to
        append one."""
        with self._lock:
            buf, self._buffer = self._buffer, ""
        return buf

    def settled(self):
        """True once no character has arrived for the settle window. A
        scanner types its whole burst with only milliseconds between keys,
        so a short quiet gap means the scan is finished; reading the
        buffer mid-burst would split one scan into a barcode now and a
        stray tail for the next part."""
        with self._lock:
            return time.monotonic() - self._last_key >= self._settle

    def flush(self):
        """Discard any completed scans and partial input. Called when the
        barcode popup opens: ignoring the scanner while the popup is
        closed is not the same as emptying it, and anything scanned while
        the system wasn't asking must not be mistaken for an answer to
        the popup that is asking now."""
        with self._lock:
            self._buffer = ""
        try:
            while True:
                self.results.get_nowait()
        except queue.Empty:
            pass

    def close(self):
        """Stop the reader thread and release the device."""
        self._stop.set()
        if self.device is not None:
            try:
                self.device.ungrab()
            except OSError:
                pass
            self.device.close()
