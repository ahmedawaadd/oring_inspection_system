"""Camera capture module.

Public API
----------
make_camera(config) -> camera object
    Returns a CameraStub when config.use_camera_stub is True,
    otherwise a PiCamera2Capture.

Both implementations expose the same three methods:
    start()             -- warm up / allocate
    capture_frame()     -- return one HxWx3 uint8 numpy array
    stop()              -- release resources
"""

from __future__ import annotations

import numpy as np

from config import Config


# ---------------------------------------------------------------------------
# Stub implementation (no hardware required)
# ---------------------------------------------------------------------------

class CameraStub:
    """Returns a synthetic noise frame; no picamera2 required.

    SWAP POINT: replace this class with PiCamera2Capture once the CSI
    cable arrives and picamera2 is installed.
    """

    def __init__(self, config: Config) -> None:
        self._width = config.frame_width
        self._height = config.frame_height
        self._started = False
        self._rng = np.random.default_rng(seed=42)

    def start(self) -> None:
        self._started = True

    def capture_frame(self) -> np.ndarray:
        if not self._started:
            raise RuntimeError("Camera.start() must be called before capture_frame()")
        return self._rng.integers(0, 256, (self._height, self._width, 3), dtype=np.uint8)

    def stop(self) -> None:
        self._started = False


# ---------------------------------------------------------------------------
# Real implementation (requires picamera2 + IMX477 hardware)
# ---------------------------------------------------------------------------

class PiCamera2Capture:
    """Wraps picamera2 for single-shot still capture.

    Requires: pip install picamera2
    """

    def __init__(self, config: Config) -> None:
        # Deferred import so the module is importable without the library
        from picamera2 import Picamera2  # type: ignore[import]

        self._cam = Picamera2()
        still_cfg = self._cam.create_still_configuration(
            main={
                "size": (config.frame_width, config.frame_height),
                "format": config.frame_format,
            }
        )
        self._cam.configure(still_cfg)

    def start(self) -> None:
        self._cam.start()

    def capture_frame(self) -> np.ndarray:
        return self._cam.capture_array()

    def stop(self) -> None:
        self._cam.stop()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_camera(config: Config) -> CameraStub | PiCamera2Capture:
    if config.use_camera_stub:
        return CameraStub(config)
    return PiCamera2Capture(config)
