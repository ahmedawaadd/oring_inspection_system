"""Tests for camera.py — runs entirely with the stub, no hardware required."""

from __future__ import annotations

import unittest

import numpy as np

from camera import CameraStub, make_camera
from config import Config


class TestCameraStub(unittest.TestCase):
    def setUp(self) -> None:
        # Use a tiny frame to keep tests fast
        self.config = Config(frame_width=64, frame_height=48, use_camera_stub=True)
        self.cam = CameraStub(self.config)

    def tearDown(self) -> None:
        self.cam.stop()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def test_make_camera_returns_stub_when_configured(self) -> None:
        cam = make_camera(self.config)
        self.assertIsInstance(cam, CameraStub)

    # ------------------------------------------------------------------
    # Normal operation
    # ------------------------------------------------------------------

    def test_capture_after_start_returns_ndarray(self) -> None:
        self.cam.start()
        frame = self.cam.capture_frame()
        self.assertIsInstance(frame, np.ndarray)

    def test_frame_has_correct_shape(self) -> None:
        self.cam.start()
        frame = self.cam.capture_frame()
        self.assertEqual(frame.shape, (self.config.frame_height, self.config.frame_width, 3))

    def test_frame_has_uint8_dtype(self) -> None:
        self.cam.start()
        frame = self.cam.capture_frame()
        self.assertEqual(frame.dtype, np.uint8)

    def test_frame_pixel_values_in_range(self) -> None:
        self.cam.start()
        frame = self.cam.capture_frame()
        self.assertGreaterEqual(int(frame.min()), 0)
        self.assertLessEqual(int(frame.max()), 255)

    def test_successive_captures_are_independent(self) -> None:
        self.cam.start()
        a = self.cam.capture_frame().copy()
        b = self.cam.capture_frame().copy()
        # Not guaranteed to differ, but the RNG seed advances so they should
        # (with overwhelming probability for a 64×48 frame)
        self.assertFalse(np.array_equal(a, b), "Two successive frames were identical")

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_capture_before_start_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            self.cam.capture_frame()

    def test_stop_and_restart_works(self) -> None:
        self.cam.start()
        self.cam.stop()
        self.cam.start()
        frame = self.cam.capture_frame()
        self.assertIsNotNone(frame)

    def test_multiple_start_calls_are_idempotent(self) -> None:
        self.cam.start()
        self.cam.start()
        frame = self.cam.capture_frame()
        self.assertIsNotNone(frame)


if __name__ == "__main__":
    unittest.main()
