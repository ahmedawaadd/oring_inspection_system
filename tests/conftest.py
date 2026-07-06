"""Shared fixtures. Pi-only hardware modules are stubbed here so the
suite runs anywhere: tests exercise the logic, not the hardware."""

import sys
import types

import numpy as np
import pytest

# Stub picamera2 before any project import so camera.py loads off-Pi.
# Tests never instantiate the real class, they use FakeCamera instead.
_picamera2 = types.ModuleType("picamera2")
_picamera2.Picamera2 = object
sys.modules.setdefault("picamera2", _picamera2)

import ui  # noqa: E402  (must come after the hardware stub)


class FakeCamera:
    """Camera double: returns queued frames from capture_array().
    Matches the only surface of Picamera2 the app touches, so faking at
    this seam keeps tests free of mock patching."""

    def __init__(self, frames):
        self._frames = iter(frames)
        self.stopped = False

    def capture_array(self):
        return next(self._frames)

    def stop(self):
        self.stopped = True


@pytest.fixture
def frame():
    """A production-size 960x1280 BGR frame."""
    return np.zeros((960, 1280, 3), dtype=np.uint8)


@pytest.fixture
def rng():
    """Seeded RNG so image-content tests are deterministic."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Isolated working directory. Persistence code uses relative paths,
    so chdir keeps test artifacts out of the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def reset_mouse_state():
    """ui.mouse is shared module state; reset it so tests can't leak
    drag state into each other."""
    ui.mouse.update(active_slot=None, drawing=False,
                    pt1=(0, 0), pt2=(0, 0), roi_ready=False)
    yield
    ui.mouse.update(active_slot=None, drawing=False,
                    pt1=(0, 0), pt2=(0, 0), roi_ready=False)
