"""YOLO inference wrapper and pass/fail decision layer.

Public API
----------
make_model(config) -> model object
    Returns a ModelStub when config.use_model_stub is True,
    otherwise a YOLOModel that loads the weights from config.model_path.

InspectionEngine(config, model=None)
    .inspect(frame: np.ndarray) -> InspectionResult
        Runs the model and applies pass/fail logic:
        exactly config.expected_oring_count detections above
        config.confidence_threshold → PASS, otherwise FAIL.

Data classes
------------
Detection        label, confidence, bbox (x1,y1,x2,y2)
InspectionResult detections, passed, reason
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import Config


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]   # (x1, y1, x2, y2) in pixels


@dataclass
class InspectionResult:
    detections: list[Detection]
    passed: bool
    reason: str


# ---------------------------------------------------------------------------
# Stub model (no weights file or GPU required)
# ---------------------------------------------------------------------------

class ModelStub:
    """Returns two synthetic O-ring detections; no Ultralytics required.

    SWAP POINT: replace with YOLOModel once the real weights are available
    and Ultralytics is installed.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def predict(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        # Two plausible O-ring bounding boxes, left and right of centre
        return [
            Detection(
                label="oring",
                confidence=0.92,
                bbox=(w * 0.25, h * 0.30, w * 0.45, h * 0.70),
            ),
            Detection(
                label="oring",
                confidence=0.88,
                bbox=(w * 0.55, h * 0.30, w * 0.75, h * 0.70),
            ),
        ]


# ---------------------------------------------------------------------------
# Real model (requires `pip install ultralytics` and a trained weights file)
# ---------------------------------------------------------------------------

class YOLOModel:
    """Thin wrapper around an Ultralytics YOLO model.

    Compatible with any YOLO generation supported by the Ultralytics library
    (YOLOv8, YOLO11, YOLO-World, etc.).
    """

    def __init__(self, config: Config) -> None:
        from ultralytics import YOLO  # type: ignore[import]  # deferred import
        self._model = YOLO(str(config.model_path))
        self._conf = config.confidence_threshold

    def predict(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(frame, conf=self._conf, verbose=False)
        detections: list[Detection] = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label: str = r.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
        return detections


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_model(config: Config) -> ModelStub | YOLOModel:
    if config.use_model_stub:
        return ModelStub(config)
    return YOLOModel(config)


# ---------------------------------------------------------------------------
# Inspection engine (pass/fail logic)
# ---------------------------------------------------------------------------

class InspectionEngine:
    def __init__(self, config: Config, model: Optional[ModelStub | YOLOModel] = None) -> None:
        self._config = config
        self._model = model if model is not None else make_model(config)

    def inspect(self, frame: np.ndarray) -> InspectionResult:
        """Run inference on *frame* and return a pass/fail result."""
        detections = self._model.predict(frame)

        qualified = [
            d for d in detections
            if d.label == "oring" and d.confidence >= self._config.confidence_threshold
        ]

        expected = self._config.expected_oring_count
        if len(qualified) == expected:
            return InspectionResult(
                detections=detections,
                passed=True,
                reason=f"{expected} O-ring(s) detected above threshold {self._config.confidence_threshold:.2f}",
            )

        return InspectionResult(
            detections=detections,
            passed=False,
            reason=(
                f"Expected {expected} O-ring(s) above threshold "
                f"{self._config.confidence_threshold:.2f}, "
                f"found {len(qualified)}"
            ),
        )
