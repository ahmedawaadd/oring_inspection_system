"""Tests for inference.py — runs entirely with ModelStub, no weights required."""

from __future__ import annotations

import unittest

import numpy as np

from config import Config
from inference import (
    Detection,
    InspectionEngine,
    InspectionResult,
    ModelStub,
    make_model,
)


def _small_frame(h: int = 48, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestMakeModel(unittest.TestCase):
    def test_returns_stub_when_configured(self) -> None:
        cfg = Config(use_model_stub=True)
        model = make_model(cfg)
        self.assertIsInstance(model, ModelStub)


class TestModelStub(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(use_model_stub=True, confidence_threshold=0.75)
        self.model = ModelStub(self.config)

    def test_returns_two_detections(self) -> None:
        dets = self.model.predict(_small_frame())
        self.assertEqual(len(dets), 2)

    def test_all_detections_labelled_oring(self) -> None:
        for d in self.model.predict(_small_frame()):
            self.assertEqual(d.label, "oring")

    def test_confidence_above_threshold(self) -> None:
        for d in self.model.predict(_small_frame()):
            self.assertGreater(d.confidence, self.config.confidence_threshold)

    def test_bboxes_have_four_components(self) -> None:
        for d in self.model.predict(_small_frame()):
            self.assertEqual(len(d.bbox), 4)

    def test_bboxes_within_frame_bounds(self) -> None:
        h, w = 48, 64
        for d in self.model.predict(_small_frame(h, w)):
            x1, y1, x2, y2 = d.bbox
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, w)
            self.assertLessEqual(y2, h)


class TestInspectionEnginePass(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(
            use_model_stub=True,
            confidence_threshold=0.75,
            expected_oring_count=2,
        )
        self.engine = InspectionEngine(self.config)

    def test_pass_with_two_high_confidence_orings(self) -> None:
        result = self.engine.inspect(_small_frame())
        self.assertIsInstance(result, InspectionResult)
        self.assertTrue(result.passed)

    def test_pass_result_contains_non_empty_reason(self) -> None:
        result = self.engine.inspect(_small_frame())
        self.assertIsInstance(result.reason, str)
        self.assertGreater(len(result.reason), 0)

    def test_pass_result_contains_detections_list(self) -> None:
        result = self.engine.inspect(_small_frame())
        self.assertIsInstance(result.detections, list)
        self.assertEqual(len(result.detections), 2)


class TestInspectionEngineFail(unittest.TestCase):
    def _engine_with(self, detections: list[Detection]) -> InspectionEngine:
        config = Config(use_model_stub=False, confidence_threshold=0.75, expected_oring_count=2)

        class _FixedModel:
            def predict(self, frame: np.ndarray) -> list[Detection]:
                return detections

        return InspectionEngine(config, model=_FixedModel())

    def test_fail_when_only_one_oring(self) -> None:
        engine = self._engine_with([Detection("oring", 0.90, (0, 0, 10, 10))])
        result = engine.inspect(_small_frame())
        self.assertFalse(result.passed)

    def test_fail_when_no_orings(self) -> None:
        engine = self._engine_with([])
        result = engine.inspect(_small_frame())
        self.assertFalse(result.passed)

    def test_fail_when_confidence_below_threshold(self) -> None:
        low_conf = [
            Detection("oring", 0.50, (0, 0, 10, 10)),
            Detection("oring", 0.45, (20, 0, 30, 10)),
        ]
        engine = self._engine_with(low_conf)
        result = engine.inspect(_small_frame())
        self.assertFalse(result.passed)

    def test_fail_when_three_orings_but_expected_two(self) -> None:
        three = [Detection("oring", 0.90, (i * 10, 0, i * 10 + 8, 10)) for i in range(3)]
        engine = self._engine_with(three)
        result = engine.inspect(_small_frame())
        self.assertFalse(result.passed)

    def test_fail_reason_mentions_expected_and_found_counts(self) -> None:
        engine = self._engine_with([Detection("oring", 0.90, (0, 0, 10, 10))])
        result = engine.inspect(_small_frame())
        self.assertIn("2", result.reason)  # expected count
        self.assertIn("1", result.reason)  # found count

    def test_non_oring_labels_excluded_from_pass_logic(self) -> None:
        mixed = [
            Detection("oring", 0.90, (0, 0, 10, 10)),
            Detection("dust", 0.95, (20, 0, 30, 10)),  # should not count
        ]
        engine = self._engine_with(mixed)
        result = engine.inspect(_small_frame())
        self.assertFalse(result.passed)


class TestInspectionEngineConfigurableThreshold(unittest.TestCase):
    def test_higher_threshold_causes_fail(self) -> None:
        config = Config(use_model_stub=True, confidence_threshold=0.95, expected_oring_count=2)
        engine = InspectionEngine(config)
        # Stub returns 0.92 and 0.88 — both below 0.95
        result = engine.inspect(_small_frame())
        self.assertFalse(result.passed)

    def test_lower_threshold_still_passes(self) -> None:
        config = Config(use_model_stub=True, confidence_threshold=0.50, expected_oring_count=2)
        engine = InspectionEngine(config)
        result = engine.inspect(_small_frame())
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
