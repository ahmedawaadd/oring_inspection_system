"""Tests for logger.py — no hardware, no external services required."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import Config
from inference import Detection, InspectionResult
from logger import InspectionLogger


def _make_config(tmp: Path, pass_sample_rate: float = 1.0) -> Config:
    return Config(
        db_path=tmp / "test.db",
        image_save_dir=tmp / "images",
        pass_sample_rate=pass_sample_rate,
    )


def _pass_result() -> InspectionResult:
    return InspectionResult(
        detections=[
            Detection("oring", 0.92, (10.0, 10.0, 50.0, 50.0)),
            Detection("oring", 0.88, (60.0, 10.0, 100.0, 50.0)),
        ],
        passed=True,
        reason="2 O-ring(s) detected above threshold 0.75",
    )


def _fail_result() -> InspectionResult:
    return InspectionResult(
        detections=[Detection("oring", 0.60, (10.0, 10.0, 50.0, 50.0))],
        passed=False,
        reason="Expected 2 O-ring(s) above threshold 0.75, found 0",
    )


def _frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


class TestLoggerInit(unittest.TestCase):
    def test_creates_db_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_config(Path(tmp))
            logger = InspectionLogger(cfg)
            logger.close()
            self.assertTrue(cfg.db_path.exists())
            self.assertTrue(cfg.image_save_dir.exists())

    def test_schema_has_inspections_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_config(Path(tmp))
            logger = InspectionLogger(cfg)
            logger.close()
            conn = sqlite3.connect(str(cfg.db_path))
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            conn.close()
            self.assertIn(("inspections",), tables)


class TestLogResult(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        cfg = _make_config(Path(self._tmp.name), pass_sample_rate=1.0)
        self.logger = InspectionLogger(cfg)
        self.db_path = cfg.db_path

    def tearDown(self) -> None:
        self.logger.close()
        self._tmp.cleanup()

    def _rows(self) -> list[tuple]:
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute("SELECT * FROM inspections").fetchall()
        conn.close()
        return rows

    def test_log_result_returns_positive_integer(self) -> None:
        row_id = self.logger.log_result(_pass_result())
        self.assertIsInstance(row_id, int)
        self.assertGreater(row_id, 0)

    def test_pass_written_correctly(self) -> None:
        self.logger.log_result(_pass_result())
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][2], 1)  # passed column

    def test_fail_written_correctly(self) -> None:
        self.logger.log_result(_fail_result())
        rows = self._rows()
        self.assertEqual(rows[0][2], 0)  # passed column

    def test_reason_stored(self) -> None:
        self.logger.log_result(_pass_result())
        rows = self._rows()
        self.assertIn("O-ring", rows[0][3])

    def test_confidence_scores_stored(self) -> None:
        self.logger.log_result(_pass_result())
        rows = self._rows()
        scores: str = rows[0][4]
        self.assertIn("oring:", scores)
        self.assertIn("0.92", scores)

    def test_timestamp_is_iso_format(self) -> None:
        from datetime import datetime
        self.logger.log_result(_pass_result())
        rows = self._rows()
        ts: str = rows[0][1]
        # Should parse without error
        datetime.fromisoformat(ts)

    def test_multiple_results_stored_in_order(self) -> None:
        for _ in range(5):
            self.logger.log_result(_pass_result())
        rows = self._rows()
        self.assertEqual(len(rows), 5)
        ids = [r[0] for r in rows]
        self.assertEqual(ids, sorted(ids))


class TestImageSaving(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _logger(self, pass_sample_rate: float = 1.0) -> InspectionLogger:
        return InspectionLogger(_make_config(self.tmp_path, pass_sample_rate))

    def _image_count(self) -> int:
        return len(list((self.tmp_path / "images").glob("*.npy")))

    def test_fail_image_always_saved(self) -> None:
        logger = self._logger(pass_sample_rate=0.0)
        logger.log_result(_fail_result(), frame=_frame())
        logger.close()
        self.assertEqual(self._image_count(), 1)

    def test_pass_image_saved_at_rate_1(self) -> None:
        logger = self._logger(pass_sample_rate=1.0)
        logger.log_result(_pass_result(), frame=_frame())
        logger.close()
        self.assertEqual(self._image_count(), 1)

    def test_pass_image_not_saved_at_rate_0(self) -> None:
        logger = self._logger(pass_sample_rate=0.0)
        logger.log_result(_pass_result(), frame=_frame())
        logger.close()
        self.assertEqual(self._image_count(), 0)

    def test_image_path_stored_in_db_for_fail(self) -> None:
        cfg = _make_config(self.tmp_path, pass_sample_rate=1.0)
        logger = InspectionLogger(cfg)
        logger.log_result(_fail_result(), frame=_frame())
        logger.close()
        conn = sqlite3.connect(str(cfg.db_path))
        row = conn.execute("SELECT image_path FROM inspections").fetchone()
        conn.close()
        self.assertIsNotNone(row[0])
        self.assertTrue(row[0].endswith(".npy"))

    def test_no_image_path_when_no_frame_provided(self) -> None:
        cfg = _make_config(self.tmp_path, pass_sample_rate=1.0)
        logger = InspectionLogger(cfg)
        logger.log_result(_fail_result(), frame=None)
        logger.close()
        conn = sqlite3.connect(str(cfg.db_path))
        row = conn.execute("SELECT image_path FROM inspections").fetchone()
        conn.close()
        self.assertIsNone(row[0])

    def test_saved_npy_file_is_valid_array(self) -> None:
        cfg = _make_config(self.tmp_path, pass_sample_rate=1.0)
        logger = InspectionLogger(cfg)
        logger.log_result(_fail_result(), frame=_frame())
        logger.close()
        images = list((self.tmp_path / "images").glob("*.npy"))
        self.assertEqual(len(images), 1)
        loaded = np.load(str(images[0]))
        self.assertEqual(loaded.shape, _frame().shape)
        self.assertEqual(loaded.dtype, np.uint8)

    def test_fail_filename_contains_FAIL_marker(self) -> None:
        cfg = _make_config(self.tmp_path)
        logger = InspectionLogger(cfg)
        logger.log_result(_fail_result(), frame=_frame())
        logger.close()
        images = list((self.tmp_path / "images").glob("*FAIL*.npy"))
        self.assertEqual(len(images), 1)

    def test_pass_filename_contains_PASS_marker(self) -> None:
        cfg = _make_config(self.tmp_path, pass_sample_rate=1.0)
        logger = InspectionLogger(cfg)
        logger.log_result(_pass_result(), frame=_frame())
        logger.close()
        images = list((self.tmp_path / "images").glob("*PASS*.npy"))
        self.assertEqual(len(images), 1)


if __name__ == "__main__":
    unittest.main()
