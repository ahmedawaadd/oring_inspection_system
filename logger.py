"""Structured inspection result logger backed by SQLite.

Every inspection is written as a row in the `inspections` table.
Images are saved to disk (as .npy arrays in stub mode; see SWAP POINT
comment to switch to PNG via Pillow once it is installed).

Schema
------
inspections
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    timestamp       TEXT    ISO-8601 UTC
    passed          INTEGER 1 = PASS, 0 = FAIL
    reason          TEXT    human-readable verdict explanation
    confidence_scores TEXT  "label:score,label:score,..."
    image_path      TEXT    path on disk, NULL if image was not saved

Public API
----------
InspectionLogger(config)
    .log_result(result, frame=None) -> int   row id
    .close()
"""

from __future__ import annotations

import logging
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from config import Config
from inference import InspectionResult

log = logging.getLogger(__name__)


class InspectionLogger:
    def __init__(self, config: Config) -> None:
        self._pass_sample_rate = config.pass_sample_rate
        self._db_path = config.db_path
        self._image_dir = config.image_save_dir

        self._image_dir.mkdir(parents=True, exist_ok=True)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._init_schema()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def log_result(
        self,
        result: InspectionResult,
        frame: Optional[np.ndarray] = None,
    ) -> int:
        """Persist one inspection result and optionally its source image.

        Returns the auto-assigned row id.
        """
        ts = datetime.now(timezone.utc).isoformat()
        confidence_scores = ",".join(
            f"{d.label}:{d.confidence:.4f}" for d in result.detections
        )
        image_path = self._maybe_save_image(frame, ts, result.passed)

        cur = self._conn.execute(
            "INSERT INTO inspections "
            "(timestamp, passed, reason, confidence_scores, image_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts, int(result.passed), result.reason, confidence_scores, image_path),
        )
        self._conn.commit()

        log.info(
            "Logged inspection id=%d passed=%s reason=%r",
            cur.lastrowid,
            result.passed,
            result.reason,
        )
        return cur.lastrowid  # type: ignore[return-value]

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                passed          INTEGER NOT NULL,
                reason          TEXT,
                confidence_scores TEXT,
                image_path      TEXT
            )
            """
        )
        self._conn.commit()

    def _maybe_save_image(
        self,
        frame: Optional[np.ndarray],
        timestamp: str,
        passed: bool,
    ) -> Optional[str]:
        if frame is None:
            return None
        # Always save failures; sample passes at the configured rate
        if passed and random.random() >= self._pass_sample_rate:
            return None
        return self._write_image(frame, timestamp, passed)

    def _write_image(self, frame: np.ndarray, timestamp: str, passed: bool) -> str:
        """Save *frame* to disk and return the file path as a string.

        SWAP POINT: replace np.save with PIL.Image.fromarray(...).save(path)
        for PNG output once Pillow is added to the environment.
        """
        safe_ts = timestamp.replace(":", "-").replace("+", "_")
        suffix = "PASS" if passed else "FAIL"
        path = self._image_dir / f"{safe_ts}_{suffix}.npy"
        np.save(str(path), frame)
        return str(path)
