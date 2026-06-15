"""Central configuration — every tunable value lives here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Config:
    # --- Camera ---
    frame_width: int = 4056
    frame_height: int = 3040
    frame_format: str = "RGB888"

    # --- Trigger ---
    trigger_pin: int = 17           # BCM pin number
    trigger_debounce_ms: int = 50

    # --- Inference ---
    model_path: Path = Path("models/oring_yolo.pt")
    confidence_threshold: float = 0.75
    expected_oring_count: int = 2
    # Optional (x, y, w, h) crop applied before inference; None = full frame
    roi: Optional[tuple[int, int, int, int]] = None

    # --- Logging ---
    db_path: Path = Path("logs/inspections.db")
    image_save_dir: Path = Path("logs/images")
    # Fraction of PASS images to persist (1.0 = keep all, 0.0 = keep none)
    pass_sample_rate: float = 0.1

    # --- Stub switches ---
    # Flip each to False once the real hardware is available
    use_camera_stub: bool = True
    use_gpio_stub: bool = True
    use_model_stub: bool = True


# Module-level singleton used by default throughout the app
CONFIG = Config()
