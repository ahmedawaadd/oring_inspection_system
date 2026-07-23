"""
storage.py

Persistence: reference images, ROI coordinates, and inspection logs."""

import csv
import json
import os
from datetime import datetime

import cv2
import numpy as np

from config import (CALIBRATION_PATH, DEFAULT_DIFF_THRESHOLD,
                    DEFAULT_NOISE_THRESHOLD, GREEN, LOGS_DIR, RED,
                    REFERENCE_PATHS, ROI_PATHS)
from vision import load_thumb, preprocess


LEGACY_LOG_HEADER = [
    "timestamp", "barcode", "overall",
    "slot1", "slot1_diff", "slot2", "slot2_diff",
]
LOG_HEADER = LEGACY_LOG_HEADER + ["noise_threshold", "diff_threshold"]


def load_thresholds():
    """Load the engineer's saved slider settings, falling back to config
    defaults if the file is absent or damaged. A bad calibration file must
    not prevent the inspection station from starting."""
    default = DEFAULT_NOISE_THRESHOLD, DEFAULT_DIFF_THRESHOLD / 10.0
    try:
        with open(CALIBRATION_PATH) as f:
            saved = json.load(f)
        noise = int(saved["noise_threshold"])
        diff = float(saved["diff_threshold"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return default
    if not 0 <= noise <= 100 or not 0.0 <= diff <= 50.0:
        return default
    return noise, diff


def save_thresholds(noise_thresh, diff_thresh):
    """Persist an authenticated calibration change for future restarts."""
    with open(CALIBRATION_PATH, "w") as f:
        json.dump({
            "noise_threshold": int(noise_thresh),
            "diff_threshold": float(diff_thresh),
        }, f, indent=2)


def load_references():
    """Load ROIs, references, and thumbnails saved by a previous session
    so the operator doesn't have to redraw regions and recapture
    references every run. Returns (rois, refs, thumbs) keyed by slot."""
    rois, refs, thumbs = {}, {}, {}
    for slot in [1, 2]:
        roi_path = ROI_PATHS[slot - 1]
        ref_path = REFERENCE_PATHS[slot - 1]
        if os.path.exists(roi_path):
            rois[slot] = tuple(np.load(roi_path).tolist())
        if os.path.exists(ref_path):
            refs[slot] = preprocess(cv2.imread(ref_path))
            thumbs[slot] = load_thumb(ref_path)
    return rois, refs, thumbs


def save_reference(slot, ref_crop, roi):
    """Persist a reference crop and its ROI coordinates for a slot."""
    cv2.imwrite(REFERENCE_PATHS[slot - 1], ref_crop)
    np.save(ROI_PATHS[slot - 1], np.array(roi))


def safe_folder_name(barcode):
    """Restrict a barcode to filesystem-safe characters. Scanned input
    becomes a folder name, so path separators and dot-only names must
    not be able to escape LOGS_DIR. Normal barcodes pass through
    unchanged."""
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in str(barcode))
    if not safe.strip("."):
        # "." or ".." would resolve to the logs dir or its parent
        safe = "_" + safe
    return safe


def _upgrade_legacy_log(log_path):
    """Add threshold columns to logs created before sensitivities were saved.
    Old rows use N/A because inventing the settings used for a past
    inspection would make the production record misleading."""
    if not os.path.exists(log_path):
        return
    with open(log_path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != LEGACY_LOG_HEADER:
        return
    rows[0] = LOG_HEADER
    for row in rows[1:]:
        row.extend(["N/A", "N/A"])
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def save_inspection(barcode, frame, per_slot, overall_passed,
                    noise_thresh, diff_thresh):
    """Save the annotated inspection image and append a CSV log row.
    Each barcode gets its own folder so results are easy to browse by part.
    Thresholds are recorded with the verdict because live slider changes
    otherwise make a historical PASS or FAIL impossible to reproduce."""
    folder = os.path.join(LOGS_DIR, safe_folder_name(barcode))
    os.makedirs(folder, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    verdict = "PASS" if overall_passed else "FAIL"
    color = GREEN if overall_passed else RED

    # Stamp the barcode and verdict onto the saved image so the result is
    # readable without opening the CSV
    out = frame.copy()
    cv2.putText(out, f"#{barcode}  {verdict}", (10, 80),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3, cv2.LINE_AA)

    img_path = os.path.join(folder, f"{ts}_{verdict}.jpg")
    cv2.imwrite(img_path, out)

    # Append one row to this barcode's CSV, writing the header first if
    # this is the first inspection for the barcode
    log_path = os.path.join(folder, "log.csv")
    _upgrade_legacy_log(log_path)
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(LOG_HEADER)
        row = [ts, barcode, verdict]
        for slot in [1, 2]:
            if slot in per_slot:
                p, d = per_slot[slot]
                row += ["PASS" if p else "FAIL", f"{d:.1f}"]
            else:
                row += ["N/A", "N/A"]
        row += [str(noise_thresh), f"{diff_thresh:.1f}"]
        writer.writerow(row)

    print(f"Saved: {img_path}")
