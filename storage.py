"""Persistence: reference images, ROI coordinates, and inspection logs."""

import csv
import os
from datetime import datetime

import cv2
import numpy as np

from config import GREEN, LOGS_DIR, RED, REFERENCE_PATHS, ROI_PATHS
from vision import load_thumb, preprocess


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


def save_inspection(barcode, frame, per_slot, overall_passed):
    """Save the annotated inspection image and append a CSV log row.
    Each barcode gets its own folder so results are easy to browse by part."""
    folder = os.path.join(LOGS_DIR, str(barcode))
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
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "barcode", "overall",
                             "slot1", "slot1_diff", "slot2", "slot2_diff"])
        row = [ts, barcode, verdict]
        for slot in [1, 2]:
            if slot in per_slot:
                p, d = per_slot[slot]
                row += ["PASS" if p else "FAIL", f"{d:.1f}"]
            else:
                row += ["N/A", "N/A"]
        writer.writerow(row)

    print(f"Saved: {img_path}")
