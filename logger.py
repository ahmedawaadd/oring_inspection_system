"""Persist inspection results: a stamped image plus a CSV row per barcode."""

import csv
import os
from datetime import datetime

import cv2

from config import LOGS_DIR, GREEN, RED


def save_inspection(barcode, frame, per_slot, overall_passed):
    # Each barcode gets its own folder so results are easy to browse by part
    folder = os.path.join(LOGS_DIR, str(barcode))
    os.makedirs(folder, exist_ok=True)

    ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    verdict = "PASS" if overall_passed else "FAIL"
    color   = GREEN if overall_passed else RED

    # Stamp the barcode and verdict directly onto the saved image so the
    # result is readable without opening the CSV
    out = frame.copy()
    cv2.putText(out, f"#{barcode}  {verdict}", (10, 80),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3, cv2.LINE_AA)

    img_path = os.path.join(folder, f"{ts}_{verdict}.jpg")
    cv2.imwrite(img_path, out)

    # Append one row to the CSV for this barcode, writing the header first
    # if this is the first inspection for this barcode
    log_path     = os.path.join(folder, "log.csv")
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

    print(f"Saved → {img_path}")
