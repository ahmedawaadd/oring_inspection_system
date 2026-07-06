"""Tests for storage.py: persistence round trips, the CSV contract, and
hostile barcode input. All tests run in an isolated workdir."""

import csv
import os

import numpy as np
import pytest

import config
import storage


@pytest.fixture
def ref_crop(rng):
    return rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)


def _read_log(barcode):
    path = os.path.join(config.LOGS_DIR, barcode, "log.csv")
    with open(path, newline="") as f:
        return list(csv.reader(f))


# Reference persistence

def test_save_then_load_reference_round_trips(workdir, ref_crop):
    storage.save_reference(1, ref_crop, (10, 20, 90, 80))
    rois, refs, thumbs = storage.load_references()
    assert rois[1] == (10, 20, 90, 80)
    assert refs[1].ndim == 2  # loaded reference comes back preprocessed
    assert thumbs[1].shape == (config.THUMB_H, config.THUMB_W, 3)


def test_load_references_skips_missing_slots(workdir, ref_crop):
    storage.save_reference(1, ref_crop, (0, 0, 80, 60))
    rois, refs, thumbs = storage.load_references()
    assert 2 not in rois and 2 not in refs and 2 not in thumbs


def test_load_references_with_roi_but_no_image(workdir, ref_crop):
    # The reference image can be deleted between sessions; the ROI must
    # still load so the UI can show the ROI SET state
    storage.save_reference(1, ref_crop, (0, 0, 80, 60))
    os.remove(config.REFERENCE_PATHS[0])
    rois, refs, _ = storage.load_references()
    assert 1 in rois
    assert 1 not in refs


def test_load_references_empty_directory(workdir):
    assert storage.load_references() == ({}, {}, {})


# Inspection logging

def test_save_inspection_writes_image_and_csv(workdir, frame):
    storage.save_inspection("ABC123", frame, {1: (True, 3.2)}, True)
    files = os.listdir(os.path.join(config.LOGS_DIR, "ABC123"))
    assert "log.csv" in files
    assert any(f.endswith("_PASS.jpg") for f in files)


def test_save_inspection_fail_verdict_in_filename(workdir, frame):
    storage.save_inspection("ABC123", frame, {1: (False, 9.9)}, False)
    files = os.listdir(os.path.join(config.LOGS_DIR, "ABC123"))
    assert any(f.endswith("_FAIL.jpg") for f in files)


def test_csv_header_matches_documented_schema(workdir, frame):
    storage.save_inspection("ABC123", frame, {1: (True, 3.2)}, True)
    rows = _read_log("ABC123")
    assert rows[0] == ["timestamp", "barcode", "overall",
                       "slot1", "slot1_diff", "slot2", "slot2_diff"]


def test_csv_header_written_only_once(workdir, frame):
    storage.save_inspection("ABC123", frame, {1: (True, 3.2)}, True)
    storage.save_inspection("ABC123", frame, {1: (False, 9.9)}, False)
    rows = _read_log("ABC123")
    assert len(rows) == 3  # one header, two data rows
    assert rows[2][2] == "FAIL"


def test_missing_slot_logged_as_na(workdir, frame):
    storage.save_inspection("ABC123", frame, {1: (True, 3.2)}, True)
    rows = _read_log("ABC123")
    assert rows[1][3:] == ["PASS", "3.2", "N/A", "N/A"]


def test_both_slots_logged(workdir, frame):
    storage.save_inspection("X", frame, {1: (True, 1.0), 2: (False, 8.5)}, False)
    rows = _read_log("X")
    assert rows[1][3:] == ["PASS", "1.0", "FAIL", "8.5"]


# Barcode sanitization: scanned input becomes a folder name

@pytest.mark.parametrize("barcode", ["ABC123", "part-42", "1.2.3", "a_b"])
def test_safe_folder_name_preserves_normal_barcodes(barcode):
    assert storage.safe_folder_name(barcode) == barcode


@pytest.mark.parametrize("barcode", ["../evil", "..", ".", "a/b", "a\\b"])
def test_safe_folder_name_neutralises_path_traversal(barcode):
    safe = storage.safe_folder_name(barcode)
    assert "/" not in safe and "\\" not in safe
    assert safe.strip(".")  # never a dot-only name that resolves upward


def test_hostile_barcode_stays_inside_logs_dir(workdir, frame):
    storage.save_inspection("../evil", frame, {1: (True, 1.0)}, True)
    logs = os.path.realpath(config.LOGS_DIR)
    for root, _, files in os.walk(logs):
        for f in files:
            assert os.path.realpath(os.path.join(root, f)).startswith(logs)
    assert not os.path.exists(workdir / "evil")
