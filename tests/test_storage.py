"""
test_storage.py

Tests for storage.py: persistence round trips, the CSV contract, and
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


# Calibration persistence

def test_thresholds_default_when_calibration_missing(workdir):
    assert storage.load_thresholds() == (config.DEFAULT_NOISE_THRESHOLD,
                                         config.DEFAULT_DIFF_THRESHOLD / 10.0)


def test_thresholds_round_trip(workdir):
    storage.save_thresholds(37, 6.4)
    assert storage.load_thresholds() == (37, 6.4)


@pytest.mark.parametrize("content", [
    "{not json",
    '{"noise_threshold": "bad", "diff_threshold": 5.0}',
    '{"noise_threshold": 101, "diff_threshold": 5.0}',
    '{"noise_threshold": 30, "diff_threshold": -0.1}',
    '{"noise_threshold": 30}',
])
def test_invalid_calibration_falls_back_to_defaults(workdir, content):
    with open(config.CALIBRATION_PATH, "w") as f:
        f.write(content)
    assert storage.load_thresholds() == (config.DEFAULT_NOISE_THRESHOLD,
                                         config.DEFAULT_DIFF_THRESHOLD / 10.0)


# Inspection logging

def test_save_inspection_writes_image_and_csv(workdir, frame):
    storage.save_inspection(
        "ABC123", frame, {1: (True, 3.2)}, True, 30, 5.0)
    files = os.listdir(os.path.join(config.LOGS_DIR, "ABC123"))
    assert "log.csv" in files
    assert any(f.endswith("_PASS.jpg") for f in files)


def test_save_inspection_fail_verdict_in_filename(workdir, frame):
    storage.save_inspection(
        "ABC123", frame, {1: (False, 9.9)}, False, 30, 5.0)
    files = os.listdir(os.path.join(config.LOGS_DIR, "ABC123"))
    assert any(f.endswith("_FAIL.jpg") for f in files)


def test_csv_header_matches_documented_schema(workdir, frame):
    storage.save_inspection(
        "ABC123", frame, {1: (True, 3.2)}, True, 30, 5.0)
    rows = _read_log("ABC123")
    assert rows[0] == ["timestamp", "barcode", "overall",
                       "slot1", "slot1_diff", "slot2", "slot2_diff",
                       "noise_threshold", "diff_threshold"]


def test_csv_header_written_only_once(workdir, frame):
    storage.save_inspection(
        "ABC123", frame, {1: (True, 3.2)}, True, 30, 5.0)
    storage.save_inspection(
        "ABC123", frame, {1: (False, 9.9)}, False, 35, 6.5)
    rows = _read_log("ABC123")
    assert len(rows) == 3  # one header, two data rows
    assert rows[2][2] == "FAIL"


def test_missing_slot_logged_as_na(workdir, frame):
    storage.save_inspection(
        "ABC123", frame, {1: (True, 3.2)}, True, 30, 5.0)
    rows = _read_log("ABC123")
    assert rows[1][3:7] == ["PASS", "3.2", "N/A", "N/A"]


def test_both_slots_logged(workdir, frame):
    storage.save_inspection(
        "X", frame, {1: (True, 1.0), 2: (False, 8.5)}, False, 30, 5.0)
    rows = _read_log("X")
    assert rows[1][3:7] == ["PASS", "1.0", "FAIL", "8.5"]


def test_sensitivity_values_logged_at_inspection_time(workdir, frame):
    storage.save_inspection(
        "ABC123", frame, {1: (True, 3.2)}, True, 37, 6.4)
    rows = _read_log("ABC123")
    assert rows[1][-2:] == ["37", "6.4"]


def test_legacy_log_is_upgraded_without_inventing_settings(workdir, frame):
    folder = os.path.join(config.LOGS_DIR, "ABC123")
    os.makedirs(folder)
    log_path = os.path.join(folder, "log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(storage.LEGACY_LOG_HEADER)
        writer.writerow([
            "2026-01-01_12-00-00", "ABC123", "PASS",
            "PASS", "1.0", "N/A", "N/A",
        ])

    storage.save_inspection(
        "ABC123", frame, {1: (False, 9.9)}, False, 42, 7.5)

    rows = _read_log("ABC123")
    assert rows[0] == storage.LOG_HEADER
    assert rows[1][-2:] == ["N/A", "N/A"]
    assert rows[2][-2:] == ["42", "7.5"]


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
    storage.save_inspection(
        "../evil", frame, {1: (True, 1.0)}, True, 30, 5.0)
    logs = os.path.realpath(config.LOGS_DIR)
    for root, _, files in os.walk(logs):
        for f in files:
            assert os.path.realpath(os.path.join(root, f)).startswith(logs)
    assert not os.path.exists(workdir / "evil")
