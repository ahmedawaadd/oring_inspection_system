"""Tests for vision.py: the comparison logic is the product, so it gets
the densest coverage, including threshold boundaries."""

import numpy as np
import pytest

import config
import vision


@pytest.mark.parametrize("pt1, pt2, expected", [
    ((10, 20), (30, 40), (10, 20, 30, 40)),  # normal top-left to bottom-right drag
    ((30, 40), (10, 20), (10, 20, 30, 40)),  # reverse drag
    ((30, 20), (10, 40), (10, 20, 30, 40)),  # diagonal drag
    ((5, 5), (5, 5), (5, 5, 5, 5)),          # zero-size click
])
def test_normalise_rect_handles_any_drag_direction(pt1, pt2, expected):
    assert vision.normalise_rect(pt1, pt2) == expected


def test_crop_matches_numpy_slicing(rng):
    img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    cropped = vision.crop(img, (10, 20, 30, 40))
    assert cropped.shape == (20, 20, 3)
    np.testing.assert_array_equal(cropped, img[20:40, 10:30])


def test_preprocess_returns_2d_grayscale(rng):
    img = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    proc = vision.preprocess(img)
    assert proc.shape == (60, 80)
    assert proc.dtype == np.uint8


def test_preprocess_blurs_noise(rng):
    # A noisy image must change under the blur, otherwise sensor noise
    # would leak straight into the comparison
    img = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    proc = vision.preprocess(img)
    gray_only = img.mean(axis=2).astype(np.uint8)
    assert not np.array_equal(proc, gray_only)


def test_compare_identical_images_pass_with_zero_diff():
    proc = np.full((50, 50), 128, dtype=np.uint8)
    passed, diff = vision.compare(proc, proc.copy(), 30, 5.0)
    assert passed
    assert diff == 0.0


def test_compare_opposite_images_fail():
    ref = np.zeros((50, 50), dtype=np.uint8)
    sample = np.full((50, 50), 255, dtype=np.uint8)
    passed, diff = vision.compare(ref, sample, 30, 50.0)
    assert not passed
    assert diff == 255.0


def test_compare_resizes_shape_mismatch():
    # Sub-pixel ROI differences can produce slightly different crop
    # sizes; compare must resize instead of raising
    ref = np.full((50, 50), 128, dtype=np.uint8)
    sample = np.full((48, 52), 128, dtype=np.uint8)
    passed, diff = vision.compare(ref, sample, 30, 5.0)
    assert passed
    assert diff == 0.0


def test_compare_noise_below_threshold_is_ignored():
    # A uniform +20 offset is under the noise threshold of 30, so the
    # binary threshold must zero it out entirely
    ref = np.full((50, 50), 100, dtype=np.uint8)
    sample = np.full((50, 50), 120, dtype=np.uint8)
    passed, diff = vision.compare(ref, sample, 30, 5.0)
    assert passed
    assert diff == 0.0


def test_compare_noise_above_threshold_counts():
    # +40 exceeds the noise threshold of 30, so every pixel survives
    # thresholding at full brightness
    ref = np.full((50, 50), 100, dtype=np.uint8)
    sample = np.full((50, 50), 140, dtype=np.uint8)
    passed, diff = vision.compare(ref, sample, 30, 5.0)
    assert not passed
    assert diff == 255.0


def test_compare_half_different_image_means_half_brightness():
    # Pin the math: half the pixels at 255 after thresholding gives a
    # mean of exactly 127.5
    ref = np.zeros((50, 50), dtype=np.uint8)
    sample = np.zeros((50, 50), dtype=np.uint8)
    sample[:25, :] = 255
    _, diff = vision.compare(ref, sample, 30, 200.0)
    assert diff == 127.5


def test_compare_diff_exactly_at_threshold_fails():
    # The verdict is strict less-than: mean_diff equal to the threshold
    # must FAIL, this is where off-by-one regressions hide
    ref = np.zeros((50, 50), dtype=np.uint8)
    sample = np.full((50, 50), 255, dtype=np.uint8)
    passed, diff = vision.compare(ref, sample, 30, 255.0)
    assert diff == 255.0
    assert not passed


def test_make_thumb_produces_configured_size(rng):
    img = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    thumb = vision.make_thumb(img)
    assert thumb.shape == (config.THUMB_H, config.THUMB_W, 3)


def test_load_thumb_missing_file_returns_none(workdir):
    assert vision.load_thumb("does_not_exist.jpg") is None


def test_load_thumb_unreadable_file_returns_none(workdir):
    (workdir / "corrupt.jpg").write_bytes(b"not an image")
    assert vision.load_thumb("corrupt.jpg") is None


def test_load_thumb_valid_file(workdir, rng):
    import cv2
    img = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    cv2.imwrite("ref.jpg", img)
    thumb = vision.load_thumb("ref.jpg")
    assert thumb.shape == (config.THUMB_H, config.THUMB_W, 3)
