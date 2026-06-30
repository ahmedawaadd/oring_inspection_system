"""Core image helpers: cropping, preprocessing, comparison, thumbnails."""

import os

import cv2
import numpy as np

from config import BLUR_KERNEL_SIZE, THUMB_W, THUMB_H


def normalise_rect(pt1, pt2):
    """Return (x1, y1, x2, y2) with top-left / bottom-right guaranteed.
    Needed because the user might drag in any direction."""
    x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
    x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
    return x1, y1, x2, y2


def crop(image, roi):
    # NumPy slicing: image[rows, cols]
    x1, y1, x2, y2 = roi
    return image[y1:y2, x1:x2]


def preprocess(image):
    # Convert to greyscale, colour isn't needed for comparison
    # Blur slightly to smooth out camera sensor noise so
    # two photos of the same thing don't look different just from sensor noise.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)


def compare(ref_proc, sample_proc, noise_thresh, diff_thresh):
    # Resize the sample to match the reference if they're slightly different
    # sizes (can happen due to sub-pixel differences in ROI coordinates)
    if ref_proc.shape != sample_proc.shape:
        sample_proc = cv2.resize(sample_proc, (ref_proc.shape[1], ref_proc.shape[0]))

    # Subtract the two images pixel by pixel (identical pixels cancel to zero
    # differences show up as bright spots
    diff = cv2.absdiff(ref_proc, sample_proc)

    # Zero out any differences smaller than noise_thresh
    _, thresh = cv2.threshold(diff, noise_thresh, 255, cv2.THRESH_BINARY)

    # Average the remaining differences. If that average exceeds diff_thresh,
    # there's enough difference to call it a FAIL.
    mean_diff = float(np.mean(thresh))
    return mean_diff < diff_thresh, mean_diff


def load_thumb(path):
    """Load a reference image from disk and shrink it to thumbnail size."""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    # INTER_AREA is the best interpolation method for shrinking images
    return cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
