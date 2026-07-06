"""Image processing: rectangle handling, preprocessing, and comparison."""

import os

import cv2
import numpy as np

from config import BLUR_KERNEL_SIZE, THUMB_H, THUMB_W


def normalise_rect(pt1, pt2):
    """Return (x1, y1, x2, y2) with guaranteed top-left and bottom-right
    order. Needed because the user might drag in any direction."""
    x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
    x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
    return x1, y1, x2, y2


def crop(image, roi):
    """Cut the ROI out of the image. NumPy slicing is image[rows, cols]."""
    x1, y1, x2, y2 = roi
    return image[y1:y2, x1:x2]


def preprocess(image):
    """Convert to greyscale and blur slightly. Colour is not needed for
    comparison, and the blur smooths sensor noise so two photos of the
    same part don't differ just from noise."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)


def compare(ref_proc, sample_proc, noise_thresh, diff_thresh):
    """Compare a preprocessed sample against a preprocessed reference.
    Returns (passed, mean_diff)."""
    # Resize the sample to match the reference if they're slightly
    # different sizes, which can happen from sub-pixel ROI differences
    if ref_proc.shape != sample_proc.shape:
        sample_proc = cv2.resize(sample_proc, (ref_proc.shape[1], ref_proc.shape[0]))

    # Subtract pixel by pixel: identical pixels cancel to zero,
    # differences show up as bright spots
    diff = cv2.absdiff(ref_proc, sample_proc)

    # Zero out any differences smaller than noise_thresh
    _, thresh = cv2.threshold(diff, noise_thresh, 255, cv2.THRESH_BINARY)

    # Average the remaining differences. Above diff_thresh means FAIL.
    mean_diff = float(np.mean(thresh))
    return mean_diff < diff_thresh, mean_diff


def make_thumb(image):
    """Shrink an image to thumbnail size. INTER_AREA is the best
    interpolation method for shrinking."""
    return cv2.resize(image, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)


def load_thumb(path):
    """Load a reference image from disk as a thumbnail, or None if the
    file is missing or unreadable."""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return make_thumb(img)
