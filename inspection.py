"""Domain logic: loading references from disk, capturing new references,
and running a comparison pass over the active reference slots."""

import os

import cv2
import numpy as np

from config import REFERENCE_PATHS, ROI_PATHS, THUMB_W, THUMB_H
from camera import capture_still
from image_ops import crop, preprocess, compare, load_thumb


def load_references():
    """Reload saved ROIs and reference images from disk so the operator
    doesn't have to redraw regions and recapture references every session.

    Returns (rois, refs, thumbs) keyed by slot (1, 2)."""
    rois, refs, thumbs = {}, {}, {}
    for slot in [1, 2]:
        roi_path = ROI_PATHS[slot - 1]
        ref_path = REFERENCE_PATHS[slot - 1]
        if os.path.exists(roi_path):
            rois[slot] = tuple(np.load(roi_path).tolist())
        if os.path.exists(ref_path):
            refs[slot]   = preprocess(cv2.imread(ref_path))
            thumbs[slot] = load_thumb(ref_path)
    return rois, refs, thumbs


def capture_reference(cam, slot, roi):
    """Capture a high-resolution still, crop it to ``roi``, persist the
    reference image and ROI coordinates to disk, and return the preprocessed
    reference array and its thumbnail for the given slot."""
    still    = capture_still(cam)
    ref_crop = crop(still, roi)
    cv2.imwrite(REFERENCE_PATHS[slot - 1], ref_crop)    # persist to disk
    np.save(ROI_PATHS[slot - 1], np.array(roi))         # persist ROI coords
    ref_proc = preprocess(ref_crop)
    # Build thumbnail directly from the in-memory crop. no need
    # to read the file we just saved back off disk
    thumb = cv2.resize(ref_crop, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
    return ref_proc, thumb


def run_inspection(still, rois, active_refs, noise_thresh, diff_thresh):
    """Compare each active slot's reference against the matching crop of
    ``still``. Returns (per_slot, sample_crops), both keyed by slot, where
    per_slot[slot] = (passed, diff_val) and sample_crops[slot] is the
    preprocessed crop (kept for live slider recomputation)."""
    per_slot     = {}
    sample_crops = {}
    for slot, ref_proc in active_refs.items():
        sample_crop        = preprocess(crop(still, rois[slot]))
        sample_crops[slot] = sample_crop
        per_slot[slot]     = compare(ref_proc, sample_crop, noise_thresh, diff_thresh)
    return per_slot, sample_crops
