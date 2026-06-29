"""The actual pass/fail logic that ties a captured still to the references.

These functions don't know anything about the camera, the screen or the
disk. You give them images and thresholds, they hand back results, which
keeps the comparison logic separate from all the wiring in the main loop.
"""

from Vision import crop, preprocess, compare


def run_inspection(still, active_refs, rois, noise_thresh, diff_thresh):
    """Compare each active slot's region of the still against its reference.

    Returns (per_slot, sample_crops) where per_slot maps slot -> (passed,
    diff_val) and sample_crops maps slot -> the preprocessed crop, kept so
    the main loop can recompute results live when a slider moves.
    """
    per_slot     = {}
    sample_crops = {}
    for slot, ref_proc in active_refs.items():
        sample_crop        = preprocess(crop(still, rois[slot]))
        sample_crops[slot] = sample_crop
        per_slot[slot]     = compare(ref_proc, sample_crop, noise_thresh, diff_thresh)
    return per_slot, sample_crops


def recompute_live_results(refs, sample_crops, noise_thresh, diff_thresh):
    """Re-run the comparison on the last captured samples so moving a slider
    updates the on-screen result without having to press SPACE again."""
    results = {}
    for slot, ref_proc in refs.items():
        if slot in sample_crops:
            results[slot] = compare(ref_proc, sample_crops[slot],
                                    noise_thresh, diff_thresh)
    return results
