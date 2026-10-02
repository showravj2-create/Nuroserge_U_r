"""Convert raw BraTS segmentation labels into binary training targets.

BraTS 2020 stores labels as {0: background, 1: necrotic/non-enhancing core,
2: peritumoral edema, 4: enhancing tumour}. Some re-exported copies of the
dataset relabel 4 -> 3, so label 3 is treated as enhancing tumour too (the
original BraTS 2020 data never uses 3, so this is safe for both variants).

The three standard evaluation regions are:

    WT  whole tumour      labels {1, 2, 3, 4}
    TC  tumour core       labels {1, 3, 4}
    ET  enhancing tumour  labels {3, 4}

Losses and metrics in this repo assume targets are binary {0, 1}. Feeding raw
labels (values up to 4) into BCE/Dice makes the loss unbounded below.
"""

import numpy as np

REGION_NAMES = ("WT", "TC", "ET")

TARGET_MODES = {
    "whole_tumor": ("WT",),
    "regions": ("WT", "TC", "ET"),
}


def target_names(mode="whole_tumor"):
    if mode not in TARGET_MODES:
        raise ValueError(
            f"Unknown target mode {mode!r}. Choose from {sorted(TARGET_MODES)}."
        )
    return TARGET_MODES[mode]


def num_targets(mode="whole_tumor"):
    return len(target_names(mode))


def label_to_regions(label):
    """Return a boolean array of shape (3, *label.shape): WT, TC, ET."""
    lab = np.rint(np.asarray(label)).astype(np.int16)
    whole = lab > 0
    core = np.isin(lab, (1, 3, 4))
    enhancing = np.isin(lab, (3, 4))
    return np.stack([whole, core, enhancing], axis=0)


def label_to_targets(label, mode="whole_tumor"):
    """Return float32 targets of shape (K, *label.shape) with values in {0, 1}."""
    names = target_names(mode)
    regions = label_to_regions(label)
    keep = [REGION_NAMES.index(name) for name in names]
    return regions[keep].astype(np.float32)


def unexpected_label_values(label):
    """Label values outside the known BraTS set (useful as a dataset sanity check)."""
    values = np.unique(np.rint(np.asarray(label)).astype(np.int16))
    return [int(v) for v in values if v not in (0, 1, 2, 3, 4)]
