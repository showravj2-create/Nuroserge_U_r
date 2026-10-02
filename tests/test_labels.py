import numpy as np

from src.data.labels import (
    label_to_regions,
    label_to_targets,
    num_targets,
    unexpected_label_values,
)


def test_targets_are_binary_for_raw_brats_labels():
    # Raw BraTS labels include 4. Feeding these to BCE/Dice caused the negative loss.
    label = np.array([[0, 1, 2, 4]], dtype=np.float32)
    targets = label_to_targets(label, "whole_tumor")

    assert targets.shape == (1, 1, 4)
    assert set(np.unique(targets)) <= {0.0, 1.0}
    assert targets.tolist() == [[[0.0, 1.0, 1.0, 1.0]]]


def test_regions_follow_brats_definitions():
    label = np.array([[0, 1, 2, 4]])
    wt, tc, et = label_to_regions(label)

    assert wt.tolist() == [[False, True, True, True]]   # 1, 2, 4
    assert tc.tolist() == [[False, True, False, True]]  # 1, 4
    assert et.tolist() == [[False, False, False, True]]  # 4


def test_relabelled_enhancing_tumour_3_is_supported():
    label = np.array([[0, 1, 2, 3]])
    _, tc, et = label_to_regions(label)

    assert tc.tolist() == [[False, True, False, True]]
    assert et.tolist() == [[False, False, False, True]]


def test_regions_mode_shape():
    label = np.zeros((5, 6, 3))
    assert label_to_targets(label, "regions").shape == (3, 5, 6, 3)
    assert num_targets("regions") == 3
    assert num_targets("whole_tumor") == 1


def test_unexpected_label_values():
    assert unexpected_label_values(np.array([0, 1, 2, 4])) == []
    assert unexpected_label_values(np.array([0, 1, 7])) == [7]
