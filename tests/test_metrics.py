import numpy as np

from src.metrics.segmentation import (
    dice_score,
    iou_score,
    hd95,
)


def test_perfect():

    x = np.array([
        [1, 0],
        [0, 1]
    ])

    assert dice_score(x, x) == 1.0
    assert iou_score(x, x) == 1.0


def test_disjoint():

    x = np.array([
        [1, 0],
        [0, 0]
    ])

    y = np.array([
        [0, 1],
        [0, 0]
    ])

    assert dice_score(x, y) < 1e-5


def test_hd95_identical():

    x = np.zeros(
        (32, 32),
        dtype=np.uint8
    )

    x[10:20, 10:20] = 1

    assert hd95(x, x) == 0.0


def test_hd95_shifted():

    x = np.zeros(
        (32, 32),
        dtype=np.uint8
    )

    y = np.zeros(
        (32, 32),
        dtype=np.uint8
    )

    x[10:20, 10:20] = 1
    y[12:22, 10:20] = 1

    distance = hd95(x, y)

    assert distance > 0.0


def test_both_empty_is_perfect_and_one_empty_is_zero():
    empty = np.zeros((8, 8), dtype=bool)
    full = np.ones((8, 8), dtype=bool)

    assert dice_score(empty, empty) == 1.0
    assert dice_score(empty, full) == 0.0
    assert dice_score(full, empty) == 0.0


def test_all_zero_prediction_scores_zero_on_tumour():
    # Regression: the old metric let empty predictions score ~1.0 on empty slices,
    # which inflated slice-averaged Dice. On a real tumour it must be 0.
    target = np.zeros((16, 16), dtype=bool)
    target[4:10, 4:10] = True

    assert dice_score(np.zeros_like(target), target) == 0.0


def test_dice_known_value():
    a = np.zeros((10,), dtype=bool)
    b = np.zeros((10,), dtype=bool)
    a[:4] = True   # 4 voxels
    b[2:8] = True  # 6 voxels, overlap 2

    assert abs(dice_score(a, b) - 2 * 2 / (4 + 6)) < 1e-12
    assert abs(iou_score(a, b) - 2 / 8) < 1e-12


def test_confusion_undefined_ratios_are_nan():
    from src.metrics.segmentation import confusion_metrics

    empty = np.zeros((4, 4), dtype=bool)
    result = confusion_metrics(empty, empty)

    assert np.isnan(result["sensitivity"])
    assert np.isnan(result["precision"])
    assert result["specificity"] == 1.0


def test_hd95_3d_respects_voxel_spacing():
    a = np.zeros((20, 20, 20), dtype=bool)
    b = np.zeros((20, 20, 20), dtype=bool)
    a[5:10, 5:10, 5:10] = True
    b[5:10, 5:10, 8:13] = True  # shifted 3 voxels along the last axis

    unit = hd95(a, b)
    scaled = hd95(a, b, spacing=(1.0, 1.0, 2.0))

    assert unit > 0
    assert scaled > unit  # 2 mm slices make the same voxel shift a larger distance


def test_hd95_undefined_when_a_mask_is_empty():
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    b[2:4, 2:4] = True

    assert np.isnan(hd95(a, b))


def test_region_metrics_flags_empty_ground_truth():
    from src.metrics.segmentation import region_metrics

    pred = np.zeros((6, 6, 6), dtype=bool)
    pred[1:3, 1:3, 1:3] = True  # false positive
    gt = np.zeros((6, 6, 6), dtype=bool)

    result = region_metrics(pred, gt, compute_hd95=False)

    assert result["gt_empty"] is True
    assert np.isnan(result["dice"])
    assert result["pred_voxels"] == 8  # false positives stay visible


def test_bootstrap_ci_ignores_nan_and_brackets_mean():
    from src.metrics.segmentation import bootstrap_ci

    values = [0.8, 0.9, float("nan"), 0.85, 0.95, 0.7]
    mean, (low, high) = bootstrap_ci(values, n_boot=500)

    assert abs(mean - np.nanmean(values)) < 1e-12
    assert low <= mean <= high

