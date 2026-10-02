"""Segmentation metrics (numpy/scipy only, no torch dependency).

Conventions
-----------
* Masks are boolean-like arrays of any shape (2D slices or 3D volumes).
* ``dice_score`` / ``iou_score`` return 1.0 when both masks are empty (nothing to
  find and nothing predicted is a perfect result), and 0.0 when exactly one is empty.
* Undefined ratios (e.g. precision with no predicted voxels) are NaN, not a
  fudge value, so they can be excluded explicitly with ``nanmean``.
* For patient-level reporting use ``region_metrics`` on the full 3D volume.
  Averaging per-slice Dice over a whole volume is dominated by empty slices and
  is not comparable with published BraTS numbers.
"""

import numpy as np
from scipy import ndimage


def _as_bool(x):
    return np.asarray(x).astype(bool)


def dice_score(pred, target):
    p = _as_bool(pred)
    t = _as_bool(target)

    denominator = p.sum() + t.sum()
    if denominator == 0:
        return 1.0

    return float(2.0 * (p & t).sum() / denominator)


def iou_score(pred, target):
    p = _as_bool(pred)
    t = _as_bool(target)

    union = (p | t).sum()
    if union == 0:
        return 1.0

    return float((p & t).sum() / union)


def confusion_metrics(pred, target):
    """Sensitivity, specificity, precision. Undefined ratios are NaN."""

    p = _as_bool(pred)
    t = _as_bool(target)

    tp = float((p & t).sum())
    tn = float((~p & ~t).sum())
    fp = float((p & ~t).sum())
    fn = float((~p & t).sum())

    def ratio(numerator, denominator):
        return numerator / denominator if denominator > 0 else float("nan")

    return {
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "precision": ratio(tp, tp + fp),
    }


def _surface(mask):
    """Boundary voxels of a binary mask (2D or 3D)."""

    mask = _as_bool(mask)

    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    structure = np.ones((3,) * mask.ndim, dtype=bool)

    eroded = ndimage.binary_erosion(
        mask,
        structure=structure,
        border_value=0,
    )

    return mask ^ eroded


def hd95(pred, target, spacing=None):
    """95th-percentile Hausdorff distance between two 2D or 3D masks.

    ``spacing`` is the voxel size per axis (e.g. mm); default is 1 per axis.
    Returns NaN if either mask is empty (the distance is undefined).
    """

    p = np.squeeze(_as_bool(pred))
    t = np.squeeze(_as_bool(target))

    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch: {p.shape} vs {t.shape}")

    if p.ndim not in (2, 3):
        raise ValueError(
            f"HD95 expects 2D or 3D masks. Got shape {p.shape}"
        )

    if spacing is not None and len(spacing) != p.ndim:
        raise ValueError(
            f"spacing has {len(spacing)} entries but masks are {p.ndim}D"
        )

    if not p.any() or not t.any():
        return float("nan")

    p_surface = _surface(p)
    t_surface = _surface(t)

    distance_to_t = ndimage.distance_transform_edt(~t_surface, sampling=spacing)
    distance_to_p = ndimage.distance_transform_edt(~p_surface, sampling=spacing)

    distances = np.concatenate([
        distance_to_t[p_surface],
        distance_to_p[t_surface],
    ])

    return float(np.percentile(distances, 95))


def region_metrics(pred, target, spacing=None, compute_hd95=True):
    """Per-patient metrics for one region on a full volume.

    When the ground-truth region is empty (e.g. no enhancing tumour in a
    low-grade case) Dice/IoU/HD95 are reported as NaN and ``gt_empty`` is True so
    the patient can be excluded from the mean; ``pred_voxels`` records any false
    positives so they are not silently forgotten.
    """

    p = _as_bool(pred)
    t = _as_bool(target)

    gt_empty = not t.any()

    result = {
        "gt_empty": bool(gt_empty),
        "gt_voxels": int(t.sum()),
        "pred_voxels": int(p.sum()),
        "dice": float("nan") if gt_empty else dice_score(p, t),
        "iou": float("nan") if gt_empty else iou_score(p, t),
        "hd95": float("nan"),
    }

    result.update(confusion_metrics(p, t))

    if compute_hd95 and not gt_empty:
        result["hd95"] = hd95(p, t, spacing=spacing)

    return result


def bootstrap_ci(values, statistic=np.mean, n_boot=2000, seed=42):
    """Percentile bootstrap 95% CI over independent units (patients).

    NaNs are dropped. Returns (point_estimate, (low, high)).
    """

    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]

    if x.size == 0:
        return float("nan"), (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)

    samples = rng.choice(x, size=(n_boot, x.size), replace=True)

    try:
        statistics = statistic(samples, axis=1)
    except TypeError:
        statistics = np.apply_along_axis(statistic, 1, samples)

    return (
        float(statistic(x)),
        (
            float(np.percentile(statistics, 2.5)),
            float(np.percentile(statistics, 97.5)),
        ),
    )
