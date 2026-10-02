"""Evaluation engine.

``evaluate_patients`` is the reference evaluation: every slice of a patient is
predicted, the probability maps are resized back to the native image grid, and
metrics are computed on the full 3D volume per region. Results are then
aggregated across patients with bootstrap confidence intervals.

``evaluate_slices`` is only used for the synthetic smoke test.
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.data.labels import label_to_targets, target_names
from src.metrics.segmentation import bootstrap_ci, dice_score, region_metrics


def _nanmean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))


@torch.no_grad()
def predict_logits(model, x, device, amp=False, batch_size=32):
    """Run the model over (D, C, h, w) slices in chunks and return CPU logits."""
    model.eval()
    chunks = []
    for start in range(0, x.shape[0], batch_size):
        batch = x[start:start + batch_size].to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(batch)
        chunks.append(logits.float().cpu())
    return torch.cat(chunks, dim=0)


def summarize(per_patient, names, losses=None):
    summary = {"n_patients": len(per_patient), "regions": {}}

    if losses:
        summary["loss"] = float(np.mean(losses))

    for name in names:

        def column(key):
            return np.array([row[f"{name}_{key}"] for row in per_patient], dtype=float)

        dice = column("dice")
        dice_mean, dice_ci = bootstrap_ci(dice)

        gt_empty = column("gt_empty").astype(bool)
        pred_voxels = column("pred_voxels")

        hd95_values = column("hd95")
        hd95_mean, hd95_ci = bootstrap_ci(hd95_values)

        summary["regions"][name] = {
            "dice_mean": dice_mean,
            "dice_ci95": list(dice_ci),
            "dice_median": float(np.nanmedian(dice)) if (~np.isnan(dice)).any() else float("nan"),
            "iou_mean": _nanmean(column("iou")),
            "sensitivity_mean": _nanmean(column("sensitivity")),
            "specificity_mean": _nanmean(column("specificity")),
            "precision_mean": _nanmean(column("precision")),
            "hd95_mean": hd95_mean,
            "hd95_ci95": list(hd95_ci),
            "n_evaluated": int((~gt_empty).sum()),
            "n_gt_empty": int(gt_empty.sum()),
            "n_false_positive_when_gt_empty": int((gt_empty & (pred_voxels > 0)).sum()),
        }

    return summary


def selection_dice(summary):
    """Mean of the per-region mean Dice (BraTS-style); used to pick checkpoints."""
    return _nanmean([r["dice_mean"] for r in summary["regions"].values()])


@torch.no_grad()
def evaluate_patients(
    model,
    dataset,
    loss_fn,
    device,
    amp=False,
    compute_hd95=False,
    drop_modalities=None,
    max_patients=None,
    batch_size=32,
    progress=True,
):
    names = target_names(dataset.target_mode)

    indices = list(range(len(dataset.patients)))
    if max_patients:
        indices = indices[:max_patients]

    iterator = (
        tqdm(indices, desc=f"Evaluating {dataset.split}", unit="patient", leave=False)
        if progress
        else indices
    )

    per_patient = []
    losses = []

    for patient_index in iterator:
        volume = dataset.load_patient_volume(patient_index)

        x = volume["x"]
        for modality in drop_modalities or []:
            x[:, dataset.channel_slice(modality)] = 0.0

        logits = predict_logits(model, x, device, amp=amp, batch_size=batch_size)
        losses.append(float(loss_fn(logits, volume["y"])))

        label = volume["label"]
        height, width = label.shape[:2]

        probabilities = torch.sigmoid(logits)  # (D, K, h, w)
        if probabilities.shape[-2:] != (height, width):
            probabilities = F.interpolate(
                probabilities,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        prediction = (probabilities > 0.5).numpy().transpose(1, 2, 3, 0)  # (K, H, W, D)
        truth = label_to_targets(label, dataset.target_mode) > 0.5

        row = {"patient": volume["patient"]}
        for k, name in enumerate(names):
            metrics = region_metrics(
                prediction[k],
                truth[k],
                spacing=volume["spacing"],
                compute_hd95=compute_hd95,
            )
            for key, value in metrics.items():
                row[f"{name}_{key}"] = value

        per_patient.append(row)

    return per_patient, summarize(per_patient, names, losses)


@torch.no_grad()
def evaluate_slices(model, loader, loss_fn, device, amp=False):
    """Slice-level loss/Dice for the synthetic smoke test only."""
    model.eval()

    total_loss = 0.0
    count = 0
    dice_values = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)

        total_loss += float(loss_fn(logits, y)) * x.size(0)
        count += x.size(0)

        predictions = (torch.sigmoid(logits.float()) > 0.5).cpu().numpy()
        targets = y.cpu().numpy()
        for prediction, target in zip(predictions, targets):
            dice_values.append(dice_score(prediction, target))

    return total_loss / count, float(np.mean(dice_values))


def format_summary(summary):
    lines = []
    if "loss" in summary:
        lines.append(f"loss {summary['loss']:.4f}")
    for name, r in summary["regions"].items():
        low, high = r["dice_ci95"]
        lines.append(
            f"{name}: Dice {r['dice_mean']:.4f} [{low:.4f}, {high:.4f}] "
            f"(n={r['n_evaluated']}, GT-empty={r['n_gt_empty']}, "
            f"FP-on-empty={r['n_false_positive_when_gt_empty']})"
        )
    return lines
