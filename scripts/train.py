import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.brats import (
    Brats2020Dataset,
    PatientGroupedSampler,
    assert_no_patient_overlap,
)
from src.data.synthetic import SyntheticSegmentationDataset
from src.evaluation import (
    evaluate_patients,
    evaluate_slices,
    format_summary,
    selection_dice,
)
from src.losses.segmentation import CompositeSegmentationLoss, check_binary_targets
from src.models.unet import UNet
from src.utils.repro import environment_info, git_state, seed_everything, write_json


def build_datasets(cfg, data_root):
    data_cfg = cfg["data"]

    if data_cfg.get("synthetic", False):
        train_dataset = SyntheticSegmentationDataset(
            size=data_cfg["image_size"],
            channels=cfg["model"]["in_channels"],
            length=data_cfg["train_samples"],
            seed=cfg["seed"],
        )
        val_dataset = SyntheticSegmentationDataset(
            size=data_cfg["image_size"],
            channels=cfg["model"]["in_channels"],
            length=data_cfg["val_samples"],
            seed=cfg["seed"] + 10000,
        )
        return train_dataset, val_dataset

    # Both datasets are built through from_config, so they share the split seed
    # and fractions. The overlap check makes patient leakage a hard error.
    train_dataset = Brats2020Dataset.from_config(cfg, "train", data_root)
    val_dataset = Brats2020Dataset.from_config(cfg, "val", data_root)
    assert_no_patient_overlap(train_dataset, val_dataset)

    return train_dataset, val_dataset


def check_model_matches_data(cfg, dataset):
    if not isinstance(dataset, Brats2020Dataset):
        return

    model_cfg = cfg["model"]

    if model_cfg["in_channels"] != dataset.in_channels:
        raise ValueError(
            f"model.in_channels={model_cfg['in_channels']} but the dataset produces "
            f"{dataset.in_channels} channels "
            f"({len(dataset.modality_names)} modalities x {dataset.slices_per_modality} slices)."
        )

    if model_cfg["out_channels"] != dataset.num_targets:
        raise ValueError(
            f"model.out_channels={model_cfg['out_channels']} but data.target_mode="
            f"'{dataset.target_mode}' has {dataset.num_targets} target channel(s)."
        )


def check_first_batch(x, y):
    if not torch.isfinite(x).all():
        raise ValueError("Input batch contains NaN or Inf.")
    check_binary_targets(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None, help="Overrides data.root in the config")
    parser.add_argument("--output-dir", default=None, help="Default: runs/<experiment>")
    parser.add_argument("--epochs", type=int, default=None, help="Overrides training.epochs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    epochs = train_cfg["epochs"]

    seed_everything(cfg["seed"])

    device_name = train_cfg["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}")

    out_dir = Path(args.output_dir or cfg.get("output_dir") or f"runs/{cfg.get('experiment', 'run')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # DATA
    # --------------------------------------------------

    train_dataset, val_dataset = build_datasets(cfg, args.data_root)
    check_model_matches_data(cfg, train_dataset)
    is_brats = isinstance(train_dataset, Brats2020Dataset)

    train_sampler = PatientGroupedSampler(train_dataset, seed=cfg["seed"]) if is_brats else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )

    val_loader = None
    if not is_brats:
        val_loader = DataLoader(val_dataset, batch_size=train_cfg["batch_size"], shuffle=False)

    if is_brats:
        print(
            f"Patients  train={len(train_dataset.patients)}  val={len(val_dataset.patients)}  "
            f"(no overlap verified)"
        )
    print(f"Train slices per epoch: {len(train_dataset)}")

    # --------------------------------------------------
    # MODEL / LOSS / OPTIMISER
    # --------------------------------------------------

    model = UNet(**cfg["model"]).to(device)
    loss_fn = CompositeSegmentationLoss(**cfg["loss"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    amp_enabled = bool(train_cfg.get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    grad_clip = train_cfg.get("grad_clip")

    # --------------------------------------------------
    # RUN METADATA (reproducibility)
    # --------------------------------------------------

    git = git_state()
    write_json(
        out_dir / "run.json",
        {
            "config": cfg,
            "git": git,
            "environment": environment_info(),
            "n_train_slices": len(train_dataset),
            "splits": train_dataset.split_manifest if is_brats else None,
        },
    )
    if git["dirty"]:
        print("Warning: working tree has uncommitted changes; the recorded commit hash does not fully describe this run.")

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    best_score = float("-inf")
    history = []

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        started = time.time()

        totals = {"total": 0.0, "dice": 0.0, "bce": 0.0, "boundary": 0.0}
        seen = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )

        for step, (x, y) in enumerate(progress):
            x = x.to(device)
            y = y.to(device)

            if epoch == 0 and step == 0:
                check_first_batch(x, y)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(x)

            # The loss runs in float32 outside autocast.
            loss, parts = loss_fn(logits, y, return_components=True)

            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_size = x.size(0)
            seen += batch_size
            totals["total"] += loss.item() * batch_size
            for name, value in parts.items():
                totals[name] += value.item() * batch_size

            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_means = {name: value / seen for name, value in totals.items()}

        # ---------------- validation ----------------
        if is_brats:
            _, summary = evaluate_patients(
                model,
                val_dataset,
                loss_fn,
                device,
                amp=amp_enabled,
                compute_hd95=False,
                max_patients=train_cfg.get("val_max_patients"),
            )
            val_loss = summary["loss"]
            val_score = selection_dice(summary)
        else:
            summary = None
            val_loss, val_score = evaluate_slices(model, val_loader, loss_fn, device, amp_enabled)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss {train_means['total']:.4f} "
            f"(dice {train_means['dice']:.3f}, bce {train_means['bce']:.3f}, "
            f"boundary {train_means['boundary']:.3f}) | "
            f"val loss {val_loss:.4f} | val Dice {val_score:.4f} | "
            f"{(time.time() - started) / 60:.1f} min"
        )
        if summary is not None:
            for line in format_summary(summary)[1:]:
                print(f"    {line}")

        history.append(
            {
                "epoch": epoch + 1,
                "train": train_means,
                "val_loss": val_loss,
                "val_dice": val_score,
                "val_summary": summary,
            }
        )
        write_json(out_dir / "history.json", history)

        checkpoint = {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": epoch + 1,
            "val_dice": val_score,
            "git": git,
        }
        torch.save(checkpoint, out_dir / "last.pt")

        if val_score > best_score:
            best_score = val_score
            torch.save(checkpoint, out_dir / "best.pt")
            print(f"  New best checkpoint (val Dice={val_score:.4f}) -> {out_dir / 'best.pt'}")

    print()
    print(f"Best validation Dice: {best_score:.4f}")
    print(f"Outputs in: {out_dir}")
    if is_brats:
        print(
            "Report final numbers on the held-out test split with:\n"
            f"  python scripts/evaluate.py --checkpoint {out_dir / 'best.pt'} --split test"
        )


if __name__ == "__main__":
    main()
