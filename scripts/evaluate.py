"""Patient-level (3D) evaluation on the held-out split.

Example:
    python scripts/evaluate.py --checkpoint runs/brats_2p5d_unet/best.pt --split test
    python scripts/evaluate.py --checkpoint runs/brats_2p5d_unet/best.pt --split test --drop-modality flair
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.brats import Brats2020Dataset
from src.evaluation import evaluate_patients, format_summary
from src.losses.segmentation import CompositeSegmentationLoss
from src.models.unet import UNet
from src.utils.repro import git_state, write_json


def data_signature(cfg):
    """Everything that determines which patients land in which split, and the inputs."""
    d = cfg["data"]
    return {
        "split_seed": d.get("split_seed", cfg["seed"]),
        "fractions": (
            d.get("train_fraction", 0.70),
            d.get("val_fraction", 0.15),
            d.get("test_fraction", 0.15),
        ),
        "target_mode": d.get("target_mode", "whole_tumor"),
        "modalities": tuple(d.get("modalities", ["t1", "t1ce", "t2", "flair"])),
        "context": d.get("num_slices_context", 3),
        "target_size": tuple(d.get("target_size", [240, 240])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="Default: the config stored in the checkpoint")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--data-root", default=None, help="Overrides data.root")
    parser.add_argument("--no-hd95", action="store_true", help="Skip HD95 (slow on full volumes)")
    parser.add_argument(
        "--drop-modality",
        action="append",
        default=[],
        help="Zero out a modality at test time (repeatable), e.g. --drop-modality flair",
    )
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # Checkpoints are produced by scripts/train.py; only load files you created.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    cfg = checkpoint["config"]
    if args.config:
        with open(args.config) as f:
            override = yaml.safe_load(f)
        if data_signature(override) != data_signature(cfg):
            raise ValueError(
                "The --config data settings differ from the ones used for training "
                "(split seed/fractions, modalities, context or target mode). Evaluating "
                "with different settings would mix patients across splits. Omit --config "
                "to use the checkpoint's own config."
            )
        cfg = override

    if cfg["data"].get("synthetic", False):
        raise ValueError("evaluate.py supports BraTS data only.")

    device_name = cfg["training"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}")

    dataset = Brats2020Dataset.from_config(cfg, args.split, args.data_root)
    print(f"{args.split} split: {len(dataset.patients)} patients")

    model = UNet(**cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    loss_fn = CompositeSegmentationLoss(**cfg["loss"])
    amp_enabled = bool(cfg["training"].get("amp", False) and device.type == "cuda")

    per_patient, summary = evaluate_patients(
        model,
        dataset,
        loss_fn,
        device,
        amp=amp_enabled,
        compute_hd95=not args.no_hd95,
        drop_modalities=args.drop_modality,
        max_patients=args.max_patients,
        batch_size=args.batch_size,
    )

    print()
    print(f"Patient-level 3D evaluation ({args.split} split)")
    print("-" * 60)
    for line in format_summary(summary):
        print(line)
    for name, region in summary["regions"].items():
        print(
            f"{name}: IoU {region['iou_mean']:.4f} | sens {region['sensitivity_mean']:.4f} | "
            f"prec {region['precision_mean']:.4f} | HD95 {region['hd95_mean']:.2f} mm"
        )

    tag = args.split + "".join(f"_drop-{m}" for m in args.drop_modality)
    out_dir = Path(args.checkpoint).resolve().parent

    write_json(
        out_dir / f"eval_{tag}.json",
        {
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint["epoch"],
            "dropped_modalities": args.drop_modality,
            "git": git_state(),
            "summary": summary,
        },
    )

    with open(out_dir / f"eval_{tag}_per_patient.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_patient[0].keys()))
        writer.writeheader()
        writer.writerows(per_patient)

    print()
    print(f"Saved: {out_dir / f'eval_{tag}.json'}")
    print(f"       {out_dir / f'eval_{tag}_per_patient.csv'}")


if __name__ == "__main__":
    main()
