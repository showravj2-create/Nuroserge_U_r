#!/usr/bin/env python3
"""Validate a BraTS folder before training.

Checks that every patient has all modalities and a segmentation, that label
values are the expected BraTS set, and prints the resulting split sizes.
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.labels import unexpected_label_values
from src.data.splits import discover_patients, has_segmentation, split_patients

REQUIRED = ["t1", "t1ce", "t2", "flair"]


def missing_modalities(patient):
    names = [p.name for p in patient.iterdir() if p.is_file()]
    return [
        modality
        for modality in REQUIRED
        if not any(n.endswith((f"_{modality}.nii", f"_{modality}.nii.gz")) for n in names)
    ]


def main():
    parser = argparse.ArgumentParser(description="Validate a BraTS 2020 dataset before training.")
    parser.add_argument("--data-root", required=True, help="Folder containing the patient folders")
    parser.add_argument("--label-samples", type=int, default=5, help="Patients whose label values are inspected")
    args = parser.parse_args()

    root = Path(args.data_root)
    try:
        all_patients = discover_patients(root)
    except Exception as exc:
        print(f"Dataset check failed: {exc}")
        raise SystemExit(1)

    with_seg = [p for p in all_patients if has_segmentation(p)]
    print(f"Dataset root:              {root}")
    print(f"Patient folders found:     {len(all_patients)}")
    print(f"  with segmentation:       {len(with_seg)}  (only these are used for train/val/test)")
    print(f"  without segmentation:    {len(all_patients) - len(with_seg)}  (e.g. BraTS validation set; ignored)")

    problems = []

    for patient in with_seg:
        missing = missing_modalities(patient)
        if missing:
            problems.append(f"{patient.name}: missing {missing}")

    for patient in with_seg[: args.label_samples]:
        seg_path = next(
            path
            for suffix in ("seg", "label", "labels")
            for ext in (".nii.gz", ".nii")
            for path in patient.glob(f"*_{suffix}{ext}")
        )
        label = np.rint(nib.load(str(seg_path)).get_fdata(dtype=np.float32))
        values = sorted(int(v) for v in np.unique(label))
        print(f"  {patient.name}: label values {values}")
        bad = unexpected_label_values(label)
        if bad:
            problems.append(f"{patient.name}: unexpected label values {bad}")

    if len(with_seg) >= 3:
        sizes = {k: len(v) for k, v in split_patients(with_seg, seed=42).items()}
        print(f"Split sizes (seed 42):     {sizes}")

    if problems:
        print("\nProblems found:")
        for problem in problems[:20]:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("\nBraTS dataset layout looks valid for training.")


if __name__ == "__main__":
    main()
