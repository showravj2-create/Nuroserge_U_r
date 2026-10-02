from pathlib import Path
import argparse
import sys

import nibabel as nib
import numpy as np

# Add repository root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import discover_patients


MODALITIES = ["t1n", "t1c", "t2w", "t2f", "seg"]


def inspect_patient(patient_dir):
    print(f"\nPatient: {patient_dir.name}")

    for modality in MODALITIES:
        matches = sorted(
            list(patient_dir.glob(f"*_{modality}.nii.gz"))
            + list(patient_dir.glob(f"*_{modality}.nii"))
        )

        if not matches:
            print(f"  {modality:4s}: MISSING")
            continue

        path = matches[0]
        image = nib.load(str(path))

        data = image.get_fdata(dtype=np.float32)

        print(
            f"  {modality:4s}: "
            f"shape={data.shape}, "
            f"spacing={image.header.get_zooms()[:3]}, "
            f"min={data.min():.3f}, "
            f"max={data.max():.3f}, "
            f"mean={data.mean():.3f}, "
            f"nonzero={np.count_nonzero(data)}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory containing patient folders",
    )

    parser.add_argument(
        "--max-patients",
        type=int,
        default=5,
        help="Maximum number of patients to inspect",
    )

    args = parser.parse_args()

    patients = discover_patients(args.data_root)

    print("=" * 70)
    print("MRI DATASET INSPECTION")
    print("=" * 70)

    print(f"Dataset root : {args.data_root}")
    print(f"Patients     : {len(patients)}")
    print(f"Inspecting   : {min(len(patients), args.max_patients)}")

    for patient in patients[:args.max_patients]:
        inspect_patient(patient)


if __name__ == "__main__":
    main()