from pathlib import Path
import random


def discover_patients(data_root, require_segmentation=False):
    """
    Discover patient directories containing MRI data.

    Supports both the legacy lab layout and the BraTS 2020 training layout:
    NIfTI volumes may be uncompressed (.nii) or compressed (.nii.gz).

    data_root/
        BraTS-001/
            BraTS-001_t1n.nii.gz
            BraTS-001_t1c.nii.gz
            BraTS-001_t2w.nii.gz
            BraTS-001_t2f.nii.gz
            BraTS-001_seg.nii.gz

    or

    data_root/
        Training/
            BraTS20_Training_001/
                BraTS20_Training_001_t1.nii.gz
                BraTS20_Training_001_t1ce.nii.gz
                BraTS20_Training_001_t2.nii.gz
                BraTS20_Training_001_flair.nii.gz
                BraTS20_Training_001_seg.nii.gz
    """

    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_root}"
        )

    nifti_files = sorted(
        list(data_root.rglob("*.nii.gz"))
        + list(data_root.rglob("*.nii"))
    )
    patients = sorted({path.parent for path in nifti_files if path.is_file()})

    if not patients:
        raise ValueError(
            f"No patient directories containing .nii or .nii.gz files "
            f"were found in {data_root}"
        )

    if require_segmentation:
        # Drop folders without ground truth (e.g. the BraTS validation set) BEFORE
        # splitting, otherwise they occupy split slots and silently shrink the
        # train/val/test sets.
        patients = [p for p in patients if has_segmentation(p)]
        if not patients:
            raise ValueError(
                f"Found patient folders under {data_root}, but none contain a "
                "segmentation file (*_seg.nii[.gz])."
            )

    return patients


def has_segmentation(patient_dir):
    patient_dir = Path(patient_dir)
    for suffix in ("seg", "label", "labels"):
        for extension in (".nii.gz", ".nii"):
            if any(patient_dir.glob(f"*_{suffix}{extension}")):
                return True
    return False


def split_patients(
    patients,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """
    Split patients into train/validation/test sets.

    Splitting is performed at patient level.
    """

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must equal 1.0"
        )

    patients = list(patients)

    if len(patients) < 3:
        raise ValueError(
            "At least 3 patients are required for train/val/test splitting."
        )

    rng = random.Random(seed)
    rng.shuffle(patients)

    n = len(patients)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Guarantee that every split gets at least one patient.
    n_train = max(1, n_train)
    n_val = max(1, n_val)

    if n_train + n_val >= n:
        n_val = 1
        n_train = n - 2

    train = patients[:n_train]
    val = patients[n_train:n_train + n_val]
    test = patients[n_train + n_val:]

    return {
        "train": train,
        "val": val,
        "test": test,
    }


def split_manifest(splits):
    """JSON-serialisable {split: [patient folder names]} for saving with a run."""
    return {name: [Path(p).name for p in members] for name, members in splits.items()}


def assert_disjoint(splits):
    """Raise if any patient appears in more than one split (patient-level leakage)."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(map(str, splits[a])) & set(map(str, splits[b]))
            if overlap:
                raise AssertionError(
                    f"Patient leakage: {len(overlap)} patient(s) appear in both "
                    f"'{a}' and '{b}' (e.g. {sorted(overlap)[:3]})."
                )
