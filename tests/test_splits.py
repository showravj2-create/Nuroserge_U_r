from pathlib import Path

from src.data.splits import discover_patients, split_patients


def test_discover_patients(tmp_path):
    data_root = tmp_path / "dataset"
    data_root.mkdir()

    for index, patient_id in enumerate(["BraTS-001", "BraTS-002", "BraTS-003"]):
        patient_dir = data_root / patient_id
        patient_dir.mkdir()

        extension = ".nii" if index == 0 else ".nii.gz"
        (patient_dir / f"{patient_id}_t1n{extension}").touch()

    patients = discover_patients(data_root)

    assert len(patients) == 3
    assert all(isinstance(p, Path) for p in patients)


def test_patient_split():
    patients = [
        Path(f"BraTS-{i:03d}")
        for i in range(20)
    ]

    splits = split_patients(
        patients,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    assert len(splits["train"]) == 14
    assert len(splits["val"]) == 3
    assert len(splits["test"]) == 3

    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])

    # Critical leakage check.
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_split_is_reproducible():
    patients = [
        Path(f"BraTS-{i:03d}")
        for i in range(20)
    ]

    split_a = split_patients(patients, seed=42)
    split_b = split_patients(patients, seed=42)

    assert split_a == split_b


def test_discover_patients_can_require_segmentation(tmp_path):
    from src.data.splits import discover_patients

    with_seg = tmp_path / "BraTS-001"
    without_seg = tmp_path / "BraTS-002"
    with_seg.mkdir()
    without_seg.mkdir()
    (with_seg / "BraTS-001_t1.nii.gz").touch()
    (with_seg / "BraTS-001_seg.nii.gz").touch()
    (without_seg / "BraTS-002_t1.nii.gz").touch()

    assert len(discover_patients(tmp_path)) == 2
    assert [p.name for p in discover_patients(tmp_path, require_segmentation=True)] == ["BraTS-001"]


def test_assert_disjoint_detects_overlap():
    import pytest
    from src.data.splits import assert_disjoint

    assert_disjoint({"train": ["a", "b"], "val": ["c"]})

    with pytest.raises(AssertionError):
        assert_disjoint({"train": ["a", "b"], "val": ["b"]})


def test_different_seeds_produce_overlapping_splits():
    # Documents the original bug: train used seed 42 and val used seed 43, so the
    # two "splits" were drawn from different shuffles and shared patients.
    patients = [Path(f"BraTS-{i:03d}") for i in range(100)]

    train = set(split_patients(patients, seed=42)["train"])
    val_other_seed = set(split_patients(patients, seed=43)["val"])

    assert len(train & val_other_seed) > 0

