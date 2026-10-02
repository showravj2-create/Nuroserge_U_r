"""Dataset regression tests on a tiny synthetic BraTS-style folder tree."""

import nibabel as nib
import numpy as np

from src.data.brats import Brats2020Dataset, assert_no_patient_overlap

MODALITIES = ["t1", "t1ce", "t2", "flair"]
SHAPE = (24, 24, 10)


def make_brats_root(root, n_patients=10, tumour_slices=(3, 6)):
    rng = np.random.default_rng(0)

    for i in range(n_patients):
        name = f"BraTS20_Training_{i:03d}"
        folder = root / name
        folder.mkdir(parents=True)

        for modality in MODALITIES:
            volume = np.zeros(SHAPE, dtype=np.float32)
            volume[2:22, 2:22, :] = rng.normal(5.0, 2.0, size=(20, 20, SHAPE[2]))
            nib.save(
                nib.Nifti1Image(volume, np.diag([2.0, 2.0, 3.0, 1.0])),
                folder / f"{name}_{modality}.nii.gz",
            )

        seg = np.zeros(SHAPE, dtype=np.float32)
        z0, z1 = tumour_slices
        seg[8:14, 8:14, z0:z1] = 2   # edema
        seg[10:12, 10:12, z0:z1] = 4  # enhancing tumour: raw BraTS label 4
        nib.save(
            nib.Nifti1Image(seg, np.eye(4)),
            folder / f"{name}_seg.nii.gz",
        )

    return root


def build(root, split="train", **kwargs):
    defaults = dict(
        split=split,
        modalities=MODALITIES,
        context=3,
        target_size=(24, 24),
        split_seed=0,
        slice_policy="all",
    )
    defaults.update(kwargs)
    return Brats2020Dataset(root, **defaults)


def test_targets_are_binary_even_though_raw_labels_contain_4(tmp_path):
    root = make_brats_root(tmp_path)
    dataset = build(root)

    for index in range(len(dataset)):
        _, y = dataset[index]
        assert set(np.unique(y.numpy())) <= {0.0, 1.0}


def test_inputs_still_show_healthy_brain_outside_the_tumour(tmp_path):
    # Regression for label leakage: with the old label-masked normalisation every
    # voxel outside the tumour was exactly zero.
    root = make_brats_root(tmp_path)
    dataset = build(root)

    checked = 0
    for index in range(len(dataset)):
        x, y = dataset[index]
        tumour = y.numpy()[0] > 0
        if not tumour.any():
            continue

        outside = ~tumour
        nonzero_fraction = np.count_nonzero(x.numpy()[0][outside]) / outside.sum()
        assert nonzero_fraction > 0.5
        checked += 1

    assert checked > 0


def test_splits_share_no_patients(tmp_path):
    root = make_brats_root(tmp_path)

    train = build(root, "train")
    val = build(root, "val")
    test = build(root, "test")

    assert_no_patient_overlap(train, val, test)
    assert len(train.patients) + len(val.patients) + len(test.patients) == 10


def test_from_config_uses_one_split_for_all_splits(tmp_path):
    # Sweep several seeds: with a mismatched split seed (the original bug) a single
    # seed can be disjoint by luck, but five seeds cannot all get lucky.
    root = make_brats_root(tmp_path)

    for seed in (1, 2, 3, 4, 5):
        cfg = {
            "seed": seed,
            "data": {
                "root": str(root),
                "modalities": MODALITIES,
                "num_slices_context": 3,
                "target_size": [24, 24],
                "slice_policy": "all",
            },
        }

        train = Brats2020Dataset.from_config(cfg, "train")
        val = Brats2020Dataset.from_config(cfg, "val")
        test = Brats2020Dataset.from_config(cfg, "test")

        assert_no_patient_overlap(train, val, test)


def test_train_slice_policy_keeps_tumour_slices_and_subsamples_the_rest(tmp_path):
    root = make_brats_root(tmp_path, tumour_slices=(3, 6))  # 3 tumour slices of 10

    only_tumour = build(root, "train", slice_policy="tumor_plus_negatives", negative_fraction=0.0)
    everything = build(root, "train", slice_policy="all")

    assert len(only_tumour) == 3 * len(only_tumour.patients)
    assert len(everything) == 10 * len(everything.patients)


def test_modality_dropout_never_removes_every_modality(tmp_path):
    root = make_brats_root(tmp_path)
    dataset = build(root, "train", modality_dropout=1.0)

    for index in range(5):
        x, _ = dataset[index]
        x = x.numpy()
        alive = [
            bool(np.any(x[dataset.channel_slice(m)] != 0)) for m in MODALITIES
        ]
        assert sum(alive) == 1


def test_regions_mode_returns_three_binary_channels(tmp_path):
    root = make_brats_root(tmp_path)
    dataset = build(root, target_mode="regions")

    _, y = dataset[4]  # a tumour slice
    y = y.numpy()

    assert y.shape[0] == 3
    assert set(np.unique(y)) <= {0.0, 1.0}
    assert y[0].sum() >= y[1].sum() >= y[2].sum() > 0  # WT >= TC >= ET


def test_load_patient_volume_shapes_and_spacing(tmp_path):
    root = make_brats_root(tmp_path)
    dataset = build(root, "val", target_size=(16, 16))

    volume = dataset.load_patient_volume(0)

    assert tuple(volume["x"].shape) == (SHAPE[2], dataset.in_channels, 16, 16)
    assert tuple(volume["y"].shape) == (SHAPE[2], 1, 16, 16)
    assert volume["label"].shape == SHAPE
    assert volume["spacing"] == (2.0, 2.0, 3.0)
