import numpy as np
import pytest

from src.data.preprocessing import (
    validate_modalities,
    normalize_volume,
    normalize_modalities,
)


def test_validate_modalities():

    images = {
        "t1n": np.zeros((16, 16, 8)),
        "t1c": np.zeros((16, 16, 8)),
        "t2w": np.zeros((16, 16, 8)),
        "t2f": np.zeros((16, 16, 8)),
    }

    assert validate_modalities(
        images
    )


def test_validate_modalities_rejects_mismatch():

    images = {
        "t1n": np.zeros((16, 16, 8)),
        "t1c": np.zeros((16, 16, 8)),
        "t2w": np.zeros((16, 16, 8)),
        "t2f": np.zeros((16, 15, 8)),
    }

    with pytest.raises(ValueError):

        validate_modalities(
            images
        )


def test_normalize_volume():

    volume = np.zeros(
        (10, 10),
        dtype=np.float32
    )

    volume[2:8, 2:8] = np.arange(
        36,
        dtype=np.float32
    ).reshape(6,6)

    normalized = normalize_volume(
        volume
    )

    valid = normalized[
        volume != 0
    ]

    assert abs(
        valid.mean()
    ) < 1e-5

    assert abs(
        valid.std() - 1.0
    ) < 1e-5

    assert np.all(
        normalized[
            volume == 0
        ] == 0
    )


def test_normalize_modalities():

    images = {
        "t1n": np.random.randn(
            8, 8, 4
        ).astype(np.float32),

        "t1c": np.random.randn(
            8, 8, 4
        ).astype(np.float32),

        "t2w": np.random.randn(
            8, 8, 4
        ).astype(np.float32),

        "t2f": np.random.randn(
            8, 8, 4
        ).astype(np.float32),
    }

    normalized = normalize_modalities(
        images
    )

    assert set(
        normalized.keys()
    ) == set(
        images.keys()
    )

    for name in images:

        assert (
            normalized[name].shape
            == images[name].shape
        )


def test_normalization_does_not_use_or_erase_by_segmentation():
    # Regression for label leakage: the old pipeline normalised inside the
    # ground-truth mask and zeroed everything outside it, so the network only
    # ever saw tumour voxels. Healthy brain must stay visible.
    rng = np.random.default_rng(0)

    volume = np.zeros((16, 16, 8), dtype=np.float32)
    volume[2:14, 2:14, :] = rng.normal(loc=5.0, scale=2.0, size=(12, 12, 8))  # "brain"

    tumour = np.zeros(volume.shape, dtype=bool)
    tumour[6:9, 6:9, 3:5] = True

    normalized = normalize_modalities({"t1": volume})["t1"]

    healthy_brain = (volume != 0) & ~tumour
    assert healthy_brain.sum() > 100
    assert np.abs(normalized[healthy_brain]).mean() > 0.1  # not zeroed
    assert np.count_nonzero(normalized[healthy_brain]) > 0.9 * healthy_brain.sum()


def test_normalization_is_independent_of_the_label():
    rng = np.random.default_rng(1)
    volume = rng.normal(3.0, 1.0, size=(8, 8, 4)).astype(np.float32)

    # normalize_modalities has no way to receive a label unless one is passed as
    # `mask`; the dataset must not do that. Same input -> same output.
    a = normalize_modalities({"t1": volume})["t1"]
    b = normalize_modalities({"t1": volume.copy()})["t1"]

    assert np.array_equal(a, b)

