import numpy as np


def validate_modalities(
    images
):
    """
    Validate that all MRI modalities have
    the same spatial dimensions.
    """

    if not images:
        raise ValueError(
            "No MRI modalities provided."
        )

    shapes = {
        name: image.shape
        for name, image in images.items()
    }

    unique_shapes = set(
        shapes.values()
    )

    if len(unique_shapes) != 1:

        raise ValueError(
            "MRI modalities have "
            "different shapes: "
            f"{shapes}"
        )

    return True


def normalize_volume(
    volume,
    mask=None,
    eps=1e-8
):
    """
    Robust z-score normalization.

    If a mask is supplied, statistics are
    calculated only inside the mask.
    """

    volume = np.asarray(
        volume,
        dtype=np.float32
    )

    if mask is None:

        valid = volume != 0

    else:

        valid = (
            np.asarray(mask).astype(bool)
        )

    values = volume[valid]

    if values.size == 0:
        return np.zeros_like(
            volume,
            dtype=np.float32
        )

    mean = values.mean()
    std = values.std()

    if std < eps:

        normalized = (
            volume - mean
        )

    else:

        normalized = (
            volume - mean
        ) / (
            std + eps
        )

    # Background should remain zero.
    normalized[~valid] = 0

    return normalized.astype(
        np.float32
    )


def normalize_modalities(
    images,
    mask=None
):
    """
    Normalize every MRI modality independently.

    IMPORTANT: ``mask`` must be an *input-derived* brain mask, never a mask
    computed from the ground-truth segmentation. Normalizing inside the label
    mask and zeroing everything outside it hands the answer to the network
    (the model would only ever see tumour voxels). When ``mask`` is None each
    modality uses its own non-zero voxels, which is the standard BraTS
    brain-mask convention.
    """

    validate_modalities(
        images
    )

    return {
        name: normalize_volume(
            image,
            mask=mask
        )
        for name, image in images.items()
    }