import numpy as np
import torch

from torch.utils.data import Dataset


def make_sample(
    size=128,
    channels=4,
    seed=42
):
    rng = np.random.default_rng(seed)

    x = rng.normal(
        size=(channels, size, size)
    ).astype("float32")

    y = np.zeros(
        (1, size, size),
        dtype="float32"
    )

    yy, xx = np.mgrid[:size, :size]

    cx, cy = rng.integers(
        int(size * 0.3),
        int(size * 0.7),
        2
    )

    rx, ry = rng.integers(
        int(size * 0.08),
        int(size * 0.18),
        2
    )

    tumor = (
        ((xx - cx) / rx) ** 2
        +
        ((yy - cy) / ry) ** 2
        < 1
    )

    y[0] = tumor.astype("float32")

    # Make the synthetic tumor visible
    # in the input MRI channels.
    x += y * 1.5

    return (
        torch.from_numpy(x),
        torch.from_numpy(y)
    )


class SyntheticSegmentationDataset(Dataset):

    def __init__(
        self,
        size=128,
        channels=4,
        length=100,
        seed=42
    ):
        self.size = size
        self.channels = channels
        self.length = length
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        return make_sample(
            size=self.size,
            channels=self.channels,
            seed=self.seed + index
        )