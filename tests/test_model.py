import torch

from src.models.unet import UNet


def test_forward():

    model = UNet(
        in_channels=4,
        out_channels=1,
        base_channels=8
    )

    x = torch.randn(
        2,
        4,
        64,
        64
    )

    y = model(x)

    assert y.shape == (
        2,
        1,
        64,
        64
    )