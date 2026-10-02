import pytest
import torch

from src.losses.segmentation import (
    CompositeSegmentationLoss,
    SoftDiceLoss,
    check_binary_targets,
)


def _random_batch(channels=1, seed=0):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(4, channels, 32, 32, generator=generator) * 5
    target = (torch.rand(4, channels, 32, 32, generator=generator) > 0.7).float()
    return logits, target


def test_loss_is_non_negative_and_components_are_bounded():
    logits, target = _random_batch()

    loss, parts = CompositeSegmentationLoss()(logits, target, return_components=True)

    assert loss.item() >= 0.0
    assert 0.0 <= parts["dice"].item() <= 1.0
    assert parts["bce"].item() >= 0.0
    assert parts["boundary"].item() >= 0.0


def test_perfect_prediction_has_near_zero_loss():
    target = torch.zeros(2, 1, 32, 32)
    target[..., 8:20, 8:20] = 1.0
    logits = (target * 2 - 1) * 20.0

    assert CompositeSegmentationLoss()(logits, target).item() < 0.05


def test_raw_brats_labels_are_rejected():
    # Regression: unbinarised labels (0/1/2/4) made the loss go strongly negative.
    raw = torch.tensor([0.0, 1.0, 2.0, 4.0]).view(1, 1, 2, 2)

    with pytest.raises(ValueError):
        check_binary_targets(raw)


def test_binary_targets_pass_the_check():
    check_binary_targets(torch.tensor([0.0, 1.0, 1.0, 0.0]))


def test_multichannel_dice_is_the_mean_of_per_channel_dice():
    logits, target = _random_batch(channels=3, seed=1)
    dice = SoftDiceLoss()

    joint = dice(logits, target)
    per_channel = torch.stack(
        [dice(logits[:, c:c + 1], target[:, c:c + 1]) for c in range(3)]
    ).mean()

    assert torch.allclose(joint, per_channel, atol=1e-6)


def test_loss_is_finite_on_empty_targets():
    logits = torch.randn(2, 1, 16, 16)
    target = torch.zeros(2, 1, 16, 16)

    assert torch.isfinite(CompositeSegmentationLoss()(logits, target))
