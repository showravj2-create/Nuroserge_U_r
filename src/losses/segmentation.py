import torch
from torch import nn
import torch.nn.functional as F


def check_binary_targets(target):
    """Raise if targets are not in [0, 1].

    BCE and Dice are only well-defined for binary targets. Raw BraTS labels
    (0/1/2/4) make the loss unbounded below, which shows up as a large negative
    training loss.
    """
    low = float(target.min())
    high = float(target.max())
    if low < 0.0 or high > 1.0:
        raise ValueError(
            f"Targets must be in [0, 1] but span [{low}, {high}]. "
            "Convert BraTS labels with src.data.labels.label_to_targets first."
        )


class SoftDiceLoss(nn.Module):
    """Soft Dice loss, computed per sample and per channel, then averaged.

    For a single channel this equals the usual per-sample Dice loss; for
    multi-region targets (WT/TC/ET) every region contributes equally.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits.float())
        y = target.float()

        dims = tuple(range(2, p.ndim))

        intersection = (p * y).sum(dims)
        denominator = p.sum(dims) + y.sum(dims)

        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)

        return (1 - dice).mean()


def boundary_loss(logits, target):
    """Gradient-magnitude boundary loss.

    L1 distance between the spatial-gradient magnitude of the predicted
    probability map and that of the target. It is a smooth, differentiable
    surrogate that sharpens contours; it is NOT the signed-distance boundary
    loss of Kervadec et al.
    """
    p = torch.sigmoid(logits.float())
    y = target.float()

    gx = F.pad(torch.abs(p[..., 1:] - p[..., :-1]), (0, 1, 0, 0))
    gy = F.pad(torch.abs(p[..., 1:, :] - p[..., :-1, :]), (0, 0, 0, 1))

    tx = F.pad(torch.abs(y[..., 1:] - y[..., :-1]), (0, 1, 0, 0))
    ty = F.pad(torch.abs(y[..., 1:, :] - y[..., :-1, :]), (0, 0, 0, 1))

    return F.l1_loss(gx + gy, tx + ty)


class CompositeSegmentationLoss(nn.Module):

    def __init__(
        self,
        dice_weight=1.0,
        bce_weight=0.5,
        boundary_weight=0.2,
    ):
        super().__init__()

        self.dw = dice_weight
        self.bw = bce_weight
        self.boundary_weight = boundary_weight
        self.dice = SoftDiceLoss()

    def forward(self, logits, target, return_components=False):
        logits = logits.float()
        target = target.float()

        dice = self.dice(logits, target)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        boundary = boundary_loss(logits, target)

        total = self.dw * dice + self.bw * bce + self.boundary_weight * boundary

        if return_components:
            return total, {
                "dice": dice.detach(),
                "bce": bce.detach(),
                "boundary": boundary.detach(),
            }

        return total
