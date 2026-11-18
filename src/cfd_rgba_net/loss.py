"""Weighted cross-entropy plus soft Dice over the core voxels."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def class_weights(
    fractions: tuple[float, float], w_max: float = 100.0
) -> torch.Tensor:
    """Inverse-frequency class weights, capped against swamping the loss.

    Capping keeps rare positives from swamping the loss.

    Parameters
    ----------
    fractions : tuple[float, float]
        Fraction of core voxels labelled vortex and shear layer in the sampled
        train patches.
    w_max : float
        Cap on a foreground weight.

    Returns
    -------
    torch.Tensor
        Weights for none, vortex and shear layer, shape ``(3,)``; none is 1.
    """
    return torch.tensor([1.0] + [min(w_max, (1 - f) / f) for f in fractions])


class SegmentationLoss(nn.Module):
    """Weighted cross-entropy plus a soft Dice term per foreground class.

    Parameters
    ----------
    weights : torch.Tensor
        Cross-entropy weights for none, vortex and shear layer, shape ``(3,)``;
        see ``class_weights``.
    eps : float
        Smoothing term of the Dice ratio.
    """

    def __init__(self, weights: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        self.register_buffer("weights", weights)
        self.eps = eps

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, core: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the loss.

        Parameters
        ----------
        logits : torch.Tensor
            Class scores ``(N, 3, D, H, W)``, as ``UNet3D`` returns them.
        labels : torch.Tensor
            Vortex and shear-layer masks, ``(N, 2, D, H, W)``, never both set at
            one voxel.
        core : torch.Tensor
            Bool mask of the voxels that count, ``(N, D, H, W)``.

        Returns
        -------
        torch.Tensor
            Scalar, the sum of the three terms.
        torch.Tensor
            The terms for logging, detached, ``(3,)``: cross-entropy, then ``1 -
            Dice`` for vortex and for shear layer.
        """
        target = (
            labels[:, 0] + 2 * labels[:, 1]
        ).long()  # 0 none, 1 vortex, 2 shear layer
        n_core = core.sum().clamp(min=1)
        ce = (
            F.cross_entropy(
                logits, target, weight=self.weights, reduction="none"
            )
            * core
        ).sum() / n_core

        # Dice sums over the whole batch: a single random patch may hold no
        # labelled voxel.
        p = logits.softmax(1) * core[:, None]
        y = F.one_hot(target, 3).movedim(-1, 1) * core[:, None]
        dims = (0, 2, 3, 4)
        dice = (2 * (p * y).sum(dims) + self.eps) / (
            p.sum(dims) + y.sum(dims) + self.eps
        )
        terms = torch.stack([ce, *(1 - dice[1:])])
        return terms.sum(), terms.detach()
