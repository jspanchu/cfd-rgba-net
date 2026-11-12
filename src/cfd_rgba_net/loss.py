"""Weighted cross-entropy plus soft Dice over the core voxels, with deep supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

DEEP_WEIGHTS = (1.0, 0.5, 0.25)  # full, half and quarter resolution


def class_weights(fractions: tuple[float, float], w_max: float = 100.0) -> torch.Tensor:
    """Inverse-frequency class weights, capped so that rare positives do not swamp the loss.

    Parameters
    ----------
    fractions : tuple[float, float]
        Fraction of core voxels labelled vortex and shear layer in the sampled train patches.
    w_max : float
        Cap on a foreground weight.

    Returns
    -------
    torch.Tensor
        Weights for none, vortex and shear layer, shape ``(3,)``; none is 1.
    """
    return torch.tensor([1.0] + [min(w_max, (1 - f) / f) for f in fractions])


class DeepSupervisionLoss(nn.Module):
    """Weighted cross-entropy plus a soft Dice term per foreground class, summed over resolutions.

    Parameters
    ----------
    weights : torch.Tensor
        Cross-entropy weights for none, vortex and shear layer, shape ``(3,)``; see ``class_weights``.
    eps : float
        Smoothing term of the Dice ratio.
    """

    def __init__(self, weights: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        self.register_buffer("weights", weights)
        self.eps = eps

    def forward(
        self, outputs: list[torch.Tensor], labels: torch.Tensor, core: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the total loss.

        Parameters
        ----------
        outputs : list[torch.Tensor]
            Logits at full, half and quarter resolution, each ``(N, 3, D, H, W)``, as ``UNet3D`` returns
            in training mode.
        labels : torch.Tensor
            Vortex and shear-layer masks, ``(N, 2, D, H, W)``, never both set at one voxel.
        core : torch.Tensor
            Bool mask of the voxels that count, ``(N, D, H, W)``.

        Returns
        -------
        torch.Tensor
            Scalar ``sum(DEEP_WEIGHTS[i] * loss(outputs[i]))``.
        torch.Tensor
            The full-resolution terms for logging, detached, ``(3,)``: cross-entropy, then ``1 - Dice``
            for vortex and for shear layer.
        """
        target = (labels[:, 0] + 2 * labels[:, 1]).long()  # 0 none, 1 vortex, 2 shear layer
        total, full = 0.0, None
        for i, (scale, logits) in enumerate(zip(DEEP_WEIGHTS, outputs)):
            s = 2**i  # labels and mask of the coarser heads are subsampled to match
            terms = self.level_loss(logits, target[:, ::s, ::s, ::s], core[:, ::s, ::s, ::s])
            total = total + scale * terms.sum()
            full = terms.detach() if full is None else full
        return total, full

    def level_loss(self, logits: torch.Tensor, target: torch.Tensor, core: torch.Tensor) -> torch.Tensor:
        # returns the cross-entropy and the two Dice losses, shape (3,)
        n_core = core.sum().clamp(min=1)
        ce = (F.cross_entropy(logits, target, weight=self.weights, reduction="none") * core).sum() / n_core

        # Dice sums over the whole batch: a single random patch may hold no labelled voxel.
        p = logits.softmax(1) * core[:, None]
        y = F.one_hot(target, 3).movedim(-1, 1) * core[:, None]
        dims = (0, 2, 3, 4)
        dice = (2 * (p * y).sum(dims) + self.eps) / (p.sum(dims) + y.sum(dims) + self.eps)
        return torch.stack([ce, *(1 - dice[1:])])
