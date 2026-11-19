"""3D UNet from n_in features in [-1, 1] on the voxel grid to n_out classes.

The pilot feeds it the rules' three quantities squashed around their
thresholds (``LabelRules.squash``), and its classes are none, vortex and
shear layer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

CHANNELS = (
    32,
    64,
    128,
    256,
    320,
)  # widths of the levels, the first one at full resolution


def make_norm(kind: str, channels: int) -> nn.Module:
    """Build the normalisation layer after a convolution.

    Parameters
    ----------
    kind : str
        ``instance`` (nnU-Net's choice, statistics per sample and channel),
        ``group`` (per sample, over groups of 4 channels) or ``batch`` (over the
        batch, with running statistics in evaluation).
    channels : int
        Number of channels to normalise.

    Returns
    -------
    torch.nn.Module
        The layer.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three.
    """
    if kind == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if kind == "group":
        return nn.GroupNorm(channels // 4, channels)
    if kind == "batch":
        return nn.BatchNorm3d(channels)
    raise ValueError(f"unknown norm {kind!r}")


def conv3(cin: int, cout: int) -> nn.Conv3d:
    return nn.Conv3d(cin, cout, 3, padding=1)  # "same" padding


def double_conv(cin: int, cout: int, norm: str = "instance") -> nn.Sequential:
    return nn.Sequential(
        conv3(cin, cout),
        make_norm(norm, cout),
        nn.ReLU(),
        conv3(cout, cout),
        make_norm(norm, cout),
        nn.ReLU(),
    )


class UNet3D(nn.Module):
    """3D UNet after Ronneberger et al. (2015) for semantic segmentation.

    The inputs are ``n_in`` feature channels on the voxel grid, each in [-1, 1];
    the net applies no scaling of its own. -1 means "no feature": the
    sliding-window inference (``evaluate.predict``) and the patch loader pad
    with it. Each level has two 3³ convolutions, each followed by a norm and a
    ReLU, with "same" padding. Max-pooling halves the grid on the way down; a
    transposed convolution doubles it on the way up, and the encoder's features
    of that level are concatenated before the two convolutions. One 1³
    convolution gives the scores of the ``n_out`` classes.

    Parameters
    ----------
    n_in : int
        Number of input channels.
    n_out : int
        Number of classes.
    levels : int
        Number of resolution levels, bottleneck included, from 2 to 5. The
        widths are the first ``levels`` entries of ``CHANNELS``.
    norm : str
        Kind of normalisation, see ``make_norm``.

    Raises
    ------
    ValueError
        If ``levels`` is out of range.
    """

    def __init__(
        self,
        n_in: int = 3,
        n_out: int = 3,
        levels: int = 5,
        norm: str = "instance",
    ):
        super().__init__()
        if not 2 <= levels <= len(CHANNELS):
            raise ValueError(
                f"levels must be between 2 and {len(CHANNELS)}, got {levels}"
            )
        c = CHANNELS[:levels]
        self.enc = nn.ModuleList(
            double_conv(cin, cout, norm)
            for cin, cout in zip((n_in, *c[:-1]), c)
        )
        # decoders, deepest first
        self.up = nn.ModuleList(
            nn.ConvTranspose3d(c[i + 1], c[i], 2, 2)
            for i in range(levels - 2, -1, -1)
        )
        self.dec = nn.ModuleList(
            double_conv(2 * c[i], c[i], norm) for i in range(levels - 2, -1, -1)
        )
        self.head = nn.Conv3d(c[0], n_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class scores.

        Parameters
        ----------
        x : torch.Tensor
            Features in [-1, 1], ``(N, n_in, D, H, W)``, with each side a
            multiple of ``2 ** (levels - 1)``.

        Returns
        -------
        torch.Tensor
            Logits (not probabilities) of shape ``(N, n_out, D, H, W)``; apply
            softmax for probabilities.
        """
        skips = []
        y = x
        for i, enc in enumerate(self.enc):
            y = enc(F.max_pool3d(y, 2) if i else y)
            skips.append(y)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips[:-1])):
            y = dec(torch.cat([up(y), skip], dim=1))
        return self.head(y)
