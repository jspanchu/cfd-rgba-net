"""3D residual UNet that maps the 6 input channels to per-voxel class scores (none, vortex, shear layer)."""

from __future__ import annotations

import torch
from torch import nn

CHANNELS = (32, 64, 128, 256, 320)  # encoder 0-3 and bottleneck
N_CLASSES = 3


def make_norm(kind: str, channels: int) -> nn.Module:
    """Build the normalisation layer after a convolution.

    Parameters
    ----------
    kind : str
        ``instance`` (nnU-Net's choice, statistics per sample and channel), ``group`` (per sample, over groups
        of 4 channels) or ``batch`` (over the batch, with running statistics in evaluation).
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


def conv3(cin: int, cout: int, stride: int = 1) -> nn.Conv3d:
    # padding=1 keeps the grid size for stride 1 and halves it exactly for stride 2
    return nn.Conv3d(cin, cout, 3, stride, padding=1)


class ResBlock(nn.Module):
    """Two 3³ convolutions with instance norm and a residual path.

    Parameters
    ----------
    cin : int
        Input channels.
    cout : int
        Output channels. A conv 1³ and a norm on the residual path match them when ``cin != cout``.
        The norm keeps that path bounded: without it, the decoder's skip convs grew until fp16 overflowed.
    norm : str
        Kind of normalisation, see ``make_norm``.
    """

    def __init__(self, cin: int, cout: int, norm: str = "instance"):
        super().__init__()
        self.conv1, self.norm1 = conv3(cin, cout), make_norm(norm, cout)
        self.conv2, self.norm2 = conv3(cout, cout), make_norm(norm, cout)
        self.skip = nn.Identity() if cin == cout else nn.Sequential(
            nn.Conv3d(cin, cout, 1, bias=False), make_norm(norm, cout))
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        return self.act(self.norm2(self.conv2(y)) + self.skip(x))


def res_blocks(cin: int, cout: int, n: int, norm: str = "instance") -> nn.Sequential:
    return nn.Sequential(ResBlock(cin, cout, norm), *(ResBlock(cout, cout, norm) for _ in range(n - 1)))


class UNet3D(nn.Module):
    """3D residual UNet with deep-supervision heads.

    Parameters
    ----------
    in_channels : int
        Number of input channels (6 in contract v1).
    levels : int
        Number of resolution levels, bottleneck included, from 4 to 5. The widths are the first ``levels``
        entries of ``CHANNELS``.
    norm : str
        Kind of normalisation, see ``make_norm``.
    """

    def __init__(self, in_channels: int = 6, levels: int = 5, norm: str = "instance"):
        super().__init__()
        if not 4 <= levels <= len(CHANNELS):
            raise ValueError(f"levels must be between 4 and {len(CHANNELS)}, got {levels}")
        self.levels = levels
        c = CHANNELS[:levels]
        # Normalized, unlike a bare conv: the raw inputs reach tens (scaled C_p), and without the norm the
        # unnormalized skip convs of the decoder blew up and the loss went NaN within 300 steps.
        self.stem = nn.Sequential(conv3(in_channels, c[0]), make_norm(norm, c[0]), nn.LeakyReLU(0.01))
        self.enc0 = res_blocks(c[0], c[0], 2, norm)
        # Encoders 1 to the bottleneck: strided conv, then 2 residual blocks.
        self.down = nn.ModuleList(conv3(c[i - 1], c[i], stride=2) for i in range(1, levels))
        self.enc = nn.ModuleList(res_blocks(c[i], c[i], 2, norm) for i in range(1, levels))
        # Decoders below the bottleneck, deepest first: transposed conv, concatenate the encoder's features,
        # 1 residual block.
        self.up = nn.ModuleList(nn.ConvTranspose3d(c[i + 1], c[i], 2, 2) for i in range(levels - 2, -1, -1))
        self.dec = nn.ModuleList(res_blocks(2 * c[i], c[i], 1, norm) for i in range(levels - 2, -1, -1))
        self.head = nn.Conv3d(c[0], N_CLASSES, 1)
        # Auxiliary heads on decoder 1 (48³) and decoder 2 (24³), used in training only.
        self.aux = nn.ModuleList(nn.Conv3d(c[i], N_CLASSES, 1) for i in (1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Predict class scores.

        Parameters
        ----------
        x : torch.Tensor
            Normalized inputs of shape (N, 6, D, H, W), with each side a multiple of ``2 ** (levels - 1)``.

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            Logits (not probabilities) of shape (N, 3, D, H, W) over none, vortex, shear layer; apply
            softmax for probabilities. In training mode, a list of the full-, half- and
            quarter-resolution logits for deep supervision.
        """
        skips = [self.enc0(self.stem(x))]
        for down, enc in zip(self.down[:-1], self.enc[:-1]):
            skips.append(enc(down(skips[-1])))
        y = self.enc[-1](self.down[-1](skips[-1]))

        decoded = []  # deepest decoder first, decoder 0 last
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            y = dec(torch.cat([up(y), skip], dim=1))
            decoded.append(y)

        out = self.head(y)
        if not self.training:
            return out
        # decoded[k] is decoder levels - 2 - k, so decoder 1 is decoded[levels - 3] and decoder 2 the one before
        return [out, self.aux[0](decoded[self.levels - 3]), self.aux[1](decoded[self.levels - 4])]
