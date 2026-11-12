"""3D residual UNet that maps the 6 input channels to per-voxel class scores (none, vortex, shear layer)."""

from __future__ import annotations

import torch
from torch import nn

CHANNELS = (32, 64, 128, 256, 320)  # encoder 0-3 and bottleneck
N_CLASSES = 3


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
        Output channels. A conv 1³ on the residual path matches them when ``cin != cout``.
    """

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1, self.norm1 = conv3(cin, cout), nn.InstanceNorm3d(cout, affine=True)
        self.conv2, self.norm2 = conv3(cout, cout), nn.InstanceNorm3d(cout, affine=True)
        self.skip = nn.Identity() if cin == cout else nn.Conv3d(cin, cout, 1)
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        return self.act(self.norm2(self.conv2(y)) + self.skip(x))


def res_blocks(cin: int, cout: int, n: int) -> nn.Sequential:
    return nn.Sequential(ResBlock(cin, cout), *(ResBlock(cout, cout) for _ in range(n - 1)))


class UNet3D(nn.Module):
    """3D residual UNet with 5 resolution levels and deep-supervision heads.

    Parameters
    ----------
    in_channels : int
        Number of input channels (6 in contract v1).
    """

    def __init__(self, in_channels: int = 6):
        super().__init__()
        c = CHANNELS
        self.stem = conv3(in_channels, c[0])
        self.enc0 = res_blocks(c[0], c[0], 2)
        # Encoders 1-3 and the bottleneck: strided conv, then 2 residual blocks.
        self.down = nn.ModuleList(conv3(c[i - 1], c[i], stride=2) for i in range(1, 5))
        self.enc = nn.ModuleList(res_blocks(c[i], c[i], 2) for i in range(1, 5))
        # Decoders 3 to 0: transposed conv, concatenate the encoder's features, 1 residual block.
        self.up = nn.ModuleList(nn.ConvTranspose3d(c[i + 1], c[i], 2, 2) for i in range(3, -1, -1))
        self.dec = nn.ModuleList(res_blocks(2 * c[i], c[i], 1) for i in range(3, -1, -1))
        self.head = nn.Conv3d(c[0], N_CLASSES, 1)
        # Auxiliary heads on decoder 1 (48³) and decoder 2 (24³), used in training only.
        self.aux = nn.ModuleList(nn.Conv3d(c[i], N_CLASSES, 1) for i in (1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Predict class scores.

        Parameters
        ----------
        x : torch.Tensor
            Normalized inputs of shape (N, 6, D, H, W), with each side a multiple of 16.

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

        decoded = []  # decoder 3, 2, 1, 0
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            y = dec(torch.cat([up(y), skip], dim=1))
            decoded.append(y)

        out = self.head(y)
        if not self.training:
            return out
        return [out, self.aux[0](decoded[2]), self.aux[1](decoded[1])]
