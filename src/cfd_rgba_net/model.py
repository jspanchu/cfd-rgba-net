"""3D UNet from the rules' three quantities on the voxel grid to class scores.

The scores are per voxel, over none, vortex and shear layer.
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
N_CLASSES = 3


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


def rule_logits(f: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Turn the squashed features into logits that argmax to the rules' classes.

    Parameters
    ----------
    f : torch.Tensor
        The squashed features ``(N, 3, D, H, W)``, see ``UNet3D.squash``.
    scale : torch.Tensor
        Scales of the vortex and shear-layer margins, ``(2,)``.

    Returns
    -------
    torch.Tensor
        ``(N, 3, D, H, W)`` logits ``[0, a_v m_v, a_s m_s]``. ``m_v = min(f0,
        f1)`` is positive in a vortex, and ``m_s = min(f2, -m_v)`` is positive
        in a shear layer that is not a vortex.
    """
    m_v = torch.minimum(f[:, 0], f[:, 1])
    m_s = torch.minimum(f[:, 2], -m_v)
    return torch.stack(
        [torch.zeros_like(m_v), scale[0] * m_v, scale[1] * m_s], dim=1
    )


class UNet3D(nn.Module):
    """3D UNet after Ronneberger et al. (2015) on the rules' three quantities.

    The inputs are lambda2*, Q* and |omega|* on the voxel grid
    (``baseline.grid_features``). The net first squashes each around its rule's
    threshold with a tanh. Each level then has two 3³ convolutions, each
    followed by a norm and a ReLU, with "same" padding. Max-pooling halves the
    grid on the way down; a transposed convolution doubles it on the way up, and
    the encoder's features of that level are concatenated before the two
    convolutions. One 1³ convolution gives the class scores.

    Parameters
    ----------
    tau_lambda2 : float
        Threshold of the vortex rule, ``lambda2* < -tau_lambda2``. Also the
        scale of the Q* feature.
    tau_omega : float
        Threshold of the shear-layer rule, ``|omega|* > tau_omega``.
    levels : int
        Number of resolution levels, bottleneck included, from 2 to 5. The
        widths are the first ``levels`` entries of ``CHANNELS``.
    norm : str
        Kind of normalisation, see ``make_norm``.
    prior : bool
        Add the rules' margins (``rule_logits``) to the logits, with learnable
        scales, and start the head at 0, so that the untrained net reproduces
        the rules and training learns a correction to them.
    prior_scale : float
        Starting value of the two learnable scales on the margins. The margins
        lie in (-1, 1) and are about -0.76 in quiet flow, so a small scale
        leaves the foreground probabilities there well above 0 and the loss far
        from the rules', although the argmax matches them.

    Raises
    ------
    ValueError
        If ``levels`` is out of range.
    """

    def __init__(
        self,
        tau_lambda2: float,
        tau_omega: float,
        levels: int = 5,
        norm: str = "instance",
        prior: bool = False,
        prior_scale: float = 4.0,
    ):
        super().__init__()
        if not 2 <= levels <= len(CHANNELS):
            raise ValueError(
                f"levels must be between 2 and {len(CHANNELS)}, got {levels}"
            )
        self.tau_lambda2, self.tau_omega, self.prior = (
            tau_lambda2,
            tau_omega,
            prior,
        )
        self.prior_scale = (
            nn.Parameter(torch.full((2,), float(prior_scale)))
            if prior
            else None
        )
        c = CHANNELS[:levels]
        self.enc = nn.ModuleList(
            double_conv(cin, cout, norm) for cin, cout in zip((3, *c[:-1]), c)
        )
        # decoders, deepest first
        self.up = nn.ModuleList(
            nn.ConvTranspose3d(c[i + 1], c[i], 2, 2)
            for i in range(levels - 2, -1, -1)
        )
        self.dec = nn.ModuleList(
            double_conv(2 * c[i], c[i], norm) for i in range(levels - 2, -1, -1)
        )
        self.head = nn.Conv3d(c[0], N_CLASSES, 1)
        if prior:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def squash(self, x: torch.Tensor) -> torch.Tensor:
        """Squash the raw features around the rules' thresholds, in float32.

        The squash runs in float32 whatever the autocast.

        Parameters
        ----------
        x : torch.Tensor
            lambda2*, Q* and |omega|*, ``(N, 3, D, H, W)``.

        Returns
        -------
        torch.Tensor
            ``tanh((-lambda2* - tau_lambda2) / tau_lambda2)``, ``tanh(Q* /
            tau_lambda2)`` and ``tanh((|omega|* - tau_omega) / tau_omega)``,
            float32 ``(N, 3, D, H, W)``. Each is 0 on its rule's threshold and
            positive on the side the rule selects.
        """
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            return torch.stack(
                [
                    torch.tanh(
                        (-x[:, 0] - self.tau_lambda2) / self.tau_lambda2
                    ),
                    torch.tanh(x[:, 1] / self.tau_lambda2),
                    torch.tanh((x[:, 2] - self.tau_omega) / self.tau_omega),
                ],
                dim=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class scores.

        Parameters
        ----------
        x : torch.Tensor
            Raw lambda2*, Q* and |omega|*, ``(N, 3, D, H, W)``, with each side a
            multiple of ``2 ** (levels - 1)``. Zero, as in padding, reads as a
            zero velocity gradient.

        Returns
        -------
        torch.Tensor
            Logits (not probabilities) of shape (N, 3, D, H, W) over none,
            vortex, shear layer; apply softmax for probabilities.
        """
        f = self.squash(x)
        skips = []
        y = f
        for i, enc in enumerate(self.enc):
            y = enc(F.max_pool3d(y, 2) if i else y)
            skips.append(y)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips[:-1])):
            y = dec(torch.cat([up(y), skip], dim=1))
        out = self.head(y)
        if self.prior:
            out = out + rule_logits(f, self.prior_scale)
        return out
