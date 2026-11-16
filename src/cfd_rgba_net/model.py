"""3D residual UNet that maps the 6 input channels to per-voxel class scores (none, vortex, shear layer)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import Contract, LabelRules
from .loader import input_scales

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


class DerivedFeatures(nn.Module):
    """Fixed stem that appends the label rules' three quantities to the inputs, each squashed around its threshold.

    The labels are thresholds on lambda2*, Q* and |omega|*, which a convolution would have to rediscover
    from the raw velocity. This stem computes them exactly as ``baseline.velocity_gradient`` does (4th-order
    central differences, 2nd order or 0 where the stencil leaves the fluid), in float32 and outside autocast.
    The three are invariant under the 48 axis-aligned orientations, so augmentation leaves them consistent.
    It has no parameters.

    Parameters
    ----------
    velocity_scale : float
        The scale that divided u, v and w, to undo.
    body_scale : float
        The scale that divided ``body_proximity``, to undo. Voxels at or above 1 are the body.
    n_per_H : int
        Voxels per reference length H, so the voxel size is ``1 / n_per_H``.
    tau_lambda2 : float
        Threshold of the vortex rule: ``lambda2* < -tau_lambda2``. Also the scale of the Q feature.
    tau_omega : float
        Threshold of the shear-layer rule: ``|omega|* > tau_omega``.
    """

    N_FEATURES = 3

    def __init__(self, velocity_scale: float, body_scale: float, n_per_H: int, tau_lambda2: float, tau_omega: float):
        super().__init__()
        self.velocity_scale, self.body_scale, self.h = velocity_scale, body_scale, 1.0 / n_per_H
        self.tau_lambda2, self.tau_omega = tau_lambda2, tau_omega

    @classmethod
    def from_configs(cls, contract: Contract, rules: LabelRules) -> "DerivedFeatures":
        """Build the stem from the contract and the label rules.

        Parameters
        ----------
        contract : Contract
            The input contract. Gives the scales of ``u`` and ``body_proximity`` and ``n_per_H``.
        rules : LabelRules
            The label thresholds.

        Returns
        -------
        DerivedFeatures
            The stem.
        """
        scales = input_scales(contract)
        return cls(float(scales[0]), float(scales[contract.channels.index("body_proximity")]),
                   contract.n_per_H, rules.tau_lambda2, rules.tau_omega)

    def gradient(self, u: torch.Tensor, fluid: torch.Tensor) -> torch.Tensor:
        """Differentiate the velocity like ``baseline.velocity_gradient``.

        Parameters
        ----------
        u : torch.Tensor
            Velocity in units of U_inf, float32 ``(N, 3, D, H, W)``.
        fluid : torch.Tensor
            Float ``(N, 1, D, H, W)``, 1 in the fluid and 0 elsewhere. Outside the tensor counts as 0.

        Returns
        -------
        torch.Tensor
            Float32 ``(N, 3, 3, D, H, W)`` with ``G[:, i, j] = du_i / dx_j``.
        """
        G = u.new_zeros(u.shape[0], 3, 3, *u.shape[2:])
        for j in range(3):
            dim = 2 + j
            n = u.shape[dim]

            def tap(a: torch.Tensor, k: int) -> torch.Tensor:
                pad = [0, 0] * (2 - j) + [2, 2] + [0, 0] * j  # F.pad lists the last dimension first
                return F.pad(a, pad).narrow(dim, 2 + k, n)

            ok2 = (tap(fluid, 1) * tap(fluid, -1)) > 0
            ok4 = ok2 & ((tap(fluid, 2) * tap(fluid, -2)) > 0)
            for i in range(3):
                f = u[:, i:i + 1]
                d2 = (tap(f, 1) - tap(f, -1)) / 2
                d4 = (8 * (tap(f, 1) - tap(f, -1)) - (tap(f, 2) - tap(f, -2))) / 12
                G[:, i, j] = (torch.where(ok4, d4, torch.where(ok2, d2, torch.zeros_like(d2))) / self.h)[:, 0]
        return G

    @staticmethod
    def lambda2(G: torch.Tensor) -> torch.Tensor:
        """Compute the middle eigenvalue of ``S^2 + W^2`` in closed form, in float64.

        Parameters
        ----------
        G : torch.Tensor
            Velocity gradients ``(N, 3, 3, ...)``.

        Returns
        -------
        torch.Tensor
            Float64 ``(N, ...)``.
        """
        g = G.double()
        gt = g.transpose(1, 2)
        S, W = 0.5 * (g + gt), 0.5 * (g - gt)
        M = torch.einsum("nik...,nkj...->nij...", S, S) + torch.einsum("nik...,nkj...->nij...", W, W)
        q = (M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]) / 3
        a, b, c = M[:, 0, 0] - q, M[:, 1, 1] - q, M[:, 2, 2] - q
        p1 = M[:, 0, 1] ** 2 + M[:, 0, 2] ** 2 + M[:, 1, 2] ** 2
        p = torch.sqrt((a * a + b * b + c * c + 2 * p1) / 6)
        # det of B = (M - qI) / p, written out for the symmetric matrix; p is clamped where M is a multiple of I
        pc = p.clamp_min(1e-30)
        det = (a * (b * c - M[:, 1, 2] ** 2) - M[:, 0, 1] * (M[:, 0, 1] * c - M[:, 1, 2] * M[:, 0, 2])
               + M[:, 0, 2] * (M[:, 0, 1] * M[:, 1, 2] - b * M[:, 0, 2])) / pc**3
        phi = torch.acos((det / 2).clamp(-1, 1)) / 3
        largest = q + 2 * p * torch.cos(phi)
        smallest = q + 2 * p * torch.cos(phi + 2 * math.pi / 3)
        return 3 * q - largest - smallest

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Append the three features to the inputs.

        Parameters
        ----------
        x : torch.Tensor
            Normalized inputs ``(N, 6, D, H, W)``: u, v, w, Cp, body_proximity, ground_proximity.

        Returns
        -------
        torch.Tensor
            The inputs with 3 more channels, ``(N, 9, D, H, W)``, in ``x``'s dtype: ``tanh((-lambda2* -
            tau_lambda2) / tau_lambda2)``, ``tanh(Q* / tau_lambda2)`` and ``tanh((|omega|* - tau_omega) /
            tau_omega)``. Each is 0 on its rule's threshold.
        """
        with torch.autocast(x.device.type, enabled=False):
            xf = x.float()
            fluid = (xf[:, 4:5] * self.body_scale < 1).float()
            G = self.gradient(xf[:, :3] * self.velocity_scale, fluid)
            Gt = G.transpose(1, 2)
            S, W = 0.5 * (G + Gt), 0.5 * (G - Gt)
            Q = 0.5 * ((W**2).sum((1, 2)) - (S**2).sum((1, 2)))
            omega = torch.sqrt((G[:, 2, 1] - G[:, 1, 2]) ** 2 + (G[:, 0, 2] - G[:, 2, 0]) ** 2
                               + (G[:, 1, 0] - G[:, 0, 1]) ** 2)
            lam2 = self.lambda2(G).float()
            feats = torch.stack([torch.tanh((-lam2 - self.tau_lambda2) / self.tau_lambda2),
                                 torch.tanh(Q / self.tau_lambda2),
                                 torch.tanh((omega - self.tau_omega) / self.tau_omega)], dim=1)
        return torch.cat([x, feats.to(x.dtype)], dim=1)


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
    features : DerivedFeatures or None
        Fixed stem that appends the rule quantities to the inputs, see ``DerivedFeatures``. The first conv
        then takes ``in_channels + 3`` channels, so weights do not load across the two settings.
    prior : bool
        Add the label rules' margins to the logits, with learnable scales, and start the heads at 0 so that
        the untrained model reproduces the rules. Needs ``features``.
    prior_scale : float
        Starting value of the two learnable scales on the margins. The margins lie in (-1, 1) and are about
        -0.76 in quiet flow, so a small scale leaves the foreground probabilities there well above 0 and the
        loss far from the rules', although the argmax matches them.

    Raises
    ------
    ValueError
        If ``levels`` is out of range, or ``prior`` is set without ``features``.
    """

    def __init__(self, in_channels: int = 6, levels: int = 5, norm: str = "instance",
                 features: DerivedFeatures | None = None, prior: bool = False, prior_scale: float = 4.0):
        super().__init__()
        if prior and features is None:
            raise ValueError("prior needs features")
        self.features, self.prior = features, prior
        # scales of the vortex and shear-layer margins; the softmax saturates away from the thresholds
        self.prior_scale = nn.Parameter(torch.full((2,), float(prior_scale))) if prior else None
        if features is not None:
            in_channels += DerivedFeatures.N_FEATURES
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
        if prior:
            for conv in (self.head, *self.aux):
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)

    def rule_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Turn the stem's features into logits that argmax to the rules' classes.

        Parameters
        ----------
        x : torch.Tensor
            The inputs with the stem's features, ``(N, 9, D, H, W)``.

        Returns
        -------
        torch.Tensor
            ``(N, 3, D, H, W)`` logits ``[0, a_v m_v, a_s m_s]``. ``m_v = min(f1, f2)`` is positive in a
            vortex, and ``m_s = min(f3, -m_v)`` is positive in a shear layer that is not a vortex.
        """
        f1, f2, f3 = x[:, 6], x[:, 7], x[:, 8]
        m_v = torch.minimum(f1, f2)
        m_s = torch.minimum(f3, -m_v)
        return torch.stack([torch.zeros_like(m_v), self.prior_scale[0] * m_v, self.prior_scale[1] * m_s], dim=1)

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
        if self.features is not None:
            x = self.features(x)
        skips = [self.enc0(self.stem(x))]
        for down, enc in zip(self.down[:-1], self.enc[:-1]):
            skips.append(enc(down(skips[-1])))
        y = self.enc[-1](self.down[-1](skips[-1]))

        decoded = []  # deepest decoder first, decoder 0 last
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            y = dec(torch.cat([up(y), skip], dim=1))
            decoded.append(y)

        out = self.head(y)
        if self.prior:
            rule = self.rule_logits(x)
            out = out + rule
        if not self.training:
            return out
        # decoded[k] is decoder levels - 2 - k, so decoder 1 is decoded[levels - 3] and decoder 2 the one before
        aux = [self.aux[0](decoded[self.levels - 3]), self.aux[1](decoded[self.levels - 4])]
        if self.prior:  # the same rules, averaged down to the half and quarter resolution
            aux = [a + F.avg_pool3d(rule, 2 ** (k + 1)) for k, a in enumerate(aux)]
        return [out, *aux]
