"""Verification: a run resumed from saved weights continues the epoch count and
the learning-rate decay."""

import json

import torch
from torch import nn
from torch.utils.data import IterableDataset

from cfd_rgba_net.loss import SegmentationLoss
from cfd_rgba_net.train import TrainConfig, train


class Net(nn.Module):
    """Stands in for the UNet: 3 features in, 3 class scores out."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(3, 3, 1)

    def forward(self, x):
        return self.conv(x)


class Patches(IterableDataset):
    seed = 0

    def __iter__(self):
        while True:
            yield {
                "inputs": torch.randn(3, 8, 8, 8),
                "labels": torch.zeros(2, 8, 8, 8, dtype=torch.uint8),
                "core": torch.ones(8, 8, 8, dtype=torch.bool),
            }


def run(cfg, tmp_path, capsys):
    loss_fn = SegmentationLoss(torch.ones(3))
    train(Net(), Patches(), loss_fn, cfg, lambda model: (0.5, 0.5), tmp_path)
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_resumed_run_continues_epochs_and_learning_rate(tmp_path, capsys):
    base = {
        "batch_size": 2,
        "steps_per_epoch": 3,
        "max_epochs": 4,
        "val_every": 10,
        "workers": 0,
    }
    whole = run(TrainConfig(**base), tmp_path / "whole", capsys)
    resumed = run(
        TrainConfig(**base, start_epoch=2), tmp_path / "resumed", capsys
    )

    assert [log["epoch"] for log in resumed] == [3, 4]
    assert [log["lr"] for log in resumed] == [log["lr"] for log in whole[2:]]


def test_warmup_ramps_the_learning_rate_then_rejoins_the_decay(
    tmp_path, capsys
):
    base = {
        "batch_size": 2,
        "steps_per_epoch": 3,
        "max_epochs": 4,
        "val_every": 10,
        "workers": 0,
    }
    plain = run(TrainConfig(**base), tmp_path / "plain", capsys)
    warm = run(TrainConfig(**base, warmup_epochs=2), tmp_path / "warm", capsys)

    assert warm[0]["lr"] < plain[0]["lr"]  # still ramping after epoch 1
    assert [log["lr"] for log in warm[2:]] == [log["lr"] for log in plain[2:]]
