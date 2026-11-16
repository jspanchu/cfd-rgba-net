"""Training loop with the Phase 1.1 starting hyperparameters (nnU-Net defaults for 3D UNets)."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    batch_size: int = 2  # patches per GPU
    lr: float = 0.01
    momentum: float = 0.99
    weight_decay: float = 3e-5
    clip_norm: float = 12.0
    steps_per_epoch: int = 250  # an epoch is a fixed number of steps, not a pass over the data
    max_epochs: int = 500
    patience: int = 50  # epochs without a better validation IoU before stopping
    val_every: int = 10  # epochs
    seed: int = 0
    workers: int = 4
    compile: bool = False  # torch.compile the training forward pass
    start_epoch: int = 0  # epochs already trained, when the model holds the weights of an earlier run
    warmup_epochs: int = 0  # the learning rate rises linearly from 0 over these epochs, counted from epoch 1


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch. Call it before building the model.

    Parameters
    ----------
    seed : int
        The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    model: nn.Module,
    dataset: torch.utils.data.IterableDataset,
    loss_fn: Callable,
    cfg: TrainConfig,
    validate: Callable[[nn.Module], tuple[float, float]],
    out: Path,
) -> None:
    """Train until the validation IoU stops improving or ``cfg.max_epochs`` is reached.

    Runs on one process, or under ``torchrun`` with one process per GPU when the process group is
    initialised. Rank 0 validates, logs and saves; the other ranks wait for its decision to stop.

    Parameters
    ----------
    model : torch.nn.Module
        The network, already on its device.
    dataset : torch.utils.data.IterableDataset
        Yields patches with ``inputs``, ``labels`` and ``core``. Its ``seed`` is set to ``cfg.seed + rank``,
        so that ranks draw different patches.
    loss_fn : callable
        ``DeepSupervisionLoss``, returning the total loss and its logged terms.
    cfg : TrainConfig
        The hyperparameters. To resume, load the earlier weights into ``model`` and set ``start_epoch`` to the
        epochs they were trained for: the epoch count and the learning rate continue from there, but the
        optimizer's momentum starts empty.
    validate : callable
        Takes the model in eval mode and returns the validation IoU of vortex and shear layer.
        Placeholder until the task 6 harness exists; task 7 wires it in, together with a command line.
    out : pathlib.Path
        The folder for ``last.pt`` and ``best.pt``, the model weights.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    dataset.seed = cfg.seed + rank
    net = DistributedDataParallel(model, device_ids=[device.index]) if dist.is_initialized() else model
    # Only the training forward is compiled: validation calls ``model`` with other shapes, which would recompile.
    forward = torch.compile(net) if cfg.compile else net
    opt = torch.optim.SGD(net.parameters(), cfg.lr, cfg.momentum, nesterov=True, weight_decay=cfg.weight_decay)
    # the run length is unknown, so the decay follows the cap; a resumed run continues the earlier run's decay
    total_steps = cfg.max_epochs * cfg.steps_per_epoch
    done_steps = cfg.start_epoch * cfg.steps_per_epoch
    warmup_steps = cfg.warmup_epochs * cfg.steps_per_epoch

    def lr_factor(step: int) -> float:
        done = done_steps + step
        ramp = min(1.0, (done + 1) / warmup_steps) if warmup_steps else 1.0  # + 1: the first step is not at 0
        return ramp * (1 - done / total_steps) ** 0.9

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    batches = iter(DataLoader(dataset, cfg.batch_size, num_workers=cfg.workers, pin_memory=cuda))
    out.mkdir(parents=True, exist_ok=True)
    best, best_epoch = -1.0, 0

    for epoch in range(cfg.start_epoch + 1, cfg.max_epochs + 1):
        net.train()
        loss_sum, terms_sum, start = 0.0, torch.zeros(3, device=device), time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(cfg.steps_per_epoch):
            batch = {k: v.to(device, non_blocking=True) for k, v in next(batches).items()}
            # bfloat16, not float16: instance norm of sparse features makes activations exceed the float16
            # range (65,504), which turned the loss NaN after a few thousand steps. bfloat16 needs no scaler.
            with torch.autocast(device.type, torch.bfloat16, enabled=cuda):
                outputs = forward(batch["inputs"])
            loss, terms = loss_fn([o.float() for o in outputs], batch["labels"], batch["core"])
            if not loss.isfinite():  # stop at once: every later step would be wasted
                raise FloatingPointError(f"non-finite loss in epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.clip_norm)
            opt.step()
            sched.step()
            loss_sum, terms_sum = loss_sum + loss.item(), terms_sum + terms
        log = {"epoch": epoch, "lr": sched.get_last_lr()[0], "loss": loss_sum / cfg.steps_per_epoch,
               "terms": (terms_sum / cfg.steps_per_epoch).tolist(),
               "steps_per_s": cfg.steps_per_epoch / (time.time() - start),
               "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None}

        stop = torch.zeros(1, device=device)
        if rank == 0:
            if epoch % cfg.val_every == 0:
                # The other ranks wait in the broadcast below, so a validation longer than the NCCL
                # timeout (10 minutes by default) kills the run: raise the timeout if it does.
                model.eval()
                val_start = time.time()
                log["val_iou"] = validate(model)
                log["val_s"] = time.time() - val_start
                score = float(np.mean(log["val_iou"]))
                if score > best:
                    best, best_epoch = score, epoch
                    torch.save(model.state_dict(), out / "best.pt")
                stop[0] = epoch - best_epoch >= cfg.patience
            torch.save(model.state_dict(), out / "last.pt")
            print(json.dumps(log), flush=True)
        if dist.is_initialized():
            dist.broadcast(stop, 0)
        if stop.item():
            break
