"""
Train the 3D UNet for the Phase 1.1 pilot (roadmap, task 7). One process, or one per GPU under torchrun:

    torchrun --nproc_per_node=2 scripts/ahmedml/pilot.py --stores /data/cfd-rgba-net-data/ahmedML/out_v1 \\
        --cache /data/cfd-rgba-net-data/ahmedML/index --out runs/pilot
    python scripts/ahmedml/pilot.py ... --overfit 216 --max-epochs 40 --out runs/overfit

By default it trains on the 100-run pilot subset of the train split and validates on a fixed 10 runs of the
validation split. ``--overfit RUN`` trains and validates on that one run instead.
"""
import argparse
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist

from cfd_rgba_net.batch import store_path
from cfd_rgba_net.config import CaseConfig, Contract, LabelRules, configs_dir
from cfd_rgba_net.evaluate import evaluate_run, metrics, pool
from cfd_rgba_net.label import CLASS_NAMES
from cfd_rgba_net.loader import PatchDataset
from cfd_rgba_net.loss import DeepSupervisionLoss, class_weights
from cfd_rgba_net.model import UNet3D
from cfd_rgba_net.train import TrainConfig, seed_everything, train

# Fraction of core voxels labelled vortex and shear layer in the sampled train patches (roadmap, task 4).
CLASS_FRACTIONS = (0.017, 0.029)
PILOT_SUBSET = 100


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stores", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True, help="the folder of sampling indices")
    ap.add_argument("--out", type=Path, required=True, help="where the weights go")
    ap.add_argument("--csv", type=Path, default=configs_dir() / "splits" / "ahmedml_v1.csv")
    ap.add_argument("--overfit", type=int, help="train and validate on this one run")
    ap.add_argument("--no-augment", action="store_true", help="keep the stored orientation of every patch")
    ap.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    ap.add_argument("--val-every", type=int, default=TrainConfig.val_every)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-max", type=float, default=100.0, help="cap on the foreground class weights of the loss")
    ap.add_argument("--levels", type=int, default=5, help="resolution levels of the UNet, 4 or 5")
    ap.add_argument("--norm", default="instance", choices=["instance", "group", "batch"], help="UNet normalisation")
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size, help="patches per GPU")
    ap.add_argument("--compile", action="store_true", help="torch.compile the training forward pass")
    args = ap.parse_args()

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(hours=1))  # rank 1 waits while rank 0 validates
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    df = pd.read_csv(args.csv)
    if args.overfit:
        train_runs = val_runs = [args.overfit]
    else:
        train_runs = df.run[(df.split == "train") & (df["subset"] <= PILOT_SUBSET)].tolist()
        val_runs = df.run[df.split == "val"].tolist()[::5]  # a fixed 10 of the 50
    cfg = TrainConfig(seed=args.seed, max_epochs=args.max_epochs, val_every=args.val_every,
                      batch_size=args.batch_size, compile=args.compile)
    dataset = PatchDataset("train", args.csv, args.stores, args.cache,
                           runs=train_runs, seed=args.seed, augment=not args.no_augment)
    ds_cfg, contract, rules = CaseConfig.load("ahmedml"), Contract.load(), LabelRules.load()

    def validate(model):
        """Pooled IoU of vortex and shear layer over the validation runs, all core voxels."""
        runs = [evaluate_run(store_path(ds_cfg, args.stores, r), model, contract, rules, device,
                             surface_dice=False)["net"] for r in val_runs]
        return tuple(metrics(pool(runs)["all"][c])["iou"] for c in CLASS_NAMES)

    seed_everything(args.seed)
    model = UNet3D(len(contract.channels), args.levels, args.norm).to(device)
    loss_fn = DeepSupervisionLoss(class_weights(CLASS_FRACTIONS, args.w_max)).to(device)
    train(model, dataset, loss_fn, cfg, validate, args.out)


if __name__ == "__main__":
    main()
