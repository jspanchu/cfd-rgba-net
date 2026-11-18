"""
Train the 3D UNet for the Phase 1.1 pilot (roadmap, task 7).

Example: use 2 gpus on a machine

    torchrun --nproc_per_node=2 scripts/ahmedml/pilot.py \\
        --stores /data2/cfd-rgba-net-data/ahmedML/out_v1 \\
        --cache /data2/cfd-rgba-net-data/ahmedML/index --out runs/pilot

Example: run the overfit test
    python scripts/ahmedml/pilot.py ... --overfit 216 --max-epochs 40 \\
        --out runs/overfit

By default it trains on the 100-run pilot subset of the train split (``--subset
368`` takes all of it) and validates on a fixed 10 runs of the validation split.
``--overfit RUN`` trains and validates on that one run instead.
"""

import argparse
import os
import shlex
import sys
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
from cfd_rgba_net.loss import SegmentationLoss, class_weights
from cfd_rgba_net.model import UNet3D
from cfd_rgba_net.train import TrainConfig, seed_everything, train

# Fraction of core voxels labelled vortex and shear layer in the sampled train
# patches (roadmap, task 4).
CLASS_FRACTIONS = (0.017, 0.029)
PILOT_SUBSET = 100


def command_line() -> str:
    """Rebuild the command that started this process, for the log's first line.

    Returns
    -------
    str
        ``# command: `` followed by the command as a shell takes it:
        ``CUDA_VISIBLE_DEVICES`` when it is set, the Python interpreter relative
        to the working directory (``torchrun`` when it launched the process),
        then the script and its arguments. The ``#`` keeps the line apart from
        the JSON lines of the epochs.
    """
    env = (
        [f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"]
        if "CUDA_VISIBLE_DEVICES" in os.environ
        else []
    )
    # torchrun sets RANK; sys.argv then holds only the script and its arguments
    if "RANK" in os.environ:
        launcher = [
            "torchrun",
            f"--nproc_per_node={os.environ['LOCAL_WORLD_SIZE']}",
        ]
    else:
        launcher = [os.path.relpath(sys.executable)]
    return "# command: " + shlex.join(env + launcher + sys.argv)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--stores", type=Path, required=True)
    ap.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="the folder of sampling indices",
    )
    ap.add_argument(
        "--out", type=Path, required=True, help="where the weights go"
    )
    ap.add_argument(
        "--csv", type=Path, default=configs_dir() / "splits" / "ahmedml_v1.csv"
    )
    ap.add_argument(
        "--overfit", type=int, help="train and validate on this one run"
    )
    ap.add_argument(
        "--subset",
        type=int,
        default=PILOT_SUBSET,
        help="train on the runs of the nested train subsets up to this size; "
        "368 is the whole train split",
    )
    ap.add_argument(
        "--no-augment",
        action="store_true",
        help="keep the stored orientation of every patch",
    )
    ap.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    ap.add_argument("--val-every", type=int, default=TrainConfig.val_every)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--w-max",
        type=float,
        default=100.0,
        help="cap on the foreground class weights of the loss",
    )
    ap.add_argument(
        "--levels",
        type=int,
        default=5,
        help="resolution levels of the UNet, 2 to 5",
    )
    ap.add_argument(
        "--norm",
        default="instance",
        choices=["instance", "group", "batch"],
        help="UNet normalisation",
    )
    ap.add_argument(
        "--prior",
        action="store_true",
        help="add the rules' margins to the logits",
    )
    ap.add_argument(
        "--prior-scale",
        type=float,
        default=4.0,
        help="starting scale of the --prior margins",
    )
    ap.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="epochs of linear learning-rate warm-up",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=TrainConfig.batch_size,
        help="patches per GPU",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=TrainConfig.patience,
        help="epochs without a better validation IoU",
    )
    ap.add_argument(
        "--resume",
        type=Path,
        help="weights of an earlier run to continue from, with --start-epoch",
    )
    ap.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="epochs the --resume weights were trained for",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the training forward pass",
    )
    args = ap.parse_args()
    if bool(args.resume) != bool(args.start_epoch):
        ap.error("--resume and --start-epoch go together")

    distributed = "RANK" in os.environ
    if (
        int(os.environ.get("RANK", "0")) == 0
    ):  # one copy in the log, not one per GPU
        print(command_line(), flush=True)
    if distributed:
        dist.init_process_group(
            "nccl", timeout=timedelta(hours=1)
        )  # rank 1 waits while rank 0 validates
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    df = pd.read_csv(args.csv)
    if args.overfit:
        train_runs = val_runs = [args.overfit]
    else:
        train_runs = df.run[
            (df.split == "train") & (df["subset"] <= args.subset)
        ].tolist()
        val_runs = df.run[df.split == "val"].tolist()[
            ::5
        ]  # a fixed 10 of the 50
    cfg = TrainConfig(
        seed=args.seed,
        max_epochs=args.max_epochs,
        val_every=args.val_every,
        batch_size=args.batch_size,
        compile=args.compile,
        patience=args.patience,
        start_epoch=args.start_epoch,
        warmup_epochs=args.warmup_epochs,
    )
    dataset = PatchDataset(
        "train",
        args.csv,
        args.stores,
        args.cache,
        runs=train_runs,
        seed=args.seed,
        augment=not args.no_augment,
    )
    ds_cfg, contract, rules = (
        CaseConfig.load("ahmedml"),
        Contract.load(),
        LabelRules.load(),
    )

    def validate(model):
        """Pooled vortex and shear-layer IoU over the validation core voxels."""
        runs = [
            evaluate_run(
                store_path(ds_cfg, args.stores, r),
                model,
                contract,
                rules,
                device,
                surface_dice=False,
            )["net"]
            for r in val_runs
        ]
        return tuple(metrics(pool(runs)["all"][c])["iou"] for c in CLASS_NAMES)

    seed_everything(args.seed)
    model = UNet3D(
        rules.tau_lambda2,
        rules.tau_omega,
        args.levels,
        args.norm,
        args.prior,
        args.prior_scale,
    ).to(device)
    if args.resume:
        model.load_state_dict(
            torch.load(args.resume, map_location=device, weights_only=True)
        )
    loss_fn = SegmentationLoss(class_weights(CLASS_FRACTIONS, args.w_max)).to(
        device
    )
    train(model, dataset, loss_fn, cfg, validate, args.out)


if __name__ == "__main__":
    main()
