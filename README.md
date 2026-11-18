# Segmentation of flow features on volumes resampled from unstructured grids

## Install

```sh
uv sync
source .venv/bin/activate
```

## Layout

```sh
configs/contract_v1.yaml        input contract v1.0: n_per_H and the SDF (signed-distance-field) band (units of H)
configs/label_rules_v1.yaml     rules v1 that contains the thresholds applied to Q*, lambda2* and |omega|* in (units of H) to generate labels
configs/datasets/*.yaml         per-dataset case parameters: files, field names and source (point or cell), U_inf, frame, ground
src/cfd_rgba_net/               # Listed in typical workflow order
    config.py                   loads the contract, label_rules and dataset YAML
    geometry.py                 Read body STL, compute voxel grid on the mesh axes, and SDF from body. 
    preprocess.py               Go from unstructured mesh to model input features (shared by training and inference)
    ingredients.py              Q*, lambda2*, |omega|* on the native mesh (label teacher)
    label.py                    applies label rules v1 as thresholds to the ingredients on the voxel grid
    io.py                       Zarr v3 case store, vti writer, provenance (git commit, input SHA-256)
    build_case.py               generate one training case, end to end (CLI: cfd-rgba-net-build-case)
    batch.py                    executes build_case.py for a large batch of runs, downloading vtu files in parallel.
    loader.py                   96^3 training patches from the case stores of one split (PyTorch dataset)
    baseline.py                 the rules on the grid's own gradients: the baseline, and the net's input features
    infer.py                    unseen case, end to end: raw mesh -> features -> probabilities -> renders (CLI: cfd-rgba-net-infer)
    render.py                   GPU volume rendering of the probabilities, one colour per feature
    scripts/ahmedml/
        fix_wedges.py               rewrites the AhmedML VTUs with VTK's wedge point order (source of jspanchu/ahmedml)
        figures.py                  progress figures from the ahmedml_run<N>.zarr stores
        make_splits.py              assigns every run to a split (configs/splits/ahmedml_v1.csv)
        build_index.py              builds and caches the sampling index of every run of a split, in parallel (restartable)
        pilot.py                    trains the 3D UNet on the pilot subset, single GPU or torchrun (--overfit RUN for one run)
        orient_eval.py              scores checkpoints on re-oriented validation cases (axis permutations and flips)
tests/                          verification suite (pytest)
```

The grid has dx = H / 32, is aligned with the mesh axes and fills the dataset's domain (`domain` in its
YAML, default the mesh bounds). There is no axis convention: the flow may run along any mesh axis, in
either sign.

## Build training data

```sh
# one case, from run files already on disk in <data>/run_<N>/
cfd-rgba-net-build-case --dataset ahmedml --run 1 --data <data> --out out_v1 --save-vti

# many runs: each run is downloaded from the dataset's Hugging Face repo (hf_repo in its YAML) into a
# temporary work folder, built, and its downloaded files deleted, so only the Zarr stores are kept
cfd-rgba-net-batch --dataset ahmedml --runs 1-499 \
    --out /data2/cfd-rgba-net-data/ahmedML/out_v1 --workers 6
cfd-rgba-net-batch --dataset windsorml --runs 0-349 --out /data2/cfd-rgba-net-data/windsorML/out_v1   # ~80 GB per build
```

## Inference on an unseen case

From a raw AhmedML-family volume to volume renderings of the vortex and shear-layer classes, one command per
method; each reads the .vtu and writes `<out>_<view>.png`, and `net` also writes `<out>_<view>_uncertainty.png`,
the entropy of its three class probabilities over ln 3 (0 certain, 0.63 an even split between two classes, 1
all three equally likely):

```sh
# get a case (here test_interp run 280)
hf download jspanchu/ahmedml run_280/volume_280.vtu run_280/ahmed_280.stl --repo-type dataset --local-dir data

common="data/run_280/volume_280.vtu --body data/run_280/ahmed_280.stl --dataset configs/datasets/ahmedml.yaml
        --contract configs/contract_v1.yaml --rules configs/label_rules_v1.yaml --views side top rear"
cfd-rgba-net-infer $common --method net --checkpoint runs/case_feat_prior_100/best.pt --out out/run280_net
cfd-rgba-net-infer $common --method baseline --out out/run280_baseline  # the rules on the grid's gradients
cfd-rgba-net-infer $common --method labels --out out/run280_labels      # the rules on the native mesh's gradients
```

## Tests

```sh
uv run pytest
```
