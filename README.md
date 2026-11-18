# Identifying flow features with learned color & opacity transfer functions

## Install

```sh
uv sync
source .venv/bin/activate
```

## Layout

```
configs/contract_v1.yaml        input contract v1.0: n_per_H and the SDF band (units of H)
configs/label_rules_v1.yaml     label rules v1 thresholds (units of H)
configs/datasets/*.yaml         per-dataset case parameters: files, field names and source (point or cell), U_inf, frame, ground
src/cfd_rgba_net/
    config.py                   loads the three kinds of YAML
    geometry.py                 body STL, voxel grid on the mesh axes, signed distance
    preprocess.py               native mesh -> model inputs (shared by training and inference)
    ingredients.py              Q*, lambda2*, |omega|* on the native mesh (label teacher)
    label.py                    label rules v1 on the voxel grid
    io.py                       Zarr v3 case store, vti writer, provenance (git commit, input SHA-256)
    build_case.py               one training case, end to end (CLI: cfd-rgba-net-build-case)
    loader.py                   96^3 training patches from the case stores of one split (PyTorch dataset)
    baseline.py                 the rules on the grid's own gradients: the baseline, and the net's input features
    infer.py                    unseen case, end to end: raw mesh -> features -> probabilities -> renders (CLI: cfd-rgba-net-infer)
    render.py                   GPU volume rendering of the probabilities, one colour per feature (CLI: cfd-rgba-net-render)
tests/                          verification suite (pytest)
scripts/ahmedml/
    fix_wedges.py               rewrites the AhmedML VTUs with VTK's wedge point order (source of jspanchu/ahmedml)
    figures.py                  progress figures from the ahmedml_run<N>.zarr stores
    make_splits.py              assigns every run to a split (configs/splits/ahmedml_v1.csv)
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

Each case writes one Zarr v3 store, `<dataset>_run<N>.zarr` (64^3 chunks in 256^3 shards, Blosc zstd):
`features` (3, nx, ny, nz) float32 (lambda2*, Q*, |omega|* from finite differences of the grid's own velocity:
the net's inputs), `labels` (2, nx, ny, nz) uint8 (vortex, shear_layer), `body_sdf` (nx, ny, nz) float32 (signed
distance to the body in units of H, negative inside), `height` (nx, ny, nz) float32 (height above the ground in
units of H) and an `ingredients/` group (the same three quantities on the native mesh, which define the
labels, and the core), with the case config, contract,
label rules, geometry, sanity checks, and the git commit and SHA-256 of every input file in the store's
attributes. `--save-vti` adds `.vti` files for
viewing. There is no stored fluid mask: take `body_sdf > 0`. The attributes also count the few voxels outside
both the mesh and the body (`checks.voxels.n_not_in_mesh_nor_body`, at most 84 per AhmedML case), where the
features and the labels are 0.

The WindsorML config is not yet verified on a downloaded case; see the TODOs in it. The DrivAerML config is
verified on run 1: its comments record what was checked and how the run's volume was prepared (concatenated
parts, rewritten in appended binary, wedges fixed with `fix_wedges.convert`).

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

`net` and `baseline` resample the velocity and take finite differences on the grid; `labels` takes the
gradient on the native mesh (vtkGradientFilter), as `build_case` does, and resamples the ingredients. All three
compute Q, |omega| and lambda2 in torch, on the GPU when there is one, with a closed-form lambda2, so `labels`
may differ from the stored labels by round-off at the thresholds. U_inf, the ground, the domain and the field
names come from the dataset config, and H from the body STL. Without `--body`, pass `--H` (mesh units); the
body is then the hole in the mesh. The classes are 0 outside the core (walls and the floor boundary layer),
where the net was neither trained nor scored. Rendering is offscreen through EGL, so no display is needed;
colours and opacity curves come from `--render-config` (copy `configs/render_default.yaml`).
`cfd-rgba-net-render` re-renders a .vti with `vortex` and `shear_layer` arrays, such as `build_case --save-vti`'s
labels.

On one AhmedML case (20 M cells, a 1077 x 216 x 151 grid; 40 cores, one RTX A5000), end to end with three
views: `net` 33 s and 5 GB of RAM (16 s features, 9 s net), `baseline` 24 s and 5 GB, `labels` 30 s and 6 GB
(22 s for the native gradient, criteria and resampling). Reading the .vtu takes 2 s and rendering about 1 s.

On DrivAerML run 1 (147 M cells, a 1088 x 216 x 151 grid, the same box as AhmedML's in units of H), the net
trained on AhmedML alone runs unchanged: `net` 3.4 min and 29 GB, `baseline` 3.1 min and 29 GB, `labels`
4.3 min and 36 GB. Its known limitations there, judged by eye against `labels` and not yet scored:

- A vortex disc floats above the roof in `net` and `baseline` but not in `labels`, so it comes from the grid
  features (resampling or finite differences on the grid), which the net cannot correct.
- At tau_lambda2 = 0.5 only the strongest vortices are labelled (0.026 % of the core against 0.33 % on
  AhmedML): A-pillar, mirrors and one trailing vortex. No single threshold suits both datasets; see
  "DrivAerML transfer" in the project roadmap (https://claude.ai/artifact/R8BnfgG8SZRQEH1j5iFB1H).
- The thresholds and n_per_H are fixed by the checkpoint; they cannot be chosen at inference.
- One case of one dataset; the trailing wake is asymmetric in all three methods and should be checked on
  more runs.

## Tests

```sh
uv run pytest
```
