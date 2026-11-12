# Identifying flow features with learned color & opacity transfer functions

## Install

```sh
uv sync
source .venv/bin/activate
```

## Layout

```
configs/contract_v1.yaml        input contract v1.0: n_per_H, channels, proximity ranges, SDF band (units of H)
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
tests/                          verification suite (pytest)
scripts/ahmedml/
    fix_wedges.py               rewrites the AhmedML VTUs with VTK's wedge point order (source of jspanchu/ahmedml)
    figures.py                  progress figures from the ahmedml_run<N>.zarr stores
    make_splits.py              assigns every run to a split (configs/splits/ahmedml_v1.csv)
    compute_scales.py           input scales of the train split, for configs/contract_v1.yaml
```

Input channels (contract v1.0) are `u, v, w` (velocity / U_inf along the mesh's own x, y, z axes), `Cp`,
and two proximity channels in place of a 0/1 body mask (sdf is the signed distance to the body, negative
inside; it is not an input itself):

- `body_proximity` b = max(0, 1 − sdf / (4 H)): 1 on the body surface, 0 at 4 H and beyond. Inside the body
  the line continues (sdf < 0), so b > 1 inside and b < 1 outside; the fluid is b < 1.
- `ground_proximity` g = max(0, 1 − h / H), h the height above the ground: 1 on the ground, 0 at 1 H and above.

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
    --out /data/cfd-rgba-net-data/ahmedML/out_v1 --workers 6
cfd-rgba-net-batch --dataset windsorml --runs 0-349 --out /data/cfd-rgba-net-data/windsorML/out_v1   # ~80 GB per build
```

Each case writes one Zarr v3 store, `<dataset>_run<N>.zarr` (64^3 chunks in 256^3 shards, Blosc zstd):
`inputs` (6, nx, ny, nz) float32 (u, v, w, Cp, body_proximity, ground_proximity), `labels`
(2, nx, ny, nz) uint8 (vortex, shear_layer) and an `ingredients/` group, with the case config, contract,
label rules, geometry, sanity checks, and the git commit and SHA-256 of every input file in the store's
attributes. `--save-vti` adds `.vti` files for
viewing. There is no stored fluid mask: take b < 1. It also counts the few voxels outside both the mesh and
the body (`checks.voxels.n_not_in_mesh_nor_body`, at most 84 per AhmedML case), where u, v, w and Cp are 0
and the labels are 0.

The WindsorML and DrivAerML configs are not yet verified on downloaded cases; see the TODOs in them.

## Tests

```sh
uv run pytest
```
