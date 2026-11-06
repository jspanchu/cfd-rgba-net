# Identifying flow features with learned color & opacity transfer functions

## Install

```sh
uv sync
source .venv/bin/activate
```

## Layout

```
configs/contract_v1.yaml        input contract v1.0: n_per_H, channels, resample box (units of H)
configs/label_rules_v1.yaml     label rules v1 thresholds (units of H)
configs/datasets/*.yaml         per-dataset case parameters: files, field names, U_inf, frame, ground
src/cfd_rgba_net/
    config.py                   loads the three kinds of YAML
    geometry.py                 body STL, voxel grid on the mesh axes, signed distance
    preprocess.py               native mesh -> model inputs (shared by training and inference)
    ingredients.py              Q*, lambda2*, |omega|* on the native mesh (label teacher)
    label.py                    label rules v1 on the voxel grid
    io.py                       Zarr v3 case store, vti writer, provenance (git commit, input SHA-256)
    build_case.py               one training case, end to end (CLI: cfd-rgba-net-build-case)
tests/                          verification suite (pytest)
scripts/ahmedml/
    fix_wedges.py               rewrites the AhmedML VTUs with VTK's wedge point order (source of jspanchu/ahmedml)
    figures.py                  progress figures from the ahmedml_run<N>.zarr stores
```

Model inputs (contract v1.0) are `u, v, w` (velocity / U_inf along the mesh's own x, y, z axes), `Cp`
and `sdf` (signed distance to the body / H), on a grid with dx = H / 32 aligned with the mesh axes that
fills the dataset's domain (`domain` in its YAML, default the mesh bounds).
There is no axis convention: the flow may run along any mesh axis, in either sign.

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
`inputs` (5, nx, ny, nz) float32, `fluid`, `labels` (2, nx, ny, nz) uint8 (vortex, shear_layer) and an
`ingredients/` group, with the case config, contract, label rules, geometry, sanity checks, and the git
commit and SHA-256 of every input file in the store's attributes. `--save-vti` adds `.vti` files for
viewing.

The WindsorML and DrivAerML configs are not yet verified on downloaded cases; see the TODOs in them.

## Tests

```sh
uv run pytest
```
