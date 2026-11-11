"""Verification: the file hash with its cache, the Zarr case store, and the vti writer."""

import hashlib

import numpy as np
import zarr
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLImageDataReader

from cfd_rgba_net.geometry import ImageSpec
from cfd_rgba_net.io import save_case_zarr, save_vti, sha256

SPEC = ImageSpec(origin=(1.0, 2.0, 3.0), dx=0.5, dims=(5, 4, 3))


def test_sha256_matches_hashlib_and_follows_the_file(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"first")
    assert sha256(f) == hashlib.sha256(b"first").hexdigest()
    assert sha256(f) == hashlib.sha256(b"first").hexdigest()     # again, now from the cache
    f.write_bytes(b"first and more")                              # a new size invalidates the cache
    assert sha256(f) == hashlib.sha256(b"first and more").hexdigest()


def test_save_case_zarr_round_trips_and_replaces_an_existing_store(tmp_path):
    path = tmp_path / "case.zarr"
    rng = np.random.default_rng(0)
    save_case_zarr(path, SPEC, arrays={"inputs": rng.random((2, *SPEC.dims), dtype=np.float32)},
                   groups={}, attrs={"tag": "first", "only_in_first": 1})
    inputs = rng.random((2, *SPEC.dims), dtype=np.float32)
    core = rng.random(SPEC.dims) > 0.5
    save_case_zarr(path, SPEC, arrays={"inputs": inputs}, groups={"ingredients": {"core": core}},
                   attrs={"tag": "second", "x": np.float32(1.5)})

    store = zarr.open_group(path, mode="r")
    np.testing.assert_array_equal(store["inputs"][...], inputs)
    np.testing.assert_array_equal(store["ingredients"]["core"][...], core)
    assert store.attrs["tag"] == "second" and store.attrs["x"] == 1.5
    assert store.attrs["grid"]["dims"] == list(SPEC.dims)
    assert "only_in_first" not in store.attrs                       # replaced, not merged
    assert [p.name for p in tmp_path.iterdir()] == ["case.zarr"]    # no .tmp or .old left behind


def test_save_vti_keeps_x_y_z_order_for_scalars_and_vectors(tmp_path):
    i, j, k = np.meshgrid(*(np.arange(n) for n in SPEC.dims), indexing="ij")
    scalar = (100 * i + 10 * j + k).astype(np.float32)             # a different value in every voxel
    vector = np.stack([scalar, -scalar])
    save_vti(tmp_path / "a.vti", SPEC, {"s": scalar, "v": vector})

    reader = vtkXMLImageDataReader(file_name=str(tmp_path / "a.vti"))
    reader.Update()
    image = reader.GetOutput()
    assert image.GetDimensions() == SPEC.dims and image.GetOrigin() == SPEC.origin
    s = vtk_to_numpy(image.GetPointData().GetArray("s"))
    v = vtk_to_numpy(image.GetPointData().GetArray("v"))
    np.testing.assert_array_equal(s.reshape(SPEC.dims, order="F"), scalar)
    assert v.shape == (scalar.size, 2)
    np.testing.assert_array_equal(v[:, 1].reshape(SPEC.dims, order="F"), -scalar)
