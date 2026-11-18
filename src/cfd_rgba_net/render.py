"""Render the net's vortex and shear-layer probabilities as a volume, optionally
with the body surface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401  registers the OpenGL (EGL or X) render window
import yaml
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import VTK_FLOAT
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkColorTransferFunction,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkVolume,
    vtkVolumeProperty,
    vtkWindowToImageFilter,
)
from vtkmodules.vtkRenderingVolume import vtkMultiVolume
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLGPUVolumeRayCastMapper

from .config import configs_dir
from .geometry import ImageSpec

CLASSES = ("none", "vortex", "shear_layer")  # channel order of the net's output

# View name -> (direction from the scene to the camera, view up). Flow is +x, so
# the wake runs to the right.
VIEWS = {
    "side": ((0, -1, 0), (0, 0, 1)),
    "top": ((0, 0, 1), (0, 1, 0)),
    "rear": ((1, -0.8, 0.5), (0, 0, 1)),
}


def load_config(path: str | Path | None = None) -> dict:
    """Read a render config.

    Parameters
    ----------
    path : str or pathlib.Path or None, optional
        The YAML file. ``None`` reads ``configs/render_default.yaml``.

    Returns
    -------
    dict
        The config; see ``configs/render_default.yaml`` for the keys.
    """
    with open(path or configs_dir() / "render_default.yaml") as fh:
        return yaml.safe_load(fh)


def _volume(
    prob: np.ndarray, spec: ImageSpec, feature: dict, unit: float
) -> tuple[vtkVolume, vtkImageData]:
    """Build the volume of one feature: its probabilities, colour and opacity
    curve.

    Parameters
    ----------
    prob : numpy.ndarray
        Probabilities of shape ``spec.dims``.
    spec : ImageSpec
        The voxel grid.
    feature : dict
        Config entry with ``color`` (RGB) and ``opacity`` (list of (probability,
        opacity) points).
    unit : float
        Opacity unit distance in mesh units.

    Returns
    -------
    volume : vtkVolume
        A volume without a mapper; ``vtkMultiVolume`` supplies the shared one.
    image : vtkImageData
        The probabilities on the grid, the mapper's input for this volume.
    """
    image = spec.to_vtk()
    scalars = numpy_to_vtk(
        np.ascontiguousarray(prob.reshape(-1, order="F"), dtype=np.float32),
        deep=True,
        array_type=VTK_FLOAT,
    )
    image.GetPointData().SetScalars(scalars)
    color = vtkColorTransferFunction()
    color.AddRGBPoint(0, *feature["color"])
    color.AddRGBPoint(1, *feature["color"])
    opacity = vtkPiecewiseFunction()
    opacity.AddPoint(0, 0)
    for p, a in feature["opacity"]:
        opacity.AddPoint(p, a)
    prop = vtkVolumeProperty(
        interpolation_type=1, scalar_opacity_unit_distance=unit
    )  # 1 = linear
    prop.SetColor(color)
    prop.SetScalarOpacity(opacity)
    return vtkVolume(property=prop), image


def _camera(ren: vtkRenderer, bounds: tuple, view: str, aspect: float):
    """Aim an orthographic camera at a box so that the box fills the image.

    Parameters
    ----------
    ren : vtkRenderer
        The renderer whose camera is set.
    bounds : tuple
        ``(xmin, xmax, ymin, ymax, zmin, zmax)`` of the region to frame.
    view : str
        A key of ``VIEWS``.
    aspect : float
        Image width over height.
    """
    to_cam, up = (np.asarray(v, float) for v in VIEWS[view])
    to_cam /= np.linalg.norm(to_cam)
    b = np.asarray(bounds, float).reshape(3, 2)
    centre = b.mean(1)
    corners = (
        np.array([[x, y, z] for x in b[0] for y in b[1] for z in b[2]]) - centre
    )
    right = np.cross(up, to_cam)
    right /= np.linalg.norm(right)
    true_up = np.cross(to_cam, right)
    half_w, half_h = (
        np.abs(corners @ right).max(),
        np.abs(corners @ true_up).max(),
    )
    cam = ren.GetActiveCamera()
    cam.SetParallelProjection(True)
    cam.SetFocalPoint(*centre)
    cam.SetPosition(*(centre + to_cam * 10 * np.abs(corners).max()))
    cam.SetViewUp(*up)
    cam.SetParallelScale(1.05 * max(half_h, half_w / aspect))
    ren.ResetCameraClippingRange(bounds)


def render_probabilities(
    probs,
    spec: ImageSpec,
    config: dict | None = None,
    views: tuple[str, ...] = ("side",),
    body=None,
) -> dict[str, np.ndarray]:
    """Render vortex and shear-layer probabilities from a volume ray caster.

    Probability fields drive that feature's opacity through an individual
    transfer function, and each feature has its own colour, both set in
    the config (see ``configs/render_default.yaml``). The reference labels
    (uint8 0/1) work too. The frame is up = +z, flow = +x, as in the AhmedML
    grid.

    Parameters
    ----------
    probs : torch.Tensor or numpy.ndarray
        Probabilities of shape ``(3, nx, ny, nz)`` over ``CLASSES``. Only vortex
        and shear_layer are drawn.
    spec : ImageSpec
        The voxel grid.
    config : dict or None, optional
        Render config, see ``load_config``. ``None`` uses the default.
    views : tuple[str, ...], optional
        Names from ``VIEWS``.
    body : vtkPolyData or None, optional
        The body surface (``geometry.read_body``). Drawn opaque; also sets the
        framed region, which is the body plus ``frame.wake`` body lengths
        downstream. Without it the whole grid is framed.

    Returns
    -------
    dict[str, numpy.ndarray]
        Per view, a uint8 array of shape ``(height, width, 3)``.
    """
    config = config or load_config()
    if hasattr(probs, "detach"):
        probs = probs.detach().cpu().numpy()
    volumes = [
        (probs[CLASSES.index(name)], feature)
        for name, feature in config["features"].items()
    ]
    return render_volumes(volumes, spec, config, views, body)


def render_volumes(
    volumes: list[tuple[np.ndarray, dict]],
    spec: ImageSpec,
    config: dict | None = None,
    views: tuple[str, ...] = ("side",),
    body=None,
) -> dict[str, np.ndarray]:
    """Render scalar volumes in [0, 1], each with its own colour and opacity
    curve, from a volume ray caster.

    The volumes are in one ``vtkMultiVolume``, so each has its own colour and
    opacity map, the rays composite all of them, and the body, an opaque
    surface, hides the volume behind it. Rendering is offscreen and runs on the
    GPU (EGL when there is no X display).

    Parameters
    ----------
    volumes : list[tuple[numpy.ndarray, dict]]
        Per volume, its values of shape ``spec.dims`` and a config entry with
        ``color`` (RGB) and ``opacity`` (list of (value, opacity) points), like
        the entries of the config's ``features``.
    spec : ImageSpec
        The voxel grid.
    config : dict or None, optional
        Render config, see ``load_config``; its ``features`` are not read here.
        ``None`` uses the default.
    views : tuple[str, ...], optional
        Names from ``VIEWS``.
    body : vtkPolyData or None, optional
        The body surface, drawn opaque and framed as in
        ``render_probabilities``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Per view, a uint8 array of shape ``(height, width, 3)``.
    """
    config = config or load_config()
    width, height = config["size"]
    ren = vtkRenderer(background=config["background"])
    win = vtkRenderWindow(off_screen_rendering=True, size=(width, height))
    win.AddRenderer(ren)

    multi = vtkMultiVolume()
    mapper = vtkOpenGLGPUVolumeRayCastMapper()
    multi.SetMapper(mapper)
    for port, (values, feature) in enumerate(volumes):
        vol, image = _volume(
            values, spec, feature, config["opacity_unit_voxels"] * spec.dx
        )
        mapper.SetInputDataObject(port, image)
        multi.SetVolume(vol, port)
    ren.AddVolume(multi)

    bounds = spec.bounds
    if body is not None:
        normals = vtkPolyDataNormals(
            input_data=body, feature_angle=40
        )  # smooth shading on the rounded edges
        actor = vtkActor(
            mapper=vtkPolyDataMapper(input_connection=normals.GetOutputPort())
        )
        actor.GetProperty().SetColor(*config["body"]["color"])
        ren.AddActor(actor)
        b = body.GetBounds()
        bounds = (b[0], b[1] + config["frame"]["wake"] * (b[1] - b[0]), *b[2:])

    images = {}
    for view in views:
        _camera(ren, bounds, view, width / height)
        win.Render()
        grab = vtkWindowToImageFilter(input=win)
        grab.Update()
        out = grab.GetOutput()
        w, h, _ = out.GetDimensions()
        images[view] = vtk_to_numpy(out.GetPointData().GetScalars()).reshape(
            h, w, -1
        )[::-1, :, :3]
    return images


def write_png(path: str | Path, image: np.ndarray):
    """Write an RGB image to a PNG file.

    Parameters
    ----------
    path : str or pathlib.Path
        The output file.
    image : numpy.ndarray
        Uint8 array of shape ``(height, width, 3)``, first row at the top.
    """
    h, w, _ = image.shape
    vtk_image = vtkImageData(dimensions=(w, h, 1))
    vtk_image.GetPointData().SetScalars(
        numpy_to_vtk(
            np.ascontiguousarray(image[::-1].reshape(-1, 3)), deep=True
        )
    )
    vtkPNGWriter(file_name=str(path), input_data=vtk_image).Write()
