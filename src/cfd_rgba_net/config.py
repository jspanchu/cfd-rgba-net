"""Input contract, label rules and per-dataset case configuration, loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import CONTRACT_VERSION, LABEL_RULES_VERSION


def configs_dir() -> Path:
    """Locate the ``configs/`` folder.

    Looks inside the installed wheel first, then at the repository root.

    Returns
    -------
    pathlib.Path
        The ``configs/`` directory.

    Raises
    ------
    FileNotFoundError
        If neither location holds a ``configs/`` directory.
    """
    here = Path(__file__).resolve().parent
    for d in (here / "configs", here.parents[1] / "configs"):
        if d.is_dir():
            return d
    raise FileNotFoundError("configs/ not found next to the package or at the repository root")


def _load(path: Path) -> dict[str, Any]:
    """Read a YAML file.

    Parameters
    ----------
    path : pathlib.Path
        The YAML file to read.

    Returns
    -------
    dict[str, Any]
        The parsed top-level mapping.
    """
    with path.open() as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Contract:
    """The input contract: what the model's input channels mean. Fixed by the model, not by the user.

    Attributes
    ----------
    contract_version : str
        Version of the contract. Must equal ``cfd_rgba_net.CONTRACT_VERSION``.
    n_per_H : int
        Voxels per reference length H, so the voxel size is ``dx = H / n_per_H``.
    channels : tuple[str, ...]
        Names of the input channels, in order.
    body_proximity_range_H : float
        Distance from the body, in units of H, at which ``body_proximity`` reaches 0.
    ground_proximity_range_H : float
        Height above the ground, in units of H, at which ``ground_proximity`` reaches 0.
    sdf_exact_band_H : float
        The signed distance to the body is exact within this distance, in units of H, of the body's
        bounding box, and a distance transform beyond it.
    """
    contract_version: str
    n_per_H: int
    channels: tuple[str, ...]
    body_proximity_range_H: float
    ground_proximity_range_H: float
    sdf_exact_band_H: float

    @classmethod
    def load(cls, path: Path | None = None) -> "Contract":
        """Load the contract from YAML.

        Parameters
        ----------
        path : pathlib.Path or None, optional
            The contract file. Defaults to ``configs/contract_v1.yaml``.

        Returns
        -------
        Contract
            The loaded contract.

        Raises
        ------
        ValueError
            If the file's ``contract_version`` differs from ``cfd_rgba_net.CONTRACT_VERSION``.
        """
        d = _load(path or configs_dir() / "contract_v1.yaml")
        version = str(d["contract_version"])
        # check first: a file of another version may not have the keys read below
        if version != CONTRACT_VERSION:
            raise ValueError(f"contract file is v{version}, code is v{CONTRACT_VERSION}")
        return cls(version, int(d["n_per_H"]), tuple(d["channels"]),
                   float(d["body_proximity_range_H"]), float(d["ground_proximity_range_H"]),
                   float(d["sdf_exact_band_H"]))

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict that serializes to JSON.

        Returns
        -------
        dict[str, Any]
            The fields by name, with ``channels`` as a list.
        """
        d = asdict(self)
        d["channels"] = list(self.channels)
        return d


@dataclass(frozen=True)
class LabelRules:
    """Thresholds of the label rules. The same rules apply to every dataset family.

    Thresholds act on non-dimensional quantities. Lengths are in units of H.

    Attributes
    ----------
    label_rules_version : str
        Version of the rules. Must equal ``cfd_rgba_net.LABEL_RULES_VERSION``.
    tau_lambda2 : float
        A voxel is a vortex when ``lambda2* < -tau_lambda2`` and ``Q* > 0``.
    tau_omega : float
        A voxel is a shear layer when ``|omega|* > tau_omega`` and it is not a vortex.
    floor_exclude_H : float
        Height above the ground, in units of H, below which voxels get no label (ground boundary layer).
    wall_erosion_H : float
        Distance from any wall, in units of H, within which voxels get no label (wall boundary layer
        dominates the resampled gradients there).
    """
    label_rules_version: str
    tau_lambda2: float
    tau_omega: float
    floor_exclude_H: float
    wall_erosion_H: float

    @classmethod
    def load(cls, path: Path | None = None) -> "LabelRules":
        """Load the label rules from YAML.

        Parameters
        ----------
        path : pathlib.Path or None, optional
            The rules file. Defaults to ``configs/label_rules_v1.yaml``.

        Returns
        -------
        LabelRules
            The loaded rules.

        Raises
        ------
        ValueError
            If the file's ``label_rules_version`` differs from ``cfd_rgba_net.LABEL_RULES_VERSION``.
        TypeError
            If the file misses a field or holds an unknown one.
        """
        d = _load(path or configs_dir() / "label_rules_v1.yaml")
        r = cls(**{k: (str(v) if k == "label_rules_version" else float(v)) for k, v in d.items()})
        if r.label_rules_version != LABEL_RULES_VERSION:
            raise ValueError(f"label rules file is v{r.label_rules_version}, code is v{LABEL_RULES_VERSION}")
        return r

    def wall_erosion_voxels(self, n_per_H: int) -> int:
        """Convert the wall erosion distance to a whole number of voxels.

        Parameters
        ----------
        n_per_H : int
            Voxels per reference length H.

        Returns
        -------
        int
            ``wall_erosion_H * n_per_H``, rounded to the nearest voxel.
        """
        return round(self.wall_erosion_H * n_per_H)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict that serializes to JSON.

        Returns
        -------
        dict[str, Any]
            The fields by name.
        """
        return asdict(self)


@dataclass(frozen=True)
class VectorField:
    """A vector array by name, or three scalar arrays (components).

    Attributes
    ----------
    name : str or None
        Name of one array that holds the vector.
    components : tuple[str, str, str] or None
        Names of three scalar arrays: the x, y and z components.
    """
    name: str | None = None
    components: tuple[str, str, str] | None = None

    def label(self) -> str:
        """Describe the field for messages and serialization.

        Returns
        -------
        str
            ``name``, or the component names joined by ``+``.
        """
        return self.name or "+".join(self.components or ())


AXIS_NAMES = "xyz"
# The voxel grid uses the mesh's own axes, so a direction that is further off an axis cannot be
# represented without rotating the data.
MAX_AXIS_ANGLE_DEG = 5.0


@dataclass(frozen=True)
class Axis:
    """A mesh axis and a sign.

    Attributes
    ----------
    index : int
        0, 1 or 2 for x, y or z.
    sign : int
        +1 or -1.
    """
    index: int
    sign: int

    def __str__(self) -> str:
        return ("+" if self.sign > 0 else "-") + AXIS_NAMES[self.index]


def axis_of(direction, what: str = "direction") -> Axis:
    """Snap a direction to the mesh axis it runs along.

    Parameters
    ----------
    direction : sequence of float
        The direction vector, three components. It need not be normalized.
    what : str, optional
        Name of the direction, used in error messages.

    Returns
    -------
    Axis
        The nearest mesh axis, signed like the direction's largest component.

    Raises
    ------
    ValueError
        If ``direction`` does not have three components, has zero length, or is more than
        ``MAX_AXIS_ANGLE_DEG`` degrees from the nearest mesh axis.
    """
    d = np.asarray(direction, dtype=float)
    if d.shape != (3,):
        raise ValueError(f"{what} {tuple(direction)} must have 3 components")
    n = np.linalg.norm(d)
    if not n > 0:
        raise ValueError(f"{what} {direction} has zero length")
    d = d / n
    i = int(np.argmax(np.abs(d)))
    angle = np.degrees(np.arccos(min(1.0, abs(d[i]))))
    if angle > MAX_AXIS_ANGLE_DEG:
        raise ValueError(f"{what} {tuple(direction)} is {angle:.1f} deg from the nearest mesh axis; the voxel grid "
                         f"uses the mesh's own axes, so it must be within {MAX_AXIS_ANGLE_DEG} deg of one")
    return Axis(i, 1 if d[i] > 0 else -1)


@dataclass(frozen=True)
class Flow:
    """Freestream and orientation of the flow case.

    Attributes
    ----------
    U_inf : float or None
        Freestream speed. ``None`` when ``reference_probe`` is set.
    p_inf : float or None
        Freestream pressure. ``None`` when ``reference_probe`` is set.
    rho : float
        Density (1 for a kinematic-pressure case).
    nu : float
        Kinematic viscosity.
    up_direction : tuple[float, float, float]
        Direction that points up, away from the ground. Must run along a mesh axis, see ``axis_of``.
    ground : float or None
        Ground coordinate along the up axis. ``None`` means the lowest mesh point along it.
    reference_probe : tuple[float, float, float] or None
        A point, in mesh coordinates, where each run's ``U_inf = |U|`` and ``p_inf = p`` are read, such
        as a tunnel's reference sensor. Replaces the fixed ``U_inf`` and ``p_inf``.
    """
    U_inf: float | None
    p_inf: float | None
    rho: float
    nu: float
    up_direction: tuple[float, float, float]
    ground: float | None
    reference_probe: tuple[float, float, float] | None = None


def check_up_axis_consistency(flow: Flow) -> None:
    """Check that the flow's up direction runs along a mesh axis.

    Parameters
    ----------
    flow : Flow
        The flow to check.

    Raises
    ------
    ValueError
        If ``flow.up_direction`` fails ``axis_of``: wrong number of components, zero length, or too far
        from every mesh axis.
    """
    axis_of(flow.up_direction, "up direction")


# Where a dataset stores the velocity and pressure fields: at the mesh points or in the cells.
FIELD_SOURCES = ("point", "cell")


@dataclass(frozen=True)
class CaseConfig:
    """Everything that describes one dataset family. Per-run values come from the files.

    Attributes
    ----------
    dataset : str
        Dataset name.
    files : dict[str, str]
        File paths relative to the data root, by role (``volume``, ``body``, ...). ``{run}`` stands for
        the run number.
    velocity : tuple[VectorField, ...]
        Candidate velocity fields. The first one found in the ``field_source`` data wins.
    pressure : tuple[str, ...]
        Candidate pressure array names. The first one found in the ``field_source`` data wins.
    field_source : str
        ``"point"`` or ``"cell"``: the mesh data that holds velocity and pressure. There is no fallback
        to the other kind. With ``"cell"``, the fields are averaged to the points.
    flow : Flow
        Freestream and orientation.
    body_scale : float
        Factor from STL units to volume-mesh units.
    body_merge_tolerance : float or None
        Merge STL vertices closer than this fraction of the body's bounding-box diagonal, for datasets
        whose STL repeats vertices with small differences. ``None`` skips the cleaning step.
    reference_length : float or None
        The reference length H. ``None`` means the body's extent along the up direction.
    domain : tuple[float, float, float, float, float, float] or None
        ``(xmin, xmax, ymin, ymax, zmin, zmax)`` of the voxel grid. ``None`` means the mesh bounds.
    floor_offset_H : float
        Distance, in units of H, of the first voxel layer above the ground. Datasets whose lowest mesh
        points sit slightly above the ground need a positive value.
    grid_inset_H : float
        Distance, in units of H, that the voxel grid stays inside the domain's side faces. A small positive
        value keeps voxels off the mesh's faces, where the resampling probe can miss points.
    check_vorticity : str or None
        Name of the dataset's own vorticity array in the cell data. The label ingredients are compared
        with it as a sanity check. ``None`` skips the check.
    check_cp : str or None
        Name of the dataset's own pressure-coefficient array in the point data. ``Cp`` is compared with
        it as a sanity check. ``None`` skips the check.
    hf_repo : str or None
        Hugging Face dataset repository that holds the run files.
    source : str or None
        Path of the YAML file this config was loaded from. Excluded from equality comparison.
    """
    dataset: str
    files: dict[str, str]
    velocity: tuple[VectorField, ...]
    pressure: tuple[str, ...]
    field_source: str
    flow: Flow
    body_scale: float = 1.0
    body_merge_tolerance: float | None = None
    reference_length: float | None = None
    domain: tuple[float, float, float, float, float, float] | None = None
    floor_offset_H: float = 0.0
    grid_inset_H: float = 0.0
    check_vorticity: str | None = None
    check_cp: str | None = None
    hf_repo: str | None = None
    source: str | None = field(default=None, compare=False)

    def field_names(self) -> set[str]:
        """List every array name the velocity and pressure fields may come from.

        Returns
        -------
        set[str]
            The pressure names, the velocity vector names and the velocity component names.
        """
        names = set(self.pressure)
        for v in self.velocity:
            names |= {v.name} if v.name else set(v.components or ())
        return names

    @classmethod
    def load(cls, path: str | Path) -> "CaseConfig":
        """Load a dataset configuration from YAML.

        Parameters
        ----------
        path : str or pathlib.Path
            The YAML file, or a bare dataset name such as ``"ahmedml"`` that resolves to
            ``configs/datasets/<name>.yaml``.

        Returns
        -------
        CaseConfig
            The loaded configuration, with ``source`` set to the resolved path.

        Raises
        ------
        ValueError
            If ``field_source`` is not ``"point"`` or ``"cell"``; if a velocity entry does not list three
            components; if ``flow`` gives both or neither of
            ``reference_probe`` and ``U_inf``; if ``reference_probe`` is not a point; if ``domain`` is
            not ``mesh`` or six numbers with min below max on each axis; or if ``up_direction`` fails
            ``check_up_axis_consistency``.
        """
        path = Path(path)
        if not path.exists() and not path.suffix:
            path = configs_dir() / "datasets" / f"{path.name}.yaml"
        d = _load(path)
        field_source = d.get("field_source")
        if field_source not in FIELD_SOURCES:
            raise ValueError(f"field_source must be one of {FIELD_SOURCES}, got {field_source!r}")
        vel = []
        for v in d["fields"]["velocity"]:
            if isinstance(v, str):
                vel.append(VectorField(name=v))
            else:
                comps = tuple(v["components"])
                if len(comps) != 3:
                    raise ValueError(f"velocity components must be 3 names, got {comps}")
                vel.append(VectorField(components=comps))  # type: ignore[arg-type]
        f = d["flow"]
        ground = f.get("ground", "auto")
        ref = d.get("body", {}).get("reference_length", "height")
        domain = d.get("domain", "mesh")
        merge_tol = d.get("body", {}).get("merge_tolerance")
        check = d.get("check_fields", {})
        probe = f.get("reference_probe")
        if probe is not None:
            if "U_inf" in f or "p_inf" in f:
                raise ValueError("flow: give either reference_probe or U_inf / p_inf, not both")
            if len(probe) != 3:
                raise ValueError(f"flow.reference_probe must be a point [x, y, z], got {probe}")
        elif "U_inf" not in f:
            raise ValueError("flow: U_inf (or reference_probe) is required")
        if domain != "mesh":
            if not isinstance(domain, list) or len(domain) != 6:
                raise ValueError(f"domain must be 'mesh' or [xmin, xmax, ymin, ymax, zmin, zmax], got {domain}")
            domain = tuple(float(v) for v in domain)
            if any(lo >= hi for lo, hi in zip(domain[0::2], domain[1::2])):
                raise ValueError(f"domain must have min < max on every axis, got {list(domain)}")
        cfg = cls(
            dataset=d["dataset"],
            files=dict(d["files"]),
            velocity=tuple(vel),
            pressure=tuple(d["fields"]["pressure"]),
            field_source=field_source,
            flow=Flow(
                U_inf=None if probe is not None else float(f["U_inf"]),
                p_inf=None if probe is not None else float(f.get("p_inf", 0.0)),
                rho=float(f.get("rho", 1.0)),
                nu=float(f["nu"]),
                up_direction=tuple(float(x) for x in f.get("up_direction", (0, 0, 1))),  # type: ignore[arg-type]
                ground=None if ground == "auto" else float(ground),
                reference_probe=None if probe is None else tuple(float(v) for v in probe),  # type: ignore[arg-type]
            ),
            body_scale=float(d.get("body", {}).get("scale", 1.0)),
            body_merge_tolerance=None if merge_tol is None else float(merge_tol),
            reference_length=None if ref == "height" else float(ref),
            domain=None if domain == "mesh" else domain,  # type: ignore[arg-type]
            floor_offset_H=float(d.get("floor_offset_H", 0.0)),
            grid_inset_H=float(d.get("grid_inset_H", 0.0)),
            check_vorticity=check.get("vorticity"),
            check_cp=check.get("cp"),
            hf_repo=d.get("hf_repo"),
            source=str(path),
        )
        check_up_axis_consistency(cfg.flow)
        return cfg

    def path(self, key: str, root: str | Path, run: int) -> Path:
        """Resolve the path of one of the case's files.

        Parameters
        ----------
        key : str
            Role of the file, a key of ``files`` such as ``"volume"``.
        root : str or pathlib.Path
            The data root that the file paths are relative to.
        run : int
            The run number that replaces ``{run}`` in the file path.

        Returns
        -------
        pathlib.Path
            ``root`` joined with the file path for this run.

        Raises
        ------
        KeyError
            If ``key`` is not in ``files``.
        """
        return Path(root) / self.files[key].format(run=run)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for the case metadata.

        Returns
        -------
        dict[str, Any]
            The fields by name, with nested dataclasses expanded and ``velocity`` as a list of labels.
        """
        d = asdict(self)
        d["velocity"] = [v.label() for v in self.velocity]
        return d
