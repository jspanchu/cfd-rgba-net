"""Input contract, label rules and per-dataset case configuration, loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import CONTRACT_VERSION, LABEL_RULES_VERSION


def configs_dir() -> Path:
    """The configs/ folder: inside the installed wheel, or at the repository root."""
    here = Path(__file__).resolve().parent
    for d in (here / "configs", here.parents[1] / "configs"):
        if d.is_dir():
            return d
    raise FileNotFoundError("configs/ not found next to the package or at the repository root")


def _load(path: Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Contract:
    contract_version: str
    n_per_H: int
    channels: tuple[str, ...]

    @classmethod
    def load(cls, path: Path | None = None) -> "Contract":
        d = _load(path or configs_dir() / "contract_v1.yaml")
        c = cls(str(d["contract_version"]), int(d["n_per_H"]), tuple(d["channels"]))
        if c.contract_version != CONTRACT_VERSION:
            raise ValueError(f"contract file is v{c.contract_version}, code is v{CONTRACT_VERSION}")
        return c

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["channels"] = list(self.channels)
        return d


@dataclass(frozen=True)
class LabelRules:
    label_rules_version: str
    tau_lambda2: float
    tau_omega: float
    floor_exclude_H: float
    wall_erosion_H: float

    @classmethod
    def load(cls, path: Path | None = None) -> "LabelRules":
        d = _load(path or configs_dir() / "label_rules_v1.yaml")
        r = cls(**{k: (str(v) if k == "label_rules_version" else float(v)) for k, v in d.items()})
        if r.label_rules_version != LABEL_RULES_VERSION:
            raise ValueError(f"label rules file is v{r.label_rules_version}, code is v{LABEL_RULES_VERSION}")
        return r

    def wall_erosion_voxels(self, n_per_H: int) -> int:
        return round(self.wall_erosion_H * n_per_H)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VectorField:
    """A vector array by name, or three scalar arrays (components)."""
    name: str | None = None
    components: tuple[str, str, str] | None = None

    def label(self) -> str:
        return self.name or "+".join(self.components or ())


AXIS_NAMES = "xyz"
MAX_AXIS_ANGLE_DEG = 5.0


@dataclass(frozen=True)
class Axis:
    """A mesh axis and a sign: index 0, 1, 2 = x, y, z; sign +1 or -1."""
    index: int
    sign: int

    def __str__(self) -> str:
        return ("+" if self.sign > 0 else "-") + AXIS_NAMES[self.index]


def axis_of(direction, what: str = "direction") -> Axis:
    """The mesh axis a direction runs along. It must be within MAX_AXIS_ANGLE_DEG of that axis."""
    d = np.asarray(direction, dtype=float)
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
    U_inf: float | None   # None when reference_probe is set
    p_inf: float | None
    rho: float
    nu: float
    up_direction: tuple[float, float, float]
    ground: float | None  # None = auto; a coordinate along the up axis
    # A point (mesh coordinates) where U_inf = |U| and p_inf = p are read for each run, e.g. a tunnel's
    # reference sensor. Replaces fixed U_inf and p_inf.
    reference_probe: tuple[float, float, float] | None = None

    @property
    def up_axis(self) -> Axis:
        return axis_of(self.up_direction, "up direction")


@dataclass(frozen=True)
class CaseConfig:
    """Everything that describes one dataset family. Per-run values come from the files."""
    dataset: str
    files: dict[str, str]
    velocity: tuple[VectorField, ...]
    pressure: tuple[str, ...]
    flow: Flow
    body_scale: float = 1.0
    reference_length: float | None = None  # None = body height along up_direction
    domain: tuple[float, float, float, float, float, float] | None = None  # None = the mesh's bounds
    floor_offset_H: float = 1e-6  # the first voxel layer sits this far (units of H) above the ground
    hf_repo: str | None = None
    source: str | None = field(default=None, compare=False)

    def field_names(self) -> set[str]:
        """Every array name the velocity and pressure fields may come from."""
        names = set(self.pressure)
        for v in self.velocity:
            names |= {v.name} if v.name else set(v.components or ())
        return names

    @classmethod
    def load(cls, path: str | Path) -> "CaseConfig":
        path = Path(path)
        if not path.exists() and not path.suffix:
            path = configs_dir() / "datasets" / f"{path.name}.yaml"
        d = _load(path)
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
        probe = f.get("reference_probe")
        if probe is not None:
            if "U_inf" in f or "p_inf" in f:
                raise ValueError("flow: give either reference_probe or U_inf / p_inf, not both")
            if len(probe) != 3:
                raise ValueError(f"flow.reference_probe must be a point [x, y, z], got {probe}")
        elif "U_inf" not in f:
            raise ValueError("flow: U_inf (or reference_probe) is required")
        if domain != "mesh" and (not isinstance(domain, list) or len(domain) != 6):
            raise ValueError(f"domain must be 'mesh' or [xmin, xmax, ymin, ymax, zmin, zmax], got {domain}")
        cfg = cls(
            dataset=d["dataset"],
            files=dict(d["files"]),
            velocity=tuple(vel),
            pressure=tuple(d["fields"]["pressure"]),
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
            reference_length=None if ref == "height" else float(ref),
            domain=None if domain == "mesh" else tuple(float(v) for v in domain),  # type: ignore[arg-type]
            floor_offset_H=float(d.get("floor_offset_H", 1e-6)),
            hf_repo=d.get("hf_repo"),
            source=str(path),
        )
        cfg.flow.up_axis  # raises if the up direction does not run along a mesh axis
        return cfg

    def path(self, key: str, root: str | Path, run: int) -> Path:
        return Path(root) / self.files[key].format(run=run)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["velocity"] = [v.label() for v in self.velocity]
        return d
