# `cfd_rgba_net` — a line-by-line developer walkthrough

This document walks through every line of `src/cfd_rgba_net/` (package version `0.1.0`, input
contract `1.0`, label rules `1`). Line numbers refer to the files as they are on the
`phase0-foundations` branch at the time of writing. Modules are presented in dependency order, so
each one only uses things already explained:

| # | Module | Lines | Role |
|---|---|---|---|
| 1 | [`__init__.py`](#1-__init__py) | 10 | version constants |
| 2 | [`config.py`](#2-configpy) | 211 | YAML → frozen dataclasses (contract, label rules, dataset config) |
| 3 | [`geometry.py`](#3-geometrypy) | 189 | body STL, voxel grid, signed distance |
| 4 | [`io.py`](#4-iopy) | 126 | provenance, Zarr v3 / VTI / JSON writers |
| 5 | [`preprocess.py`](#5-preprocesspy) | 221 | native mesh → 5-channel model inputs |
| 6 | [`ingredients.py`](#6-ingredientspy) | 101 | Q*, λ2*, \|ω\|* on the native mesh (teacher only) |
| 7 | [`label.py`](#7-labelpy) | 59 | label rules v1 → vortex / shear-layer masks |
| 8 | [`build_case.py`](#8-build_casepy) | 168 | one case end-to-end, CLI `cfd-rgba-net-build-case` |
| 9 | [`batch.py`](#9-batchpy) | 124 | many cases with HF download, CLI `cfd-rgba-net-batch` |

A final section, [Cross-cutting notes and caveats](#10-cross-cutting-notes-and-caveats), collects
behaviours that span modules, unused code, and sharp edges found while reading.

---

## 0. The big picture

The package turns one CFD run (an unstructured `.vtu` volume plus an `.stl` of the body) into one
training sample for a network that predicts per-voxel flow-feature probabilities (which a renderer
later maps to RGBA). The data flow, exactly as `build_case.build_case` executes it:

```
volume_N.vtu ──read_volume──▶ vtkUnstructuredGrid (only the needed arrays)
                               │
                               ▼ prepare_mesh
                     PreparedMesh: U_nd (point), Cp (point), U_cell (cell, optional),
                                   U_inf, p_inf, ground
                               │
body_N.stl ──read_body──▶ vtkPolyData ──case_geometry──▶ CaseGeometry (H, ground, up axis)
                                                          │
                                                          ▼ image_spec(contract, domain)
                                                    ImageSpec (origin, dx = H/32, dims)
                               │
                               ▼ add_ingredients (on the native mesh)
                     + point arrays Q_nd, lambda2_nd, omega_mag_nd
                               │
                               ▼ resample (vtkResampleToImage onto ImageSpec)
                     dict of (nx,ny,nz) / (3,nx,ny,nz) arrays + "valid"
                               │
                 ┌─────────────┴──────────────┐
                 ▼ assemble_inputs            ▼ make_labels
       inputs (5,nx,ny,nz): u v w Cp sdf/H   labels (2,nx,ny,nz): vortex, shear_layer
       fluid  (nx,ny,nz)                     core   (nx,ny,nz)
                 └─────────────┬──────────────┘
                               ▼ io.save_case_zarr (+ optional io.save_vti)
                         <dataset>_run<N>.zarr
```

Three design rules show up everywhere and explain many individual lines:

1. **Nothing is rotated.** The voxel grid uses the mesh's own x, y, z axes. "Up" only decides the
   reference length H, where the ground is, and which layers are excluded near the floor.
2. **Everything the model sees is non-dimensional.** Velocities are divided by `U_inf`, pressure
   becomes `Cp`, lengths are in units of `H`, and the voxel size is fixed at `H / n_per_H`.
3. **Training and inference share code.** `preprocess.py` and `geometry.signed_distance` are the
   "contract"; any numerical change there must bump `CONTRACT_VERSION`.

Array layout convention: every voxel array is indexed `[i, j, k] = (x, y, z)`, with optional
leading channel axes, e.g. `(C, nx, ny, nz)`. VTK stores image data with x varying fastest, which
is NumPy's Fortran order. That is why every reshape between VTK and NumPy uses `order="F"`.

---

## 1. `__init__.py`

```python
1  """cfd-rgba-net: learned per-voxel flow-feature probabilities for CFD volume rendering."""
```
Package docstring. "RGBA" in the name refers to the eventual use: the predicted per-voxel class
probabilities drive a color + opacity transfer function in a volume renderer.

```python
3  __version__ = "0.1.0"
```
The package version. Kept in sync by hand with `version = "0.1.0"` in `pyproject.toml`. It is
recorded in every case's provenance (`io.provenance`).

```python
5-7  CONTRACT_VERSION = "1.0"
```
The version of the **input contract**: the meaning, order and units of the model's input channels.
The comment states the rule: any preprocessing change that alters the numbers a model sees must
bump this. `config.Contract.load` refuses a `contract_v1.yaml` whose `contract_version` differs, and
the value is stored in each case's provenance.

```python
9-10  LABEL_RULES_VERSION = "1"
```
Version of the label rules, i.e. `configs/label_rules_v1.yaml` together with the code in
`label.py`. Checked by `config.LabelRules.load` and recorded in provenance.

Both versions are strings (not numbers) so that comparisons against YAML values are exact string
compares (the loaders `str()` the YAML value first, so `1.0` written unquoted in YAML would become
`"1.0"`, but `1` would become `"1"`).

---

## 2. `config.py`

Purpose: load the three kinds of YAML (contract, label rules, per-dataset case config) into
immutable dataclasses, and validate them early.

### Imports (lines 1–12)

```python
3  from __future__ import annotations
```
Makes every annotation a lazily-evaluated string. This allows `-> "Contract"`-style forward
references and the `X | None` syntax without runtime cost. (It is used in every module.)

```python
5  from dataclasses import dataclass, field, asdict
6  from pathlib import Path
7  from typing import Any
9  import numpy as np
10 import yaml
12 from . import CONTRACT_VERSION, LABEL_RULES_VERSION
```
`dataclass`/`field`/`asdict` build and serialize the config objects. `numpy` is needed only by
`axis_of`. `yaml` is PyYAML. The two version constants come from `__init__.py`.

### `configs_dir()` (lines 15–21)

```python
15 def configs_dir() -> Path:
17     here = Path(__file__).resolve().parent
18     for d in (here / "configs", here.parents[1] / "configs"):
19         if d.is_dir():
20             return d
21     raise FileNotFoundError(...)
```
- Line 17: `here` is the package directory, e.g. `<repo>/src/cfd_rgba_net` (`resolve()` follows
  symlinks so editable installs work).
- Line 18: two candidates, tried in order:
  1. `here / "configs"` — the **installed wheel** case. `pyproject.toml` has
     `[tool.hatch.build.targets.wheel.force-include] "configs" = "cfd_rgba_net/configs"`, which
     copies the repo's `configs/` into the package.
  2. `here.parents[1] / "configs"` — the **source checkout** case. `parents[0]` is `src/`,
     `parents[1]` is the repo root.
- Line 21: neither exists → explicit error.

### `_load(path)` (lines 24–26)

Opens the file and returns `yaml.safe_load(...)` — a plain `dict`. `safe_load` refuses arbitrary
Python object tags. `Path(path)` lets callers pass strings.

### `Contract` (lines 29–46)

```python
29 @dataclass(frozen=True)
30 class Contract:
31     contract_version: str
32     n_per_H: int
33     channels: tuple[str, ...]
```
`frozen=True` makes instances immutable and hashable. Fields mirror `configs/contract_v1.yaml`:
`n_per_H` = voxels per reference length (32 in v1, so `dx = H/32`), and `channels` =
`[u, v, w, Cp, sdf]`. `channels` is a tuple (not a list) because frozen dataclasses should hold
immutable values.

```python
35 @classmethod
36 def load(cls, path: Path | None = None) -> "Contract":
37     d = _load(path or configs_dir() / "contract_v1.yaml")
38     c = cls(str(d["contract_version"]), int(d["n_per_H"]), tuple(d["channels"]))
39     if c.contract_version != CONTRACT_VERSION:
40         raise ValueError(...)
41     return c
```
- Line 37: default file is `contract_v1.yaml` in `configs_dir()`.
- Line 38: explicit coercions (`str`, `int`, `tuple`) so YAML typing quirks do not leak in. Any extra
  keys in the YAML are ignored; a missing key raises `KeyError`.
- Lines 39–40: the code/config version guard.

```python
43 def to_dict(self):
44     d = asdict(self)
45     d["channels"] = list(self.channels)
46     return d
```
For metadata. `asdict` keeps tuples as tuples; line 45 turns `channels` into a list (JSON-friendly,
and symmetric with the YAML).

### `LabelRules` (lines 49–69)

```python
50 class LabelRules:
51     label_rules_version: str
52     tau_lambda2: float        # vortex threshold on -lambda2*
53     tau_omega: float          # shear threshold on |omega|*
54     floor_exclude_H: float    # exclude this height above the ground (units of H)
55     wall_erosion_H: float     # erode the fluid mask by this distance (units of H)
```
Mirror of `configs/label_rules_v1.yaml` (v1 values: 0.5, 2.0, 0.12, 0.03125).

```python
58 def load(cls, path=None):
59     d = _load(path or configs_dir() / "label_rules_v1.yaml")
60     r = cls(**{k: (str(v) if k == "label_rules_version" else float(v)) for k, v in d.items()})
61     if r.label_rules_version != LABEL_RULES_VERSION: raise ValueError(...)
63     return r
```
- Line 60: unlike `Contract.load`, every YAML key is passed straight to the constructor via `**`.
  Consequence: an **unknown key in the YAML raises `TypeError`** (unexpected keyword), and a missing
  key also raises `TypeError`. The YAML is therefore schema-checked by the dataclass signature.
  Every value except the version is coerced to `float`.
- Lines 61–62: version guard, same as the contract.

```python
65 def wall_erosion_voxels(self, n_per_H: int) -> int:
66     return round(self.wall_erosion_H * n_per_H)
```
Converts the erosion distance from units of H to an integer number of voxels:
`0.03125 * 32 = 1.0 → 1`. Python's `round` is banker's rounding (`round(2.5) == 2`), which only
matters if someone picks a value landing on exactly .5 voxels.

`to_dict` (68–69) is a plain `asdict`.

### `VectorField` (lines 72–79)

```python
73 class VectorField:
74     """A vector array by name, or three scalar arrays (components)."""
75     name: str | None = None
76     components: tuple[str, str, str] | None = None
78     def label(self) -> str:
79         return self.name or "+".join(self.components or ())
```
One candidate source for the velocity. AhmedML stores velocity as a 3-component array (`UMean`);
WindsorML stores three scalars (`velocityxavg`, `velocityyavg`, `velocityzavg`). Exactly one of the
two fields is set by `CaseConfig.load`. `label()` gives a human-readable name, e.g. `"UMean"` or
`"velocityxavg+velocityyavg+velocityzavg"`; the same `+`-joined string is what
`preprocess._find_vector` reports as the source name.

### Axes (lines 82–108)

```python
82 AXIS_NAMES = "xyz"
83 MAX_AXIS_ANGLE_DEG = 5.0
```
Index → letter lookup, and the maximum allowed misalignment between a configured direction and a
mesh axis. Because the grid is never rotated, a direction must essentially *be* a mesh axis.

```python
86 @dataclass(frozen=True)
87 class Axis:
89     index: int   # 0, 1, 2 = x, y, z
90     sign: int    # +1 or -1
92     def __str__(self):
93         return ("+" if self.sign > 0 else "-") + AXIS_NAMES[self.index]
```
A signed axis; prints as `+z`, `-y`, etc. Used for the up direction.

```python
96  def axis_of(direction, what="direction") -> Axis:
98      d = np.asarray(direction, dtype=float)
99      n = np.linalg.norm(d)
100     if not n > 0:
101         raise ValueError(f"{what} {direction} has zero length")
102     d = d / n
103     i = int(np.argmax(np.abs(d)))
104     angle = np.degrees(np.arccos(min(1.0, abs(d[i]))))
105     if angle > MAX_AXIS_ANGLE_DEG:
106         raise ValueError(...)
108     return Axis(i, 1 if d[i] > 0 else -1)
```
- Line 100: `not n > 0` (rather than `n == 0`) also catches `NaN`, since every comparison with NaN
  is false.
- Line 102: normalize.
- Line 103: the dominant component picks the axis.
- Line 104: angle between the direction and that axis is `arccos(|d_i|)`. `min(1.0, …)` guards
  against `|d_i|` being `1.0000000002` from rounding, which would make `arccos` return NaN.
- Lines 105–107: reject anything more than 5° off.
- Line 108: the sign of the dominant component.

### `Flow` (lines 111–126)

```python
112 class Flow:
113     U_inf: float | None   # None when reference_probe is set
114     p_inf: float | None
115     rho: float
116     nu: float
117     freestream_direction: tuple[float, float, float]
118     up_direction: tuple[float, float, float]
119     ground: float | None  # None = auto; a coordinate along the up axis
122     reference_probe: tuple[float, float, float] | None = None
```
Freestream/reference quantities for one dataset:
- `U_inf`, `p_inf`: reference speed and pressure. Either fixed numbers from the YAML, or `None` when
  `reference_probe` is given, in which case they are measured per run at that point
  (`preprocess.probe_point`). WindsorML uses the probe because the tunnel speed varies per run.
- `rho`: density used in `Cp = 2(p − p_inf)/(ρ U_inf²)`. `1.0` for kinematic pressure (OpenFOAM).
- `nu`: kinematic viscosity — used only to report `Re_H` in the metadata.
- `freestream_direction`: parsed and stored but **not used anywhere** in the package (see §10).
- `up_direction`: drives H, the ground, and floor exclusion.
- `ground`: a coordinate along the up axis, or `None` meaning "auto" (lowest mesh point along up).

```python
124 @property
125 def up_axis(self) -> Axis:
126     return axis_of(self.up_direction, "up direction")
```
Computed on each access (cheap). Raises if `up_direction` is not within 5° of an axis.

### `CaseConfig` (lines 129–211)

```python
130 class CaseConfig:
132     dataset: str                           # e.g. "AhmedML"
133     files: dict[str, str]                  # key -> path template with {run}
134     velocity: tuple[VectorField, ...]      # candidates, first found wins
135     pressure: tuple[str, ...]              # candidates, first found wins
136     flow: Flow
137     body_scale: float = 1.0                # STL units -> mesh units
138     reference_length: float | None = None  # None = body height along up
139     domain: tuple[6 floats] | None = None  # None = the mesh's bounds
140     floor_offset_H: float = 1e-6           # first voxel layer above ground, units of H
141     hf_repo: str | None = None             # Hugging Face dataset repo for batch.py
142     source: str | None = field(default=None, compare=False)
```
Everything that is constant across all runs of one dataset family. `source` remembers the YAML path
it came from (for provenance hashing); `compare=False` excludes it from `==` so two configs loaded
from different paths but with identical content compare equal.

Note: `files` is a `dict` inside a frozen dataclass. Frozen prevents reassigning the attribute, but
the dict itself is still mutable; the dataclass is therefore not actually hashable in practice
(hashing would fail on the dict).

```python
144 def field_names(self) -> set[str]:
146     names = set(self.pressure)
147     for v in self.velocity:
148         names |= {v.name} if v.name else set(v.components or ())
149     return names
```
All array names that might be needed from the `.vtu`. `preprocess.read_volume` uses this to load
only those arrays (the volumes are tens of GB).

#### `CaseConfig.load` (lines 151–203)

```python
153 path = Path(path)
154 if not path.exists() and not path.suffix:
155     path = configs_dir() / "datasets" / f"{path.name}.yaml"
```
`--dataset ahmedml` resolves to `configs/datasets/ahmedml.yaml`; `--dataset my/file.yaml` is used
as-is. The rule: if the argument is not an existing file **and** has no extension, treat it as a
name. (A bare name that happens to match an existing file/dir in the CWD would be read as a path.)

```python
156 d = _load(path)
157 vel = []
158 for v in d["fields"]["velocity"]:
159     if isinstance(v, str):
160         vel.append(VectorField(name=v))
161     else:
162         comps = tuple(v["components"])
163         if len(comps) != 3: raise ValueError(...)
165         vel.append(VectorField(components=comps))
```
Each entry of `fields.velocity` is either a string (vector array name) or a mapping
`{components: [ux, uy, uz]}`. Order is preserved: it is the priority order.

```python
166 f = d["flow"]
167 ground = f.get("ground", "auto")
168 ref = d.get("body", {}).get("reference_length", "height")
169 domain = d.get("domain", "mesh")
170 probe = f.get("reference_probe")
```
Read the optional keys with their sentinel defaults: `"auto"`, `"height"`, `"mesh"`, `None`.

```python
171 if probe is not None:
172     if "U_inf" in f or "p_inf" in f:
173         raise ValueError("flow: give either reference_probe or U_inf / p_inf, not both")
174     if len(probe) != 3:
175         raise ValueError(...)
176 elif "U_inf" not in f:
177     raise ValueError("flow: U_inf (or reference_probe) is required")
178 if domain != "mesh" and (not isinstance(domain, list) or len(domain) != 6):
179     raise ValueError(...)
```
Mutual-exclusion validation: probe XOR fixed `U_inf` (`p_inf` is optional and defaults to 0).
`domain` must be the string `"mesh"` or a 6-element list `[xmin, xmax, ymin, ymax, zmin, zmax]`.
An absent `domain` key and `domain: mesh` mean the same thing: use the mesh bounds.

```python
180 cfg = cls(
181     dataset=d["dataset"],
182     files=dict(d["files"]),
183     velocity=tuple(vel),
184     pressure=tuple(d["fields"]["pressure"]),
185     flow=Flow(
186         U_inf=None if probe is not None else float(f["U_inf"]),
187         p_inf=None if probe is not None else float(f.get("p_inf", 0.0)),
188         rho=float(f.get("rho", 1.0)),
189         nu=float(f["nu"]),
190         freestream_direction=tuple(float(x) for x in f.get("freestream_direction", (1, 0, 0))),
191         up_direction=tuple(float(x) for x in f.get("up_direction", (0, 0, 1))),
192         ground=None if ground == "auto" else float(ground),
193         reference_probe=None if probe is None else tuple(float(v) for v in probe),
194     ),
195     body_scale=float(d.get("body", {}).get("scale", 1.0)),
196     reference_length=None if ref == "height" else float(ref),
197     domain=None if domain == "mesh" else tuple(float(v) for v in domain),
198     floor_offset_H=float(d.get("floor_offset_H", 1e-6)),
199     hf_repo=d.get("hf_repo"),
200     source=str(path),
201 )
```
Construction with defaults:

| YAML key | Required | Default |
|---|---|---|
| `dataset` | yes | — |
| `files` | yes | — |
| `fields.velocity`, `fields.pressure` | yes | — |
| `flow.U_inf` | unless `reference_probe` | — |
| `flow.p_inf` | no | `0.0` |
| `flow.rho` | no | `1.0` |
| `flow.nu` | yes | — |
| `flow.freestream_direction` | no | `(1, 0, 0)` |
| `flow.up_direction` | no | `(0, 0, 1)` |
| `flow.ground` | no | `auto` |
| `flow.reference_probe` | no | `None` |
| `body.scale` | no | `1.0` |
| `body.reference_length` | no | `height` |
| `domain` | no | `mesh` |
| `floor_offset_H` | no | `1e-6` |
| `hf_repo` | no | `None` |

The `# type: ignore[arg-type]` comments exist because `tuple(generator)` is typed as
`tuple[float, ...]`, not the fixed-length `tuple[float, float, float]` the field declares.

```python
202 cfg.flow.up_axis  # raises if the directions do not run along mesh axes
203 return cfg
```
A bare attribute access executed only for its side effect: `axis_of` validates `up_direction`
at load time rather than mid-build. Despite the plural in the comment, only `up_direction` is
validated; `freestream_direction` is not.

```python
205 def path(self, key, root, run) -> Path:
206     return Path(root) / self.files[key].format(run=run)
```
`cfg.path("volume", "/data/ahmedML/data", 7)` → `/data/ahmedML/data/run_7/volume_7.vtu`.

```python
208 def to_dict(self):
209     d = asdict(self)
210     d["velocity"] = [v.label() for v in self.velocity]
211     return d
```
For metadata. `asdict` recurses into `Flow`; the velocity candidates are flattened to their labels.

---

## 3. `geometry.py`

Purpose: read the body, define the voxel grid, and compute the inside mask and signed distance.

### Imports (lines 1–17)

```python
8  import numpy as np
9  from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
10 from vtkmodules.vtkCommonCore import vtkPoints
11 from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
12 from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkImplicitPolyDataDistance
13 from vtkmodules.vtkImagingHybrid import vtkSampleFunction
14 from vtkmodules.vtkImagingStencil import vtkImageStencilToImage, vtkPolyDataToImageStencil
15 from vtkmodules.vtkIOGeometry import vtkSTLReader
17 from .config import Axis, Contract
```
Import specific VTK modules (not the monolithic `import vtk`) to keep import time and loaded
libraries down. `scipy.ndimage` is imported lazily inside `signed_distance` (line 177).

A VTK idiom used throughout the package: **keyword-argument constructors**, e.g.
`vtkSTLReader(file_name=...)`. Since VTK 9.4 the Python wrappers accept any settable property as a
snake_case keyword (`file_name=` → `SetFileName`, `input_connection=` → `SetInputConnection`,
`input_data=` → `SetInputData`). `reader.output_port` is the property form of `GetOutputPort()`.

### `STL_MERGE_TOLERANCE` (line 22)

```python
22 STL_MERGE_TOLERANCE = 1e-5
```
Fraction of the bounding-box diagonal under which STL vertices are merged. STL stores each triangle
with its own three vertex copies; some datasets (WindsorML) have copies of the "same" vertex that
differ by ~1e-6 m, leaving cracks. Cracks break the inside test (scanline parity) and the distance.

### `read_body(path, scale)` (lines 25–40)

```python
26 reader = vtkSTLReader(file_name=str(path))
29 clean = vtkCleanPolyData(input_connection=reader.output_port, tolerance=STL_MERGE_TOLERANCE,
30                          tolerance_is_absolute=False, point_merging=True, convert_polys_to_lines=False,
31                          convert_lines_to_points=False, convert_strips_to_polys=False)
32 clean.Update()
```
- `tolerance_is_absolute=False`: the tolerance is relative to the bounding-box diagonal.
- `point_merging=True`: merge coincident points.
- The three `convert_*=False` settings: by default, `vtkCleanPolyData` turns a triangle whose
  vertices merge into a line or vertex. `vtkImplicitPolyDataDistance` crashes on non-triangle
  cells, so degenerate triangles are dropped instead (comment, lines 27–28).

```python
33 body = vtkPolyData()
34 body.DeepCopy(clean.GetOutput())
```
Detach the result from the pipeline so the reader/cleaner can be garbage-collected and later
modifications do not re-execute anything.

```python
35 if scale != 1.0:
36     pts = vtk_to_numpy(body.GetPoints().GetData()).astype(np.float64) * scale
37     new = vtkPoints()
38     new.SetData(numpy_to_vtk(pts, deep=True))
39     body.SetPoints(new)
40 return body
```
Optional unit conversion (e.g. mm → m). Points are promoted to `float64` (STL points are `float32`)
before scaling; `deep=True` copies, so the VTK array owns its memory and outlives `pts`. Scaling
happens after cleaning, which is fine because the merge tolerance is relative.

### `ImageSpec` (lines 43–62)

```python
44 @dataclass(frozen=True)
45 class ImageSpec:
46     """A uniform voxel grid on the mesh's own axes. Arrays are indexed [i, j, k] = (x, y, z)."""
47     origin: tuple[float, float, float]
48     dx: float
49     dims: tuple[int, int, int]
```
A **point-centred** uniform grid: voxel `(i, j, k)` sits at `origin + dx * (i, j, k)`. "Voxel
centre" in the docstrings means exactly these grid points. Spacing is isotropic.

```python
51 @property
52 def bounds(self):
53     o, d, n = self.origin, self.dx, self.dims
54     return (o[0], o[0] + (n[0] - 1) * d, o[1], ..., o[2], o[2] + (n[2] - 1) * d)
```
VTK-style bounds `(xmin, xmax, ymin, ymax, zmin, zmax)` of the **first and last sample points** (not
the voxel cell faces): `n` points span `(n − 1) dx`. These bounds are passed to
`vtkResampleToImage` and `vtkSampleFunction`, which, given matching dims, place their samples
exactly on this grid.

```python
55 def axis(self, i: int) -> np.ndarray:
56     return self.origin[i] + self.dx * np.arange(self.dims[i])
```
1-D coordinates of the samples along axis `i`.

```python
58 def to_vtk(self) -> vtkImageData:
59     return vtkImageData(dimensions=self.dims, spacing=(self.dx,) * 3, origin=self.origin)
```
An empty `vtkImageData` with the same geometry, used by the `.vti` writer.

```python
61 def to_dict(self):
62     return dict(origin=list(self.origin), dx=self.dx, dims=list(self.dims))
```
Stored as the `grid` attribute in every Zarr store, so a consumer can map voxel indices back to mesh
coordinates.

### `GRID_INSET_H` (line 68)

```python
68 GRID_INSET_H = 1e-6
```
The grid is pulled `1e-6·H` inside each face of the domain. Points exactly on a face of the mesh are
found or missed by the probe at random (floating-point ties in the point-in-cell test), so the
outermost layer would be noisy. The ground side uses the dataset's `floor_offset_H` instead.

### `CaseGeometry` (lines 71–108)

```python
72 class CaseGeometry:
74     body_bounds: tuple[6 floats]
75     ground: float      # coordinate along the up axis
76     H: float           # reference length
77     up: Axis
```

```python
79 def height_above_ground(self, spec: ImageSpec) -> np.ndarray:
81     h = self.up.sign * (spec.axis(self.up.index) - self.ground)
82     shape = [1, 1, 1]
83     shape[self.up.index] = -1
84     return h.reshape(shape)
```
Signed height of each grid layer above the ground. Multiplying by `sign` makes it positive above
the ground even when up is `-y` or `-z`. Line 82–84 reshape to e.g. `(1, 1, nz)` for up = ±z, so
it broadcasts against `(nx, ny, nz)` masks without allocating a full 3-D array. Used by
`label.make_labels` for floor exclusion.

#### `image_spec(contract, domain, floor_offset_H)` (lines 86–108)

```python
92  dx = self.H / contract.n_per_H
93  eps = GRID_INSET_H * self.H
94  floor = floor_offset_H * self.H
95  lo, n = [0.0] * 3, [0] * 3
```
Voxel size (H/32 in v1), side inset, ground offset, and accumulators for origin and counts.

```python
96  for a in range(3):
97      start, end = domain[2 * a] + eps, domain[2 * a + 1] - eps
```
Default extent along axis `a`: the domain, inset by `eps` on both sides.

```python
98      if a == self.up.index:
99          if self.up.sign > 0:
100             start = self.ground + floor
101         else:
102             start, end = domain[2 * a] + eps, self.ground - floor
```
Along the up axis the ground side is replaced:
- up positive: the grid starts `floor` above the ground; the top stays at the domain top − `eps`.
- up negative: "above the ground" means *lower* coordinates, so the grid ends `floor` below the
  ground coordinate and starts at the domain minimum.

Note: when up is positive, `start` ignores `domain[2a]` entirely, so a `domain` whose lower bound
along up is above the ground still gets a grid starting at the ground.

```python
103     if not end > start:
104         raise ValueError(f"empty domain along {'xyz'[a]}: ...")
```
`not end > start` (rather than `end <= start`) also catches NaN.

```python
105     n[a] = int(np.floor((end - start) / dx + 1e-9)) + 1
```
How many samples fit: `floor(length/dx) + 1` points span at most `length`. The `+1e-9` absorbs
round-off when the length is an exact multiple of `dx` (e.g. `31.9999999999` → 32). The last
sample therefore sits at or just before `end`; the leftover (< dx) is at the far side.

```python
107     lo[a] = end - (n[a] - 1) * dx if (a == self.up.index and self.up.sign < 0) else start
108 return ImageSpec(tuple(float(v) for v in lo), dx, tuple(n))
```
Where to anchor the grid: normally at `start` (the low face, or the ground for up positive). For a
negative up axis the ground is at `end`, so the grid is anchored there and the leftover goes to the
low face. Either way **the first layer above the ground is always exactly `floor` from the ground**,
which keeps the floor-exclusion behaviour consistent across datasets regardless of orientation.
`float(v)` strips NumPy scalar types so the spec is JSON-friendly.

### `case_geometry(body, ground, reference_length, up)` (lines 111–120)

```python
112 b = body.GetBounds()
113 j = up.index
114 lowest, highest = (b[2 * j], b[2 * j + 1]) if up.sign > 0 else (-b[2 * j + 1], -b[2 * j])
```
Body extent along up, in "up-signed" coordinates (multiply by `sign`). For up = −y, the body's
lowest point is the one with the largest y, hence `-b[max]`.

```python
115 H = reference_length if reference_length is not None else highest - lowest
116 if not H > 0:
117     raise ValueError(...)
```
H is the configured number, or the body height along up (AhmedML: body height; WindsorML: includes
the support pins down to the ground).

```python
118 if lowest < up.sign * ground - 1e-6 * H:
119     raise ValueError(f"body reaches below the ground along {up}: ...")
120 return CaseGeometry(tuple(b), float(ground), float(H), up)
```
Sanity check: the body must not poke below the ground (with a `1e-6·H` tolerance). Catches a wrong
`ground`, wrong `up_direction`, or wrong `body.scale`.

### `body_inside(body, spec)` (lines 123–131)

```python
125 n = spec.dims
126 stencil = vtkPolyDataToImageStencil(input_data=body, output_origin=spec.origin,
            output_spacing=(spec.dx,) * 3,
127         output_whole_extent=(0, n[0] - 1, 0, n[1] - 1, 0, n[2] - 1))
128 to_image = vtkImageStencilToImage(input_connection=stencil.output_port, inside_value=1,
            outside_value=0,
129         output_scalar_type=3)  # unsigned char
130 to_image.Update()
131 return vtk_to_numpy(...GetScalars()).reshape(n, order="F").astype(bool)
```
- `vtkPolyDataToImageStencil` voxelizes a closed surface by casting scanlines along x and toggling
  inside/outside at each crossing (**parity**). This depends only on the surface being closed, not
  on triangle orientation or normals — hence "robust to the surface's normals".
- The stencil is converted to a `uint8` image (type code 3 = `VTK_UNSIGNED_CHAR`) with 1 inside.
- `order="F"` because VTK's x index varies fastest.

### `_exact_distance(body, spec)` (lines 134–141)

```python
136 dist = vtkImplicitPolyDataDistance()
137 dist.SetInput(body)
138 sampler = vtkSampleFunction(implicit_function=dist, model_bounds=spec.bounds,
            sample_dimensions=spec.dims,
139         compute_normals=False, output_scalar_type=10)  # float
140 sampler.Update()
141 return np.abs(vtk_to_numpy(...)).reshape(spec.dims, order="F")
```
- `vtkImplicitPolyDataDistance` evaluates the exact distance to the nearest triangle (with a cell
  locator), signed by the surface normal.
- `vtkSampleFunction` evaluates it at every grid point. With `model_bounds = spec.bounds` and
  `sample_dimensions = spec.dims`, the samples coincide with the grid exactly.
- `output_scalar_type=10` = `VTK_FLOAT` (float32). `compute_normals=False` skips the gradient.
- `np.abs`: the sign is discarded here and re-derived from `body_inside` (see below). This is the
  slow part: cost ∝ number of grid points × log(triangles).

### `signed_distance(body, spec, band, info)` (lines 144–189)

Returns `(sdf, inside)`: signed distance in mesh units (negative inside) as `float32`, and the bool
inside mask.

**Why the sign comes from parity** (docstring 148–149): the implicit function's sign comes from the
nearest triangle's normal; near sharp edges, far-field points can pick a triangle whose normal
points the wrong way, flipping the sign. Parity has no such failure.

```python
158 inside = body_inside(body, spec)
159 if band is None or not inside.any():
160     d = _exact_distance(body, spec)
161     if info is not None:
162         info.update(method="exact")
163     return np.where(inside, -d, d).astype(np.float32), inside
```
Exact path, used when no band is requested **or** when no voxel is inside the body (the distance
transform below would have nothing to measure from). `np.where(inside, -d, d)` applies the sign.

**Band path** — exact near the body, cheap far away:

```python
166 b = body.GetBounds()
167 sl = []
168 for a in range(3):
169     ax = spec.axis(a)
170     i0 = int(np.searchsorted(ax, b[2 * a] - band, side="left"))
171     i1 = int(np.searchsorted(ax, b[2 * a + 1] + band, side="right"))
172     sl.append(slice(max(i0, 0), min(i1, spec.dims[a])))
```
For each axis, the index range of grid samples inside `[body_min − band, body_max + band]`.
`side="left"` for the start gives the first sample ≥ the low edge; `side="right"` for the stop gives
one past the last sample ≤ the high edge. Clipped to the grid.

```python
173 sub = ImageSpec(tuple(spec.origin[a] + spec.dx * sl[a].start for a in range(3)), spec.dx,
174                 tuple(sl[a].stop - sl[a].start for a in range(3)))
175 exact = _exact_distance(body, sub)
```
A sub-grid with identical spacing and aligned samples, covering just the band box; the exact
distance is computed only there.

```python
177 from scipy.ndimage import binary_erosion, distance_transform_edt
178 d = distance_transform_edt(~inside, sampling=spec.dx).astype(np.float32)
```
Euclidean distance transform over the **whole** grid: for each non-zero element of `~inside`
(outside voxels), the distance to the nearest zero element (an inside voxel), in mesh units thanks
to `sampling=dx`. This approximates the surface distance but is biased high: the nearest inside
voxel centre is a fraction of a voxel deeper than the surface.

```python
180 in_band = np.zeros(spec.dims, bool)
181 in_band[tuple(sl)] = True
182 rim = in_band & ~binary_erosion(in_band, border_value=0) & ~inside
183 rim_sub = rim[tuple(sl)]
184 offset = float(np.median(exact[rim_sub] - d[tuple(sl)][rim_sub])) if rim_sub.any() else 0.0
```
Calibrate that bias:
- Line 182: `rim` is the outermost voxel layer of the band box (box minus its 1-voxel erosion),
  excluding inside voxels. `border_value=0` treats outside the array as False, so if the box is
  clipped by the grid, the box faces lying on the grid boundary also count as rim.
- Line 184: the median difference "exact − EDT" over the rim is the offset (typically a small
  negative number, a fraction of `dx`). The median is robust to the few rim voxels where the EDT
  and exact distance disagree more. If the rim is empty, no offset.

```python
185 d += np.float32(offset)
186 d[tuple(sl)] = exact
```
Shift the far-field EDT by the offset so it meets the exact values continuously at the band
boundary, then overwrite the band box with the exact distances. Inside voxels are always within
the band box (it contains the body bounding box), so the negative side is always exact.

```python
187 if info is not None:
188     info.update(method="exact in band, distance transform outside", band=band, edt_offset=offset)
189 return np.where(inside, -d, d).astype(np.float32), inside
```
Record what was done (stored in the case metadata under `checks.voxels.sdf`) and apply the sign.

Callers use `band = SDF_EXACT_BAND_H · H = 1·H` (`preprocess.py`). Since training and inference both
call this function, both see identical SDF values.

---

## 4. `io.py`

Purpose: provenance (git state, input hashes, versions) and writers.

### Imports (lines 1–19)

```python
5  import hashlib, json, os, shutil, subprocess
10 from pathlib import Path
12 import numpy as np
13 import zarr
14 from zarr.codecs import BloscCodec
15 from vtkmodules.util.numpy_support import numpy_to_vtk
16 from vtkmodules.vtkIOXML import vtkXMLImageDataWriter
18 from . import CONTRACT_VERSION, LABEL_RULES_VERSION, __version__
19 from .geometry import ImageSpec
```
`zarr` is Zarr-Python 3 (`zarr>=3.4.2`), needed for v3 sharding.

### `git_state(repo)` (lines 22–31)

```python
23 repo = repo or Path(__file__).resolve().parents[2]
```
`io.py` → `parents[0]` = `cfd_rgba_net/`, `[1]` = `src/`, `[2]` = repo root.

```python
25 commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
       text=True, check=True).stdout.strip()
27 dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain",
       "--untracked-files=no"], ...).stdout.strip())
29 return dict(commit=commit, dirty=dirty)
30 except (OSError, subprocess.CalledProcessError):
31     return dict(commit=None, dirty=None)
```
- `rev-parse HEAD` → current commit SHA.
- `status --porcelain --untracked-files=no` → any output means tracked files are modified. Untracked
  files do **not** count as dirty.
- `OSError` covers "git not installed"; `CalledProcessError` covers "not a git repo". Both give
  `None`s rather than failing a build.
- When installed as a wheel, `parents[2]` is somewhere inside the environment (e.g.
  `.venv/lib/python3.12`). If that venv lives inside a git checkout, git walks up and reports that
  checkout's state; otherwise `None`s.

### `sha256(path, chunk)` (lines 34–60)

```python
39 path = Path(path)
40 st = path.stat()
41 cache = path.with_name(path.name + ".sha256")
42 key = dict(size=st.st_size, mtime_ns=st.st_mtime_ns)
```
A sidecar cache file `<file>.sha256` keyed by size and nanosecond mtime. (This is why
`configs/datasets/ahmedml.yaml.sha256` and `windsorml.yaml.sha256` exist: the dataset YAML is one of
the hashed inputs.)

```python
43 try:
44     c = json.loads(cache.read_text())
45     if {k: c.get(k) for k in key} == key:
46         return c["sha256"]
47 except (OSError, ValueError, KeyError):
48     pass
```
Cache hit if size and mtime match. Missing file (`OSError`), corrupt JSON (`ValueError`), or
missing field (`KeyError`) silently fall through to recomputing.

```python
49 h = hashlib.sha256()
50 with path.open("rb") as fh:
51     while b := fh.read(chunk):
52         h.update(b)
53 digest = h.hexdigest()
```
Streaming hash in 16 MiB chunks (`1 << 24`), so memory stays flat on tens-of-GB volumes. The walrus
loop stops on the empty `bytes` at EOF.

```python
54 try:
55     tmp = cache.with_name(cache.name + f".{os.getpid()}")
56     tmp.write_text(json.dumps(dict(**key, sha256=digest)))
57     tmp.replace(cache)
58 except OSError:
59     pass
60 return digest
```
Write the cache atomically: per-PID temp file, then `replace` (atomic rename on POSIX). Parallel
builds in `batch.py` can hash the same file without corrupting the cache. Read-only data
directories just skip caching.

### `provenance(inputs)` (lines 63–70)

```python
64 return dict(
65     package_version=__version__,
66     contract_version=CONTRACT_VERSION,
67     label_rules_version=LABEL_RULES_VERSION,
68     git=git_state(),
69     inputs={k: dict(path=str(p), sha256=sha256(p)) for k, p in inputs.items()
             if p is not None and Path(p).exists()},
70 )
```
Everything needed to reproduce a case. Inputs that are `None` or missing are skipped silently.

### Zarr layout constants (lines 73–75)

```python
73 CHUNK = 64     # voxels per chunk edge
74 SHARD = 256    # voxels per shard edge
75 COMPRESSOR = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
```
- **Chunk** (64³ voxels): the unit of compression and of reading. A training patch only decodes
  the chunks it overlaps.
- **Shard** (256³ voxels = 64 chunks): Zarr v3 packs many chunks into one file, so a 300M-voxel
  array becomes a handful of files instead of thousands.
- Blosc with zstd level 5 and byte-shuffle: shuffle groups the same byte of every float together,
  which compresses smooth float fields much better.

### `_put(group, name, a)` (lines 78–83)

```python
80 lead = a.shape[:-3]
81 arr = group.create_array(name, shape=a.shape, dtype=a.dtype, chunks=lead + (CHUNK,) * 3,
82                          shards=lead + (SHARD,) * 3, compressors=COMPRESSOR)
83 arr[...] = a
```
Arrays are `(..., nx, ny, nz)`. Leading axes (`lead`, e.g. `(5,)` for inputs or `(2,)` for labels)
are kept whole in every chunk, so one chunk read returns all channels of a 64³ block. `arr[...] = a`
writes the whole array (Zarr encodes shards in parallel internally). Arrays smaller than a
chunk/shard are fine; edge chunks are padded internally.

### `jsonable(obj)` (lines 86–87)

```python
87 return json.loads(json.dumps(obj, default=float))
```
Round-trips through JSON to get plain Python types: tuples → lists, and any non-JSON object
(NumPy scalars like `np.float32`, `np.int64`) → `float(obj)`. Zarr attributes must be JSON. Note it
turns NumPy integers into floats too, and would fail on a NumPy array with more than one element.

### `save_case_zarr(path, spec, arrays, groups, attrs)` (lines 90–109)

```python
97  path = Path(path)
98  tmp = path.with_name(path.name + ".tmp")
99  shutil.rmtree(tmp, ignore_errors=True)
100 root = zarr.open_group(tmp, mode="w", zarr_format=3)
```
Write into `<name>.zarr.tmp`, after removing any leftover from a crashed build. `mode="w"` creates
(overwriting), `zarr_format=3` forces v3 (sharding needs it).

```python
101 for name, a in arrays.items():
102     _put(root, name, a)
103 for gname, members in groups.items():
104     g = root.create_group(gname)
105     for name, a in members.items():
106         _put(g, name, a)
107 root.attrs.update(jsonable(dict(grid=spec.to_dict(), **attrs)))
```
Root arrays (`inputs`, `fluid`, `labels`), sub-groups (`ingredients/…`), then the root attributes:
`grid` plus all metadata from `build_case`. (If `attrs` contained a `grid` key, `dict(grid=..., **attrs)`
would raise `TypeError`.)

```python
108 shutil.rmtree(path, ignore_errors=True)
109 tmp.rename(path)
```
Replace any older store and rename into place. The invariant: **a store that exists is complete** —
`batch.py` relies on this to skip finished runs. (There is a tiny window between `rmtree` and
`rename` in which no store exists, only relevant when rebuilding an existing case.)

Resulting layout:

```
<tag>.zarr/
  zarr.json                 # root group + attributes (grid, metadata)
  inputs/                   # (5, nx, ny, nz) float32
  fluid/                    # (nx, ny, nz) bool
  labels/                   # (2, nx, ny, nz) uint8
  ingredients/
    Q_nd/  lambda2_nd/  omega_mag_nd/   # float32
    core/                               # bool
```

### `save_vti(path, spec, arrays)` (lines 112–121)

```python
113 image = spec.to_vtk()
114 for name, a in arrays.items():
115     flat = (a.reshape(-1, order="F") if a.ndim == 3
116             else np.stack([a[i].reshape(-1, order="F") for i in range(a.shape[0])], 1))
117     v = numpy_to_vtk(np.ascontiguousarray(flat), deep=True)
118     v.SetName(name)
119     image.GetPointData().AddArray(v)
```
For viewing in ParaView. A 3-D array is flattened in Fortran order (x fastest, VTK's layout). A 4-D
`(C, nx, ny, nz)` array becomes an `(N, C)` multi-component VTK array (e.g. `U_nd` with 3
components). `ascontiguousarray` because `numpy_to_vtk` requires C-contiguous memory; `deep=True`
copies.

```python
120 writer = vtkXMLImageDataWriter(file_name=str(path), input_data=image)
121 writer.Write()
```
XML `.vti` with VTK's defaults (appended binary, compressed).

### `save_json(path, obj)` (lines 124–126)

Pretty-printed JSON with `default=float` for NumPy scalars. **Not called anywhere in the package**
(metadata goes into Zarr attributes).

---

## 5. `preprocess.py`

Purpose: native mesh → model inputs. The module docstring (lines 1–16) is the recipe, and it is
what inference must also follow:

```python
mesh  = read_volume(path, case_cfg.field_names())
prep  = prepare_mesh(mesh, case_cfg)              # select fields, non-dimensionalize, find the ground
body  = read_body(stl, case_cfg.body_scale)
geo   = case_geometry(body, prep.ground, case_cfg.reference_length, flow.up_axis)
spec  = geo.image_spec(contract, case_cfg.domain or mesh.GetBounds())   # the whole domain
image = resample(prep.mesh, spec)                 # point arrays on the voxel grid
inp   = assemble_inputs(image, body, geo, spec)   # (5, nx, ny, nz) + fluid mask
```
Lines 13–15 restate the no-rotation rule.

### Imports (lines 18–30)

```python
24 from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
25 from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkUnstructuredGrid
26 from vtkmodules.vtkFiltersCore import vtkCellDataToPointData, vtkResampleToImage
27 from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
29 from .config import CaseConfig
30 from .geometry import CaseGeometry, ImageSpec, signed_distance
```

### Constants (lines 33–38)

```python
33 U_ND = "U_nd"      # point data, velocity / U_inf, mesh axes
34 CP = "Cp"          # point data
35 U_CELL = "U_cell"  # cell data, velocity / U_inf (gradients only); may be absent
38 SDF_EXACT_BAND_H = 1.0
```
Canonical array names on the prepared mesh (imported by `ingredients.py` and `build_case.py`), and
the exact-SDF band width in units of H.

### `read_volume(path, arrays)` (lines 41–54)

```python
47 reader = vtkXMLUnstructuredGridReader(file_name=str(path))
48 if arrays is not None:
49     reader.UpdateInformation()
50     for sel in (reader.GetPointDataArraySelection(), reader.GetCellDataArraySelection()):
51         for name in [sel.GetArrayName(i) for i in range(sel.GetNumberOfArrays())]:
52             sel.SetArraySetting(name, int(name in arrays))
53 reader.Update()
54 return reader.GetOutput()
```
- `UpdateInformation()` parses only the XML header, which populates the array-selection objects
  with every array name in the file.
- Lines 50–52: enable only the requested names, disable the rest — both point and cell data. The
  list comprehension snapshots names before modifying settings.
- `Update()` then reads geometry plus only the enabled arrays. Requested names that do not exist are
  simply never enabled (no error).

### `PreparedMesh` (lines 57–65)

```python
58 class PreparedMesh:
59     mesh: vtkUnstructuredGrid
60     U_inf: float                # from the config, or read at flow.reference_probe
61     p_inf: float
62     ground: float               # coordinate along the up axis
63     velocity_source: str        # which array the velocity came from (for metadata)
64     pressure_source: str
65     checks: dict = field(default_factory=dict)
```
A mutable dataclass (not frozen: `build_case` does `del prep.mesh` to free memory).
`default_factory=dict` avoids a shared default dict. `checks` is currently never filled.

### `_find_vector(ug, specs, assoc)` (lines 68–77)

```python
69 data = ug.GetPointData() if assoc == "point" else ug.GetCellData()
70 for s in specs:
71     if s.name is not None and data.HasArray(s.name):
72         a = vtk_to_numpy(data.GetArray(s.name))
73         if a.ndim == 2 and a.shape[1] == 3:
74             return s.name, a
75     elif s.components is not None and all(data.HasArray(c) for c in s.components):
76         return "+".join(s.components), np.stack([vtk_to_numpy(data.GetArray(c)) for c in s.components], 1)
77 return None, None
```
Walk the configured candidates in priority order; first match wins. A named array must be
3-component (an array with the right name but wrong shape is skipped silently and the next
candidate is tried). Component arrays are stacked into `(N, 3)`. `vtk_to_numpy` returns a view
(no copy); `np.stack` copies. Returns `(None, None)` if nothing matches.

Subtle point: the `elif` belongs to the outer `if`, so a candidate with `name` set and present but of
the wrong shape does not fall into the `components` branch (it has none anyway).

### `_find_scalar(ug, names, assoc)` (lines 80–85)

Same idea for a scalar: first name present wins, returns `(name, view)` or `(None, None)`.

### `_vtk(name, a)` (lines 88–91)

NumPy → named VTK array, contiguous and deep-copied (so it does not depend on the NumPy buffer
staying alive).

### `prepare_mesh(ug, cfg, keep_checks)` (lines 94–152)

```python
102 f = cfg.flow
103 u_name, U = _find_vector(ug, cfg.velocity, "point")
104 p_name, p = _find_scalar(ug, cfg.pressure, "point")
105 uc_name, Uc = _find_vector(ug, cfg.velocity, "cell")
```
Look for point velocity, point pressure, and cell velocity. The cell velocity is wanted
independently: OpenFOAM stores fields at cell centres, and gradients of the cell field are more
faithful (see `ingredients.py`).

```python
106 if U is None or p is None:
107     pc_name, pc = _find_scalar(ug, cfg.pressure, "cell")
108     if Uc is None or pc is None:
109         raise KeyError(...)
111     tmp = vtkUnstructuredGrid()
112     tmp.ShallowCopy(ug)
113     tmp.GetPointData().Initialize()
114     tmp.GetCellData().Initialize()
115     tmp.GetCellData().AddArray(_vtk("U", Uc))
116     tmp.GetCellData().AddArray(_vtk("p", pc))
117     c2p = vtkCellDataToPointData(input_data=tmp)
118     c2p.Update()
119     U = vtk_to_numpy(c2p.GetOutput().GetPointData().GetArray("U"))
120     p = vtk_to_numpy(c2p.GetOutput().GetPointData().GetArray("p"))
121     u_name, p_name = f"cell:{uc_name} -> points", f"cell:{pc_name} -> points"
```
Fallback when point data is incomplete (WindsorML has cell data only):
- If *either* point velocity or point pressure is missing, **both** are taken from cell data (a
  point array that was found is discarded). Both cell arrays must exist, else `KeyError`.
- `ShallowCopy` shares points and cells with `ug` (no geometry copy); `Initialize()` on the copy's
  attribute containers empties them without touching `ug`'s.
- Only `U` and `p` are put on `tmp`, so `vtkCellDataToPointData` averages just those two (it would
  otherwise average every cell array). Each point gets the average of its adjacent cells.
- The source names record the conversion, e.g. `"cell:velocityxavg+velocityyavg+velocityzavg -> points"`.

```python
123 out = vtkUnstructuredGrid()
124 out.ShallowCopy(ug)
125 pd, cd = out.GetPointData(), out.GetCellData()
126 pd.Initialize()
127 cd.Initialize()
128 for name in keep_checks:
129     for src, dst in ((ug.GetPointData(), pd), (ug.GetCellData(), cd)):
130         if src.HasArray(name):
131             dst.AddArray(src.GetArray(name))
```
The output grid shares geometry with `ug` and starts with empty attribute data. Arrays named in
`keep_checks` (dataset-provided vorticity / Cp, used only for sanity checks) are copied over **by
reference** (same VTK array object), from point or cell data, wherever they are found.

```python
133 if f.reference_probe is not None:
134     U_ref, p_ref = probe_point(ug, f.reference_probe, U, p)
135     U_inf, p_inf = float(np.linalg.norm(U_ref)), float(p_ref)
136 else:
137     U_inf, p_inf = float(f.U_inf), float(f.p_inf)
```
Reference quantities: either measured at the probe (speed = |U|, pressure = p, interpolated from
the **dimensional point** arrays) or taken from the config.

```python
138 U_nd = (U.astype(np.float32) / np.float32(U_inf))
139 Cp = (2.0 * (p.astype(np.float64) - p_inf) / (f.rho * U_inf**2)).astype(np.float32)
140 pd.AddArray(_vtk(U_ND, U_nd))
141 pd.AddArray(_vtk(CP, Cp))
```
Non-dimensionalization:
- `u* = U / U_inf`, in float32.
- `Cp = 2 (p − p_inf) / (ρ U_inf²)`, computed in **float64** then stored as float32. The subtraction
  `p − p_inf` can cancel badly in float32 when `p_inf` is large (WindsorML: p ≈ 37,000 Pa while the
  interesting differences are hundreds of Pa).

```python
142 if Uc is not None:
143     Uc_nd = Uc.astype(np.float32) / np.float32(U_inf)
144     cd.AddArray(_vtk(U_CELL, Uc_nd))
```
If a cell velocity exists, keep a non-dimensional copy as cell data `U_cell` for the gradient.

```python
146 if f.ground is not None:
147     ground = f.ground
148 else:  # the lowest mesh point along the up direction
149     up = f.up_axis
150     b = out.GetBounds()
151     ground = float(b[2 * up.index] if up.sign > 0 else b[2 * up.index + 1])
152 return PreparedMesh(out, U_inf, p_inf, ground, u_name, p_name)
```
Ground: configured, or "auto" = the mesh bound on the low side of up (min for +up, max for −up). This
is wrong for meshes whose bounding box extends below the floor (WindsorML's reservoir), which is why
`windsorml.yaml` sets `ground: 0.0`.

### `probe_point(ug, x, U, p)` (lines 155–166)

```python
157 from vtkmodules.vtkCommonDataModel import vtkGenericCell, vtkStaticCellLocator
158 loc = vtkStaticCellLocator(data_set=ug)
159 loc.BuildLocator()
160 cell, pcoords, weights = vtkGenericCell(), [0.0] * 3, [0.0] * 64
161 cid = loc.FindCell(list(map(float, x)), 1e-12, cell, pcoords, weights)
162 if cid < 0:
163     raise ValueError(f"flow.reference_probe {tuple(x)} is not inside any mesh cell")
164 ids = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
165 w = np.array(weights[:len(ids)])
166 return (w[:, None] * U[ids]).sum(0), float((w * p[ids]).sum())
```
- Lazy import (only needed with a probe).
- A static cell locator over the whole mesh (built once per call; one call per case).
- `FindCell(x, tol2, cell, pcoords, weights)`: finds the containing cell; `1e-12` is the squared
  tolerance; `weights` receives the interpolation weights of the cell's points. Python lists are
  passed as output buffers; 64 slots cover any standard cell (hexahedron: 8; octree/polyhedral
  cells beyond 64 points would overflow).
- Returns interpolated `U` (3-vector) and `p` (scalar) from the provided point arrays.

### `resample(mesh, spec, names)` (lines 169–189)

```python
174 resampler = vtkResampleToImage(input_data_object=mesh, sampling_dimensions=spec.dims,
175                                sampling_bounds=spec.bounds, use_input_bounds=False)
176 resampler.Update()
177 pd = resampler.GetOutput().GetPointData()
```
`vtkResampleToImage` probes the unstructured mesh at every grid point (point-in-cell search +
interpolation of **point** arrays with the cell's shape functions). `use_input_bounds=False` with
`sampling_bounds=spec.bounds` makes the samples coincide with the `ImageSpec`. It is SMP-parallel
(threads set by `vtkSMPTools` in `build_case.setup_threads`). It also emits `vtkValidPointMask`: 1
where a containing cell was found.

```python
178 out = {}
179 for name in names or [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]:
180     a = vtk_to_numpy(pd.GetArray(name))
181     if name == "vtkValidPointMask":
182         continue
183     out[name] = (a.reshape(spec.dims, order="F") if a.ndim == 1
184                  else np.stack([a[:, c].reshape(spec.dims, order="F") for c in range(a.shape[1])]))
```
For each requested array (default: all output arrays), reshape to `(nx, ny, nz)` or, for
multi-component arrays, `(C, nx, ny, nz)` — channel-first, so `U_nd` becomes `(3, nx, ny, nz)`.
The mask is skipped here and handled below. (If `names` lists an array that is missing,
`GetArray` returns `None` and `vtk_to_numpy` raises.)

```python
185 valid = vtk_to_numpy(pd.GetArray("vtkValidPointMask"))
186 if valid.dtype.kind == "S":
187     valid = valid.view(np.uint8)
188 out["valid"] = valid.reshape(spec.dims, order="F") > 0
189 return out
```
The mask is a `vtkCharArray`. With VTK 9.7.1, `vtk_to_numpy` returns it as `int8` (verified), so
lines 186–187 are a defensive branch for builds that expose it as a bytes dtype (`S1`); `view(uint8)`
reinterprets such bytes as numbers without copying. Converted to bool and stored as `"valid"`.

### `Inputs` (lines 192–196)

```python
193 class Inputs:
194     inputs: np.ndarray       # (5, nx, ny, nz) float32: u, v, w, Cp, sdf/H
195     fluid: np.ndarray        # (nx, ny, nz) bool: inside the mesh and outside the body
196     checks: dict
```

### `assemble_inputs(image, body, geo, spec)` (lines 199–221)

```python
200 sdf_info = {}
201 sdf, inside = signed_distance(body, spec, band=SDF_EXACT_BAND_H * geo.H, info=sdf_info)
```
Signed distance (mesh units) with an exact band of 1·H around the body, plus the inside mask.

```python
202 valid = image["valid"]
203 fluid = valid & ~inside
```
**Fluid** = a mesh cell contains the voxel **and** the voxel is not inside the body. Both conditions
are needed: body-fitted meshes (AhmedML) have no cells inside the body, but immersed-boundary meshes
(WindsorML) do.

```python
204 U = image[U_ND].astype(np.float32).copy()
205 Cp = image[CP].astype(np.float32).copy()
206 U[:, ~fluid] = 0.0
207 Cp[~fluid] = 0.0
```
Copies (so the `image` dict is not mutated), then zero velocity and Cp outside the fluid (contract
rule: "u, v, w and Cp are 0 outside the fluid"). `U[:, ~fluid]` broadcasts the 3-D mask over the
channel axis. The SDF is **not** zeroed; it is defined everywhere.

```python
210 outside_both = ~valid & ~inside
211 checks = dict(
212     n_voxels=int(valid.size),
213     n_fluid=int(fluid.sum()),
214     n_body=int(inside.sum()),
215     n_not_in_mesh_nor_body=int(outside_both.sum()),
216     n_in_mesh_and_body=int((valid & inside).sum()),  # > 0 for immersed-boundary meshes
217     Cp_max_fluid=float(Cp[fluid].max()) if fluid.any() else None,  # ~1 at the stagnation point
218     sdf=sdf_info,
219 )
```
Diagnostics stored in metadata:
- `n_not_in_mesh_nor_body`: voxels the mesh does not cover and that are not body — the box leaves
  the CFD domain (e.g. WindsorML's yawed tunnel inside an axis-aligned box) or the mesh has holes.
  `build_case` warns when > 0.
- `n_in_mesh_and_body`: positive for immersed-boundary meshes, ~0 for body-fitted ones.
- `Cp_max_fluid`: should be ≈ 1 (stagnation point); a quick check of `U_inf`, `p_inf`, `rho`.

```python
220 inputs = np.concatenate([U, Cp[None], (sdf / geo.H)[None]], axis=0).astype(np.float32)
221 return Inputs(inputs, fluid, checks)
```
Stack into `(5, nx, ny, nz)` in contract order `u, v, w, Cp, sdf/H` (`[None]` adds the channel axis).

---

## 6. `ingredients.py`

Purpose: compute the **label ingredients** — non-dimensional Q-criterion, λ2, and vorticity
magnitude — on the native mesh. They are "teacher only": used to make labels, never fed to the model.
Computing gradients on the native mesh (not the voxel grid) uses the full CFD resolution.

Non-dimensionalization (docstring line 3): the gradient is of `u* = u/U_inf` with respect to
`x`, then multiplied by `H`, i.e. `∇* u* = (H/U_inf) ∇u`. Hence `Q* = Q H²/U_inf²`,
`λ2* = λ2 H²/U_inf²`, `|ω|* = |ω| H/U_inf`.

### Imports (lines 6–15)

```python
9  from scipy.stats import spearmanr
11 from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkUnstructuredGrid
12 from vtkmodules.vtkFiltersCore import vtkCellDataToPointData
13 from vtkmodules.vtkFiltersGeneral import vtkGradientFilter
15 from .preprocess import CP, U_CELL, U_ND
```

```python
17 INGREDIENTS = ("Q_nd", "lambda2_nd", "omega_mag_nd")
```
Canonical names, reused by `build_case` and `label`.

### Gradient convention

`vtkGradientFilter` stores a vector gradient as 9 components `du/dx, du/dy, du/dz, dv/dx, …`, so
after `reshape(-1, 3, 3)`, **`G[n, i, j] = ∂u_i/∂x_j`**. Then `S = ½(G + Gᵀ)` (strain rate) and
`W = ½(G − Gᵀ)` (rotation rate).

### `lambda2(G, n_chunk)` (lines 20–29)

```python
22 out = np.empty(G.shape[0], dtype=np.float32)
23 for i in range(0, G.shape[0], n_chunk):
25     g = G[i:i + n_chunk].astype(np.float64)
26     gt = np.transpose(g, (0, 2, 1))
27     S, W = 0.5 * (g + gt), 0.5 * (g - gt)
28     out[i:i + n_chunk] = np.linalg.eigvalsh(S @ S + W @ W)[:, 1]
29 return out
```
- λ2 (Jeong & Hussain): a vortex core is where the **middle** eigenvalue of `S² + W²` is negative.
- Chunks of 2M gradients bound memory: each `(2M, 3, 3)` float64 temporary is ~144 MB, and several
  exist at once.
- float64 (comment line 24): `S² + W²` subtracts terms of similar magnitude, and λ2 is often near
  zero, so float32 would be noisy.
- `S² + W²` is symmetric, so `eigvalsh` (symmetric solver, batched over the leading axis) applies;
  it returns eigenvalues in **ascending** order, hence `[:, 1]` is the middle one.

### `q_and_vorticity(G)` (lines 32–38)

```python
34 gt = np.transpose(G, (0, 2, 1))
35 S, W = 0.5 * (G + gt), 0.5 * (G - gt)
36 Q = 0.5 * (np.sum(W**2, axis=(1, 2)) - np.sum(S**2, axis=(1, 2)))
37 w = np.stack([G[:, 2, 1] - G[:, 1, 2], G[:, 0, 2] - G[:, 2, 0], G[:, 1, 0] - G[:, 0, 1]], 1)
38 return Q.astype(np.float32), np.linalg.norm(w, axis=1).astype(np.float32)
```
- Q-criterion: `Q = ½(‖W‖²_F − ‖S‖²_F)`; positive where rotation dominates strain.
- Vorticity `ω = ∇ × u`: `ω_x = ∂w/∂y − ∂v/∂z = G[2,1] − G[1,2]`, etc.
- Computed in the dtype of `G` (float32) without chunking — simpler and less sensitive than λ2.

### `add_ingredients(mesh, H, dataset_vorticity, dataset_cp)` (lines 41–101)

```python
48 use_cell = mesh.GetCellData().HasArray(U_CELL)
49 grad = vtkGradientFilter(input_data=mesh, result_array_name="gradU", compute_divergence=False,
50                          compute_q_criterion=False, compute_vorticity=False)
51 if use_cell:
52     grad.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_CELLS, U_CELL)
53 else:
54     grad.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, U_ND)
55 grad.Update()
```
- Prefer the gradient of the **cell** velocity when `prepare_mesh` kept one (OpenFOAM stores
  cell-centred data; the cell gradient avoids an extra cell→point smoothing first). Otherwise use
  the point velocity.
- VTK's own Q/vorticity outputs are disabled: we compute them ourselves in NumPy (and need λ2,
  which VTK does not offer).
- `SetInputArrayToProcess(idx=0, port=0, connection=0, association, name)` selects the field.
- For a cell field, `vtkGradientFilter` computes a per-cell gradient from neighbouring cell values;
  for a point field, a per-cell gradient by shape-function derivatives averaged to points. The
  output association matches the input.

```python
56 gout = grad.GetOutput()
57 gdata = gout.GetCellData() if use_cell else gout.GetPointData()
58 G = (vtk_to_numpy(gdata.GetArray("gradU")).reshape(-1, 3, 3) * H).astype(np.float32)
59 bad = ~np.isfinite(G).all(axis=(1, 2))
60 G[bad] = 0.0
```
- Fetch the gradient, reshape to `(N, 3, 3)`, multiply by H (non-dimensionalize), float32.
- Non-finite gradients (degenerate cells) are zeroed so they produce Q = λ2 = |ω| = 0 (never
  labelled) rather than NaN that would spread through interpolation.

```python
61 Q, omega = q_and_vorticity(G)
62 lam2 = lambda2(G)
63 del G
64 checks = {"gradient_of": "cell U" if use_cell else "point U", "n_nonfinite_gradient": int(bad.sum())}
```
Compute the three ingredients, free `G` (≈ 36 bytes × N; hundreds of MB to GB), start the checks.
(`gout` still holds the VTK gradient array until the function returns.)

```python
66 if dataset_vorticity and mesh.GetCellData().HasArray(dataset_vorticity) and use_cell:
67     ref = np.linalg.norm(vtk_to_numpy(mesh.GetCellData().GetArray(dataset_vorticity)), axis=1)
69     rng = np.random.default_rng(0)
70     idx = rng.choice(ref.size, min(2_000_000, ref.size), replace=False)
71     sel = idx[ref[idx] > np.median(ref)]
72     checks["omega_vs_dataset"] = dict(
73         array=dataset_vorticity,
74         spearman_r=float(spearmanr(omega[sel], ref[sel]).statistic),
75         note="per cell, 2M-cell random subsample above the median; scale differs by H/U_inf",
76     )
```
Sanity check against the dataset's own vorticity (e.g. AhmedML `vorticityMean`), only when both are
per-cell (so they align element-wise):
- The dataset array is dimensional, so values are not compared directly; Spearman rank correlation
  compares the *pattern*.
- Deterministic subsample (seed 0) of up to 2M cells, restricted to cells above the median dataset
  vorticity (the near-zero far field would add noise to ranks). Should be close to 1.

```python
78 if use_cell:
79     tmp = vtkUnstructuredGrid()
80     tmp.ShallowCopy(mesh)
81     tmp.GetPointData().Initialize()
82     tmp.GetCellData().Initialize()
83     for name, a in (("Q_nd", Q), ("lambda2_nd", lam2), ("omega_mag_nd", omega)):
84         v = numpy_to_vtk(a, deep=True)
85         v.SetName(name)
86         tmp.GetCellData().AddArray(v)
87     c2p = vtkCellDataToPointData(input_data=tmp)
88     c2p.Update()
89     point = {n: vtk_to_numpy(c2p.GetOutput().GetPointData().GetArray(n)).astype(np.float32)
90              for n in ("Q_nd", "lambda2_nd", "omega_mag_nd")}
91 else:
92     point = {"Q_nd": Q, "lambda2_nd": lam2, "omega_mag_nd": omega}
```
`vtkResampleToImage` interpolates **point** data only, so per-cell ingredients are averaged to the
points (same shallow-copy-and-clear trick as in `prepare_mesh`, so only these three arrays are
converted). Note: the ingredients are derived per cell, *then* averaged — averaging Q/λ2 is not the
same as computing them from an averaged gradient, but it is consistent between cases.

```python
94 if dataset_cp and mesh.GetPointData().HasArray(dataset_cp):
95     Cp = vtk_to_numpy(mesh.GetPointData().GetArray(CP))
96     checks["Cp_vs_dataset_maxabs"] = float(np.abs(Cp - vtk_to_numpy(mesh.GetPointData().GetArray(dataset_cp))).max())
```
Second sanity check: our `Cp` against the dataset's point Cp (AhmedML `static(p)_coeffMean`), as
the max absolute difference. This lives here (rather than in `preprocess`) because this function
already has the prepared mesh with both arrays.

```python
97  for name, a in point.items():
98      v = numpy_to_vtk(np.ascontiguousarray(a), deep=True)
99      v.SetName(name)
100     mesh.GetPointData().AddArray(v)
101 return checks
```
Add the three ingredient arrays to the prepared mesh's point data **in place**, ready for
`resample`. Return the checks.

---

## 7. `label.py`

The module docstring (lines 1–7) is the specification:

```
vortex        : lambda2* < -tau_lambda2 AND Q* > 0     AND core
shear_layer   : |omega|* > tau_omega AND NOT vortex     AND core
core = fluid eroded by wall_erosion_H AND height above the ground >= floor_exclude_H * H
```

### Imports and constants (lines 9–19)

```python
13 import numpy as np
14 import scipy.ndimage as ndi
16 from .config import LabelRules
17 from .geometry import CaseGeometry, ImageSpec
19 CLASS_NAMES = ("vortex", "shear_layer")
```
`CLASS_NAMES` fixes the channel order of the `labels` array and the names in metadata/VTI.

### `LabelResult` (lines 22–27)

```python
24 labels: np.ndarray        # (2, nx, ny, nz) uint8, CLASS_NAMES order
25 core: np.ndarray          # (nx, ny, nz) bool, where labels may be set
26 fluid_eroded: np.ndarray  # (nx, ny, nz) bool
27 fractions: dict           # per class, over fluid voxels
```

### `make_labels(ing, fluid, spec, geo, rules, n_per_H)` (lines 30–48)

```python
36 erosion = rules.wall_erosion_voxels(n_per_H)
37 fluid_eroded = ndi.binary_erosion(fluid, iterations=erosion) if erosion > 0 else fluid.copy()
```
Erode the fluid mask by `erosion` voxels (1 in v1). Rationale (label rules YAML): gradients
resampled next to walls are dominated by the wall boundary layer, which would otherwise be labelled
"shear layer" everywhere on the body. `binary_erosion` uses the default 6-connected cross and
`border_value=0`, so fluid voxels on the grid's outer faces are eroded as well, and so are fluid
voxels next to non-mesh (`valid == False`) regions. `.copy()` keeps the result independent of
`fluid` when there is no erosion.

```python
38 above_floor = geo.height_above_ground(spec) >= rules.floor_exclude_H * geo.H
39 core = fluid_eroded & above_floor
```
Exclude the ground boundary layer: layers lower than `0.12·H` above the ground. `above_floor` has
shape e.g. `(1, 1, nz)` and broadcasts.

```python
41 vortex = (ing["lambda2_nd"] < -rules.tau_lambda2) & (ing["Q_nd"] > 0) & core
42 shear = (ing["omega_mag_nd"] > rules.tau_omega) & ~vortex & core
```
The two rules. Requiring both λ2* < −0.5 and Q* > 0 makes vortex labels conservative. Shear layers
are high-vorticity regions that are not vortex cores, so the classes are mutually exclusive by
construction.

```python
44 labels = np.stack([vortex, shear]).astype(np.uint8)
45 n_fluid = max(int(fluid.sum()), 1)
46 fractions = {c: float(labels[i][fluid].sum() / n_fluid) for i, c in enumerate(CLASS_NAMES)}
47 fractions["none"] = float(((labels.sum(0) == 0) & fluid).sum() / n_fluid)
48 return LabelResult(labels, core, fluid_eroded, fractions)
```
- Stack as `(2, nx, ny, nz)` `uint8` (0/1; uint8 rather than bool for convenient summing and VTI).
- Class fractions relative to **all fluid voxels** (not just core); `max(…, 1)` avoids division by
  zero. `"none"` = fluid voxels with no label. The three fractions sum to 1 (classes are exclusive).
  These are useful for class-balancing during training.

### `describe(rules, n_per_H)` (lines 51–59)

Formats the rules as human-readable strings with this configuration's numbers, e.g.
`"lambda2* < -0.5 & Q* > 0 & core"` and
`"fluid eroded by 1 voxel(s) & height above ground >= 0.12 H"`. Stored in metadata next to the raw
thresholds so a store is self-describing.

---

## 8. `build_case.py`

Module docstring (lines 1–14) documents the CLI and the output store layout (see §4 for the
layout). One discrepancy: line 7 says `sdf/H (0 outside the fluid)`; in the code, only u, v, w and
Cp are zeroed outside the fluid, while sdf/H is defined everywhere (consistent with
`configs/contract_v1.yaml`).

### Imports (lines 16–31)

```python
18 import argparse, csv, os, time
22 from pathlib import Path
24 from vtkmodules.vtkCommonCore import vtkSMPTools
26 from . import io
27 from .config import CaseConfig, Contract, LabelRules
28 from .geometry import case_geometry, read_body
29 from .ingredients import INGREDIENTS, add_ingredients
30 from .label import CLASS_NAMES, describe, make_labels
31 from .preprocess import CP, U_ND, assemble_inputs, prepare_mesh, read_volume, resample
```
`from . import io` imports the package module `cfd_rgba_net.io` (shadowing the stdlib `io` name
only inside this module).

### Constants (lines 34–35)

```python
34 DATASET_VORTICITY = "vorticityMean"
35 DATASET_CP = "static(p)_coeffMean"
```
Names of optional dataset arrays (present in AhmedML) used only by the sanity checks in
`add_ingredients`. Hard-coded rather than per-dataset config.

### `setup_threads(n)` (lines 38–44)

```python
39 n = n or os.cpu_count() or 1
40 vtkSMPTools.SetBackend("STDThread")
41 vtkSMPTools.Initialize(n)
42 for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
43     os.environ.setdefault(var, str(n))
44 return n
```
- Default to all cores.
- VTK's SMP filters (resample, gradient, stencil, sampler) use the `STDThread` backend (portable,
  no TBB dependency) with `n` threads.
- Lines 42–43 set BLAS/OpenMP thread variables only if the user has not. Caveat: these are read when
  the BLAS library initializes, and NumPy is already imported by the time `main()` runs, so in
  practice they mostly affect child processes, not this one (see §10).

### `read_geo_parameters(path)` (lines 47–51)

```python
48 if not path.exists():
49     return {}
50 rows = list(csv.reader(path.open()))
51 return {k.strip(): float(v) for k, v in zip(rows[0], rows[1])}
```
Reads the dataset's per-run design parameters (e.g. AhmedML's slant angle, lengths): a two-row CSV,
header and values. Keys are whitespace-stripped; values must be numeric. Stored in metadata for
later conditioning/analysis. (The file handle is not explicitly closed; CPython closes it on
garbage collection.)

### `build_case(cfg, run, data, out, contract, rules, save_vti, log, max_voxels)` (lines 54–149)

```python
56 t0 = time.time()
58 def step(msg):
59     log(f"[{time.time() - t0:6.0f}s] {msg}")
```
Logger that prefixes elapsed seconds. `log` is injectable (tests capture output; the CLI flushes).

```python
61 out.mkdir(parents=True, exist_ok=True)
62 tag = f"{cfg.dataset.lower()}_run{run}"
63 volume_path = cfg.path("volume", data, run)
64 body_path = cfg.path("body", data, run)
65 geo_path = cfg.path("geo_parameters", data, run) if "geo_parameters" in cfg.files else None
```
Output directory, case tag (e.g. `ahmedml_run7`), and the input paths from the config templates.
`volume` and `body` are required keys in `files`; `geo_parameters` is optional.

```python
67 ug = read_volume(volume_path, cfg.field_names() | {DATASET_VORTICITY, DATASET_CP})
68 step(...)
69 prep = prepare_mesh(ug, cfg, keep_checks=(DATASET_VORTICITY, DATASET_CP))
70 del ug
```
Read only the velocity/pressure candidates plus the two check arrays; prepare the mesh. `del ug`
drops our reference; the raw velocity/pressure arrays become collectable (the prepared mesh shares
only the geometry and the check arrays).

```python
71 up = cfg.flow.up_axis
72 domain = cfg.domain or prep.mesh.GetBounds()
73 ref = "reference probe " + str(cfg.flow.reference_probe) if cfg.flow.reference_probe else "config"
74 step(f"prepared mesh: velocity from ..., U_inf = ..., p_inf = ... ({ref}), up {up}, ground at ..., domain ...")
```
The voxel domain is the configured box or the mesh bounds. Line 73 parses as
`("reference probe " + str(...)) if ... else "config"`. The log line records where U_inf/p_inf came
from and the domain.

```python
78 body = read_body(body_path, cfg.body_scale)
79 geo = case_geometry(body, prep.ground, cfg.reference_length, up)
80 spec = geo.image_spec(contract, domain, cfg.floor_offset_H)
81 n_vox = spec.dims[0] * spec.dims[1] * spec.dims[2]
82 step(f"H = ..., body bounds ..., grid {spec.dims} ({n_vox:,} voxels), dx = ...")
84 if n_vox > max_voxels:
85     raise ValueError(...)
```
Body → geometry → grid. The voxel-count guard (default 300M ≈ 6 GB for the float32 inputs alone)
fails fast **before** the expensive gradient and resampling steps, with a hint to set `domain`.

```python
88 checks = add_ingredients(prep.mesh, geo.H, DATASET_VORTICITY, DATASET_CP)
89 step(f"ingredients ready: {checks}")
```
Q*, λ2*, |ω|* added to the mesh's point data; sanity checks returned.

```python
92 keep = {U_ND, CP, *INGREDIENTS}
93 prep.mesh.GetCellData().Initialize()
94 pd = prep.mesh.GetPointData()
95 for name in [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]:
96     if name not in keep:
97         pd.RemoveArray(name)
98 image = resample(prep.mesh, spec, [U_ND, CP, *INGREDIENTS])
99 del prep.mesh
100 step("resampled to the voxel grid")
```
`vtkResampleToImage` interpolates **every** point array onto the grid (comment line 91), so first
strip the mesh to the five arrays needed (U_nd, Cp, Q_nd, lambda2_nd, omega_mag_nd): drop all cell
data (U_cell, check arrays) and any other point arrays (the Cp check array). Names are snapshotted
before removal. Then resample and free the mesh.

```python
102 inp = assemble_inputs(image, body, geo, spec)
103 checks["voxels"] = inp.checks
104 step(f"inputs assembled: {inp.checks}")
105 if inp.checks["n_not_in_mesh_nor_body"] > 0:
106     step("[warn] ... voxels are outside both the mesh and the body ...; they are not fluid")
```
Build the 5-channel inputs and fluid mask; nest the voxel checks; warn about uncovered voxels
(expected for WindsorML's wedges between the yawed tunnel walls and the axis-aligned box).

```python
109 ing = {k: image[k] for k in INGREDIENTS}
110 lab = make_labels(ing, inp.fluid, spec, geo, rules, contract.n_per_H)
111 step("class fractions over fluid voxels: ...")
```
Resampled ingredients → labels.

```python
113 if save_vti:
114     io.save_vti(out / f"{tag}_inputs.vti", spec, {"U_nd": inp.inputs[:3], "Cp": inp.inputs[3],
115                 "sdf_H": inp.inputs[4], "fluid": inp.fluid.astype("uint8")})
116     io.save_vti(out / f"{tag}_ingredients.vti", spec, {**ing, "core": lab.core.astype("uint8")})
117     io.save_vti(out / f"{tag}_labels.vti", spec, {**{c: lab.labels[i] for i, c in enumerate(CLASS_NAMES)},
118                 "n_labels": lab.labels.sum(0).astype("uint8")})
119 if save_vti:
120     step("wrote vti")
```
Optional ParaView files. `inputs[:3]` is a `(3, nx, ny, nz)` view → a 3-component `U_nd` array. Bool
masks are cast to `uint8` (VTK has no bool array). `n_labels` (0 or 1, since classes are exclusive)
is a convenience. The two consecutive `if save_vti:` blocks are equivalent to one.

```python
122 meta = dict(
123     tag=tag,
124     dataset=cfg.dataset,
125     run=run,
126     case_config=cfg.to_dict(),
127     contract=contract.to_dict(),
128     label_rules=dict(thresholds=rules.to_dict(), text=describe(rules, contract.n_per_H)),
129     class_names=list(CLASS_NAMES),
130     class_fractions_fluid=lab.fractions,
131     geometry=dict(H=geo.H, body_bounds=list(geo.body_bounds), ground=geo.ground,
132                   up_axis=str(up), domain=list(domain),
133                   Re_H=prep.U_inf * geo.H / cfg.flow.nu,
134                   geo_parameters=read_geo_parameters(geo_path) if geo_path else {}),
135     channel_names=list(contract.channels),
136     reference=dict(U_inf=prep.U_inf, p_inf=prep.p_inf, rho=cfg.flow.rho,
137                    source=f"probe at {list(cfg.flow.reference_probe)}" if cfg.flow.reference_probe else "config"),
138     fields=dict(velocity=prep.velocity_source, pressure=prep.pressure_source),
139     checks=checks,
140     provenance=io.provenance(dict(volume=volume_path, body=body_path, case_config=Path(cfg.source or ""))),
141     vtk_threads=vtkSMPTools.GetEstimatedNumberOfThreads(),
142     runtime_s=time.time() - t0,
143 )
```
All metadata, stored as the store's root attributes:

| Key | Content |
|---|---|
| `tag`, `dataset`, `run` | identity |
| `case_config` | the full dataset config as loaded |
| `contract` | version, n_per_H, channels |
| `label_rules` | raw thresholds + human-readable rules |
| `class_names`, `class_fractions_fluid` | label channel order and fractions |
| `geometry` | H, body bounds, ground, up axis, domain, Reynolds number `Re_H = U_inf·H/ν`, design parameters |
| `channel_names` | input channel order (`u, v, w, Cp, sdf`) |
| `reference` | U_inf, p_inf, rho and where they came from |
| `fields` | which arrays velocity/pressure were read from |
| `checks` | gradient/vorticity/Cp checks + `voxels` (counts, Cp max, SDF method/offset) |
| `provenance` | package/contract/rules versions, git commit + dirty flag, SHA-256 of volume, body, and case YAML |
| `vtk_threads`, `runtime_s` | performance |

Note that hashing the volume (tens of GB) happens on the first build only; `io.sha256` caches it.

```python
144 io.save_case_zarr(out / f"{tag}.zarr", spec,
145                   arrays=dict(inputs=inp.inputs, fluid=inp.fluid, labels=lab.labels),
146                   groups=dict(ingredients={**{k: v.astype("float32") for k, v in ing.items()}, "core": lab.core}),
147                   attrs=meta)
148 step(f"wrote {tag}.zarr")
149 return dict(grid=spec.to_dict(), **meta)
```
Write the store atomically. The ingredients go into a sub-group to mark them as teacher-only. The
return value mirrors the store's attributes (used by tests).

### `main(argv)` (lines 152–164)

```python
153 ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
154 ap.add_argument("--dataset", required=True, help="dataset config: a name in configs/datasets/ or a YAML path")
155 ap.add_argument("--run", type=int, required=True)
156 ap.add_argument("--data", type=Path, required=True, help="data root holding run_<N>/")
157 ap.add_argument("--out", type=Path, required=True)
158 ap.add_argument("--save-vti", action="store_true", help="also write .vti files for viewing")
159 ap.add_argument("--threads", type=int, default=None, help="VTK SMP threads (default: all cores)")
160 ap.add_argument("--max-voxels", type=int, default=300_000_000, help="refuse grids larger than this")
161 args = ap.parse_args(argv)
162 setup_threads(args.threads)
163 build_case(CaseConfig.load(args.dataset), args.run, args.data, args.out, Contract.load(), LabelRules.load(),
164            args.save_vti, log=lambda m: print(m, flush=True), max_voxels=args.max_voxels)
```
`RawDescriptionHelpFormatter` keeps the docstring's formatting in `--help`. `argv=None` means
`sys.argv[1:]`; passing a list makes `main` testable. The contract and label rules always come from
the packaged `configs/` (not overridable from the CLI). `flush=True` makes log lines appear
immediately when stdout is redirected to a file (as `batch.py` does).

Entry points: `cfd-rgba-net-build-case` (from `pyproject.toml`'s `[project.scripts]`) or
`python -m cfd_rgba_net.build_case` (lines 167–168).

---

## 9. `batch.py`

Module docstring (lines 1–17): build many runs of one dataset, downloading each run from Hugging
Face on demand into a work directory (default in the system temp dir, so not on the output disk),
building in a subprocess, and deleting the downloads after a successful build. Failed runs keep
their downloads for diagnosis. Runs whose store already exists are skipped.

### Imports (lines 19–32)

```python
21 import argparse, os, shutil, subprocess, sys, tempfile, threading, time
29 from concurrent.futures import ThreadPoolExecutor
30 from pathlib import Path
32 from .config import CaseConfig
```
`pandas` and `huggingface_hub` are imported lazily inside the functions that need them.

### `parse_runs(spec)` (lines 35–44)

```python
37 if spec.endswith(".csv"):
38     import pandas as pd
39     return pd.read_csv(spec)["run"].astype(int).tolist()
40 runs = []
41 for part in spec.split(","):
42     a, _, b = part.partition("-")
43     runs += list(range(int(a), int(b) + 1)) if b else [int(a)]
44 return runs
```
Accepts a CSV with a `run` column (e.g. AhmedML's `geo_parameters_all.csv`), or a comma-separated
list of numbers and inclusive ranges: `"1-3,10"` → `[1, 2, 3, 10]`. `partition("-")` splits at the
first dash; if there is none, `b` is `""`. Duplicates are kept; a reversed range (`"5-3"`) yields
nothing; negative numbers are not supported.

### `download(cfg, run, work)` (lines 47–59)

```python
48 from huggingface_hub import hf_hub_download
49 if not cfg.hf_repo:
50     raise ValueError(f"{cfg.dataset}: no hf_repo in the dataset config")
51 for key, template in cfg.files.items():
52     rel = template.format(run=run)
53     target = work / rel
54     if target.exists() and target.stat().st_size > 0:
55         continue
56     t0 = time.time()
57     hf_hub_download(repo_id=cfg.hf_repo, filename=rel, local_dir=work, repo_type="dataset")
58     print(f"[dl ] {cfg.hf_repo}: {rel}  {target.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)
```
Every file in `files` (volume, body, geo_parameters) is fetched from the HF **dataset** repo into
`work/<rel>`, preserving the repo-relative path, so `work` becomes a valid `--data` root for
`build_case`. Non-empty existing files are not re-downloaded (resumable). With `hf-xet` installed,
transfers use the Xet backend. Note: DrivAerML's volume is split into `.part` files on HF; this
function does not handle that (the DrivAerML config is marked unverified).

### `store_path(cfg, out, run)` (lines 62–63)

`out / "<dataset lower>_run<N>.zarr"` — must match the `tag` in `build_case` (line 62 there).

### `build(cfg, run, args)` (lines 66–77)

```python
67 log = args.out / f"log_{cfg.dataset.lower()}_run{run}.txt"
68 print(f"[run] build run_{run} (log {log})", flush=True)
69 cmd = [sys.executable, "-u", "-m", "cfd_rgba_net.build_case", "--dataset", args.dataset, "--run", str(run),
70        "--data", str(args.work), "--out", str(args.out), "--threads", str(args.threads)]
71 cmd += (["--save-vti"] if args.save_vti else []) + args.extra
72 t0 = time.time()
73 with log.open("w") as fh:
74     r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
75 ok = r.returncode == 0 and store_path(cfg, args.out, run).exists()
76 print(f"[run] run_{run}: {'ok' if ok else f'FAILED (exit {r.returncode})'} in {time.time() - t0:.0f}s", flush=True)
77 return ok
```
- Each build runs in a **separate Python process** (same interpreter, `sys.executable`), which
  isolates memory (a 100M-cell mesh needs tens of GB, returned to the OS when the process exits),
  isolates VTK crashes (a segfault fails one run, not the batch), and sidesteps the GIL.
- `-u` = unbuffered output, so the per-run log is live.
- `args.dataset` is passed as given (name or path); the child resolves it again.
- stdout and stderr go to one log file per run in `--out`.
- Success requires exit code 0 **and** the store existing.
- `args.extra` forwards anything after `--`, e.g. `-- --max-voxels 500000000`.

### `main(argv)` (lines 80–120)

```python
81 ap = argparse.ArgumentParser(...)
82 ap.add_argument("--dataset", required=True, ...)
83 ap.add_argument("--runs", required=True, ...)
84 ap.add_argument("--out", type=Path, required=True, ...)
85 ap.add_argument("--work", type=Path, default=None, ...)
87 ap.add_argument("--keep", action="store_true", ...)
88 ap.add_argument("--workers", type=int, default=1, ...)
89 ap.add_argument("--threads", type=int, default=None, ...)
90 ap.add_argument("--save-vti", action="store_true", ...)
91 ap.add_argument("extra", nargs="*", ...)
92 args = ap.parse_args(argv)
```
CLI. `extra` is a positional catch-all; argparse puts everything after `--` into it.

```python
93 cfg = CaseConfig.load(args.dataset)
94 args.work = args.work or Path(tempfile.gettempdir()) / f"cfd-rgba-net-{cfg.dataset.lower()}"
95 args.threads = args.threads or max(1, (os.cpu_count() or 1) // args.workers)
```
Load the config once here (for `files`, `hf_repo` and naming). Default work dir e.g.
`/tmp/cfd-rgba-net-ahmedml`. Default threads per build = cores / workers, at least 1, so concurrent
builds do not oversubscribe the CPU. Results are written back onto `args`, which `build` reads.

```python
96 runs = parse_runs(args.runs)
97 todo = [r for r in runs if not store_path(cfg, args.out, r).exists()]
98 print(f"{cfg.dataset}: {len(runs)} runs requested, {len(todo)} to do; downloads in {args.work}", flush=True)
99 args.out.mkdir(parents=True, exist_ok=True)
```
Skip finished runs (safe thanks to the atomic store rename), report, create the output dir.

```python
101 results: dict[int, bool] = {}
102 lock = threading.Lock()
104 def worker(r: int):
105     try:
106         download(cfg, r, args.work)
107         ok = build(cfg, r, args)
108     except Exception as e:  # noqa: BLE001  (report and go on with the next run)
109         print(f"[run] run_{r}: {e}", flush=True)
110         ok = False
111     if ok and not args.keep:
112         shutil.rmtree(args.work / f"run_{r}", ignore_errors=True)
113     with lock:
114         results[r] = ok
115         bad = sorted(k for k, v in results.items() if not v)
116         print(f"done: {sum(results.values())}/{len(results)} ok; failed: {bad}", flush=True)
```
Per-run job: download then build. Any exception (network, missing `hf_repo`, …) marks the run failed
and moves on; the `noqa` silences the linter's broad-except warning on purpose. On success, the
run's download folder is removed (it assumes files live under `run_<N>/`, as in all current configs).
Results and the progress line are updated under a lock so concurrent workers do not interleave the
dict update and the print.

```python
118 with ThreadPoolExecutor(max_workers=args.workers) as pool:
119     list(pool.map(worker, todo))
120 return 0 if all(results.values()) else 1
```
Threads are enough here: the heavy work is in subprocesses and downloads, both of which release the
GIL. `list(...)` drains the iterator so any unexpected exception would surface (none can, since
`worker` catches them). Exit code 0 only if every attempted run succeeded (and 0 if there was nothing
to do, since `all([])` is `True`). Each worker downloads then builds sequentially, so with
`--workers N` up to N downloads/builds overlap.

Lines 123–124: `sys.exit(main())` propagates the exit code when run as `python -m cfd_rgba_net.batch`;
the console script `cfd-rgba-net-batch` does the same via `[project.scripts]`.

---

## 10. Cross-cutting notes and caveats

Behaviours worth knowing that are not obvious from any single function:

1. **Memory lifecycle in `build_case`.** The largest objects are the raw mesh arrays, the gradient
   `G`, and the voxel arrays. The code frees in a fixed order: raw arrays (`del ug` after
   `prepare_mesh`), `G` (inside `add_ingredients`), every unneeded mesh array before resampling, then
   the mesh itself (`del prep.mesh`). Peak memory is usually the gradient step or `resample`.
2. **Point vs. cell data.** Inputs are always resampled from *point* data; ingredients are computed
   from *cell* velocity when available and averaged to points. `prepare_mesh` falls back to
   cell→point averaging for velocity/pressure when point data is incomplete.
3. **`freestream_direction` is unused.** It is parsed into `Flow` and stored in metadata via
   `case_config`, but no code reads it and it is not validated (the comment at `config.py:202`
   suggests both directions are checked; only `up_direction` is).
4. **`io.save_json` and `PreparedMesh.checks` are unused.**
5. **Docstring vs. code on the SDF channel.** `build_case.py:7` says sdf/H is 0 outside the fluid; the
   code (and the contract YAML) only zero u, v, w, Cp.
6. **`image_spec` with up positive ignores `domain`'s lower bound along up**; the grid always starts
   `floor_offset_H·H` above the ground.
7. **Thread environment variables** in `setup_threads` are set after NumPy/SciPy are imported (they
   are imported at module load by `io`, `geometry`, etc.), so OpenBLAS/MKL/OpenMP in the current
   process have usually already sized their pools; the variables mainly affect child processes.
   VTK's thread count (`vtkSMPTools.Initialize`) does take effect.
8. **SDF exact-path fallback.** If the body has no voxel inside the grid (`inside.any()` is false),
   `signed_distance` computes the exact distance on the *whole* grid, which can be very slow for
   large domains.
9. **`probe_point` weight buffer** has 64 slots; a containing polyhedral cell with more points would
   overflow it.
10. **Provenance with `cfg.source = None`.** `build_case` passes `Path(cfg.source or "")`, which is
    `Path(".")`, an existing directory; `io.sha256` would then fail to open it. This cannot happen
    through `CaseConfig.load` (which always sets `source`), only with hand-built configs.
11. **Sidecar hash files** (`*.sha256`) are written next to every hashed input, including
    `configs/datasets/*.yaml`; they show up as untracked files in the repo.
12. **Atomicity contract.** `save_case_zarr` writes to `.tmp` and renames; `batch.py`'s skip logic
    and `build`'s success check both depend on "exists ⇒ complete".
13. **Versioning discipline.** Change anything in `preprocess.py`, `geometry.signed_distance`,
    `geometry.image_spec`, or `contract_v1.yaml` that alters input numbers → bump
    `CONTRACT_VERSION` (and the YAML). Change thresholds or `label.py` logic → bump
    `LABEL_RULES_VERSION` (and the YAML file name/version).
