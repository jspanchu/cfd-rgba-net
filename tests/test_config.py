import pytest

from cfd_rgba_net.config import CaseConfig, Contract, LabelRules, axis_of


def write_case(tmp_path, flow="U_inf: 1.0", velocity="[U]", extra="", source="field_source: point"):
    """Write a minimal dataset YAML. `flow`, `source` and `extra` are spliced in as YAML text."""
    y = tmp_path / "c.yaml"
    y.write_text(f"""dataset: T
files: {{volume: v, body: b}}
fields: {{velocity: {velocity}, pressure: [p]}}
{source}
flow:
  {flow}
  nu: 1.0e-5
{extra}
""")
    return y


def test_default_contract_and_rules_load():
    Contract.load()
    r = LabelRules.load()
    # the unit conversion reproduces the earlier voxel-based setting at n_per_H = 32
    assert r.wall_erosion_voxels(32) == 1


@pytest.mark.parametrize("loader,text", [
    (Contract.load, 'contract_version: "0.9"\nn_per_H: 32\nchannels: [u]\n'),
    (LabelRules.load, 'label_rules_version: "0"\ntau_lambda2: 0.5\ntau_omega: 2.0\n'
                      'floor_exclude_H: 0.1\nwall_erosion_H: 0.03\n'),
])
def test_version_mismatch_is_rejected(tmp_path, loader, text):
    y = tmp_path / "v.yaml"
    y.write_text(text)
    with pytest.raises(ValueError, match="code is v"):
        loader(y)


@pytest.mark.parametrize("name", ["ahmedml", "windsorml", "drivaerml"])
def test_dataset_configs_load(name):
    cfg = CaseConfig.load(name)
    assert (cfg.flow.U_inf or 0) > 0 or cfg.flow.reference_probe is not None
    assert {"volume", "body"} <= set(cfg.files)
    assert cfg.path("volume", "/root", 7).as_posix().endswith("run_7/volume_7.vtu")


@pytest.mark.parametrize("flow,ok", [
    ("U_inf: 1.0", True),
    ("reference_probe: [0, 0, 0]", True),
    ("reference_probe: [0, 0, 0]\n  U_inf: 1.0", False),   # both
    ("rho: 1.0", False),                                    # neither
])
def test_reference_probe_xor_fixed_values(tmp_path, flow, ok):
    y = write_case(tmp_path, flow=flow)
    if ok:
        CaseConfig.load(y)
    else:
        with pytest.raises(ValueError):
            CaseConfig.load(y)


@pytest.mark.parametrize("source", ["", "field_source: face", "field_source: Point"])
def test_field_source_must_be_point_or_cell(tmp_path, source):
    with pytest.raises(ValueError, match="field_source"):
        CaseConfig.load(write_case(tmp_path, source=source))


def test_grid_and_stl_settings_default_to_off(tmp_path):
    cfg = CaseConfig.load(write_case(tmp_path))
    assert (cfg.body_merge_tolerance, cfg.floor_offset_H, cfg.grid_inset_H) == (None, 0.0, 0.0)


def test_grid_and_stl_settings_are_read_from_the_dataset(tmp_path):
    cfg = CaseConfig.load(write_case(tmp_path, extra="floor_offset_H: 1.0e-4\ngrid_inset_H: 2.0e-6\n"
                                                      "body: {merge_tolerance: 1.0e-5}"))
    assert (cfg.body_merge_tolerance, cfg.floor_offset_H, cfg.grid_inset_H) == (1e-5, 1e-4, 2e-6)


def test_check_fields_are_optional_and_read_from_the_dataset(tmp_path):
    cfg = CaseConfig.load(write_case(tmp_path))
    assert (cfg.check_vorticity, cfg.check_cp) == (None, None)
    cfg = CaseConfig.load(write_case(tmp_path, extra="check_fields: {vorticity: vortMean, cp: cpMean}"))
    assert (cfg.check_vorticity, cfg.check_cp) == ("vortMean", "cpMean")


@pytest.mark.parametrize("domain", [
    "[0, 1, 0, 1]",          # wrong length
    "[0, 1, 0, 1, 2, 1]",    # zmin > zmax
    "[0, 1, 0, 0, 0, 1]",    # ymin == ymax
])
def test_malformed_domain_is_rejected(tmp_path, domain):
    with pytest.raises(ValueError, match="domain"):
        CaseConfig.load(write_case(tmp_path, extra=f"domain: {domain}"))


@pytest.mark.parametrize("up", ["[0, 1]", "[1, 1, 0]", "[0, 0, 0]"])
def test_up_direction_must_run_along_a_mesh_axis(tmp_path, up):
    with pytest.raises(ValueError, match="up direction"):
        CaseConfig.load(write_case(tmp_path, flow=f"U_inf: 1.0\n  up_direction: {up}"))


def test_velocity_components_must_be_three_names(tmp_path):
    y = write_case(tmp_path, velocity="[{components: [ux, uy]}]")
    with pytest.raises(ValueError, match="3 names"):
        CaseConfig.load(y)


@pytest.mark.parametrize("direction,expected", [
    ((1, 0, 0), "+x"), ((0, 1, 0), "+y"), ((0, 0, -1), "-z"), ((0.99904, 0, -0.0438), "+x"),  # WindsorML yaw
])
def test_axis_of(direction, expected):
    assert str(axis_of(direction)) == expected


@pytest.mark.parametrize("direction", [(1, 1, 0), (0, 0, 0), (0, 1)])
def test_axis_of_rejects_off_axis(direction):
    with pytest.raises(ValueError):
        axis_of(direction)
