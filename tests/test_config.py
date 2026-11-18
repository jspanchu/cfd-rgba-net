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


@pytest.mark.parametrize("kwargs,match", [
    (dict(flow="reference_probe: [0, 0, 0]\n  U_inf: 1.0"), "reference_probe|U_inf"),     # both
    (dict(flow="rho: 1.0"), "reference_probe|U_inf"),                                      # neither
    (dict(source=""), "field_source"),
    (dict(source="field_source: Point"), "field_source"),
    (dict(extra="domain: [0, 1, 0, 1]"), "domain"),                                        # wrong length
    (dict(extra="domain: [0, 1, 0, 1, 2, 1]"), "domain"),                                  # zmin > zmax
    (dict(flow="U_inf: 1.0\n  up_direction: [1, 1, 0]"), "up direction"),
    (dict(velocity="[{components: [ux, uy]}]"), "3 names"),
], ids=lambda v: v if isinstance(v, str) else "")
def test_invalid_case_config_is_rejected(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        CaseConfig.load(write_case(tmp_path, **kwargs))


def test_optional_settings_default_off_and_are_read_when_given(tmp_path):
    cfg = CaseConfig.load(write_case(tmp_path))
    assert (cfg.body_merge_tolerance, cfg.floor_offset_H, cfg.grid_inset_H) == (None, 0.0, 0.0)
    assert (cfg.check_vorticity, cfg.check_cp) == (None, None)
    cfg = CaseConfig.load(write_case(tmp_path, extra="floor_offset_H: 1.0e-4\ngrid_inset_H: 2.0e-6\n"
                                                      "body: {merge_tolerance: 1.0e-5}\n"
                                                      "check_fields: {vorticity: vortMean, cp: cpMean}"))
    assert (cfg.body_merge_tolerance, cfg.floor_offset_H, cfg.grid_inset_H) == (1e-5, 1e-4, 2e-6)
    assert (cfg.check_vorticity, cfg.check_cp) == ("vortMean", "cpMean")


@pytest.mark.parametrize("direction,expected", [((0, 0, -1), "-z"), ((0.99904, 0, -0.0438), "+x")])  # WindsorML yaw
def test_axis_of_picks_the_dominant_axis_and_rejects_off_axis_directions(direction, expected):
    assert str(axis_of(direction)) == expected
    with pytest.raises(ValueError):
        axis_of((1, 1, 0))
