import pytest

from cfd_rgba_net import CONTRACT_VERSION, LABEL_RULES_VERSION
from cfd_rgba_net.config import CaseConfig, Contract, LabelRules


def test_contract_and_rules_load():
    c = Contract.load()
    r = LabelRules.load()
    assert c.contract_version == CONTRACT_VERSION
    assert r.label_rules_version == LABEL_RULES_VERSION
    assert c.channels == ("u", "v", "w", "Cp", "body_proximity", "ground_proximity")
    # unit conversions reproduce the previous voxel-based settings at n_per_H = 32
    assert r.wall_erosion_voxels(32) == 1


@pytest.mark.parametrize("name", ["ahmedml", "windsorml", "drivaerml"])
def test_dataset_configs_load(name):
    cfg = CaseConfig.load(name)
    assert (cfg.flow.U_inf or 0) > 0 or cfg.flow.reference_probe is not None
    assert {"volume", "body"} <= set(cfg.files)
    assert cfg.path("volume", "/root", 7).as_posix().endswith("run_7/volume_7.vtu")


def test_windsor_velocity_is_components():
    cfg = CaseConfig.load("windsorml")
    assert cfg.velocity[0].components == ("velocityxavg", "velocityyavg", "velocityzavg")


def test_windsor_reference_probe_domain_and_floor_offset():
    cfg = CaseConfig.load("windsorml")
    assert cfg.flow.reference_probe == (-2.0, 1.3, 0.0) and cfg.flow.U_inf is None and cfg.flow.p_inf is None
    assert cfg.domain == (-5.0, 6.0, 0.0, 1.32, -1.223, 1.18) and cfg.floor_offset_H == 1e-4
    assert str(cfg.flow.up_axis) == "+y"
    ahmed = CaseConfig.load("ahmedml")
    assert ahmed.domain is None and ahmed.floor_offset_H == 1e-6 and ahmed.flow.reference_probe is None


@pytest.mark.parametrize("flow,ok", [
    ("U_inf: 1.0", True),
    ("reference_probe: [0, 0, 0]", True),
    ("reference_probe: [0, 0, 0]\n  U_inf: 1.0", False),   # both
    ("rho: 1.0", False),                                    # neither
])
def test_reference_probe_xor_fixed_values(tmp_path, flow, ok):
    y = tmp_path / "c.yaml"
    y.write_text(f"""dataset: T
files: {{volume: v, body: b}}
fields: {{velocity: [U], pressure: [p]}}
flow:
  {flow}
  nu: 1.0e-5
""")
    if ok:
        CaseConfig.load(y)
    else:
        with pytest.raises(ValueError):
            CaseConfig.load(y)


def test_parse_runs():
    from cfd_rgba_net.batch import parse_runs
    assert parse_runs("1,5,7") == [1, 5, 7]
    assert parse_runs("0-3,10") == [0, 1, 2, 3, 10]
