import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from cfd_rgba_net.config import CaseConfig, Flow, VectorField  # noqa: E402


@pytest.fixture
def flow_cfg():
    def make(U_inf=1.0, p_inf=0.0, rho=1.0, up=(0, 0, 1), velocity=None, ground=None):
        return CaseConfig(
            dataset="Synthetic",
            files={},
            velocity=velocity or (VectorField(name="UMean"),),
            pressure=("pMean",),
            flow=Flow(U_inf=U_inf, p_inf=p_inf, rho=rho, nu=1e-5, up_direction=up, ground=ground),
        )
    return make
