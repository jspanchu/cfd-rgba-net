import math

import pytest
import torch

from cfd_rgba_net.infer import normalized_entropy


def test_normalized_entropy_spans_certain_to_uniform_and_is_zero_outside_the_core():
    # four voxels: certain, even split between two classes, all three equally likely, unscored (all zero)
    p = torch.tensor([[1.0, 0.5, 1 / 3, 0.0],
                      [0.0, 0.5, 1 / 3, 0.0],
                      [0.0, 0.0, 1 / 3, 0.0]]).reshape(3, 4, 1, 1)
    u = normalized_entropy(p).flatten()
    assert u.tolist() == pytest.approx([0.0, math.log(2) / math.log(3), 1.0, 0.0], abs=1e-6)
