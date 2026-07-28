import torch
import torch.nn as nn

from thermograph import SGTENode
from thermograph.core.functional import SGTE_polynomial


class TrackingTemperatureNode(nn.Module):
    def __init__(self):
        super().__init__()
        self.called = False

    def forward(self, state_tensor):
        self.called = True
        return state_tensor[0]


def test_sgte_node_uses_temperature_child_and_matches_functional_core():
    temperature_node = TrackingTemperatureNode()
    coeffs = {"a": 1.0, "b": 2.0, "c": 0.5, "d": 0.1, "e": 0.01, "f": 3.0}
    node = SGTENode(temperature_node=temperature_node, coeffs=coeffs)
    state = torch.tensor([1000.0], dtype=torch.float64)

    result = node(state)
    expected = SGTE_polynomial(
        state[0],
        node.a,
        node.b,
        node.c,
        node.d,
        node.e,
        node.f,
    )

    assert temperature_node.called is True
    assert torch.allclose(result, expected, atol=1e-10)


def test_sgte_node_defaults_to_zero_coefficients():
    temperature_node = TrackingTemperatureNode()
    node = SGTENode(temperature_node=temperature_node, coeffs=None)
    state = torch.tensor([500.0], dtype=torch.float64)

    result = node(state)

    assert torch.allclose(result, torch.tensor(0.0, dtype=torch.float64), atol=1e-12)
