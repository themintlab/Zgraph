import pytest
import torch

from zgraph import FactorNode, SignalNodes
from zgraph.transforms import graph_to_function, legendre_transform


def _build_binary_graph():
    r = 8.314
    t, mu1, mu2 = SignalNodes(0, 1, 2)

    rt = FactorNode([[r]], [t])
    mu1a = FactorNode([2, -1], [rt, mu1])
    mu2a = FactorNode([-1], [mu2])
    mu1b = FactorNode([-1], [mu1])
    mu2b = FactorNode([1, -1], [rt, mu2])

    phase_a = FactorNode(torch.eye(2), [mu1a, mu2a], beta=rt)
    phase_b = FactorNode(torch.eye(2), [mu1b, mu2b], beta=rt)
    system = FactorNode(torch.eye(2), [phase_a, phase_b], beta=0)
    return phase_a, phase_b, system


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile is unavailable")
def test_graph_and_legendre_paths_compile_and_execute():
    phase_a, phase_b, system = _build_binary_graph()

    t_val = torch.tensor(298.15)
    mu1_vals = torch.linspace(-10.0, 10.0, steps=32)
    mu2_vals = -mu1_vals
    t_flat = t_val.expand_as(mu1_vals)
    input_tensor = torch.stack([t_flat, mu1_vals, mu2_vals], dim=-1)

    phase_a_fn, phase_b_fn, system_fn = graph_to_function(
        [phase_a, phase_b, system],
        compile=True,
        compile_mode="reduce-overhead",
    )
    base_out = system_fn(input_tensor)

    legendre_modules = legendre_transform([phase_a, phase_b, system], [1, 2])
    leg_a_fn, leg_b_fn, leg_s_fn = graph_to_function(
        legendre_modules,
        compile=True,
        compile_mode="reduce-overhead",
    )
    psi, x_dual = leg_s_fn(input_tensor)

    assert base_out.shape == torch.Size([32])
    assert psi.shape == torch.Size([32])
    assert x_dual.shape == torch.Size([32, 3])
    assert torch.isfinite(base_out).all()
    assert torch.isfinite(psi).all()
    assert torch.isfinite(x_dual).all()
