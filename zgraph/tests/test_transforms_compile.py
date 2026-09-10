import pytest
import jax.numpy as jnp
import jax

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

    phase_a = FactorNode(jnp.eye(2), [mu1a, mu2a], beta=rt)
    phase_b = FactorNode(jnp.eye(2), [mu1b, mu2b], beta=rt)
    system = FactorNode(jnp.eye(2), [phase_a, phase_b], beta=0)
    return phase_a, phase_b, system


def test_graph_and_legendre_paths_compile_and_execute():
    phase_a, phase_b, system = _build_binary_graph()

    t_val = jnp.array(298.15)
    mu1_vals = jnp.linspace(-10.0, 10.0, num=32)
    mu2_vals = -mu1_vals
    t_flat = jnp.broadcast_to(t_val, mu1_vals.shape)
    input_tensor = jnp.stack([t_flat, mu1_vals, mu2_vals], axis=-1)

    phase_a_fn, phase_b_fn, system_fn = graph_to_function(
        [phase_a, phase_b, system],
        compile=True,
    )
    base_out = system_fn(input_tensor)

    legendre_modules = legendre_transform([phase_a, phase_b, system], [1, 2])
    leg_a_fn, leg_b_fn, leg_s_fn = graph_to_function(
        legendre_modules,
        compile=True,
    )
    psi, x_dual = leg_s_fn(input_tensor)

    assert base_out.shape == (32,)
    assert psi.shape == (32,)
    assert x_dual.shape == (32, 3)
    assert jnp.isfinite(base_out).all()
    assert jnp.isfinite(psi).all()
    assert jnp.isfinite(x_dual).all()
