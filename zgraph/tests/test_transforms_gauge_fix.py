import jax
import jax.numpy as jnp

from zgraph import FactorNode, SignalNodes
from zgraph.solvers import gauge_fix


def _build_simple_graph():
    x0, x1 = SignalNodes(0, 1)
    return FactorNode([[1.0, 2.0]], [x0, x1], beta=1.0)


def test_gauge_fix_solves_properly():
    module = _build_simple_graph()
    primal_x = jnp.array([0.2, -0.3])
    projected_x = gauge_fix(module, primal_x, [0, 1])

    assert projected_x.shape == primal_x.shape
    assert jnp.isfinite(projected_x).all()

