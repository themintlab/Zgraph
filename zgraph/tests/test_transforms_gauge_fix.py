import torch

from zgraph import FactorNode, SignalNodes
from zgraph.transforms import gauge_fix


def _build_simple_graph():
    x0, x1 = SignalNodes(0, 1)
    return FactorNode([[1.0], [2.0]], [x0, x1], beta=1.0)


def test_gauge_fix_single_module_returns_projected_tuple():
    module = _build_simple_graph()
    fixer = gauge_fix(module, [0, 1])

    primal_x = torch.tensor([0.2, -0.3])
    target, projected_x = fixer(primal_x)

    assert target.ndim == 0
    assert projected_x.shape == primal_x.shape
    assert torch.isfinite(target)
    assert torch.isfinite(projected_x).all()


def test_gauge_fix_preserves_container_type_for_multiple_modules():
    module = _build_simple_graph()

    as_list = gauge_fix([module, module], [0])
    as_tuple = gauge_fix((module, module), [0])

    assert isinstance(as_list, list)
    assert isinstance(as_tuple, tuple)
    assert len(as_list) == 2
    assert len(as_tuple) == 2
