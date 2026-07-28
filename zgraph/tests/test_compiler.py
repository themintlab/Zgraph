import torch
import pytest

from zgraph import ConstantNode, FactorNode, SignalNode, SignalNodes
from zgraph.transforms import compile_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_identity_factor(beta=1.0):
    """2-cluster identity FactorNode: energy = logsumexp([w0, w1])."""
    return FactorNode(
        M_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        subgraph_list=[ConstantNode(1.0), ConstantNode(2.0)],
        beta=beta,
    )


def make_signal_factor():
    """FactorNode whose cluster energies come from two input signals."""
    s0, s1 = SignalNodes(0, 1)
    return FactorNode(
        M_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        subgraph_list=[s0, s1],
        beta=1.0,
    )


# ---------------------------------------------------------------------------
# Return type — plain callable
# ---------------------------------------------------------------------------

def test_compile_graph_returns_callable():
    node = make_identity_factor()
    fn = compile_graph(node, use_torch_compile=False)
    assert callable(fn)


def test_compile_graph_list_returns_list_of_callables():
    nodes = [make_identity_factor(), ConstantNode(3.0)]
    result = compile_graph(nodes, use_torch_compile=False)
    assert isinstance(result, list)
    assert all(callable(f) for f in result)


def test_compile_graph_tuple_returns_tuple_of_callables():
    nodes = (make_identity_factor(),)
    result = compile_graph(nodes, use_torch_compile=False)
    assert isinstance(result, tuple)
    assert all(callable(f) for f in result)


def test_compile_graph_dict_returns_dict_of_callables():
    nodes = {"a": make_identity_factor(), "b": ConstantNode(0.5)}
    result = compile_graph(nodes, use_torch_compile=False)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"a", "b"}
    assert all(callable(f) for f in result.values())


# ---------------------------------------------------------------------------
# Unbatched correctness
# ---------------------------------------------------------------------------

def test_unbatched_constant_factor_matches_expected():
    """Single-sample forward should match direct module call."""
    node = make_identity_factor(beta=1.0)
    fn = compile_graph(node, batched=False, use_torch_compile=False)

    x = torch.tensor([0.0])
    assert torch.allclose(fn(x), node(x), atol=1e-6)


def test_unbatched_signal_factor_matches_expected():
    """Signals extracted from state tensor should work in unbatched mode."""
    node = make_signal_factor()
    fn = compile_graph(node, batched=False, use_torch_compile=False)

    x = torch.tensor([1.0, 2.0])
    assert torch.allclose(fn(x), node(x), atol=1e-6)


def test_unbatched_constant_node_matches_expected():
    node = ConstantNode(5.0)
    fn = compile_graph(node, batched=False, use_torch_compile=False)
    x = torch.tensor([0.0])
    assert torch.allclose(fn(x), node(x), atol=1e-6)


# ---------------------------------------------------------------------------
# Batched correctness
# ---------------------------------------------------------------------------

def test_batched_output_shape():
    node = make_signal_factor()
    fn = compile_graph(node, batched=True, use_torch_compile=False)

    batch = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert fn(batch).shape == (3,)


def test_batched_results_match_unbatched():
    """Each row of a batched call must equal the corresponding unbatched call."""
    node = make_signal_factor()
    fn_batched = compile_graph(node, batched=True, use_torch_compile=False)
    fn_single = compile_graph(node, batched=False, use_torch_compile=False)

    xs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.5, 0.5]])
    batched_out = fn_batched(xs)

    for i, x in enumerate(xs):
        single_out = fn_single(x)
        assert torch.allclose(batched_out[i], single_out, atol=1e-6), (
            f"Mismatch at row {i}: batched={batched_out[i]}, single={single_out}"
        )


def test_batched_constant_factor():
    """ConstantNode-only graph: all batch outputs should equal the constant."""
    node = ConstantNode(7.0)
    fn = compile_graph(node, batched=True, use_torch_compile=False)

    batch = torch.zeros(4, 1)
    result = fn(batch)

    assert result.shape == (4,)
    assert torch.allclose(result, torch.full((4,), 7.0), atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient flow through live parameters
# ---------------------------------------------------------------------------

def test_gradient_flows_through_params_unbatched():
    node = ConstantNode(3.0)
    fn = compile_graph(node, batched=False, use_torch_compile=False)

    fn(torch.tensor([0.0])).backward()
    assert node.value.grad is not None


def test_gradient_flows_through_params_batched():
    """Loss over a batch should propagate gradients back to node parameters."""
    node = make_identity_factor(beta=1.0)
    fn = compile_graph(node, batched=True, use_torch_compile=False)

    fn(torch.zeros(5, 1)).sum().backward()
    assert node.beta.value.grad is not None


def test_fn_attribute_works_with_torch_func_grad():
    """fn attribute can be composed with torch.func.grad for parameter gradients."""
    node = ConstantNode(2.0)
    fn = compile_graph(node, batched=False, use_torch_compile=False)

    params = dict(node.named_parameters())
    x = torch.tensor([0.0])

    grad_fn = torch.func.grad(fn.fn, argnums=0)
    g = grad_fn(params, x)

    # ConstantNode.forward returns self.value regardless of x, so d(value)/d(value) = 1
    assert isinstance(g, dict)
    assert "value" in g
    assert torch.allclose(g["value"], torch.tensor(1.0), atol=1e-6)


# ---------------------------------------------------------------------------
# Buffers are baked in (structural correctness)
# ---------------------------------------------------------------------------

def test_buffers_baked_into_closure():
    """
    signal_indices (a buffer) must be correctly baked in at compile-time so
    the result matches the direct module call without passing buffers explicitly.
    """
    node = make_signal_factor()
    fn = compile_graph(node, batched=False, use_torch_compile=False)

    x = torch.tensor([10.0, 20.0])
    assert torch.allclose(fn(x), node(x), atol=1e-6)

