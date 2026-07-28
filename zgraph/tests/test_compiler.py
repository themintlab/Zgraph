import torch
import pytest

from zgraph import ConstantNode, FactorNode, SignalNode, SignalNodes
from zgraph.transforms import finalize, FinalizedGraph


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
# FinalizedGraph return type
# ---------------------------------------------------------------------------

def test_finalize_returns_finalized_graph():
    node = make_identity_factor()
    fg = finalize(node, compile_graph=False)
    assert isinstance(fg, FinalizedGraph)


def test_finalize_list_returns_list_of_finalized_graphs():
    nodes = [make_identity_factor(), ConstantNode(3.0)]
    result = finalize(nodes, compile_graph=False)
    assert isinstance(result, list)
    assert all(isinstance(f, FinalizedGraph) for f in result)


def test_finalize_tuple_returns_tuple_of_finalized_graphs():
    nodes = (make_identity_factor(),)
    result = finalize(nodes, compile_graph=False)
    assert isinstance(result, tuple)


def test_finalize_dict_returns_dict_of_finalized_graphs():
    nodes = {"a": make_identity_factor(), "b": ConstantNode(0.5)}
    result = finalize(nodes, compile_graph=False)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"a", "b"}


# ---------------------------------------------------------------------------
# Unbatched correctness
# ---------------------------------------------------------------------------

def test_unbatched_constant_factor_matches_expected():
    """Single-sample forward should match direct module call."""
    node = make_identity_factor(beta=1.0)
    fg = finalize(node, batched=False, compile_graph=False)

    x = torch.tensor([0.0])
    result = fg(x)
    expected = node(x)

    assert torch.allclose(result, expected, atol=1e-6)


def test_unbatched_signal_factor_matches_expected():
    """Signals extracted from state tensor should work in unbatched mode."""
    node = make_signal_factor()
    fg = finalize(node, batched=False, compile_graph=False)

    x = torch.tensor([1.0, 2.0])
    result = fg(x)
    expected = node(x)

    assert torch.allclose(result, expected, atol=1e-6)


def test_unbatched_constant_node_matches_expected():
    node = ConstantNode(5.0)
    fg = finalize(node, batched=False, compile_graph=False)
    x = torch.tensor([0.0])
    assert torch.allclose(fg(x), node(x), atol=1e-6)


# ---------------------------------------------------------------------------
# Batched correctness
# ---------------------------------------------------------------------------

def test_batched_output_shape():
    node = make_signal_factor()
    fg = finalize(node, batched=True, compile_graph=False)

    batch = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = fg(batch)

    assert result.shape == (3,)


def test_batched_results_match_unbatched():
    """Each row of a batched call must equal the corresponding unbatched call."""
    node = make_signal_factor()
    fg_batched = finalize(node, batched=True, compile_graph=False)
    fg_single = finalize(node, batched=False, compile_graph=False)

    xs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.5, 0.5]])
    batched_out = fg_batched(xs)

    for i, x in enumerate(xs):
        single_out = fg_single(x)
        assert torch.allclose(batched_out[i], single_out, atol=1e-6), \
            f"Mismatch at row {i}: batched={batched_out[i]}, single={single_out}"


def test_batched_constant_factor():
    """ConstantNode-only graph: all batch outputs should equal the constant."""
    node = ConstantNode(7.0)
    fg = finalize(node, batched=True, compile_graph=False)

    # ConstantNode ignores its input, so we need in_dims to handle a dummy batch.
    # The node's forward ignores x; provide a (N, 1) batch.
    batch = torch.zeros(4, 1)
    result = fg(batch)

    assert result.shape == (4,)
    assert torch.allclose(result, torch.full((4,), 7.0), atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient flow through live parameters
# ---------------------------------------------------------------------------

def test_gradient_flows_through_params_unbatched():
    node = ConstantNode(3.0)
    fg = finalize(node, batched=False, compile_graph=False)

    x = torch.tensor([0.0])
    out = fg(x)
    out.backward()

    assert node.value.grad is not None


def test_gradient_flows_through_params_batched():
    """Loss over a batch should propagate gradients back to node parameters."""
    node = make_identity_factor(beta=1.0)
    fg = finalize(node, batched=True, compile_graph=False)

    batch = torch.zeros(5, 1)
    loss = fg(batch).sum()
    loss.backward()

    # beta is a ConstantNode whose .value is a Parameter
    beta_param = node.beta.value
    assert beta_param.grad is not None


def test_fn_attribute_works_with_torch_func_grad():
    """fg.fn can be composed with torch.func.grad for parameter gradients."""
    node = ConstantNode(2.0)
    fg = finalize(node, batched=False, compile_graph=False)

    params = dict(node.named_parameters())
    x = torch.tensor([0.0])

    grad_fn = torch.func.grad(fg.fn, argnums=0)
    g = grad_fn(params, x)

    # ConstantNode.forward returns self.value regardless of x, so d(value)/d(value) = 1
    assert isinstance(g, dict)
    assert "value" in g
    assert torch.allclose(g["value"], torch.tensor(1.0), atol=1e-6)


# ---------------------------------------------------------------------------
# Buffers are baked in (structural correctness)
# ---------------------------------------------------------------------------

def test_buffers_do_not_appear_in_fn_signature():
    """
    Verify that signal_indices (a buffer) is correctly baked in and does not
    need to be passed explicitly — the result matches the direct module call.
    """
    node = make_signal_factor()
    fg = finalize(node, batched=False, compile_graph=False)

    x = torch.tensor([10.0, 20.0])
    assert torch.allclose(fg(x), node(x), atol=1e-6)


# ---------------------------------------------------------------------------
# Module proxy
# ---------------------------------------------------------------------------

def test_module_property_returns_original_module():
    node = make_identity_factor()
    fg = finalize(node, compile_graph=False)
    assert fg.module is node


def test_parameters_proxy_yields_same_as_module():
    node = make_identity_factor()
    fg = finalize(node, compile_graph=False)
    fg_params = list(fg.parameters())
    mod_params = list(node.parameters())
    assert len(fg_params) == len(mod_params)
    for fp, mp in zip(fg_params, mod_params):
        assert fp is mp
