import torch
import pytest

from zgraph import ConstantNode, FactorNode, ProductNode


def test_factor_node_matches_expected_logsumexp():
    node = FactorNode(
        M_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        subgraph_list=[ConstantNode(1.0), ConstantNode(2.0)],
        beta=1.0,
    )

    result = node(torch.tensor([0.0]))
    expected = torch.logsumexp(torch.tensor([1.0, 2.0]), dim=-1)

    assert torch.allclose(result, expected, atol=1e-6)


def test_factor_node_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="Dimension mismatch"):
        FactorNode(
            M_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            subgraph_list=[ConstantNode(1.0)],
        )


def test_factor_node_rejects_non_2d_matrix_after_normalization():
    with pytest.raises(ValueError, match="must be a 2D tensor"):
        FactorNode(
            M_matrix=torch.ones(2, 2, 2),
            subgraph_list=[ConstantNode(1.0), ConstantNode(2.0)],
        )


def test_factor_node_accepts_1d_matrix_by_unsqueezing():
    node = FactorNode(
        M_matrix=torch.tensor([1.0, 1.0]),
        subgraph_list=[ConstantNode(1.5), ConstantNode(0.5)],
        beta=1.0,
    )

    result = node(torch.tensor([0.0]))
    expected = torch.tensor(2.0)

    assert torch.allclose(result, expected, atol=1e-6)


def test_factor_node_rejects_invalid_beta_type():
    with pytest.raises(TypeError, match="beta must be"):
        FactorNode(
            M_matrix=torch.tensor([[1.0]]),
            subgraph_list=[ConstantNode(1.0)],
            beta="invalid",
        )


def test_factor_node_clamps_zero_beta_to_minimum():
    node = FactorNode(
        M_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        subgraph_list=[ConstantNode(0.0), ConstantNode(0.0)],
        beta=0.0,
    )

    result = node(torch.tensor([0.0]))
    expected = FactorNode._MIN_BETA * torch.log(torch.tensor(2.0))

    assert torch.isfinite(result)
    assert torch.allclose(result, expected, atol=1e-10)


def test_product_node_multiplies_subgraph_outputs():
    node = ProductNode([ConstantNode(2.0), ConstantNode(3.0)])
    result = node(torch.tensor([0.0]))
    assert torch.allclose(result, torch.tensor(6.0))


def test_product_node_rejects_empty_subgraphs():
    with pytest.raises(ValueError, match="at least one subgraph"):
        ProductNode([])
