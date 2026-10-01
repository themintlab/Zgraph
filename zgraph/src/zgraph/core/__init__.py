from .tropical_nodes import (
    TropicalPolynomialNode, 
    TropicalPowerNode,
    TropicalMatMulNode,
    TropicalZeroNode,
    TropicalIdentityNode
)
from .utility_nodes import PiecewiseNode
from .leaf_nodes import ConstantNode, SignalNode, SignalNodes

import jax.numpy as jnp

# Convenience Aliases for users wanting standard probability/math naming
FactorNode = TropicalPolynomialNode
ProductNode = TropicalPowerNode
MatMulNode = TropicalMatMulNode

# Backwards compatibility wrappers
def AdditionNode(subgraph_list, weights=None):
    if weights is None:
        weights = jnp.ones(len(subgraph_list), dtype=jnp.float32)
    else:
        weights = jnp.array(weights, dtype=jnp.float32)
    M = jnp.expand_dims(weights, 0)
    return TropicalPolynomialNode(M_matrix=M, subgraph_list=subgraph_list)

def DivisionNode(numerator, denominator):
    return TropicalPolynomialNode(M_matrix=jnp.array([[1.0, -1.0]], dtype=jnp.float32), subgraph_list=[numerator, denominator])

__all__ = [
    "FactorNode",
    "ProductNode",
    "AdditionNode",
    "DivisionNode",
    "MatMulNode",
    "TropicalPolynomialNode",
    "TropicalPowerNode",
    "TropicalMatMulNode",
    "TropicalZeroNode",
    "TropicalIdentityNode",
    "PiecewiseNode",
    "ConstantNode",
    "SignalNode",
    "SignalNodes",
]
