import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Union, Optional
from . import functional as F
from .base import ZGraphNode
from .leaf_nodes import ConstantNode

class TropicalPolynomialNode(ZGraphNode):
    """
    Tropical Polynomial.
    Computes a full tropical polynomial by performing a dense linear transformation
    (Tropical Products) followed by a logsumexp collapse (Tropical Addition).
    """
    _MIN_BETA: float = eqx.field(static=True, default=1e-4)
    
    M: jax.Array
    beta: eqx.Module
    subgraphs: List[eqx.Module]

    def __init__(self, 
                 M_matrix: Optional[Union[jax.Array, List[List[float]]]] = None, 
                 subgraph_list: Optional[List[eqx.Module]] = None, 
                 beta: Optional[Union[eqx.Module, float, int, jax.Array]] = None):
        if subgraph_list is None:
            subgraph_list = []
            
        if M_matrix is None:
            M_matrix_tensor = jnp.eye(len(subgraph_list), dtype=jnp.float32)
        elif isinstance(M_matrix, list):
            M_matrix_tensor = jnp.array(M_matrix, dtype=jnp.float32)
        else:
            M_matrix_tensor = jnp.array(M_matrix, dtype=jnp.float32)
        
        if M_matrix_tensor.ndim == 1:
            M_matrix_tensor = jnp.expand_dims(M_matrix_tensor, 0)
        
        if M_matrix_tensor.ndim != 2:
            raise ValueError(f"M_matrix must be a 2D tensor, got {M_matrix_tensor.ndim}D.")
            
        num_clusters = M_matrix_tensor.shape[1]
        if num_clusters != len(subgraph_list):
            raise ValueError(
                f"Dimension mismatch: M_matrix expects {num_clusters} clusters (columns), "
                f"but received {len(subgraph_list)} subgraphs."
            )

        self.M = M_matrix_tensor
        
        if beta is None:
            self.beta = ConstantNode(1.0)
        elif isinstance(beta, (int, float, jax.Array)):
            self.beta = ConstantNode(beta)
        elif isinstance(beta, eqx.Module):
            self.beta = beta
        else:
            raise TypeError("beta must be an eqx.Module, a numeric value (int/float), or a scalar jax.Array.")
            
        self.subgraphs = list(subgraph_list)

    def logits(self, signals: jax.Array) -> jax.Array:
        w = jnp.stack([subgraph(signals) for subgraph in self.subgraphs], axis=-1)
        return jnp.matmul(self.M, w)
    
    def probabilities(self, signals: jax.Array) -> jax.Array:
        energy_landscape = self.logits(signals)
        beta_val = jnp.maximum(self.beta(signals), self._MIN_BETA)
        return jax.nn.softmax(energy_landscape / beta_val, axis=-1)

    def evaluate(self, local_signals: jax.Array) -> jax.Array:
        energy_landscape = self.logits(local_signals)
        beta_val = jnp.maximum(self.beta(local_signals), self._MIN_BETA)
        return F.marginalize(energy_landscape, beta_val)





class TropicalMatMulNode(ZGraphNode):
    """
    Tropical Matrix Multiplication (Min-Sum / Soft-Min-Sum).
    C_{ij} = ⨁_k (A_{ik} ⊗ B_{kj}) => logsumexp_k(A_{ik} + B_{kj})
    Acts as a routing or message-passing layer between distinct sets of microstates.
    """
    _MIN_BETA: float = eqx.field(static=True, default=1e-4)
    node_A: eqx.Module
    node_B: eqx.Module
    beta: eqx.Module

    def __init__(self, node_A: eqx.Module, node_B: eqx.Module, 
                 beta: Optional[Union[eqx.Module, float, int, jax.Array]] = None):
        if not isinstance(node_A, eqx.Module) or not isinstance(node_B, eqx.Module):
            raise TypeError("Both node_A and node_B must be eqx.Module instances.")
        self.node_A = node_A
        self.node_B = node_B
        
        if beta is None:
            self.beta = ConstantNode(1.0)
        elif isinstance(beta, (int, float, jax.Array)):
            self.beta = ConstantNode(beta)
        elif isinstance(beta, eqx.Module):
            self.beta = beta
        else:
            raise TypeError("beta must be an eqx.Module, a numeric value (int/float), or a scalar jax.Array.")

    def evaluate(self, local_signals: jax.Array) -> jax.Array:
        A = self.node_A(local_signals)
        B = self.node_B(local_signals)
        
        A_expanded = jnp.expand_dims(A, axis=-1)
        B_expanded = jnp.expand_dims(B, axis=-3)
        
        summed = A_expanded + B_expanded
        
        # Move K to last axis for marginalize
        summed_k_last = jnp.moveaxis(summed, -2, -1)
        
        beta_val = jnp.maximum(self.beta(local_signals), self._MIN_BETA)
        return F.marginalize(summed_k_last, beta_val)


