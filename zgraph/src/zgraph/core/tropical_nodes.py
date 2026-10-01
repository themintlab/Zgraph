import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Union, Optional
from . import functional as F
from .base import ZGraphNode
from .leaf_nodes import ConstantNode

class TropicalAdditionNode(ZGraphNode):
    """
    Tropical Addition (⊕) / Soft Minimum.
    Computes the logsumexp (soft-max/min) over parallel microstates to collapse them
    into a partition function. In the min-plus tropical semiring over energy space, 
    this is the addition operation.
    """
    _MIN_BETA: float = eqx.field(static=True, default=1e-4)
    
    M: jax.Array
    beta: eqx.Module
    subgraphs: List[eqx.Module]

    def __init__(self, 
                 M_matrix: Union[jax.Array, List[List[float]]], 
                 subgraph_list: List[eqx.Module], 
                 beta: Optional[Union[eqx.Module, float, int, jax.Array]] = None):
        if isinstance(M_matrix, list):
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
        

class TropicalProductNode(ZGraphNode):
    """
    Tropical Product (⊗) / Standard Addition.
    Computes a weighted sum of independent subgraphs: sum(w_i * subgraph_i(signals)).
    In energy space, adding energy terms is equivalent to multiplying their underlying 
    probabilities, making this the min-plus tropical product operation.
    """
    weights: jax.Array
    subgraphs: List[eqx.Module]
    
    def __init__(self, subgraph_list: List[eqx.Module], weights: Optional[Union[jax.Array, List[float]]] = None):
        if len(subgraph_list) == 0:
            raise ValueError("subgraph_list must contain at least one subgraph.")
        for subgraph in subgraph_list:
            if not isinstance(subgraph, eqx.Module):
                raise TypeError("Each entry in subgraph_list must be an eqx.Module.")
        self.subgraphs = list(subgraph_list)
        if weights is None:
            self.weights = jnp.ones(len(subgraph_list), dtype=jnp.float32)
        else:
            weights_arr = jnp.array(weights, dtype=jnp.float32)
            if weights_arr.shape != (len(subgraph_list),):
                raise ValueError("weights must be a 1D array matching the number of subgraphs.")
            self.weights = weights_arr

    def evaluate(self, local_signals: jax.Array) -> jax.Array:
        values = jnp.stack([subgraph(local_signals) for subgraph in self.subgraphs], axis=0)
        return jnp.tensordot(self.weights, values, axes=1)


class TropicalPowerNode(ZGraphNode):
    """
    Tropical Power / Standard Multiplication.
    Multiplies a list of subgraph outputs elementwise. In energy space, scalar 
    multiplication corresponds to exponentiating the underlying probability.
    """
    subgraphs: List[eqx.Module]
    
    def __init__(self, subgraph_list: List[eqx.Module]):
        if len(subgraph_list) == 0:
            raise ValueError("subgraph_list must contain at least one subgraph.")
        for subgraph in subgraph_list:
            if not isinstance(subgraph, eqx.Module):
                raise TypeError("Each entry in subgraph_list must be an eqx.Module.")
        self.subgraphs = list(subgraph_list)

    def evaluate(self, local_signals: jax.Array) -> jax.Array:
        values = jnp.stack([subgraph(local_signals) for subgraph in self.subgraphs], axis=0)
        return jnp.prod(values, axis=0)
