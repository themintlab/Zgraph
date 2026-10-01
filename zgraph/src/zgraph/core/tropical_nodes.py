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

class TropicalDivisionNode(ZGraphNode):
    """
    Tropical Division (⊘) / Standard Subtraction.
    Subtracts the output of the denominator subgraph from the numerator subgraph.
    In energy space, standard subtraction equates to tropical division (dividing probabilities).
    Essential for calculating relative free energies, defect formation energies, 
    or isolating excess components.
    """
    numerator: eqx.Module
    denominator: eqx.Module
    
    def __init__(self, numerator: eqx.Module, denominator: eqx.Module):
        if not isinstance(numerator, eqx.Module) or not isinstance(denominator, eqx.Module):
            raise TypeError("Both numerator and denominator must be eqx.Module instances.")
        self.numerator = numerator
        self.denominator = denominator

    def evaluate(self, local_signals: jax.Array) -> jax.Array:
        return self.numerator(local_signals) - self.denominator(local_signals)

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

class TropicalZeroNode(ZGraphNode):
    """
    Tropical Zero Element (for ⊕).
    Outputs a sufficiently high energy barrier (representing +inf).
    Useful for dynamically masking out forbidden microstates, unreachable phases, 
    or infinite potential walls without breaking the gradient graph.
    """
    barrier_value: jax.Array
    
    def __init__(self, barrier_value: Union[float, int, jax.Array] = 1e9):
        if isinstance(barrier_value, jax.Array):
            self.barrier_value = barrier_value.astype(jnp.float32)
        else:
            self.barrier_value = jnp.array(barrier_value, dtype=jnp.float32)

    def evaluate(self, local_signals: jax.Array) -> jax.Array:
        return self.barrier_value

class TropicalIdentityNode(ZGraphNode):
    """
    Tropical Identity Element (for ⊗).
    Outputs 0.0.
    """
    def evaluate(self, local_signals: jax.Array) -> jax.Array:
        return jnp.array(0.0, dtype=jnp.float32)
