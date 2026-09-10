import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Union, Optional
from . import functional as F
from .leaf_nodes import ConstantNode

class FactorNode(eqx.Module):
    # A minimum allowed beta to prevent division by zero in logsumexp
    _MIN_BETA: float = eqx.field(static=True, default=1e-4)
    
    M: jax.Array
    beta: eqx.Module
    subgraphs: List[eqx.Module]

    def __init__(self, 
                 M_matrix: Union[jax.Array, List[List[float]]], 
                 subgraph_list: List[eqx.Module], 
                 beta: Optional[Union[eqx.Module, float, int, jax.Array]] = None):
        """
        Args:
            M_matrix (Union[jax.Array, List[List[float]]]): 2D matrix of shape (num_microstates, num_clusters).
            subgraph_list (list[eqx.Module]): A list of subgraph modules. The order
                of modules in this list MUST match the order of the cluster
                columns in the M_matrix.
            beta (Optional[Union[eqx.Module, float, int, jax.Array]]): A module that extracts or provides the 
                rationality/temperature parameter (e.g. SignalNode or ConstantNode).
                Defaults to ConstantNode(1.0).
        """

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
        """
        The Logits / Uncollapsed Energy Vector.
        Evaluates the subgraphs to build the cluster inputs (w), and maps them to microstates via M.
        Returns a vector of size (num_microstates).
        """
        w = jnp.stack([subgraph(signals) for subgraph in self.subgraphs], axis=-1)
        return jnp.matmul(self.M, w)
    
    def probabilities(self, signals: jax.Array) -> jax.Array:
        """
        The Local Marginal Probabilities (SoftMin weights).
        Returns a normalized vector of size (num_microstates) representing the probability/weight of each state.
        """
        energy_landscape = self.logits(signals)
        beta_val = jnp.maximum(self.beta(signals), self._MIN_BETA)
        
        # Softmax applies the exact exponential weighting used in the partition function
        return jax.nn.softmax(energy_landscape / beta_val, axis=-1)

    def __call__(self, local_signals: jax.Array) -> jax.Array:
        """
        The Strict Axiom: The Partition Function Collapse.
        Returns Rank 0 Tensor (Scalar).
        """
        energy_landscape = self.logits(local_signals)
        beta_val = jnp.maximum(self.beta(local_signals), self._MIN_BETA)
        return F.marginalize(energy_landscape, beta_val)
        

class ProductNode(eqx.Module):
    """Multiplies a list of subgraph outputs elementwise (tropical power)."""
    subgraphs: List[eqx.Module]
    
    def __init__(self, subgraph_list: List[eqx.Module]):
        if len(subgraph_list) == 0:
            raise ValueError("subgraph_list must contain at least one subgraph.")
        for subgraph in subgraph_list:
            if not isinstance(subgraph, eqx.Module):
                raise TypeError("Each entry in subgraph_list must be an eqx.Module.")
        self.subgraphs = list(subgraph_list)

    def __call__(self, local_signals: jax.Array) -> jax.Array:
        values = jnp.stack([subgraph(local_signals) for subgraph in self.subgraphs], axis=0)
        return jnp.prod(values, axis=0)
