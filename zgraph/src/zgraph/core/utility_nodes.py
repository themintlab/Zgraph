import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Union
from .base import ZGraphNode

class PiecewiseNode(ZGraphNode):
    """
    Utility Node: Evaluates a single subgraph based on a signal's value and a set of transition bounds.
    Used for routing execution mathematically rather than defining structural tropical geometry.
    """
    bounds: jax.Array
    subgraphs: List[eqx.Module]
    signal_index: int = eqx.field(static=True)
    
    def __init__(self, bounds: Union[jax.Array, List[float]], subgraph_list: List[eqx.Module], signal_index: int = 0):
        bounds_arr = jnp.array(bounds, dtype=jnp.float32)
        if bounds_arr.ndim != 1:
            raise ValueError("bounds must be a 1D array of transition values.")
        if len(subgraph_list) != len(bounds_arr) + 1:
            raise ValueError("Number of subgraphs must be exactly one more than the number of bounds.")
        for subgraph in subgraph_list:
            if not isinstance(subgraph, eqx.Module):
                raise TypeError("Each entry in subgraph_list must be an eqx.Module.")
        
        self.bounds = bounds_arr
        self.subgraphs = list(subgraph_list)
        self.signal_index = signal_index

    def evaluate(self, local_signals: jax.Array) -> jax.Array:
        signal_val = local_signals[self.signal_index]
        idx = jnp.searchsorted(self.bounds, signal_val, side='right')
        branches = [lambda s, sub=sub: sub(s) for sub in self.subgraphs]
        return jax.lax.switch(idx, branches, local_signals)
