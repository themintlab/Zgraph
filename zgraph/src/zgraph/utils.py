import numpy as np
from jax.tree_util import tree_map

def to_numpy(pytree):
    """Strips all JAX tracking from a PyTree, returning clean numpy arrays."""
    return tree_map(lambda x: np.asarray(x).squeeze() if hasattr(x, "shape") else x, pytree)

__all__ = ["to_numpy"]
