import jax
import jax.numpy as jnp
import equinox as eqx
from jax import value_and_grad
import zgraph.transforms.functional as F
from typing import List, Optional, Tuple, Any, Union, Iterable, Dict, TypeVar

# T is a TypeVar for the apply_transform recursive container generic
T = TypeVar('T')

class LegendreTransform(eqx.Module):
    """
    Transforms a base thermodynamic module via a Legendre transform on specified indices.
    """
    base_model: eqx.Module
    idx: jax.Array

    def __init__(self, base_model: eqx.Module, transform_indices: List[int]):
        self.base_model = base_model
        self.idx = jnp.atleast_1d(jnp.array(transform_indices, dtype=jnp.int32))

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        phi, grad = value_and_grad(self.base_model)(x)
        psi, dual_coords = F.apply_legendre(x, phi, grad, self.idx)
        return psi, dual_coords




from jax.tree_util import tree_map

def legendre_transform(modules: Any, transform_indices: List[int]) -> Any:
    """Maps LegendreTransform over an arbitrary PyTree of modules."""
    return tree_map(lambda m: LegendreTransform(m, transform_indices), modules, is_leaf=lambda x: isinstance(x, eqx.Module))
