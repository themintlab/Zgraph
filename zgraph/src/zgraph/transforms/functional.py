import jax
import jax.numpy as jnp
from typing import Tuple

def apply_legendre(primal_x: jax.Array, raw_phi: jax.Array, 
                   full_grad: jax.Array, lt_idx: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Pure subfunction to compute the multivariate Legendre dual.
    Takes the evaluated energy (phi) and gradients, and constructs the dual state.
    
    Args:
        primal_x (jax.Array): The input primal coordinates.
        raw_phi (jax.Array): The evaluated primal energy.
        full_grad (jax.Array): The gradient vector.
        lt_idx (jax.Array): Indices for the Legendre transform (1D integer tensor).
        
    Returns:
        Tuple[jax.Array, jax.Array]: The dual energy (psi) and the dual coordinates.
    """
    if lt_idx.size == 0:
        return raw_phi, primal_x
        
    # Use jnp.dot since batching is deferred to vmap/jit
    psi = raw_phi - jnp.dot(primal_x[lt_idx], full_grad[lt_idx])
    
    dual_x = primal_x.at[lt_idx].set(full_grad[lt_idx])
    return psi, dual_x
