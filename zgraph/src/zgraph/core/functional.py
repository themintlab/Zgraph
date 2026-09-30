import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from typing import Tuple, Union

def marginalize(microstates: jax.Array, beta: Union[float, jax.Array] = 1.0) -> jax.Array:
    """
    The stateless mathematical core of the ZGraph engine.
    Executes the SoftMin collapse (Partition Function).
    
    Args:
        energy_landscape (jax.Array): The dynamic Energy Vector across microstates.
                                         Shape: (..., Num_Microstates)
        beta (Union[float, jax.Array]): The thermodynamic smoothing parameter (-kT).
                                           beta -> 0 triggers hardmax (T->0 limit).
                                           Shape: () (Scalar)
                                 
    Returns:
        jax.Array: The renormalized scalar Free Energy. Shape: (...,)
    """
    # The 1-state bypass is optimal for JAX (resolved at JIT-compile time, zero GPU overhead)
    # and safely prevents inf * 0 = NaN cases when beta=inf.
    if microstates.shape[-1] == 1:
        return jnp.squeeze(microstates, axis=-1)
        
    # Explicit logsumexp formulation for numerical stability.
    # By subtracting the max BEFORE dividing by beta, we prevent float32 overflows
    # that occur when beta is very small and energy is large.
    energy_max = jnp.max(microstates, axis=-1, keepdims=True)
    
    # Safely handle beta=0 (hardmax limit) to prevent division by exactly zero.
    # The formula naturally evaluates to exactly energy_max.
    safe_beta = jnp.where(beta == 0.0, 1e-10, beta)
    
    centered_energy = (microstates - energy_max) / safe_beta
    sum_exp = jnp.sum(jnp.exp(centered_energy), axis=-1, keepdims=True)
    
    res = energy_max + beta * jnp.log(sum_exp)
    
    return jnp.squeeze(res, axis=-1)