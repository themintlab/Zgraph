import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from typing import Tuple, Union

def marginalize(energy_landscape: jax.Array, beta: Union[float, jax.Array] = 1.0) -> jax.Array:
    """
    The stateless mathematical core of the ZGraph engine.
    Executes the SoftMin collapse (Partition Function).
    
    Args:
        energy_landscape (jax.Array): The dynamic Energy Vector across microstates.
                                         Shape: (Num_Microstates,)
        beta (Union[float, jax.Array]): The thermodynamic smoothing parameter (-kT).
                                           Beta=inf triggers hardmax.
                                           Shape: () (Scalar)
                                 
    Returns:
        jax.Array: The renormalized scalar Free Energy. Shape: () (Scalar)
    """
    # Calculate the partition function / free energy as a scalar.
    return beta * logsumexp(energy_landscape / beta, axis=-1)