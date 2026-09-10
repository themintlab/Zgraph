import jax
import jax.numpy as jnp
from typing import Callable, List
import zgraph.solvers.functional as F

def gauge_fix(compiled_model_fn: Callable[[jax.Array], jax.Array], 
              sweep_signals: jax.Array, 
              shift_indices: List[int], 
              target_val: float = 0.0) -> jax.Array:
    """
    Project the input state onto a target equilibrium manifold (default 0.0) 
    by analytically finding a uniform shift in the specified indices.
    
    Args:
        compiled_model_fn: A compiled graph or forward function.
        sweep_signals: Batched tensor of input coordinates.
        shift_indices: List of indices to shift.
        target_val: The target manifold value (default: 0.0).
        
    Returns:
        jax.Array: The shifted coordinates.
    """
    batched_phi = compiled_model_fn(sweep_signals)
    target_tensor = jnp.array(target_val, dtype=batched_phi.dtype).squeeze()
    shift_idx = jnp.atleast_1d(jnp.array(shift_indices, dtype=jnp.int32))
    
    # Delegate pure tensor math to the core layer
    return F.apply_gauge_shift(sweep_signals, batched_phi, shift_idx, target_tensor)
