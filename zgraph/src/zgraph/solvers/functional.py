import jax
import jax.numpy as jnp
from typing import Tuple

def apply_gauge_shift(primal_x: jax.Array, raw_phi: jax.Array, 
                      shift_idx: jax.Array, target_val: jax.Array) -> jax.Array:
    """
    Pure subfunction to apply an invariant shift.
    Takes the evaluated energy (phi) and applies the exact shift to coordinates.
    
    Args:
        primal_x (jax.Array): The input coordinates.
        raw_phi (jax.Array): The evaluated energy.
        shift_idx (jax.Array): Indices to shift (must be 1D integer tensor).
        target_val (jax.Array): The target invariant value.
        
    Returns:
        jax.Array: The shifted coordinates.
    """
    if shift_idx.size == 0:
        return primal_x
        
    shift_amount = target_val - raw_phi
    shifted_x = primal_x.at[..., shift_idx].add(jnp.expand_dims(shift_amount, -1))
    
    return shifted_x

def find_crossover_indices(batched_logits: jax.Array, ranks: Tuple[int, int] = (0, 1)) -> Tuple[jax.Array, jax.Array]:
    """
    Pure subfunction to find the exact tensor indices where two sorted states crossover.
    
    Args:
        batched_logits (jax.Array): Evaluated logits across a sweep. Shape: [Sweep_Steps, N_States]
        ranks (Tuple[int, int]): Which two rank-ordered states to find the crossover for.
        
    Returns:
        Tuple[jax.Array, jax.Array]: 
            - The integer index in Sweep_Steps where the intersection occurs.
            - The actual state indices [2] of the states at the intersection.
    """
    k_needed = max(ranks) + 1
    # largest=False means we want the lowest "costs" (energies/tolls/utilities)
    # jax.lax.top_k returns the largest, so we negate batched_logits
    top_values_neg, top_indices = jax.lax.top_k(-batched_logits, k_needed)
    top_values = -top_values_neg
    
    # The boundary occurs where the difference between the two specified ranks is 0
    delta_omega = top_values[..., ranks[0]] - top_values[..., ranks[1]]
    boundary_idx = jnp.argmin(jnp.abs(delta_omega), axis=-1) # Scalar index
    
    # Extract the original state indices of the two crossing phases
    # top_indices has shape [..., Sweep_Steps, k_needed]
    # boundary_idx has shape [...]
    boundary_idx_expanded = jnp.broadcast_to(jnp.expand_dims(boundary_idx, (-1, -2)), (*boundary_idx.shape, 1, k_needed))
    selected_top_indices = jnp.take_along_axis(top_indices, boundary_idx_expanded, axis=-2).squeeze(-2)
    
    active_phases = jnp.stack([
        selected_top_indices[..., ranks[0]], 
        selected_top_indices[..., ranks[1]]
    ], axis=-1)
    
    return boundary_idx, active_phases
