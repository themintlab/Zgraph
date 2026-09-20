import jax
import jax.numpy as jnp
from jax import vmap, jacrev
from zgraph.solvers import extract_decision_boundary
from zgraph.core.operation_nodes import FactorNode
from typing import Tuple

class PhaseDiagramCompiler:
    """
    Wraps a ZGraph system root node (FactorNode) to extract physical phase 
    boundaries (level-set zeros) and tie-lines using the uncollapsed logits.
    
    This operates strictly on the latent PGM distribution to extract exact 
    compositions without SoftMin smoothing.
    """
    def __init__(self, system_node: FactorNode):
        # thermograph takes the pure zgraph engine as an engine block
        self.system = system_node

    def extract_boundaries(self, sweep_signals: jax.Array, mu_index: int = 1) -> jax.Array:
        """
        Uses zgraph's universal boundary solver to find physical phase transitions.
        
        Args:
            sweep_signals: A batched tensor of input signals [..., Sweep_Steps, N_Signals].
                           The decision boundary is searched along the Sweep_Steps dimension.
            mu_index: The index of the chemical potential in the signal tensor.
            
        Returns:
            jax.Array: The tie-line compositions [..., 2] at equilibrium.
        """
        # Save original shape and flatten all batch dimensions to a single vector for vmap
        orig_shape = sweep_signals.shape
        flat_signals = sweep_signals.reshape(-1, orig_shape[-1])
        
        # 1. Domain creates the batched callable
        def batched_logits_fn(x_nd):
            x_flat = x_nd.reshape(-1, orig_shape[-1])
            logits_flat = vmap(self.system.logits)(x_flat)
            return logits_flat.reshape(*orig_shape[:-1], -1)

        # 2. Call the universal math solver to find the intersection sub-domain
        # The solver expects the N-dim signals and searches along the last batch dimension
        boundary_idx, active_phases = extract_decision_boundary(batched_logits_fn, sweep_signals)
        
        # 3. Domain applies physical meaning (compositions) via jacrev at the intersection
        # Extract the exact signal vectors where the crossover occurred
        # We use take_along_axis to select the correct signal vectors along the Sweep_Steps dimension
        boundary_idx_expanded = jnp.broadcast_to(
            jnp.expand_dims(boundary_idx, (-1, -2)), 
            (*boundary_idx.shape, 1, orig_shape[-1])
        )
        eq_signals = jnp.take_along_axis(sweep_signals, boundary_idx_expanded, axis=-2).squeeze(-2)
        
        def compute_branch_x(signals):
            return jacrev(self.system.logits)(signals)[:, mu_index]
            
        # We flatten eq_signals for vmap, then reshape
        eq_signals_flat = eq_signals.reshape(-1, orig_shape[-1])
        batched_x_flat = vmap(compute_branch_x)(eq_signals_flat)
        batched_x = batched_x_flat.reshape(*eq_signals.shape[:-1], -1)
        
        # Return tie-lines for the specific active phases
        # batched_x has shape [..., N_Phases], active_phases has shape [..., 2]
        active_phases_expanded = jnp.expand_dims(active_phases, -2) # Just for take_along_axis, shape [..., 1, 2]
        
        # Actually take_along_axis on the last dimension is simpler:
        return jnp.take_along_axis(batched_x, active_phases, axis=-1)
