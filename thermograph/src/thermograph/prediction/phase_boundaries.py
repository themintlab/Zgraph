import jax
import jax.numpy as jnp
import equinox as eqx
from jax import vmap, jacrev
from zgraph.solvers import extract_decision_boundary
from zgraph.core.operation_nodes import FactorNode

class PhaseBoundaryPredictor(eqx.Module):
    """
    Predicts physical phase boundaries (compositions) by finding the thermodynamic 
    equilibrium manifold in the latent PGM distribution.
    
    By inheriting from `eqx.Module`, this predictor is a pure JAX PyTree.
    This allows it to be seamlessly vmapped over a posterior trace of parameters 
    for Uncertainty Quantification (UQ).
    """
    system: FactorNode

    def __init__(self, system_node: FactorNode):
        self.system = system_node

    def predict_compositions(self, sweep_signals: jax.Array, mu_index: int = 1) -> jax.Array:
        """
        Extracts tie-lines by finding crossovers in the uncollapsed logits and 
        applying the physical Legendre transform via the Jacobian.
        
        Args:
            sweep_signals: A batched tensor of input signals [..., Sweep_Steps, N_Signals].
                           The decision boundary is searched along the Sweep_Steps dimension.
            mu_index: The index of the chemical potential in the signal tensor.
            
        Returns:
            jax.Array: The tie-line compositions [..., 2] at equilibrium.
        """
        orig_shape = sweep_signals.shape
        
        def batched_logits_fn(x_nd):
            x_flat = x_nd.reshape(-1, orig_shape[-1])
            logits_flat = vmap(self.system.logits)(x_flat)
            return logits_flat.reshape(*orig_shape[:-1], -1)

        # zgraph universal solver finds where the energy levels cross
        boundary_idx, active_phases = extract_decision_boundary(batched_logits_fn, sweep_signals)
        
        # Gather the exact signal vectors at the crossover
        boundary_idx_expanded = jnp.broadcast_to(
            jnp.expand_dims(boundary_idx, (-1, -2)), 
            (*boundary_idx.shape, 1, orig_shape[-1])
        )
        eq_signals = jnp.take_along_axis(sweep_signals, boundary_idx_expanded, axis=-2).squeeze(-2)
        
        def compute_branch_x(signals):
            return jacrev(self.system.logits)(signals)[:, mu_index]
            
        eq_signals_flat = eq_signals.reshape(-1, orig_shape[-1])
        batched_x_flat = vmap(compute_branch_x)(eq_signals_flat)
        batched_x = batched_x_flat.reshape(*eq_signals.shape[:-1], -1)
        
        # Return tie-lines
        return jnp.take_along_axis(batched_x, active_phases, axis=-1)

