import torch
from torch.func import grad_and_value
from zgraph.transforms.base import GraphTransform


def legendre_transform(module, idx):
    """
    Returns a Module (or container of Modules) evaluating partial Legendre transform(s).

    Args:
        module: A nn.Module (or list/tuple of nn.Modules) taking a 1D tensor and returning a scalar energy.
        idx: The indices of the primal variables to be transformed.

    Returns:
        LegendreTransformModule (or list/tuple of LegendreTransformModule) taking primal variables and returning
        the transformed energy and the updated state variables.
    """
    return GraphTransform.map_factory(module, lambda m: LegendreTransformModule(m, idx))

class LegendreTransformModule(GraphTransform):
    """
    Transforms a base thermodynamic module via a Legendre transform on specified indices.
    Designed strictly for a single unbatched sample (1D tensor).
    """
    def __init__(self, base_model: GraphTransform, transform_indices: list):
        super().__init__()
        # 1. Store the base engine. 
        # This automatically exposes base_model's parameters to optimizers.
        self.base_model = base_model
        
        # 2. Store indices as a registered buffer.
        # This ensures the indices automatically move to the GPU if the model is moved,
        # preventing device mismatch errors during compiled execution.
        self.register_buffer(
            'idx_tensor',
            torch.atleast_1d(torch.as_tensor(transform_indices, dtype=torch.long))
        )

    def _compute_transform(self, primal_x: torch.Tensor):
        # primal_x must be a 1D tensor (e.g., shape [3] for [T, P, mu])

        # Evaluate potential and coordinates from the incoming state in one pass.
        def potential_with_coords(x: torch.Tensor):
            phi, coords = self.base_model(x)
            return phi, coords

        # 3. Compute gradient and carry forward transformed coordinates.
        full_grad, (phi_in, coords_in) = grad_and_value(
            potential_with_coords,
            has_aux=True,
        )(primal_x)
        
        # 4. Execute the Legendre Math
        # psi = phi - SUM(x_i * y_i)
        idx = self.get_buffer("idx_tensor")
        x_I = coords_in[idx]
        y_I = full_grad[idx]
        psi = phi_in - torch.dot(x_I, y_I)
        
        # 5. Construct and return the dual coordinate vector alongside the energy
        coords_out = coords_in.clone()
        coords_out[idx] = y_I
        
        return psi, coords_out