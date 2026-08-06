import torch
from zgraph.transforms.base import GraphTransform


def gauge_fix(module, idx):
    """
    Returns a Module (or container of Modules) applying gauge-fix projection(s).

    Args:
        module: A nn.Module (or list/tuple of nn.Modules) taking a 1D tensor and returning
            either a scalar energy tensor or a tuple whose first element is that scalar energy.
        idx: The indices of primal variables that will receive the uniform shift.

    Returns:
        GaugeFix (or list/tuple of GaugeFix) returning
        (target_value, projected_coordinates).
    """
    return GraphTransform.map_factory(module, lambda m: GaugeFix(m, idx))

class GaugeFix(GraphTransform):
    """
    Analytically projects the state onto a target manifold (default 0.0) 
    by uniformly shifting a list of invariant indices.
    
    Returns:
        tuple: (target_val, exact_coordinates) to maintain API consistency 
               with other thermodynamic graph modifiers.
    """
    def __init__(self, base_model: GraphTransform, shift_indices: list):
        super().__init__()
        self.base_model = base_model
        self.register_buffer(
            'idx', 
            torch.atleast_1d(torch.as_tensor(shift_indices, dtype=torch.long))
        )

    def _compute_transform(self, primal_x: torch.Tensor, target_val: float = 0.0):
        # 1. Evaluate the base model
        raw_energy, coords_in = self.base_model(primal_x)
                   
        # 2. Calculate the universal shift
        target_tensor = torch.as_tensor(target_val, dtype=primal_x.dtype, device=primal_x.device).squeeze()
        shift_amount = target_tensor - raw_energy
        
        # Shift the incoming coordinate state to preserve transform composability.
        exact_x = coords_in.clone()
        idx = self.get_buffer("idx")
        exact_x[idx] += shift_amount
        
        # 4. Return consistent tuple: (Scalar Value, Coordinate Tensor)
        return target_tensor, exact_x