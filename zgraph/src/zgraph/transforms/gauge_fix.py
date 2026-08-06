import torch
import torch.nn as nn


def gauge_fix(module, idx):
    """
    Returns a Module (or container of Modules) applying gauge-fix projection(s).

    Args:
        module: A nn.Module (or list/tuple of nn.Modules) taking a 1D tensor and returning
            either a scalar energy tensor or a tuple whose first element is that scalar energy.
        idx: The indices of primal variables that will receive the uniform shift.

    Returns:
        GaugeFixModule (or list/tuple of GaugeFixModule) returning
        (target_value, projected_coordinates).
    """
    if isinstance(module, (list, tuple)):
        modules = [gauge_fix(m, idx) for m in module]
        return type(module)(modules)

    if isinstance(module, nn.Module):
        return GaugeFixModule(module, idx)

    raise TypeError(
        "gauge_fix expects an nn.Module or a list/tuple of nn.Module instances."
    )

class GaugeFixModule(nn.Module):
    """
    Analytically projects the state onto a target manifold (default 0.0) 
    by uniformly shifting a list of invariant indices.
    
    Returns:
        tuple: (target_val, exact_coordinates) to maintain API consistency 
               with other thermodynamic graph modifiers.
    """
    def __init__(self, base_model: nn.Module, shift_indices: list):
        super().__init__()
        self.base_model = base_model
        self.register_buffer(
            'idx_tensor', 
            torch.atleast_1d(torch.as_tensor(shift_indices, dtype=torch.long))
        )

    def forward(self, primal_x: torch.Tensor, target_val: float = 0.0):
        # 1. Evaluate the base model
        raw_output = self.base_model(primal_x)
        
        # Handle tuple inputs if stacked directly after LegendreTransform
        if isinstance(raw_output, tuple):
            raw_energy = raw_output[0]
        else:
            raw_energy = raw_output
            
        # 2. Calculate the universal shift
        shift_amount = target_val - raw_energy
        
        # 3. Clone to preserve functional purity and Autograd history
        idx_dev = self.get_buffer("idx_tensor").to(device=primal_x.device)
        exact_x = primal_x.clone()
        exact_x[idx_dev] += shift_amount
        
        # 4. Return consistent tuple: (Scalar Value, Coordinate Tensor)
        # We ensure target_val is a tensor so it plays nicely with PyTorch ops
        target_tensor = torch.as_tensor(target_val, dtype=primal_x.dtype, device=primal_x.device)
        
        return target_tensor, exact_x