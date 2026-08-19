import torch
from typing import Callable, List
import zgraph.solvers.functional as F

def gauge_fix(compiled_model_fn: Callable[[torch.Tensor], torch.Tensor], 
              sweep_signals: torch.Tensor, 
              shift_indices: List[int], 
              target_val: float = 0.0) -> torch.Tensor:
    """
    Project the input state onto a target equilibrium manifold (default 0.0) 
    by analytically finding a uniform shift in the specified indices.
    
    Args:
        compiled_model_fn: A compiled graph or forward function.
        sweep_signals: Batched tensor of input coordinates.
        shift_indices: List of indices to shift.
        target_val: The target manifold value (default: 0.0).
        
    Returns:
        torch.Tensor: The shifted coordinates.
    """
    batched_phi = compiled_model_fn(sweep_signals)
    target_tensor = torch.as_tensor(target_val, dtype=batched_phi.dtype, device=batched_phi.device).squeeze()
    shift_idx = torch.atleast_1d(torch.as_tensor(shift_indices, dtype=torch.long, device=sweep_signals.device))
    
    # Delegate pure tensor math to the core layer
    return F.apply_gauge_shift(sweep_signals, batched_phi, shift_idx, target_tensor)
