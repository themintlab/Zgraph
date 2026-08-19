import torch
from typing import Tuple

def apply_gauge_shift(primal_x: torch.Tensor, raw_phi: torch.Tensor, 
                      shift_idx: torch.Tensor, target_val: torch.Tensor) -> torch.Tensor:
    """
    Pure subfunction to apply an invariant shift.
    Takes the evaluated energy (phi) and applies the exact shift to coordinates.
    
    Args:
        primal_x (torch.Tensor): The input coordinates.
        raw_phi (torch.Tensor): The evaluated energy.
        shift_idx (torch.Tensor): Indices to shift (must be 1D integer tensor).
        target_val (torch.Tensor): The target invariant value.
        
    Returns:
        torch.Tensor: The shifted coordinates.
    """
    if shift_idx.numel() == 0:
        return primal_x
        
    shift_amount = target_val - raw_phi
    shifted_x = primal_x.clone()
    shifted_x[..., shift_idx] += shift_amount.unsqueeze(-1)
    
    return shifted_x

def find_crossover_indices(batched_logits: torch.Tensor, ranks: Tuple[int, int] = (0, 1)) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure subfunction to find the exact tensor indices where two sorted states crossover.
    
    Args:
        batched_logits (torch.Tensor): Evaluated logits across a sweep. Shape: [Sweep_Steps, N_States]
        ranks (Tuple[int, int]): Which two rank-ordered states to find the crossover for.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - The integer index in Sweep_Steps where the intersection occurs.
            - The actual state indices [2] of the states at the intersection.
    """
    k_needed = max(ranks) + 1
    # largest=False means we want the lowest "costs" (energies/tolls/utilities)
    top_values, top_indices = torch.topk(batched_logits, k=k_needed, dim=-1, largest=False)
    
    # The boundary occurs where the difference between the two specified ranks is 0
    delta_omega = top_values[..., ranks[0]] - top_values[..., ranks[1]]
    boundary_idx = torch.argmin(torch.abs(delta_omega), dim=-1) # Scalar index
    
    # Extract the original state indices of the two crossing phases
    # top_indices has shape [..., Sweep_Steps, k_needed]
    # boundary_idx has shape [...]
    boundary_idx_expanded = boundary_idx.unsqueeze(-1).unsqueeze(-1).expand(*boundary_idx.shape, 1, k_needed)
    selected_top_indices = torch.gather(top_indices, dim=-2, index=boundary_idx_expanded).squeeze(-2)
    
    active_phases = torch.stack([
        selected_top_indices[..., ranks[0]], 
        selected_top_indices[..., ranks[1]]
    ], dim=-1)
    
    return boundary_idx, active_phases
