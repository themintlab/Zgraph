import torch
from typing import Tuple, Union

def marginalize(energy_landscape: torch.Tensor, beta: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
    """
    The stateless mathematical core of the ZGraph engine.
    Executes the SoftMin collapse (Partition Function).
    
    Args:
        energy_landscape (torch.Tensor): The dynamic Energy Vector across microstates.
                                         Shape: (Num_Microstates,)
        beta (Union[float, torch.Tensor]): The thermodynamic smoothing parameter (-kT).
                                           Beta=inf triggers hardmax.
                                           Shape: () (Scalar)
                                 
    Returns:
        torch.Tensor: The renormalized scalar Free Energy. Shape: () (Scalar)
    """
    # Calculate the partition function / free energy as a scalar.
    return beta * torch.logsumexp(energy_landscape / beta, dim=-1)

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

def apply_legendre(primal_x: torch.Tensor, raw_phi: torch.Tensor, 
                   full_grad: torch.Tensor, lt_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure subfunction to compute the multivariate Legendre dual.
    Takes the evaluated energy (phi) and gradients, and constructs the dual state.
    
    Args:
        primal_x (torch.Tensor): The input primal coordinates.
        raw_phi (torch.Tensor): The evaluated primal energy.
        full_grad (torch.Tensor): The gradient vector.
        lt_idx (torch.Tensor): Indices for the Legendre transform (1D integer tensor).
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The dual energy (psi) and the dual coordinates.
    """
    if lt_idx.numel() == 0:
        return raw_phi, primal_x
        
    # Use torch.dot since batching is deferred to vmap/torch.compile
    psi = raw_phi - torch.dot(primal_x[lt_idx], full_grad[lt_idx])
    
    dual_x = primal_x.clone()
    dual_x[lt_idx] = full_grad[lt_idx]
    
    return psi, dual_x