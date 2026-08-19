import torch
from typing import Tuple

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
