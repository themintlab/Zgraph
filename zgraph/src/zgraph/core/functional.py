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