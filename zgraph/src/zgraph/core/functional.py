import torch

def marginalize(M_matrix, w_vector, beta = 1.0):
    """
    The stateless mathematical core of the ZGraph engine.
    Executes the Tropical Polynomial and SoftMin collapse.
    
    Args:
        M_matrix (torch.Tensor): The static Configuration Matrix. 
                                 Shape: (Num_Microstates, Num_Clusters)
        w_vector (torch.Tensor): The dynamic Energy Vector from the subgraphs.
                                 Shape: (Num_Clusters,)
        beta (torch.Tensor):     The thermodynamic smoothing parameter (-kT).
                                 Beta=inf triggers hardmax.
                                 Shape: () (Scalar)
                                 
    Returns:
        torch.Tensor: The renormalized scalar Free Energy. Shape: () (Scalar)
    """

    # ---------------------------------------------------------
    # STEP 1: The Landscape Construction (Tropical Polynomial)
    # ---------------------------------------------------------
    # We map the cluster energies to the allowed microstates.
    # Perform a standard matrix-vector multiplication for the energy landscape.
    # M_matrix: (Microstates, Clusters), w_vector: (Clusters,) -> (Microstates,)
    energy_landscape = M_matrix @ w_vector

    # Calculate the partition function / free energy as a scalar.
    return beta * torch.logsumexp(energy_landscape / beta, dim=-1)

def apply_gauge_shift(primal_x: torch.Tensor, raw_phi: torch.Tensor, 
                      shift_idx: torch.Tensor, target_val: torch.Tensor):
    """
    Pure subfunction to apply an invariant shift.
    Takes the evaluated energy (phi) and applies the exact shift to coordinates.
    """
    if shift_idx.numel() == 0:
        return primal_x
        
    shift_amount = target_val - raw_phi
    shifted_x = primal_x.clone()
    shifted_x[shift_idx] += shift_amount
    
    return shifted_x

def apply_legendre(primal_x: torch.Tensor, raw_phi: torch.Tensor, 
                   full_grad: torch.Tensor, lt_idx: torch.Tensor):
    """
    Pure subfunction to compute the multivariate Legendre dual.
    Takes the evaluated energy (phi) and gradients, and constructs the dual state.
    """
    if lt_idx.numel() == 0:
        return raw_phi, primal_x
        
    # Use torch.dot since batching is deferred to vmap/torch.compile
    psi = raw_phi - torch.dot(primal_x[lt_idx], full_grad[lt_idx])
    
    dual_x = primal_x.clone()
    dual_x[lt_idx] = full_grad[lt_idx]
    
    return psi, dual_x