import torch

def marginalize(M_matrix, w_vector, beta = 1.0):
    """
    The stateless mathematical core of the ZNet engine.
    Executes the Tropical Polynomial and SoftMin collapse.
    
    Args:
        M_matrix (torch.Tensor): The static Configuration Matrix. 
                                 Shape: (Num_Microstates, Num_Clusters)
        w_vector (torch.Tensor): The dynamic Energy Vector from the subgraphs.
                                 Shape: (*Batch, Num_Clusters)
        beta (torch.Tensor):     The thermodynamic smoothing parameter (-kT).
                                 Beta=inf triggers hardmax.
                                 Shape: (*Batch, 1)
                                 
    Returns:
        torch.Tensor: The renormalized scalar Free Energy. Shape: (*Batch, 1)
    """

    # ---------------------------------------------------------
    # STEP 1: The Landscape Construction (Tropical Polynomial)
    # ---------------------------------------------------------
    # We map the cluster energies to the allowed microstates.
    # 'mc'   = Microstates x Clusters (Static Matrix)
    # '...c' = Batch x Clusters (Dynamic Vector)
    # '...m' = Batch x Microstates (Output Landscape)
    energy_landscape = torch.einsum('mc,...c->...m', M_matrix, w_vector)

    # ---------------------------------------------------------
    # STEP 2: The Renormalization (Log-Partition Collapse)
    # ---------------------------------------------------------
    if isinstance(beta, (int, float)):
        if beta == 1 or beta == 1.0:
            return torch.logsumexp(energy_landscape, dim=-1, keepdim=True)
        if beta == 0 or beta == 0.0:
            return torch.amax(energy_landscape, dim=-1, keepdim=True)

    return beta * torch.logsumexp(energy_landscape / beta, dim=-1, keepdim=True)