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