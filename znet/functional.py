from torch import Tensor, logsumexp
from .constants import DEFAULT_KB

def build_energy_tensor(phi_vectors, interaction_tensor=None):
    """
    Constructs the total energy tensor for a system of interacting subsystems.
    Performs an 'Outer Sum' of component potentials and adds interaction energy.

    Args:
        phi_vectors (list of torch.Tensor): List of N tensors. 
            Each tensor must have shape (Batch, D_i), where D_i is the 
            number of species/states for that child node.
        interaction_tensor (torch.Tensor, optional): The Enthalpy Matrix (Omega).
            Shape must be broadcastable to (Batch, D_1, D_2, ..., D_N).
            Usually shape is (D_1, D_2, ..., D_N) for static interactions,
            or (Batch, D_1, ..., D_N) for context-dependent interactions.

    Returns:
        torch.Tensor: E_total = phi_1 + ... + phi_n + Omega
    """
    
    num_children = len(phi_vectors)
    
    # Pre-calculate base shape: [Batch, 1, 1, ..., 1]
    base_shape = [-1] + [1] * num_children

    # 1. Initialize Base Energy
    if interaction_tensor is None:
        # Ideal Mixing Optimization: Start with the first vector
        # This avoids allocating a massive Zeros tensor.
        start_index = 1
        
        first_shape = list(base_shape)
        first_shape[1] = phi_vectors[0].shape[1] # Set D_1
        total_energy = phi_vectors[0].view(*first_shape)
    else:
        # Interaction Case: Start with the Enthalpy Matrix
        start_index = 0
        total_energy = interaction_tensor
    
    # 2. Add Remaining Potential Vectors
    for i in range(start_index, num_children):
        phi = phi_vectors[i]
        
        shape = list(base_shape)
        shape[i + 1] = phi.shape[1] # Set dimension D_i
        
        # In-place addition (+=) is faster and saves memory
        total_energy = total_energy + phi.view(*shape)

    return total_energy


def softmin_energy(energy_tensor, dim=None, temperature=293.15, k_b=DEFAULT_KB):
    """
    Computes the Free Energy (Phi) by marginalizing out specific dimensions 
    of an Energy Tensor using the SoftMin operator.
    
    Math: Phi = -k_B * T * ln( sum( exp( -E / (k_B * T) ) ) )

    Args:
        energy_tensor (torch.Tensor): Input energy grid.
            Shape: (Batch, D1, D2, ..., DN)
        dim (int or tuple of ints, optional): The dimension(s) to collapse/integrate out.
            Example: (1, 2) collapses the first two sublattices. If None, all dimensions
            except the batch dimension (dim 0) will be collapsed.
        temperature (float or torch.Tensor): System temperature.
            - If float: Applied globally.
            - If Tensor: Must be broadcastable to (Batch,).
        k_b (float): Boltzmann constant (default eV/K).

    Returns:
        torch.Tensor: The effective potential / Free Energy.
    """

    # If dim is None, collapse all dimensions except the batch dimension
    if dim is None:
        dim_to_collapse = tuple(range(1, energy_tensor.dim()))
    else:
        dim_to_collapse = dim

    # Temperature Broadcasting: if T is a tensor (e.g., shape (Batch,)), we need to align it with energy_tensor
    if isinstance(temperature, Tensor):
        # Create a view that adds singleton dimensions for every dim in energy_tensor
        # except the batch dim (0).
        # Example: T(Batch) -> T(Batch, 1, 1, ...)
        view_shape = [temperature.shape[0]] + [1] * (energy_tensor.dim() - 1)
        T = temperature.view(*view_shape)
    else:
        T = temperature

    # We compute 1/beta * log( sum( exp( -E * beta ) ) )
    beta = 1.0 / (k_b * T)
    scaled_energy = -energy_tensor * beta
    log_z = logsumexp(scaled_energy, dim=dim_to_collapse, keepdim=False)
    free_energy = -log_z / beta     # broadcasts correctly if T is a tensor.
    
    return free_energy