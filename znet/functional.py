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
        torch.Tensor: The combined energy tensor of shape (Batch, D_1, D_2, ..., D_N).
            Represents E_total = phi_1 + phi_2 + ... + phi_n + Omega
    """
    
    # 1. Validation
    if not phi_vectors:
        raise ValueError("phi_vectors list cannot be empty")
    
    batch_size = phi_vectors[0].shape[0]
    num_children = len(phi_vectors)
    
    # 1b. Normalize Shapes: Ensure all vectors are at least (Batch, D)
    # If a vector is (Batch,), treat it as (Batch, 1).
    #TODO: Ensure this is efficient!
    normalized_phis = []
    for i, v in enumerate(phi_vectors):
        if v.dim() == 1:
            # Check consistency (optional but good practice)
            if v.shape[0] != batch_size:
                 raise ValueError(f"Batch dimension mismatch at index {i}. Expected {batch_size}, got {v.shape[0]}")
            normalized_phis.append(v.view(-1, 1))
        else:
            normalized_phis.append(v)
            
    phi_vectors = normalized_phis

    # Collect state dimensions (D_1, D_2, ...)
    # Each vec is (Batch, D_i), so we take dim 1
    state_dims = [v.shape[1] for v in phi_vectors]
    
    # 2. Reshape for Broadcasting (The "Outer Sum" Setup)
    # We want to transform vectors into shapes:
    # Vec 1: (Batch, D_1, 1,   1,   ...)
    # Vec 2: (Batch, 1,   D_2, 1,   ...)
    # Vec 3: (Batch, 1,   1,   D_3, ...)
    
    reshaped_phis = []
    for i, phi in enumerate(phi_vectors):
        # Verify batch size consistency
        if phi.shape[0] != batch_size:
            raise ValueError(f"Batch dimension mismatch at index {i}. Expected {batch_size}, got {phi.shape[0]}")
            
        # Create view shape: [Batch] + [1, 1, ...]
        view_shape = [batch_size] + [1] * num_children
        # Set the target dimension to D_i
        view_shape[i + 1] = state_dims[i]
        
        reshaped_phis.append(phi.view(*view_shape))

    # 3. Compute Non-Interacting Energy (Ideal Mixing)
    # Start with the first vector
    total_energy = reshaped_phis[0]
    
    # Add the rest (Broadcasting handles the expansion)
    for i in range(1, num_children):
        total_energy = total_energy + reshaped_phis[i]

    # 4. Add Enthalpic Coupling (Interaction)
    if interaction_tensor is not None:
        # Interaction tensor is added to the ideal mixing background.
        # PyTorch broadcasting handles cases where interaction_tensor 
        # is (D1, D2...) (shared across batch) or (Batch, D1, D2...)
        
        # Sanity check for non-batched interaction tensor dimensions
        if interaction_tensor.dim() == num_children:
             # If interaction is static (no batch dim), ensure dims match
            if list(interaction_tensor.shape) != state_dims:
                 # It might be broadcastable (e.g. 1s in shape), so we warn/check carefully
                 pass 
        
        total_energy = total_energy + interaction_tensor

    return total_energy


def softmin_energy(energy_tensor, dim=None, temperature=293.15, k_b=DEFAULT_KB, keepdim=False):
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
        keepdim (bool): Whether to retain the collapsed dimensions as size 1.
            Default is False (standard marginalization).

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
    log_z = logsumexp(scaled_energy, dim=dim_to_collapse, keepdim=keepdim)
    free_energy = -log_z / beta     # broadcasts correctly if T is a tensor.
    
    return free_energy