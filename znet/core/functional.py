import torch

def scaled_logsumexp(energy, dim, beta=1):
    """
    Computes beta * log( sum( exp( energy / beta ) ) ).
    """
    # Use torch.logsumexp explicitly if import is shadowed, 
    # Add keep dims to do a full overload of torch.logsumexp?
    # but here we renamed the function so 'logsumexp' refers to the import at the top.

    return beta * torch.logsumexp(energy / beta, dim=dim)

def outer_addition(tensor_list):
    """Atomic Op 1: Builds the uncoupled energy landscape (The 'AND' geometry)."""
    num_subs = len(tensor_list)
    aligned_tensors = [
            t.view(*t.shape[:-1], *[1]*i, t.shape[-1], *[1]*(num_subs - i - 1))
            for i, t in enumerate(tensor_list)
        ]
    return sum(aligned_tensors)

def collapse(grid, num_sublattices):
    """Renormalization / collapse degrees of freedom (Trace of Hamiltonian)."""
    dims_to_collapse = tuple(range(-num_sublattices, 0))
    # Un-squeeze to maintain the (*Batch, 1) scalar format
    return torch.logsumexp(grid, dim=dims_to_collapse).unsqueeze(-1)