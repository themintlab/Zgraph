import torch

def outer_addition(tensor_list):
    """Atomic Op 1: Builds the uncoupled energy landscape (The 'AND' geometry)."""
    num_subs = len(tensor_list)
    aligned_tensors = [
            t.view(*t.shape[:-1], *[1]*i, t.shape[-1], *[1]*(num_subs - i - 1))
            for i, t in enumerate(tensor_list)
        ]
    return sum(aligned_tensors)

def collapse(grid, num_sublattices, scale=1.0):
    """
    Renormalization / collapse degrees of freedom (Trace of Hamiltonian).

    Args:
        grid: Energy landscape tensor.
        num_sublattices: Number of trailing state dimensions to collapse.
        scale: Softness/temperature-like scale. scale=1 uses the fastest path.
    """
    dims_to_collapse = tuple(range(-num_sublattices, 0))

    # Fast path for the common case to avoid extra divide/multiply kernels.
    if scale == 1 or scale == 1.0:
        return torch.logsumexp(grid, dim=dims_to_collapse).unsqueeze(-1)

    # Un-squeeze to maintain the (*Batch, 1) scalar format.
    return (scale * torch.logsumexp(grid / scale, dim=dims_to_collapse)).unsqueeze(-1)


