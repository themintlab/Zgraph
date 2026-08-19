from torch.utils._pytree import tree_map

def to_numpy(pytree):
    """Strips all PyTorch tracking from a PyTree, returning clean numpy arrays."""
    return tree_map(lambda x: x.numpy(force=True).squeeze() if hasattr(x, "numpy") else x, pytree)

__all__ = ["to_numpy"]
