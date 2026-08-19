import torch
import torch.nn as nn
from torch.func import grad_and_value
import zgraph.transforms.functional as F
from typing import List, Optional, Tuple, Any, Union, Iterable, Dict, TypeVar

# T is a TypeVar for the apply_transform recursive container generic
T = TypeVar('T')

class LegendreTransform(nn.Module):
    """
    Transforms a base thermodynamic module via a Legendre transform on specified indices.
    """
    def __init__(self, base_model: nn.Module, transform_indices: List[int]):
        super().__init__()
        self.base_model = base_model
        self.register_buffer(
            'idx',
            torch.atleast_1d(torch.as_tensor(transform_indices, dtype=torch.long))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad, phi = grad_and_value(self.base_model)(x)
        psi, dual_coords = F.apply_legendre(x, phi, grad, self.idx)
        return psi, dual_coords




from torch.utils._pytree import tree_map

def legendre_transform(modules: Any, transform_indices: List[int]) -> Any:
    """Maps LegendreTransform over an arbitrary PyTree of modules."""
    return tree_map(lambda m: LegendreTransform(m, transform_indices), modules)
