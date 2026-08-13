import torch
import torch.nn as nn
from torch.func import grad_and_value
import zgraph.core.functional as F
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


def gauge_fix(system_model: Any, x: torch.Tensor, shift_indices: List[int], target_val: float = 0.0) -> torch.Tensor:
    """
    Project the input state onto a target equilibrium manifold (default 0.0) 
    by analytically finding a uniform shift in the specified indices.
    """
    phi = system_model(x)
    target_tensor = torch.as_tensor(target_val, dtype=phi.dtype, device=phi.device).squeeze()
    shift_idx = torch.atleast_1d(torch.as_tensor(shift_indices, dtype=torch.long, device=x.device))
    return F.apply_gauge_shift(x, phi, shift_idx, target_tensor)

# --- Helper Functions for Container Mapping ---

def apply_transform(transform_class: type, modules: Union[T, Dict[Any, T], List[T], Tuple[T, ...]], *args: Any, **kwargs: Any) -> Union[nn.Module, Dict[Any, nn.Module], List[nn.Module], Tuple[nn.Module, ...]]:
    """Applies a transform class to a single module or a container of modules."""
    if isinstance(modules, (list, tuple)):
        # Correctly instantiating the tuple or list class
        return type(modules)(apply_transform(transform_class, m, *args, **kwargs) for m in modules) # type: ignore
    if isinstance(modules, dict):
        return {k: apply_transform(transform_class, v, *args, **kwargs) for k, v in modules.items()}
    return transform_class(modules, *args, **kwargs)


def legendre_transform(modules: Union[T, Dict[Any, T], List[T], Tuple[T, ...]], transform_indices: List[int]) -> Union[nn.Module, Dict[Any, nn.Module], List[nn.Module], Tuple[nn.Module, ...]]:
    """Maps LegendreTransform over a module or container of modules."""
    return apply_transform(LegendreTransform, modules, transform_indices)

