import torch
import torch.nn as nn
from torch.func import grad_and_value
import zgraph.core.functional as F
from typing import List, Optional, Tuple, Any, Union, Iterable, Dict, TypeVar

# T is a TypeVar for the apply_transform recursive container generic
T = TypeVar('T')

class GaugeFix(nn.Module):
    """
    Analytically projects the state onto a target manifold (default 0.0) 
    by uniformly shifting a list of invariant indices.
    """
    def __init__(self, base_model: nn.Module, shift_indices: List[int], target_val: float = 0.0):
        super().__init__()
        self.base_model = base_model
        self.target_val = target_val
        self.register_buffer(
            'idx', 
            torch.atleast_1d(torch.as_tensor(shift_indices, dtype=torch.long))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        phi = self.base_model(x)
        target_tensor = torch.as_tensor(self.target_val, dtype=phi.dtype, device=phi.device).squeeze()
        coords = F.apply_gauge_shift(x, phi, self.idx, target_tensor)
        return target_tensor, coords


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


class GF_and_LT(nn.Module):
    """
    Applies both Legendre and Gauge transforms in a single efficient pass,
    safely exploiting value_and_grad without destroying gradients.
    """
    def __init__(self, 
                 base_model: nn.Module, 
                 gauge_indices: Optional[List[int]] = None, 
                 legendre_indices: Optional[List[int]] = None, 
                 target_val: float = 0.0):
        super().__init__()
        self.base_model = base_model
        self.target_val = target_val
        
        g_idx = [] if gauge_indices is None else gauge_indices
        lt_idx = [] if legendre_indices is None else legendre_indices
        
        self.register_buffer('g_idx', torch.atleast_1d(torch.as_tensor(g_idx, dtype=torch.long)))
        self.register_buffer('lt_idx', torch.atleast_1d(torch.as_tensor(lt_idx, dtype=torch.long)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Evaluate base landscape and optionally gradients
        if self.lt_idx.numel() > 0:
            grad, phi = grad_and_value(self.base_model)(x)
        else:
            phi = self.base_model(x)
            grad = torch.zeros_like(x) # Fallback to appease static type checkers, never used

        coords = x

        # 2. Apply Gauge Fix FIRST
        if self.g_idx.numel() > 0:
            target_tensor = torch.as_tensor(self.target_val, dtype=phi.dtype, device=phi.device).squeeze()
            coords = F.apply_gauge_shift(coords, phi, self.g_idx, target_tensor)
            phi = target_tensor

        # 3. Apply Legendre Transform SECOND
        if self.lt_idx.numel() > 0:
            phi, coords = F.apply_legendre(coords, phi, grad, self.lt_idx)

        return phi, coords


# --- Helper Functions for Container Mapping ---

def apply_transform(transform_class: type, modules: Union[T, Dict[Any, T], List[T], Tuple[T, ...]], *args: Any, **kwargs: Any) -> Union[nn.Module, Dict[Any, nn.Module], List[nn.Module], Tuple[nn.Module, ...]]:
    """Applies a transform class to a single module or a container of modules."""
    if isinstance(modules, (list, tuple)):
        # Correctly instantiating the tuple or list class
        return type(modules)(apply_transform(transform_class, m, *args, **kwargs) for m in modules) # type: ignore
    if isinstance(modules, dict):
        return {k: apply_transform(transform_class, v, *args, **kwargs) for k, v in modules.items()}
    return transform_class(modules, *args, **kwargs)

def gauge_fix(modules: Union[T, Dict[Any, T], List[T], Tuple[T, ...]], shift_indices: List[int], target_val: float = 0.0) -> Union[nn.Module, Dict[Any, nn.Module], List[nn.Module], Tuple[nn.Module, ...]]:
    """Maps GaugeFix over a module or container of modules."""
    return apply_transform(GaugeFix, modules, shift_indices, target_val=target_val)

def legendre_transform(modules: Union[T, Dict[Any, T], List[T], Tuple[T, ...]], transform_indices: List[int]) -> Union[nn.Module, Dict[Any, nn.Module], List[nn.Module], Tuple[nn.Module, ...]]:
    """Maps LegendreTransform over a module or container of modules."""
    return apply_transform(LegendreTransform, modules, transform_indices)

def gf_and_lt(modules: Union[T, Dict[Any, T], List[T], Tuple[T, ...]], gauge_indices: Optional[List[int]] = None, legendre_indices: Optional[List[int]] = None, target_val: float = 0.0) -> Union[nn.Module, Dict[Any, nn.Module], List[nn.Module], Tuple[nn.Module, ...]]:
    """Maps GF_and_LT over a module or container of modules."""
    return apply_transform(GF_and_LT, modules, gauge_indices=gauge_indices, legendre_indices=legendre_indices, target_val=target_val)
