import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class GraphTransform(nn.Module, ABC):
    """
    Base class enforcing the Universal ZGraph Contract.
    Output is strictly guaranteed to be: (0D Scalar Potential, 1D Coordinate Vector).
    """
    
    # 1. The Base Class controls `forward` to guarantee the output contract.
    def forward(self, x: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Call the subclass's math implementation
        output = self._compute_transform(x, *args, **kwargs)
        
        # --- ZERO-COST COMPILER ENFORCEMENT ---
        # 1. Enforce Tuple Contract
        assert isinstance(output, tuple) and len(output) == 2, \
            f"{self.__class__.__name__} must return a tuple of length 2."
            
        potential, coords = output
        
        # 2. Enforce Scalar Potential Contract
        assert potential.dim() == 0, \
            f"{self.__class__.__name__} potential must be a 0D scalar tensor, got {potential.dim()}D."
            
        # 3. Enforce Coordinate Shape Contract (Must match input shape)
        assert coords.shape == x.shape, \
            f"{self.__class__.__name__} coordinates must retain input shape {x.shape}, got {coords.shape}."

        return potential, coords

    @staticmethod
    def ensure_transform(module: nn.Module) -> "GraphTransform":
        """Wrap plain nn.Modules so all transforms can rely on a shared contract."""
        if isinstance(module, GraphTransform):
            return module
        if isinstance(module, nn.Module):
            return GraphAdapter(module)
        raise TypeError("Expected an nn.Module, list/tuple/dict of nn.Module, or GraphTransform.")

    @classmethod
    def map_factory(cls, module, factory):
        """Apply a transform factory while preserving common container types."""
        if isinstance(module, list):
            return [cls.map_factory(m, factory) for m in module]
        if isinstance(module, tuple):
            return tuple(cls.map_factory(m, factory) for m in module)
        if isinstance(module, dict):
            return {k: cls.map_factory(v, factory) for k, v in module.items()}
        return factory(cls.ensure_transform(module))

    # Subclasses MUST implement this instead of `forward`.
    @abstractmethod
    def _compute_transform(self, x: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Developers write the mathematical transform here.
        Must return (potential, transformed_coordinates).
        """
        pass


class GraphAdapter(GraphTransform):
    def __init__(self, zgraph: nn.Module):
        super().__init__()
        self.graph = zgraph

    # Implement _compute_transform instead of forward
    def _compute_transform(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        energy = self.graph(x)
        if isinstance(energy, tuple):
            energy = energy[0]
        # Squeeze ensures a scalar if the wrapped module returned shape [1]
        return energy.squeeze(), x