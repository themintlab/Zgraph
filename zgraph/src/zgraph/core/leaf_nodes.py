import torch
import torch.nn as nn
from typing import Optional, Union, List, Callable, Dict, Any, Tuple

class BaseLeafNode(nn.Module):
    """
    Abstract base class for all standard leaf nodes in ZGraph.
    Subclasses MUST explicitly define their mathematical forward() method
    and register their learnable parameters via nn.Parameter.
    """
    def __init__(self, signal_indices: Optional[List[int]] = None):
        super().__init__()
        indices_to_register = signal_indices if signal_indices is not None else []
        self.register_buffer(
            'signal_indices',
            torch.tensor(indices_to_register, dtype=torch.long)
        )

    def forward(self, local_signals: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward()")

class DynamicLeafNode(BaseLeafNode):
    """
    Evaluation-only node that accepts arbitrary pure PyTorch functions.
    Ideal for rapid prototyping. Should NOT be used for performance-critical training.
    """
    def __init__(self, energy_function: Callable[..., torch.Tensor], signal_indices: Optional[List[int]] = None, **constants: Any):
        """
        Args:
            energy_function (callable): The pure math equation.
            signal_indices (list[int], optional): Hardcoded indices for early testing.
            **constants: Constant parameters passed to the function.
        """
        super().__init__(signal_indices)
        self.energy_function = energy_function
        
        # We don't register them as Parameters because this is evaluation-only.
        # But we do need to pass them to the function, so we register them as buffers.
        self.constant_keys: List[str] = []
        for key, val in constants.items():
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32)
            self.register_buffer(key, val)
            self.constant_keys.append(key)

    def forward(self, full_local_signals: torch.Tensor) -> torch.Tensor:
        # Strictly vector input: (Channels,) -> scalar output: ()
        sliced_signals = full_local_signals[self.signal_indices]
        kwargs = {k: getattr(self, k) for k in self.constant_keys}
        return self.energy_function(sliced_signals, **kwargs)

class ConstantNode(nn.Module):
    """The simplest physics model: a trainable constant (or constants)."""
    def __init__(self, init_val: Union[float, int, torch.Tensor] = 1.0):
        super().__init__()
        
        if torch.is_tensor(init_val):
            tensor_val = init_val.clone().detach().to(dtype=torch.float32)
        else:
            tensor_val = torch.tensor(init_val, dtype=torch.float32)
            
        self.value = nn.Parameter(tensor_val)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return self.value
    
class SignalNode(nn.Module):
    """A node that extracts specific signal indices from the input."""
    def __init__(self, signal_index: int):
        super().__init__()
        try:
            signal_index = int(signal_index)
        except (TypeError, ValueError):
            raise TypeError("signal_index must be an integer. Use SignalNodes() for multiple nodes.")
        self.signal_index = signal_index

    def forward(self, local_signals: torch.Tensor) -> torch.Tensor:
        return local_signals.select(0, self.signal_index)

from torch.utils._pytree import tree_map

def SignalNodes(*indices: Any) -> Any:
    """
    Convenience factory for generating SignalNodes.
    Accepts flat args: mu1, mu2 = SignalNodes(1, 2)
    Or a PyTree: nodes = SignalNodes({'T': 0, 'mu': [1, 2]})
    """
    if len(indices) == 1 and isinstance(indices[0], (list, dict, tuple)):
        pytree = indices[0]
    else:
        pytree = indices
    return tree_map(lambda i: SignalNode(int(i)), pytree)
