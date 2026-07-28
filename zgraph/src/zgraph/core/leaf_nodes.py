import torch
import torch.nn as nn

class BaseLeafNode(nn.Module):
    """
    Abstract base class for all standard leaf nodes in ZGraph.
    Subclasses MUST explicitly define their mathematical forward() method
    and register their learnable parameters via nn.Parameter.
    """
    def __init__(self, signal_indices=None):
        super().__init__()
        indices_to_register = signal_indices if signal_indices is not None else []
        self.register_buffer(
            'signal_indices',
            torch.tensor(indices_to_register, dtype=torch.long)
        )

    def forward(self, local_signals):
        raise NotImplementedError("Subclasses must implement forward()")

class DynamicLeafNode(nn.Module):
    """
    Evaluation-only node that accepts arbitrary pure PyTorch functions.
    Ideal for rapid prototyping. Should NOT be used for performance-critical training.
    """
    def __init__(self, energy_function, signal_indices=None, **constants):
        """
        Args:
            energy_function (callable): The pure math equation.
            signal_indices (list[int], optional): Hardcoded indices for early testing.
            **constants: Constant parameters passed to the function.
        """
        super().__init__()
        self.energy_function = energy_function
        
        indices_to_register = signal_indices if signal_indices is not None else []
        self.register_buffer(
            'signal_indices',
            torch.tensor(indices_to_register, dtype=torch.long)
        )
        
        # We don't register them as Parameters because this is evaluation-only.
        # But we do need to pass them to the function, so we register them as buffers.
        self.constant_keys = tuple(constants.keys())
        for key, val in constants.items():
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32)
            self.register_buffer(key, val)

    def forward(self, full_local_signals):
        # Strictly vector input: (Channels,) -> scalar output: ()
        sliced_signals = full_local_signals[self.signal_indices]
        kwargs = {k: getattr(self, k) for k in self.constant_keys}
        return self.energy_function(sliced_signals, **kwargs)

class ConstantNode(nn.Module):
    """The simplest physics model: a trainable constant (or constants)."""
    def __init__(self, init_val=1.):
        super().__init__()
        
        if torch.is_tensor(init_val):
            tensor_val = init_val.clone().detach().to(dtype=torch.float32)
        else:
            tensor_val = torch.tensor(init_val, dtype=torch.float32)
            
        self.value = nn.Parameter(tensor_val)

    def forward(self, signals):
        return self.value
    
class SignalNode(nn.Module):
    """A node that extracts specific signal indices from the input."""
    def __init__(self, signal_index):
        super().__init__()
        try:
            signal_index = int(signal_index)
        except (TypeError, ValueError):
            raise TypeError("signal_index must be an integer. Use SignalNodes() for multiple nodes.")
        self.register_buffer('signal_index', torch.tensor(signal_index, dtype=torch.long))

    def forward(self, local_signals):
        return local_signals[self.signal_index]

def SignalNodes(*indices):
    """
    Convenience factory for generating multiple SignalNodes simultaneously.
    Usage: mu1, mu2 = SignalNodes(1, 2)
    """
    return [SignalNode(i) for i in indices]
