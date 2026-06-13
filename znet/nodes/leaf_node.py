import torch
import torch.nn as nn

class ConstantNode(nn.Module):
    """The simplest physics model: a trainable constant (or constants)."""
    def __init__(self, init_val=0.0):
        super().__init__()
        
        if torch.is_tensor(init_val):
            tensor_val = init_val.clone().detach().to(dtype=torch.float32)
        else:
            tensor_val = torch.tensor(init_val, dtype=torch.float32)
            
        self.value = nn.Parameter(torch.atleast_1d(tensor_val))


    def forward(self, signals):
        return self.value
    
class SignalNode(nn.Module):
    """A node that extracts specific signal indices from the input."""
    def __init__(self, signal_indices):
        super().__init__()
        # Passing a single int creates a 0-D tensor (reduces dimension on slice).
        # Passing a list creates a 1-D tensor (preserves dimension on slice).
        self.register_buffer('signal_indices', torch.tensor(signal_indices, dtype=torch.long))

    def forward(self, local_signals):
        return local_signals[..., self.signal_indices]




class LeafNode(nn.Module):
    def __init__(self, energy_function, signal_indices=None, **initial_guesses):
        """
        Args:
            energy_function (callable): The pure math equation.
            signal_indices (list[int], optional): Hardcoded indices for early testing.
            **initial_guesses: Trainable parameters.
        """
        super().__init__()
        self.energy_function = energy_function
        
        indices_to_register = signal_indices if signal_indices is not None else []
        self.register_buffer(
            'signal_indices',
            torch.tensor(indices_to_register, dtype=torch.long)
        )
        
        # Dynamically register parameters
        # Note: Might cause issues with torch.script but should be okay with torch.compile
        self.theta = nn.ParameterDict({
            key: nn.Parameter(torch.tensor([val], dtype=torch.float32))
            for key, val in initial_guesses.items()
        })

    def forward(self, full_local_signals):
        # Slice and execute with parameters defined positionally
        sliced_signals = full_local_signals[..., self.signal_indices]
        return self.energy_function(sliced_signals, **self.theta)