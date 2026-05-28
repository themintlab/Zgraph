import torch
import torch.nn as nn

class ConstantReference(nn.Module):
    """The simplest physics model: a trainable constant."""
    def __init__(self, init_val=0.0):
        super().__init__()
        self.value = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))


    def forward(self, signals):
        return self.value
    



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