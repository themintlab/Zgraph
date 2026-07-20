import torch
import torch.nn as nn
from . import functional as F

class FactorNode(nn.Module):
    def __init__(self, M_matrix, subgraph_list, beta_index=0, beta_factor = 1):
        """
        Args:
            M_matrix (torch.Tensor): 2D Tensor of shape (num_microstates, num_clusters).
            subgraph_list (list[nn.Module]): A list of subgraph modules. The order
                of modules in this list MUST match the order of the cluster
                columns in the M_matrix.
            beta_index: an integer index of the rationality parameter from the signal. 
            beta_factor: a scalar multiplier of the rationality parameter
        """
        super().__init__()

        if isinstance(M_matrix, list):
            M_matrix = torch.tensor(M_matrix)
        
        if M_matrix.ndim == 1:
            M_matrix = M_matrix.unsqueeze(0)
        
        if M_matrix.ndim != 2:
            raise ValueError(f"M_matrix must be a 2D tensor, got {M_matrix.ndim}D.")
            
        num_clusters = M_matrix.shape[1]
        if num_clusters != len(subgraph_list):
            raise ValueError(
                f"Dimension mismatch: M_matrix expects {num_clusters} clusters (columns), "
                f"but received {len(subgraph_list)} subgraphs."
            )

        self.register_buffer('M', M_matrix.to(torch.get_default_dtype()))
        self.register_buffer('beta_index', torch.tensor(beta_index, dtype=torch.long))
        self.register_buffer('beta_factor', torch.tensor(beta_factor, dtype=torch.float32))    
        self.subgraphs = nn.ModuleList(subgraph_list)

    def forward(self, local_signals):
        energy_vector = torch.stack([subgraph(local_signals) for subgraph in self.subgraphs])
        beta = torch.clamp(self.beta_factor * local_signals[self.beta_index], min = 1.2e-7)
        return F.marginalize(self.M, energy_vector, beta)


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
    def __init__(self, signal_indices):
        super().__init__()
        # Passing a single int creates a 0-D tensor (reduces dimension on slice).
        # Passing a list creates a 1-D tensor (preserves dimension on slice).
        self.register_buffer('signal_indices', torch.tensor(signal_indices, dtype=torch.long))

    def forward(self, local_signals):
        # Summing selected signals implements Tropical Multiplication.
        # Slicing a 1D vector with indices returns a vector; sum() makes it a scalar.
        return local_signals[self.signal_indices].sum()

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
            key: nn.Parameter(torch.tensor(val, dtype=torch.float32))
            for key, val in initial_guesses.items()
        })

    def forward(self, full_local_signals):
        # Strictly vector input: (Channels,) -> scalar output: ()
        sliced_signals = full_local_signals[self.signal_indices]
        return self.energy_function(sliced_signals, **self.theta)
