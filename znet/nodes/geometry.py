import torch
import torch.nn as nn
from ..core import functional as F

class FactorNode(nn.Module):
    def __init__(self, M_matrix, subgraph_list):
        """
        Args:
            M_matrix (torch.Tensor): 2D Tensor of shape (num_microstates, num_clusters).
            subgraph_list (list[nn.Module]): A list of subgraph modules. The order
                of modules in this list MUST match the order of the cluster
                columns in the M_matrix.
        """
        super().__init__()
        
        # Ensure M is a float tensor so it responds to .to(dtype) operations
        self.register_buffer('M', M_matrix.to(torch.get_default_dtype()))        
        self.subgraphs = nn.ModuleList(subgraph_list)

    def forward(self, local_signals):
        # Execute each subgraph and ensure its output has a trailing dimension for concatenation.
        # The order of subgraphs is critical and must match the columns of M_matrix.
        cluster_energies = [
            subgraph(local_signals).view(local_signals.shape[:-1] + (1,))
            for subgraph in self.subgraphs
        ]

        # torch.cat handles the memory allocation natively in C++
        energy_vector = torch.cat(cluster_energies, dim=-1)
    
        T = local_signals[..., 0:1]

        beta = - 8.617e-5 * T
        return F.marginalize(self.M, energy_vector)


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
        
        # ==========================================
        # THE BINDING LOGIC
        # ==========================================
        if signal_indices is not None:
            # EARLY BINDING (For prototype testing)
            # Register the provided indices immediately
            self.register_buffer(
                'signal_indices', 
                torch.tensor(signal_indices, dtype=torch.long)
            )
        else:
            # LATE BINDING (For production assembly)
            # Initialize an empty buffer waiting for the parent to link it
            self.register_buffer(
                'signal_indices', 
                torch.empty(0, dtype=torch.long)
            )
        
        # Dynamically register parameters
        self.theta = nn.ParameterDict({
            key: nn.Parameter(torch.tensor([val], dtype=torch.float32))
            for key, val in initial_guesses.items()
        })

    def forward(self, full_local_signals):
        # Slice and execute remains completely unchanged
        sliced_signals = full_local_signals[..., self.signal_indices]
        return self.energy_function(sliced_signals, **self.theta)

    def bind_indices(self, new_indices):
        """
        Allows the parent PhaseNode to safely overwrite the indices later,
        even if they were hardcoded during early testing.
        """
        device = self.signal_indices.device 
        self.signal_indices = torch.tensor(new_indices, dtype=torch.long, device=device)
