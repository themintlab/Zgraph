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
        
        if M_matrix.ndim != 2:
            raise ValueError(f"M_matrix must be a 2D tensor, got {M_matrix.ndim}D.")
            
        num_clusters = M_matrix.shape[1]
        if num_clusters != len(subgraph_list):
            raise ValueError(
                f"Dimension mismatch: M_matrix expects {num_clusters} clusters (columns), "
                f"but received {len(subgraph_list)} subgraphs."
            )

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