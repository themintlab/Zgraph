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
