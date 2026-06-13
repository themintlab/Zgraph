import torch
import torch.nn as nn
from ..core import functional as F

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
        self.register_buffer('beta_index', torch.tensor(beta_index, dtype=torch.long))
        self.register_buffer('beta_factor', torch.tensor(beta_factor, dtype=torch.float32))    
        self.subgraphs = nn.ModuleList(subgraph_list)

    def forward(self, local_signals):
        # Execute each subgraph and ensure its output has a trailing dimension for concatenation.
        # The order of subgraphs is critical and must match the columns of M_matrix.
        # This is a strict-tree architecture - leaves cannot be shared. In the future, the w_vector
        # might be best passed into the forward call by an orchestrator to facilitate sharing
        cluster_energies = [
            subgraph(local_signals).view(local_signals.shape[:-1] + (1,))
            for subgraph in self.subgraphs
        ]

        # torch.cat handles the memory allocation natively in C++
        energy_vector = torch.cat(cluster_energies, dim=-1)
    
        beta = self.beta_factor * local_signals[..., self.beta_index].unsqueeze(-1)
  
        return F.marginalize(self.M, energy_vector, beta)


class TProdNode(nn.Module):
    def __init__(self, subgraph_list):
        """
        Args:
            subgraph_list (list[nn.Module]): A list of subgraph modules whose outputs will be summed.
        """
        super().__init__()
        if not subgraph_list:
            raise ValueError("subgraph_list cannot be empty.")
            
        self.subgraphs = nn.ModuleList(subgraph_list)

    def forward(self, local_signals):
        outputs = [subgraph(local_signals) for subgraph in self.subgraphs]
        return sum(outputs[1:], start=outputs[0])


class TPowNode(nn.Module):
    """Multiplies the output of a subgraph by a constant factor."""
    def __init__(self, subgraph, factor=-1.0):
        super().__init__()
        self.subgraph = subgraph
        
        if torch.is_tensor(factor):
            tensor_val = factor.clone().detach().to(dtype=torch.float32)
        else:
            tensor_val = torch.tensor(factor, dtype=torch.float32)
            
        self.register_buffer('factor', tensor_val)

    def forward(self, local_signals):
        return self.factor * self.subgraph(local_signals)
