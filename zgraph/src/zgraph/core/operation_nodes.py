import torch
import torch.nn as nn
from . import functional as F
from .leaf_nodes import ConstantNode

class FactorNode(nn.Module):
    def __init__(self, M_matrix, subgraph_list, beta=None):
        """
        Args:
            M_matrix (torch.Tensor): 2D Tensor of shape (num_microstates, num_clusters).
            subgraph_list (list[nn.Module]): A list of subgraph modules. The order
                of modules in this list MUST match the order of the cluster
                columns in the M_matrix.
            beta (nn.Module, optional): A module that extracts or provides the 
                rationality/temperature parameter (e.g. SignalNode or ConstantNode).
                Defaults to ConstantNode(1.0).
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
        
        if beta is None:
            beta = ConstantNode(1.0)
        elif isinstance(beta, (int, float, torch.Tensor)):
            beta = ConstantNode(beta)
        elif not isinstance(beta, nn.Module):
            raise TypeError("beta must be an nn.Module, a numeric value (int/float), or a scalar torch.Tensor.")
            
        self.beta = beta
        self.subgraphs = nn.ModuleList(subgraph_list)

    def forward(self, local_signals):
        energy_vector = torch.stack([subgraph(local_signals) for subgraph in self.subgraphs])
        beta = torch.clamp(self.beta(local_signals), min = 1.2e-7)
        return F.marginalize(self.M, energy_vector, beta)


class ProductNode(nn.Module):
    """Multiplies a list of subgraph outputs elementwise (tropical power)."""
    def __init__(self, subgraph_list):
        super().__init__()
        if len(subgraph_list) == 0:
            raise ValueError("subgraph_list must contain at least one subgraph.")
        for subgraph in subgraph_list:
            if not isinstance(subgraph, nn.Module):
                raise TypeError("Each entry in subgraph_list must be an nn.Module.")
        self.subgraphs = nn.ModuleList(subgraph_list)

    def forward(self, local_signals):
        values = torch.stack([subgraph(local_signals) for subgraph in self.subgraphs], dim=0)
        return torch.prod(values, dim=0)
