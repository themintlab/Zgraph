import torch
import torch.nn as nn
from typing import List, Union, Optional
from . import functional as F
from .leaf_nodes import ConstantNode

class FactorNode(nn.Module):
    # Minimum allowed beta value to prevent numerical instability in exponential calculations
    # Specifically prevents division by zero when calculating exp(-E / beta) in marginalization.
    _MIN_BETA = 1.2e-7

    def __init__(self, 
                 M_matrix: Union[torch.Tensor, List[List[float]]], 
                 subgraph_list: List[nn.Module], 
                 beta: Optional[Union[nn.Module, float, int, torch.Tensor]] = None):
        """
        Args:
            M_matrix (Union[torch.Tensor, List[List[float]]]): 2D matrix of shape (num_microstates, num_clusters).
            subgraph_list (list[nn.Module]): A list of subgraph modules. The order
                of modules in this list MUST match the order of the cluster
                columns in the M_matrix.
            beta (Optional[Union[nn.Module, float, int, torch.Tensor]]): A module that extracts or provides the 
                rationality/temperature parameter (e.g. SignalNode or ConstantNode).
                Defaults to ConstantNode(1.0).
        """
        super().__init__()

        if isinstance(M_matrix, list):
            M_matrix_tensor = torch.tensor(M_matrix, dtype=torch.get_default_dtype())
        else:
            M_matrix_tensor = M_matrix.clone().detach().to(dtype=torch.get_default_dtype())
        
        if M_matrix_tensor.ndim == 1:
            M_matrix_tensor = M_matrix_tensor.unsqueeze(0)
        
        if M_matrix_tensor.ndim != 2:
            raise ValueError(f"M_matrix must be a 2D tensor, got {M_matrix_tensor.ndim}D.")
            
        num_clusters = M_matrix_tensor.shape[1]
        if num_clusters != len(subgraph_list):
            raise ValueError(
                f"Dimension mismatch: M_matrix expects {num_clusters} clusters (columns), "
                f"but received {len(subgraph_list)} subgraphs."
            )

        self.register_buffer('M', M_matrix_tensor)
        
        if beta is None:
            self.beta = ConstantNode(1.0)
        elif isinstance(beta, (int, float, torch.Tensor)):
            self.beta = ConstantNode(beta)
        elif isinstance(beta, nn.Module):
            self.beta = beta
        else:
            raise TypeError("beta must be an nn.Module, a numeric value (int/float), or a scalar torch.Tensor.")
            
        self.subgraphs = nn.ModuleList(subgraph_list)

    def logits(self, signals: torch.Tensor) -> torch.Tensor:
        """
        The Logits / Uncollapsed Energy Vector.
        Evaluates the subgraphs to build the cluster inputs (w), and maps them to microstates via M.
        Returns a vector of size (num_microstates).
        """
        w = torch.stack([subgraph(signals) for subgraph in self.subgraphs], dim=-1)
        return torch.matmul(self.M, w)
    
    def probabilities(self, signals: torch.Tensor) -> torch.Tensor:
        """
        The Local Marginal Probabilities (SoftMin weights).
        Returns a normalized vector of size (num_microstates) representing the probability/weight of each state.
        """
        energy_landscape = self.logits(signals)
        beta_val = torch.clamp(self.beta(signals), min=self._MIN_BETA)
        
        # Softmax applies the exact exponential weighting used in the partition function
        return torch.softmax(energy_landscape / beta_val, dim=-1)

    def forward(self, local_signals: torch.Tensor) -> torch.Tensor:
        """
        The Strict Axiom: The Partition Function Collapse.
        Returns Rank 0 Tensor (Scalar).
        """
        energy_landscape = self.logits(local_signals)
        beta_val = torch.clamp(self.beta(local_signals), min=self._MIN_BETA)
        return F.marginalize(energy_landscape, beta_val)
        

class ProductNode(nn.Module):
    """Multiplies a list of subgraph outputs elementwise (tropical power)."""
    def __init__(self, subgraph_list: List[nn.Module]):
        super().__init__()
        if len(subgraph_list) == 0:
            raise ValueError("subgraph_list must contain at least one subgraph.")
        for subgraph in subgraph_list:
            if not isinstance(subgraph, nn.Module):
                raise TypeError("Each entry in subgraph_list must be an nn.Module.")
        self.subgraphs = nn.ModuleList(subgraph_list)

    def forward(self, local_signals: torch.Tensor) -> torch.Tensor:
        values = torch.stack([subgraph(local_signals) for subgraph in self.subgraphs], dim=0)
        return torch.prod(values, dim=0)
