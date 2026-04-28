import torch
import torch.nn as nn
import warnings
from ..core import functional as F

class FactorNode(nn.Module):
    def __init__(self, M_matrix, microstate_labels, cluster_labels, subgraphs_dict):
        """
        Args:
            M_matrix (Tensor): 2D Tensor of shape (num_microstates, num_clusters)
            microstate_labels (list[str]): Names of the rows (Allowed configurations)
            column_labels (list[str]): Names of the columns (Geometric clusters)
            subgraphs_dict (dict): Dictionary mapping labels to nn.Modules
        """
        super().__init__()
        
        # 1. The Geometry Math (Registered for GPU acceleration)
        # Ensure M_matrix is stored as a float so model.to(dtype) conversions work
        self.register_buffer('M', M_matrix.to(torch.get_default_dtype()))
        
        # 2. The Introspection Metadata (Kept as plain Python objects)
        self.microstate_labels = microstate_labels
        self.column_labels = column_labels
        
        # 3. The Physics Subgraphs
        self.subgraphs = nn.ModuleDict(subgraphs_dict)

    def forward(self, local_signals):
        
        # Pythonic List Comprehension
        # We execute the subgraphs, enforce the trailing dimension, and collect them
        cluster_energies = [
            self.subgraphs[label](local_signals).view(local_signals.shape[:-1] + (1,))
            for label in self.cluster_labels
        ]
    
        # torch.cat handles the memory allocation natively in C++
        w_vector = torch.cat(cluster_energies, dim=-1)
    
        print(w_vector)
        T = local_signals[..., 0:1]

        beta = - 8.617e-5 * T
        return F.marginalize(self.M, w_vector, beta)