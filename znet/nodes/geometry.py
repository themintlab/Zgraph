import torch
import torch.nn as nn
import warnings
from ..core import functional as F
from ..algebra import _GraphAlgebra

class OrNode(_GraphAlgebra, nn.Module):
    """
    Combines mutually exclusive components (Species) into a single state vector.
    Logic: [Phi_A, Phi_B] -> Tensor([Phi_A, Phi_B])
    """
    def __init__(self, sub_nodes):
        super().__init__()
        self.sub_nodes = nn.ModuleList(sub_nodes)
        
    def forward(self, inputs):
        # Gather all scalar potentials: [ (Batch, 1), (Batch, 1), ... ]
        scalars = [child(inputs) for child in self.sub_nodes]
        
        # Concatenate along the last dimension to make a vector: (Batch, N_species)
        return torch.cat(scalars, dim=-1)

class AndNode(_GraphAlgebra, nn.Module):
    """
    Recursive node that combines multiple child nodes with optional interaction.
    Logic: [Output_A, Output_B, ...] + Interaction -> Combined_Output
    """
    def __init__(self, 
                 sub_nodes, 
                 enthalpy=None,  
                 ):
        """
        Args:
            sub_nodes (list): List of sub-nodes.
            enthalpy (node, optional): Optional interaction term callable node.
        """
        super().__init__()
        if enthalpy is None:
            warnings.warn(
                "AndNode initialized with enthalpy=None; Consider collapsing sub_nodes first.",
                RuntimeWarning,
            )
        self.enthalpy = enthalpy
        
        # Ensure sub_nodes is always a list, even if a single node is provided
        if not isinstance(sub_nodes, (list, tuple)):
            sub_nodes = [sub_nodes]
        
        self.sub_nodes = nn.ModuleList(sub_nodes)     
        self.num_sub_nodes = len(sub_nodes)   

    def forward(self, inputs):
        # 1. Gather Inputs (Recursive)
        child_outputs = [mod(inputs) for mod in self.sub_nodes]
        if not child_outputs:
            raise ValueError("AndNode requires at least one sub-node.")

        grid = F.outer_addition(child_outputs) 
        # Apply interaction term only when present.
        if self.enthalpy is None:
            return grid
        return grid + self.enthalpy(inputs)

class CollapseNode(_GraphAlgebra, nn.Module):
    """
    Renormalization / dimensionality reduction node.
    Traces out the internal structure using logsumexp.
    """
    def __init__(self, sub_node, scale = 1.0):
        """
        Args:
            sub_node: Node to collapse/renormalize.
            scale: Softness parameter. scale=1 uses the fastest path.
        """
        super().__init__()
        self.sub_node = sub_node
        self.scale = scale

    def forward(self, inputs):
        state_tensor = self.sub_node(inputs)

        batch_rank = inputs.dim() - 1 
        landscape_rank = state_tensor.dim()
        num_state_dims = landscape_rank - batch_rank
        return F.collapse(state_tensor, num_state_dims, scale=self.scale)