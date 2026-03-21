import torch
import torch.nn as nn
from ..core import functional as F
from ..algebra import ThermoAlgebra

class StackNode(ThermoAlgebra, nn.Module):
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

class MixingNode(ThermoAlgebra, nn.Module):
    """
    Recursive node that combines multiple child nodes into a single output potential,
        including enthalpic mixing.
    Logic: [Phi_A, Phi_B, ...] + Omega -> Phi_total
    Provides functions for legendre transforms
    """
    def __init__(self, 
                 sub_nodes, 
                 enthalpy=None,
                 scale = 1,  
                 ):
        """
        Args:
            sub_nodes (list): List of sub-nodes.
            enthalpy (node): Enthalpy node
        """
        super().__init__()
        self.scale = scale
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
            raise ValueError("MixingNode requires at least one sub-node.")

        if self.enthalpy is None:
            # Shortcut for no enthalpy: logsumexp(A + B) = logsumexp(A) + logsumexp(B)
            #reduced_children = [torch.logsumexp(child, dim=-1) for child in child_outputs]
            reduced_children = [F.collapse(child, 1) for child in child_outputs]
            phi = torch.stack(reduced_children, dim=0).sum(dim=0).unsqueeze(-1)
        else: 
            grid = F.outer_addition(child_outputs) 
            
            # Apply Enthalpy and Collapse
            enthalpy = self.enthalpy(inputs)
            phi = F.collapse(grid+enthalpy, self.num_sub_nodes) #= torch.logsumexp(grid + enthalpy, dim=tuple(range(-num_subs, 0)))

        # Return as (*Batch, 1) to maintain the scalar potential format 
        # so it can be fed into a SystemNode (Competition)
        return phi

class CollapseNode(ThermoAlgebra, nn.Module):
    """
    The Universal Collapser (Renormalization Node).
    It takes an internal structural graph (a Stack or a Mix) and traces it out.
    """
    def __init__(self, sub_node):
                 #sub_nodes, scale = 1, ):
        """
        Args:
            sub_nodes (list): List of sub-nodes.
        """
        super().__init__()
        #self.scale = scale
        # Ensure sub_nodes is always a list, even if a single node is provided
        # if not isinstance(sub_nodes, (list, tuple)):
        #     sub_nodes = [sub_nodes]
        
        # #TODO: Check that sub_nodes are all nodes. 

        # self.sub_nodes = nn.ModuleList(sub_nodes)    
        self.sub_node = sub_node

    def forward(self, inputs):
        # child_outputs = [mod(inputs) for mod in self.sub_nodes]
        # if not child_outputs:
        #     raise ValueError("Collapse requires at least one sub-node.")
        state_tensor = self.sub_node(inputs)

        batch_rank = inputs.dim() - 1 
        landscape_rank = state_tensor.dim()
        num_state_dims = landscape_rank - batch_rank

        return F.collapse(state_tensor, num_state_dims)