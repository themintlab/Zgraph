# thermograph/layers.py
import torch
import torch.nn as nn
from . import functional as F
from .constants import DEFAULT_KB

class SourceNode(nn.Module):
    def __init__(self, key, index = None):
        """
        Leaf node: Retreives a potential from inputs and adds a reference energy.
        
        Physics: Phi_out = Input_Potential(key) + Reference_Enthalpy
        
        Args:
            key (str): The key to look up in the input dictionary.
            energy_init (float): Initial value for the reference energy parameter.
        """
        super().__init__()
        self.key = key
        self.index = index
    
    def bind_to_bus(self, global_registry):
        """Called during the compile step to lock in the routing."""
        if self.key not in global_registry:
            raise ValueError(f"Species '{self.key}' not found in registry.")
        self.index = global_registry[self.key]

    def forward(self, global_state_tensor, temperature = 293.15):
        if self.index is None:
            raise RuntimeError("Graph was not compiled! Call .compile() on the root node.")
        
        # Fast, zero-copy routing
        return -global_state_tensor[..., self.index : self.index + 1]
    
    # def forward(self, inputs, temperature=293.15):
    #     # if self.key not in inputs:
    #     #      raise KeyError(f"SourceNode '{self.key}' input missing.")
        
    #     return -inputs[self.key]
        
    def __repr__(self):
        return f"SourceNode('{self.key}')"



class StackNode(nn.Module):
    """
    Combines mutually exclusive components (Species) into a single state vector.
    Logic: [Phi_A, Phi_B] -> Tensor([Phi_A, Phi_B])
    """
    def __init__(self, sub_nodes):
        super().__init__()
        self.sub_nodes = nn.ModuleList(sub_nodes)
        
    def forward(self, inputs, temperature = 293.15):
        # Gather all scalar potentials: [ (Batch, 1), (Batch, 1), ... ]
        scalars = [child(inputs, temperature=temperature) for child in self.sub_nodes]
        
        # Concatenate along the last dimension to make a vector: (Batch, N_species)
        return torch.cat(scalars, dim=-1)