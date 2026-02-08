# thermograph/layers.py
import torch
import torch.nn as nn
from . import functional as F
from .constants import DEFAULT_KB

class SourceNode(nn.Module):
    def __init__(self, key, shape=(1,), energy_init=0.0):
        """
        Leaf node: Retreives a potential from inputs and adds a reference energy.
        
        Physics: Phi_out = Input_Potential(key) + Reference_Enthalpy
        
        Args:
            key (str): The key to look up in the input dictionary.
            shape (tuple): Shape of the reference energy tensor (e.g., (1,) for scalar).
            energy_init (float): Initial value for the reference energy parameter.
        """
        super().__init__()
        self.key = key
        
        # The "Enthalpic Interaction" at the leaf level is the Reference Energy (E_ref)
        # We make it a parameter so it can be learned or fixed.
        self.reference_energy = nn.Parameter(torch.full(shape, energy_init))

    def forward(self, inputs, temperature=None):
        # 1. Zero-Cost Lookup
        if self.key not in inputs:
             raise KeyError(f"SourceNode '{self.key}' input missing.")
        
        phi_in = inputs[self.key]
        
        # 2. Vectorized Addition (Bias)
        # Broadcasting handles (Batch, D) + (D,)
        return phi_in + self.reference_energy



class MixingNode(nn.Module):
    def __init__(self, children, interaction_shape=None, keep_dims=None, k_b=DEFAULT_KB):
        """
        Branch node: Mixes children nodes + Interaction + SoftMin.
        
        Args:
            children (list of nn.Module): Can be SourceNodes or other MixingNodes.
            interaction_shape (tuple): Shape of the interaction matrix (D1, D2...).
            keep_dims (tuple): Dimensions to preserve (None = collapse all).
        """
        super().__init__()
        self.k_b = k_b
        self.keep_dims = keep_dims
        
        # PURE ModuleList - PyTorch automatically tracks these parameters
        self.children = nn.ModuleList(children)

        # Interaction Enthalpy (Omega)
        if interaction_shape is not None:
            self.enthalpy = nn.Parameter(torch.zeros(*interaction_shape))
        else:
            self.enthalpy = None

    def forward(self, inputs, temperature):
        # 1. Pure Recursive Gather
        # No parsing. Just execution.
        child_phis = [child(inputs, temperature) for child in self.children]
        
        # 2. Build Tensor (Outer Sum)
        total_energy = F.build_energy_tensor(child_phis)
        
        # 3. Add Interaction
        if self.enthalpy is not None:
            total_energy = total_energy + self.enthalpy
            
        # 4. Marginalize
        num_children = len(child_phis)
        
        if self.keep_dims is None:
            dims_to_collapse = tuple(range(1, num_children + 1))
            keep_flag = False
        else:
            # Logic to calculate collapse dims...
            all_dims = set(range(num_children))
            keep_set = set(self.keep_dims)
            dims_to_collapse = tuple(d + 1 for d in (all_dims - keep_set))
            keep_flag = False

        if not dims_to_collapse:
            return total_energy

        return F.softmin_energy(
            total_energy, 
            dim=dims_to_collapse, 
            temperature=temperature, 
            k_b=self.k_b,
            keepdim=keep_flag
        )
    


***
Source nodes integrate reference potentials directly or as enthalpic contributions after? 
Mixing nodes / enthalpic interaction tensor. 
**