# thermograph/layers.py
import torch
import torch.nn as nn
from . import functional as F
from .constants import DEFAULT_KB

class SourceNode(nn.Module):
    def __init__(self, key):
        """
        Leaf node: Retreives a potential from inputs and adds a reference energy.
        
        Physics: Phi_out = Input_Potential(key) + Reference_Enthalpy
        
        Args:
            key (str): The key to look up in the input dictionary.
            energy_init (float): Initial value for the reference energy parameter.
        """
        super().__init__()
        self.key = key

    def forward(self, inputs, temperature=None):
        if self.key not in inputs:
             raise KeyError(f"SourceNode '{self.key}' input missing.")
        return inputs[self.key]
        
    def __repr__(self):
        return f"SourceNode('{self.key}')"


class MixingNode(nn.Module):
    """
    Recursive Branch Node.
    Couples children nodes via an Enthalpy Matrix and marginalizes.
    """
    def __init__(self, 
                 children, 
                 interaction=None, 
                 trainable=False, 
                 keep_dims=None, 
                 k_b=DEFAULT_KB):
        """
        Args:
            children (list): List of child Modules.
            interaction (Tensor/tuple/None): 
                - None: Ideal Mixing (Trigger Shortcut).
                - Tensor: Fixed/Initial Enthalpy Matrix.
                - Tuple: Shape of Enthalpy Matrix to learn from scratch.
            trainable (bool): If True, enthalpy is a learnable parameter.
            keep_dims (tuple): Dimensions to preserve. None = Full Collapse.
        """
        super().__init__()
        self.k_b = k_b
        self.keep_dims = keep_dims
        self.children = nn.ModuleList(children)

        # --- Enthalpy Initialization ---
        if interaction is None:
            self.enthalpy = None # Flag for Ideal Mixing Shortcut
            
        elif isinstance(interaction, (tuple, list)):
            # Learn from scratch (Initialize to 0)
            self.enthalpy = nn.Parameter(torch.zeros(*interaction))
            
        elif isinstance(interaction, torch.Tensor):
            # Physics-Informed (Clone provided tensor)
            self.enthalpy = nn.Parameter(interaction.clone())
            
        else:
            raise ValueError("Interaction must be None, tuple (shape), or Tensor.")

        # --- Freeze/Thaw Physics ---
        if self.enthalpy is not None:
            self.enthalpy.requires_grad = trainable

    def forward(self, inputs, temperature):
        # 1. Gather Inputs (Recursive)
        child_outputs = [child(inputs, temperature) for child in self.children]
        
        # --- SHORTCUT: Ideal Mixing Optimization ---
        # Condition: No Enthalpy AND Full Collapse (Scalar Output)
        # If we are just mixing independent systems, Phi_total = Sum(Phi_children)
        if self.enthalpy is None and self.keep_dims is None:
            # We assume children return Scalars (Batch, 1) or compatible shapes
            # We simply sum them up. 
            # This avoids O(D^N) tensor construction entirely.
            print("Ideal caImplement laterse detected. ")
            # total_phi = child_outputs[0]
            # for i in range(1, len(child_outputs)):
            #     total_phi = total_phi + child_outputs[i]
            # return total_phi

        # --- STANDARD PATH: Interacting Systems ---
        
        # 2. Build Energy Tensor (Outer Sum)
        # Combines children into orthogonal dimensions: (Batch, D1, D2...)
        total_energy = F.build_energy_tensor(child_outputs)
        
        # 3. Add Interaction Enthalpy (Coupling)
        if self.enthalpy is not None:
            total_energy = total_energy + self.enthalpy
            
        # 4. Marginalize (SoftMin)
        if not child_outputs: return total_energy # Edge case: Pure enthalpy node
        
        num_children = len(child_outputs)
        
        # Determine collapse dimensions (1..N correspond to children)
        if self.keep_dims is None:
            dims_to_collapse = tuple(range(1, num_children + 1))
            keep_flag = False
        else:
            all_dims = set(range(num_children))
            keep_set = set(self.keep_dims)
            # Map logical index (0..N-1) to tensor index (1..N)
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

# ***
# Source nodes integrate reference potentials directly or as enthalpic contributions after? 
# Mixing nodes / enthalpic interaction tensor. 
# **