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

    #TODO: Note negative has been added to reflect application of chemical potential. Fix when adding reference potential?
    def forward(self, inputs, temperature=293.15):
        if self.key not in inputs:
             raise KeyError(f"SourceNode '{self.key}' input missing.")
        return -inputs[self.key]
        
    def __repr__(self):
        return f"SourceNode('{self.key}')"


class MixingNode(nn.Module):
    """
    Recursive Branch Node.
    Couples children nodes via an Enthalpy Matrix and marginalizes.
    """
    def __init__(self, 
                 sub_nodes, 
                 enthalpy=None, 
                 trainable=False, 
                 keep_dims=None, 
                 k_b=DEFAULT_KB):
        """
        Args:
            sub_nodes (list): List of sub-nodes.
            enthalpy (Tensor/tuple/None): 
                - None: Ideal Mixing (Trigger Shortcut).
                - Tensor: Fixed/Initial Enthalpy Matrix.
                - Tuple: Shape of Enthalpy Matrix to learn from scratch.
            trainable (bool): If True, enthalpy is a learnable parameter.
            keep_dims (tuple): Dimensions to preserve. None = Full Collapse.
        """
        super().__init__()
        self.k_b = k_b
        self.keep_dims = keep_dims
        
        # Ensure sub_nodes is always a list, even if a single node is provided
        if not isinstance(sub_nodes, (list, tuple)):
            sub_nodes = [sub_nodes]
        
        self.sub_nodes = nn.ModuleList(sub_nodes)

        # --- Enthalpy Initialization ---
        
        if enthalpy is None:
            self.enthalpy = None # Flag for Ideal Mixing Shortcut
        else:
            if isinstance(enthalpy, (tuple, list)):
                # Learn from scratch (Initialize to 0)
                enthalpy_tensor = torch.zeros(*enthalpy)
            elif isinstance(enthalpy, (int, float)):
                # Scalar offset
                enthalpy_tensor = torch.tensor(float(enthalpy))
            elif isinstance(enthalpy, torch.Tensor):
                # Physics-Informed (Clone provided tensor)
                enthalpy_tensor = enthalpy.clone()
            else:
                raise ValueError("Enthalpy must be None, tuple (shape), Tensor, or scalar.")

            # Create Parameter and set trainability
            self.enthalpy = nn.Parameter(enthalpy_tensor)
            self.enthalpy.requires_grad = trainable

    def forward(self, inputs, temperature=293.15):
        # 1. Gather Inputs (Recursive)
        child_outputs = [mod(inputs, temperature) for mod in self.sub_nodes]

        
        # --- SHORTCUT: Ideal Mixing Optimization ---
        # Condition: No Enthalpy AND Full Collapse (Scalar Output) 
        ## Also ensure all children are scalars (dimension 1), otherwise we need to marginalize.
        #if self.enthalpy is None and self.keep_dims is None):
        # Also ensure all children are scalars (dimension 1), otherwise we need to marginalize.
        # TODO: Something funny with this shortcut. Test with full marginalization and then return. 
        # if self.enthalpy is None and self.keep_dims is None and all(c.shape[1] == 1 for c in child_outputs):
        #     # We assume children return Scalars (Batch, 1) or compatible shapes
        #     # We simply sum them up. 
        #     # This avoids O(D^N) tensor construction entirely.
        #     # print("Ideal case detected. Implement later.")
        #     total_phi = child_outputs[0]
        #     for i in range(1, len(child_outputs)):
        #         total_phi = total_phi + child_outputs[i]
        #     return total_phi

        # --- STANDARD PATH: Interacting Systems ---
        
        # 2. Build Energy Tensor (Outer Sum)
        # Combines children into orthogonal dimensions: (Batch, D1, D2...)
        # If self.enthalpy is None, build_energy_tensor handles it as Ideal Mixing.
        total_energy = F.build_energy_tensor(child_outputs, self.enthalpy)
                  
        
        # num_children = len(child_outputs)
        
        # # Determine collapse dimensions (1..N correspond to children)
        # if self.keep_dims is None:
        #     dims_to_collapse = tuple(range(1, num_children + 1))
        # else:
        #     all_dims = set(range(num_children))
        #     keep_set = set(self.keep_dims)
        #     # Map logical index (0..N-1) to tensor index (1..N)
        #     dims_to_collapse = tuple(d + 1 for d in (all_dims - keep_set))

        # if not dims_to_collapse:
        #     # Flatten to (Batch, D_total) to maintain (Batch, D) contract
        #     return total_energy.view(total_energy.shape[0], -1)

        return F.softmin_energy(
            total_energy, 
            dim=dims_to_collapse, 
            temperature=temperature, 
            k_b=self.k_b
        )

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
        scalars = [child(inputs, temperature) for child in self.sub_nodes]
        
        # Concatenate along the last dimension to make a vector: (Batch, N_species)
        return torch.cat(scalars, dim=-1)