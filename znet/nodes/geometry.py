import torch
import torch.nn as nn
from ..algebra import ThermoAlgebra

class StackNode(ThermoAlgebra, nn.Module):
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

    def forward(self, inputs, temperature=293.15):
        # 1. Gather Inputs (Recursive)
        child_outputs = [mod(inputs, temperature) for mod in self.sub_nodes]
        num_subs = len(child_outputs)

        # Shortcut for no enthalpy: logsumexp(A + B) = logsumexp(A) + logsumexp(B)
        if self.enthalpy is None:
            phi = sum(torch.logsumexp(child, dim=-1) for child in child_outputs)
            # return phi.unsqueeze(-1)
        # Build the Grid via broadcasting
        # Reshape each tensor to broadcast across its designated sublattice dimension
        else: 
            aligned_tensors = [
                t.view(*t.shape[:-1], *[1]*i, t.shape[-1], *[1]*(num_subs - i - 1))
                for i, t in enumerate(child_outputs)
            ]
            grid = sum(aligned_tensors)
            
            # Apply Enthalpy and Collapse
            enthalpy = self.enthalpy(inputs, temperature)
            phi = torch.logsumexp(grid + enthalpy, dim=tuple(range(-num_subs, 0)))

        # Return as (*Batch, 1) to maintain the scalar potential format 
        # so it can be fed into a SystemNode (Competition)
        return phi.unsqueeze(-1)
        




        # If enthalpy is applied, proceed to build grid and apply enthalpy. 
        # # 2. Dynamic right-alignment - Builds grid for broadcasting
        # aligned_tensors = []
        # for i, tensor in enumerate(child_outputs):
        #     batch_shape = tensor.shape[:-1]
        #     D_i = tensor.shape[-1]
            
        #     # Create the trailing grid: [1, 1, ..., 1]
        #     trailing_shape = [1] * num_subs
        #     trailing_shape[i] = D_i # Slot this sublattice into its unique dimension
            
        #     # target_shape: (*Batch, 1, D_i, 1)
        #     target_shape = list(batch_shape) + trailing_shape
        #     aligned_tensors.append(tensor.view(*target_shape))
        
        # # 3. Build the Grid (Pure broadcasting addition)
        # # Result shape: (*Batch, D_0, D_1, ..., D_N)
        # grid = aligned_tensors[0]
        # for t in aligned_tensors[1:]:
        #     grid = grid + t
        
        # # 4. Apply Enthalpy (Scaling eV to Dimensionless)
        # enthalpy = self.enthalpy(inputs, temperature) # Should be (Batch, 1) or compatible shape
        # grid = grid + enthalpy

        # # 5. Collapse the Interaction Dimensions via LogSumExp
        # # We want to collapse the exact number of sublattices we added to the right side
        # dims_to_collapse = tuple(range(-num_subs, 0))
        # phi = torch.logsumexp(grid, dim=dims_to_collapse)