import torch
import torch.nn as nn
from ..algebra import _GraphAlgebra

class SourceNode(_GraphAlgebra, nn.Module):
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

    def forward(self, inputs):
        if self.index is None:
            raise RuntimeError("Graph was not compiled! Call .compile() on the root node.")
        #print(self.key, self.index)
        return inputs[..., self.index : self.index + 1]
        
    def __repr__(self):
        return f"SourceNode('{self.key}')"
   
class ConstantNode(_GraphAlgebra, nn.Module):
    def __init__(self, value):
        """
        Constant node: Returns a constant. 
                
        Args:
            value (float): The constant value to return.
        """
        super().__init__()

        #dtype=global_state_tensor.dtype, device=global_state_tensor.device
    
        if isinstance(value, (tuple, list)):
            # Learn from scratch (Initialize to 0)
            value_tensor = torch.zeros(*value)
        elif isinstance(value, (int, float)):
            # Scalar offset
            value_tensor = torch.tensor(float(value))
        elif isinstance(value, torch.Tensor):
            # Physics-Informed (Clone provided tensor)
            value_tensor = value.clone()
        else:
            raise ValueError("Value must be None, tuple (shape), Tensor, or scalar.")

        # Create Parameter and set trainability
        self.value = nn.Parameter(value_tensor)
        self.register_buffer("_value_tensor", value_tensor.detach().clone())
        #self.value.requires_grad = trainable

        # Keep a tensor-backed constant on the module so it is initialized once.
        
    
    def forward(self, inputs):
        return self.value
        
    def __repr__(self):
        return f"ConstantNode({self.value})"

