import torch
import torch.nn as nn
from ..algebra import ThermoAlgebra

class SourceNode(ThermoAlgebra, nn.Module):
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

    def forward(self, global_state_tensor):
        if self.index is None:
            raise RuntimeError("Graph was not compiled! Call .compile() on the root node.")
        #print(self.key, self.index)
        return global_state_tensor[..., self.index : self.index + 1]
        
    def __repr__(self):
        return f"SourceNode('{self.key}')"
   
class ConstantNode(ThermoAlgebra, nn.Module):
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

class PolynomialEnthalpy(nn.Module):
    """
    Evaluates a temperature-dependent SGTE polynomial.
    Can handle scalar bonds, vectors, or N-dimensional interaction matrices.
    """
    def __init__(self, a=0.0, b=0.0, c=0.0, d=0.0, e=0.0, f=0.0):
        super().__init__()
        # We wrap inputs in torch.tensor and make them learnable Parameters.
        # If the user passes a 2x2 list for 'a', self.a becomes a 2x2 Parameter matrix!
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float64))
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float64))
        self.d = nn.Parameter(torch.tensor(d, dtype=torch.float64))
        self.e = nn.Parameter(torch.tensor(e, dtype=torch.float64))
        self.f = nn.Parameter(torch.tensor(f, dtype=torch.float64))

    def forward(self, state_tensor, T):
        # PyTorch Safety Measure: log(T) will return NaN if T <= 0.
        # During ML optimization, T might slightly dip below 0. 
        # We clamp it to a tiny positive number to prevent gradient explosions.
        T_safe = torch.clamp(T, min=1e-6)
        
        # The SGTE Polynomial
        # Because self.a, self.b etc. are broadcastable tensors, 
        # this single line evaluates the polynomial for every bond in the matrix
        # across the entire Temperature batch simultaneously!
        H_grid = (
            self.a 
            + self.b * T_safe 
            + self.c * T_safe * torch.log(T_safe) 
            + self.d * (T_safe ** 2) 
            + self.e * (T_safe ** 3) 
            + self.f * (1.0 / T_safe)
        )
        
        return H_grid