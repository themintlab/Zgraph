import torch
import torch.nn as nn
from ..core import functional as F

class SGTENode(nn.Module):
    """
    Evaluates a temperature-dependent SGTE polynomial.
    Can handle scalar bonds, vectors, or N-dimensional interaction matrices.
    """
    def __init__(self, temperature_node, coeffs : dict = None):
        super().__init__()
        self.temperature_node = temperature_node
        coeffs = coeffs or {}
        # We wrap inputs in torch.tensor and make them learnable Parameters.
        # If the user passes a 2x2 list for 'a', self.a becomes a 2x2 Parameter matrix!
        self.a = nn.Parameter(torch.tensor(coeffs.get('a', 0.0), dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(coeffs.get('b', 0.0), dtype=torch.float64))
        self.c = nn.Parameter(torch.tensor(coeffs.get('c', 0.0), dtype=torch.float64))
        self.d = nn.Parameter(torch.tensor(coeffs.get('d', 0.0), dtype=torch.float64))
        self.e = nn.Parameter(torch.tensor(coeffs.get('e', 0.0), dtype=torch.float64))
        self.f = nn.Parameter(torch.tensor(coeffs.get('f', 0.0), dtype=torch.float64))

    def forward(self, state_tensor):
        # Get temperature from child node
        T = self.temperature_node(state_tensor)
        return F.SGTE_polynomial(T, self.a, self.b, self.c, self.d, self.e, self.f)

