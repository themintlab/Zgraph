import torch
import torch.nn as nn
from . import functional as F
from .constants import DEFAULT_KB

class ZNet(nn.Module):
    def __init__(self, root_node):
        super().__init__()
        self.root_node = root_node

    def forward(self, inputs=None, temperature=293.15, **kwargs):
        """
        Args:
            inputs (dict, optional): Dictionary of inputs.
            temperature (float): Temperature in Kelvin.
            **kwargs: Alternative way to pass inputs (e.g., model(pressure=1.0)).
        """
        # 1. Consolidate inputs from dict and kwargs
        if inputs is None:
            inputs = kwargs
        elif kwargs:
            # Merge kwargs into inputs (non-destructive)
            inputs = {**inputs, **kwargs}
            
        if not inputs:
            raise ValueError("No inputs provided to ZNet.")

        # 2. Check for raw data (Convenience helper)
        # If the first value is NOT a tensor, assume we need to convert.
        if not isinstance(next(iter(inputs.values())), torch.Tensor):
             return self._convert_inputs(inputs, temperature)

        # 3. Fast Path (Standard Training)
        output = self.root_node(inputs, temperature)
        
        # Auto-Mix: If the output implies unmixed states (vector/tensor), 
        # collapse them to a scalar Free Energy.
        # Check if we have more dimensions than (Batch, 1) or if the last dim > 1.
        if output.ndim > 2 or (output.ndim == 2 and output.shape[1] > 1):
            output = F.softmin_energy(output, temperature=temperature, k_b=DEFAULT_KB)
            
        return output
    
    def _convert_inputs(self, raw_inputs, temperature):
        """Internal helper to sanitize inputs on the fly."""
        try:  
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device('cpu') # Default to CPU if no parameters exist

        clean_inputs = {}
        for k, v in raw_inputs.items():
            t = torch.tensor(v, dtype=torch.float32, device=device)
            # Ensure (Batch, 1) shape for scalars
            if t.ndim == 0: t = t.view(1, 1)
            elif t.ndim == 1: t = t.view(-1, 1)
            clean_inputs[k] = t
        #return self.root_node(clean_inputs, temperature)
        return self.forward(clean_inputs, temperature)

    def to_Helmholtz(self, inputs):
        """Converts an Energy Tensor to a Helmholtz Free Energy via the Legendre Transform."""
        inputs_grad = {
            k: v.clone().detach().requires_grad_(True) 
            for k, v in inputs.items()
        }

        # 2. Forward Pass: Calculate Grand Potential (Omega)
        Omega = self.forward(inputs_grad)

        # 3. Calculate Conjugate Variables (Particle Numbers N)
        # Thermodynamics: N_i = - d(Omega) / d(mu_i)
        # usage of torch.autograd.grad handles the differentiation
        # sum() is needed because .grad() only works on scalar outputs
        grads = torch.autograd.grad(
            outputs=Omega.sum(), 
            inputs=list(inputs_grad.values()), 
            create_graph=False
        )
        # Map gradients back to their keys ('mu_A', 'mu_B')
        N_counts = {k: -g for k, g in zip(inputs_grad.keys(), grads)}
        # 4. Perform Legendre Transform
        # F = Omega + sum(mu_i * N_i)
        diff_terms = sum(inputs_grad[k] * N_counts[k] for k in inputs_grad)
        Helmholtz_F = Omega + diff_terms
        
        return Helmholtz_F, N_counts