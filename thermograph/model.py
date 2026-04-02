import torch
import torch.nn as nn
from znet import ZNet
from .constants import DEFAULT_KB


class ZNetThermo(ZNet):
    """Thermodynamic wrapper around ZNet graph with temperature support."""
    
    def forward(self, inputs=None, temperature=293.15, **kwargs):
        """
        Args:
            inputs (dict or Tensor, optional): Dictionary of inputs or tensor.
            temperature (float): Temperature in Kelvin (passed to temperature-aware nodes).
            **kwargs: Alternative way to pass inputs (e.g., model(pressure=1.0)).
        """
        # 1. Handle tensor input directly (compiled graph case)
        if isinstance(inputs, torch.Tensor):
            # For compiled graphs, just call the node - temperature is handled internally
            # via temperature nodes in the graph if needed
            return self.root_node(inputs)
        
        # 2. Consolidate inputs from dict and kwargs
        if inputs is None:
            inputs = kwargs
        elif kwargs:
            # Merge kwargs into inputs (non-destructive)
            inputs = {**inputs, **kwargs}
            
        if not inputs:
            raise ValueError("No inputs provided to ZNetThermo.")

        # 3. Check for raw data (Convenience helper)
        # If the first value is NOT a tensor, assume we need to convert.
        if not isinstance(next(iter(inputs.values())), torch.Tensor):
             return self._convert_inputs(inputs, temperature)

        # 4. Fast Path (Standard Training)
        # Note: temperature is passed to nodes via input if needed (e.g., ConstantNode)
        output = self.root_node(inputs)
            
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
        return self.forward(clean_inputs, temperature)

    def to_Helmholtz(self, inputs):
        """Converts a Grand Potential to Helmholtz Free Energy via the Legendre Transform."""
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
