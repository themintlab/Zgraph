import torch
import torch.nn as nn


class ZNet(nn.Module):
    def __init__(self, root_node):
        super().__init__()
        self.root_node = root_node

    def forward(self, inputs=None, **kwargs):
        """
        Args:
            inputs (dict or Tensor, optional): Dictionary of inputs or tensor.
            **kwargs: Alternative way to pass inputs (e.g., model(pressure=1.0)).
        """
        # 1. Handle tensor input directly (compiled graph case)
        if isinstance(inputs, torch.Tensor):
            return self.root_node(inputs)
        
        # 2. Consolidate inputs from dict and kwargs
        if inputs is None:
            inputs = kwargs
        elif kwargs:
            # Merge kwargs into inputs (non-destructive)
            inputs = {**inputs, **kwargs}
            
        if not inputs:
            raise ValueError("No inputs provided to ZNet.")

        # 3. Check for raw data (Convenience helper)
        # If the first value is NOT a tensor, assume we need to convert.
        if not isinstance(next(iter(inputs.values())), torch.Tensor):
             return self._convert_inputs(inputs)

        # 4. Fast Path (Standard Training)
        output = self.root_node(inputs)
        
            
        return output
    
    def _convert_inputs(self, raw_inputs):
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
        return self.forward(clean_inputs)