import torch
import torch.nn as nn

class BaseLeafNode(nn.Module):
    """
    Abstract base class for all standard leaf nodes in ZGraph.
    Subclasses MUST explicitly define their mathematical forward() method
    and register their learnable parameters via nn.Parameter.
    """
    def __init__(self, signal_indices=None):
        super().__init__()
        indices_to_register = signal_indices if signal_indices is not None else []
        self.register_buffer(
            'signal_indices',
            torch.tensor(indices_to_register, dtype=torch.long)
        )

    def forward(self, local_signals):
        raise NotImplementedError("Subclasses must implement forward()")

class DynamicLeafNode(BaseLeafNode):
    """
    Evaluation-only node that accepts arbitrary pure PyTorch functions.
    Ideal for rapid prototyping. Should NOT be used for performance-critical training.
    """
    def __init__(self, energy_function, signal_indices=None, **constants):
        """
        Args:
            energy_function (callable): The pure math equation.
            signal_indices (list[int], optional): Hardcoded indices for early testing.
            **constants: Constant parameters passed to the function.
        """
        super().__init__(signal_indices)
        self.energy_function = energy_function
        
        # We don't register them as Parameters because this is evaluation-only.
        # But we do need to pass them to the function, so we register them as buffers.
        self.constant_keys = []
        for key, val in constants.items():
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32)
            self.register_buffer(key, val)
            self.constant_keys.append(key)

    def forward(self, full_local_signals):
        # Strictly vector input: (Channels,) -> scalar output: ()
        sliced_signals = full_local_signals[self.signal_indices]
        kwargs = {k: getattr(self, k) for k in self.constant_keys}
        return self.energy_function(sliced_signals, **kwargs)

class ConstantNode(nn.Module):
    """The simplest physics model: a trainable constant (or constants)."""
    def __init__(self, init_val=1.):
        super().__init__()
        
        if torch.is_tensor(init_val):
            tensor_val = init_val.clone().detach().to(dtype=torch.float32)
        else:
            tensor_val = torch.tensor(init_val, dtype=torch.float32)
            
        self.value = nn.Parameter(tensor_val)

    def forward(self, signals):
        return self.value
    
class SignalNode(BaseLeafNode):
    """A node that extracts specific signal indices from the input.
    Can extract a single scalar signal or a vector of signals."""
    def __init__(self, signal_indices):
        if isinstance(signal_indices, int):
            signal_indices = [signal_indices]
        super().__init__(signal_indices)

    def forward(self, local_signals):
        res = local_signals[self.signal_indices]
        if len(self.signal_indices) == 1:
            return res.squeeze(-1)
        return res

    def export_nodes(self):
        """
        Exports a list of individual scalar SignalNodes, one for each index.
        Replaces the old SignalNodes() factory function.
        Usage: T, mu1, mu2 = SignalNode([0, 1, 2]).export_nodes()
        """
        return [SignalNode(idx.item()) for idx in self.signal_indices]
