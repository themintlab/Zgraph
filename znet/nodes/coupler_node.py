import torch
import torch.nn as nn

class PotentialCoupler(nn.Module):
    def __init__(self, reference_node, signal_index):
        """
        Args:
            reference_node: A node that describes mu0
            signal_index: The index in local_signals that describes the corresponding mu
        """
        super().__init__()
        
        self.reference_node = reference_node
        self.register_buffer('signal_index', torch.tensor(signal_index, dtype=torch.long))

    def forward(self, local_signals):
        mu0 = self.reference_node(local_signals)
        mu = local_signals[..., self.signal_index]
        return mu0-mu 