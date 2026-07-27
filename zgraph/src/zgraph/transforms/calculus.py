import torch
import torch.nn as nn
from torch.func import grad_and_value

class LegendreStep(nn.Module):
    def __init__(self, fcn, idx):
        super().__init__()
        self.fcn = fcn
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx, dtype=torch.long)
        self.register_buffer('idx_tensor', torch.atleast_1d(idx))
        self.gv_fcn = grad_and_value(fcn)

    def forward(self, x):
        grads, energy = self.gv_fcn(x)
        idx_dev = self.idx_tensor.to(device=x.device)
        y_partial = grads[idx_dev]
        x_partial = x[idx_dev]
        
        transformed_energy = energy - torch.dot(y_partial, x_partial)
        new_x_stack = x.scatter(0, idx_dev, y_partial)
        return transformed_energy, new_x_stack


def legendre_transform(fcn, idx):
    """
    Returns a Module (or container of Modules) evaluating partial Legendre transform(s).

    Args:
        fcn: A callable (or list/tuple/dict of callables) taking a 1D tensor and returning a scalar energy.
        idx: The indices of the primal variables to be transformed.

    Returns:
        LegendreStep (or list/tuple/dict of LegendreStep) taking primal variables and returning 
        the transformed energy and the updated state variables.
    """
    if isinstance(fcn, (list, tuple)):
        funcs = [legendre_transform(f, idx) for f in fcn]
        return type(fcn)(funcs)

    if isinstance(fcn, dict):
        return {key: legendre_transform(value, idx) for key, value in fcn.items()}

    return LegendreStep(fcn, idx)
