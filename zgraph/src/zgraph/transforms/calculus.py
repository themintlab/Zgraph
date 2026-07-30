import torch
import torch.nn as nn
from torch.func import grad_and_value


def legendre_transform(module, idx):
    """
    Returns a Module (or container of Modules) evaluating partial Legendre transform(s).

    Args:
        module: A nn.Module (or list/tuple of nn.Modules) taking a 1D tensor and returning a scalar energy.
        idx: The indices of the primal variables to be transformed.

    Returns:
        LegendreTransformModule (or list/tuple of LegendreTransformModule) taking primal variables and returning
        the transformed energy and the updated state variables.
    """
    if isinstance(module, (list, tuple)):
        modules = [legendre_transform(m, idx) for m in module]
        return type(module)(modules)

    if isinstance(module, nn.Module):
        return LegendreTransformModule(module, idx)

    raise TypeError(
        "legendre_transform expects an nn.Module or a list/tuple of nn.Module instances."
    )

class LegendreTransformModule(nn.Module):
    """
    Transforms a base thermodynamic module via a Legendre transform on specified indices.
    Designed strictly for a single unbatched sample (1D tensor).
    """
    def __init__(self, base_model: nn.Module, transform_indices: list):
        super().__init__()
        # 1. Store the base engine. 
        # This automatically exposes base_model's parameters to optimizers.
        self.base_model = base_model
        
        # 2. Store indices as a registered buffer.
        # This ensures the indices automatically move to the GPU if the model is moved,
        # preventing device mismatch errors during compiled execution.
        self.register_buffer(
            'idx_tensor',
            torch.atleast_1d(torch.as_tensor(transform_indices, dtype=torch.long))
        )

    def forward(self, primal_x: torch.Tensor):
        # primal_x must be a 1D tensor (e.g., shape [3] for [T, P, mu])

        # 3. Compute gradient w.r.t. input in torch.func style.
        full_grad, phi = grad_and_value(self.base_model)(primal_x)
        
        # 4. Execute the Legendre Math
        # psi = phi - SUM(x_i * y_i)
        idx_dev = self.get_buffer("idx_tensor").to(device=primal_x.device)
        x_I = primal_x[idx_dev]
        y_I = full_grad[idx_dev]
        psi = phi - torch.dot(x_I, y_I)
        
        # 5. Construct and return the dual coordinate vector alongside the energy
        transformed_x = primal_x.clone()
        transformed_x[idx_dev] = y_I
        
        return psi, transformed_x