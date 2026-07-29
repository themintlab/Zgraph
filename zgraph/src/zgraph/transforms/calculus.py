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




import torch
import torch.nn as nn
from torch.func import functional_call, value_and_grad

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
            'idx', 
            torch.atleast_1d(torch.as_tensor(transform_indices, dtype=torch.long))
        )

    def forward(self, primal_x: torch.Tensor):
        # primal_x must be a 1D tensor (e.g., shape [3] for [T, P, mu])
        
        # 3. Dynamically extract state.
        # We grab both parameters AND buffers (like fixed constants) to ensure 
        # the functional call has the complete context of the base model.
        state = {
            **dict(self.base_model.named_parameters()), 
            **dict(self.base_model.named_buffers())
        }
        
        # 4. Define the pure function mapping for torch.func
        def pure_fwd(params_and_buffers, x_single):
            return functional_call(self.base_model, params_and_buffers, (x_single,))
        
        # 5. Safely compute the Math Gradient
        # argnums=1 tells PyTorch to take the derivative w.r.t x_single, not the parameters.
        phi, full_grad = value_and_grad(pure_fwd, argnums=1)(state, primal_x)
        
        # 6. Execute the Legendre Math
        # psi = phi - SUM(x_i * y_i)
        x_I = primal_x[self.idx]
        y_I = full_grad[self.idx]
        psi = phi - torch.dot(x_I, y_I)
        
        # 7. Construct and return the dual coordinate vector alongside the energy
        transformed_x = primal_x.clone()
        transformed_x[self.idx] = y_I
        
        return psi, transformed_x