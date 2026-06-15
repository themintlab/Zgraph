import torch
from torch.func import grad_and_value

def legendre_transform(fcn, idx):
    """
    Returns a pure function that evaluates the partial Legendre transform.

    Args:
        fcn: A callable taking a 1D tensor and returning a scalar energy.
        idx: The indices of the primal variables to be transformed.

    Returns:
        A function that takes primal variables and returns the 
        transformed energy and the updated state variables.
    """
    idx = torch.atleast_1d(torch.as_tensor(idx, dtype=torch.long))

    def legendre_step(x):
        grads, energy = grad_and_value(fcn)(x)
        
        y_partial = grads[idx]
        x_partial = x[idx]
        
        transformed_energy = energy - torch.dot(y_partial, x_partial)

        new_x_stack = x.clone()
        new_x_stack[idx] = y_partial
        return transformed_energy, new_x_stack

    return legendre_step