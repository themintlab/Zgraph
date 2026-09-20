import jax.numpy as jnp
from zgraph.core.leaf_nodes import TemplateNode
from typing import List, Tuple

def _piecewise_sgte_kernel(signals, params):
    """
    Pure JAX mathematical kernel for a piecewise SGTE polynomial equation.
    Signals: [T]
    Params: (bounds, coeffs_matrix)
        bounds: Array of shape (N-1,) containing transition temperatures.
        coeffs_matrix: Array of shape (N, 8) containing coefficients for each range.
    Computes: G = a + bT + cT*ln(T) + dT^2 + eT^-1 + fT^3 + iT^7 + jT^-9
    """
    T = signals[0]
    bounds, coeffs_matrix = params
    
    # Efficiently find the index of the temperature range without branching
    idx = jnp.searchsorted(bounds, T, side='right')
    
    # Extract the 8 coefficients for the active range using dynamic slicing
    a, b, c, d, e, f, i, j = coeffs_matrix[idx]
    
    # We pad the polynomial up to 8 parameters to support standard Unary50 elements like Cu
    # Prevent log(T) and T^-n from returning nan/inf at absolute zero.
    T_safe = jnp.maximum(T, 1e-10)
    return a + b*T + c*T_safe*jnp.log(T_safe) + d*T**2 + e*T_safe**-1 + f*T**3 + i*T**7 + j*T_safe**-9

class SGTENode:
    """
    Domain-specific builder for the SGTE piecewise polynomial thermodynamic model.
    Compiles into a pure zgraph execution node.
    """
    def __init__(self, piecewise_data: List[Tuple[float, List[float]]], T_index: int = 0):
        """
        Args:
            piecewise_data: A list of tuples, each representing a temperature range.
                Format: [(T_max, [a, b, c, d, e, f, i, j]), ...]
                The last tuple should have T_max as jnp.inf or the highest valid temperature.
            T_index: The index of Temperature in the signals array.
        """
        self.bounds = []
        self.coeffs = []
        
        for i, (t_max, coeffs) in enumerate(piecewise_data):
            if len(coeffs) != 8:
                raise ValueError(f"SGTE kernel requires exactly 8 coefficients. Got {len(coeffs)}.")
            self.coeffs.append(coeffs)
            # The last T_max isn't a transition bound, it's just the end of the domain
            if i < len(piecewise_data) - 1:
                self.bounds.append(t_max)
                
        self.T_index = T_index

    def compile_zgraph_engine(self) -> TemplateNode:
        """
        Compiles the piecewise SGTE parameters into a purely numerical zgraph TemplateNode.
        """
        bounds_arr = jnp.array(self.bounds, dtype=jnp.float32)
        coeffs_matrix = jnp.array(self.coeffs, dtype=jnp.float32)
        
        # params must be a tuple to satisfy zgraph's no-dictionary rule
        params = (bounds_arr, coeffs_matrix)
        
        return TemplateNode(
            kernel_fn=_piecewise_sgte_kernel,
            params=params,
            signal_indices=[self.T_index]
        )

