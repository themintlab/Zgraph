import jax.numpy as jnp
from zgraph.core.leaf_nodes import TemplateNode
from zgraph.core.base import ZGraphNode, Ensemble
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

class SGTENode(ZGraphNode):
    """
    Domain-specific model for the SGTE piecewise polynomial thermodynamic equation.
    """
    engine: TemplateNode
    
    def __init__(self, piecewise_data: List[Tuple[float, List[float]]], T_index: int = 0):
        """
        Args:
            piecewise_data: A list of tuples, each representing a temperature range.
                Format: [(T_max, [a, b, c, d, e, f, i, j]), ...]
                The last tuple should have T_max as jnp.inf or the highest valid temperature.
            T_index: The index of Temperature in the signals array.
        """
        bounds = []
        coeffs = []
        
        for i, (t_max, coeffs_arr) in enumerate(piecewise_data):
            if len(coeffs_arr) != 8:
                raise ValueError(f"SGTE kernel requires exactly 8 coefficients. Got {len(coeffs_arr)}.")
            coeffs.append(coeffs_arr)
            # The last T_max isn't a transition bound, it's just the end of the domain
            if i < len(piecewise_data) - 1:
                bounds.append(t_max)
                
        bounds_arr = jnp.array(bounds, dtype=jnp.float32)
        coeffs_matrix = jnp.array(coeffs, dtype=jnp.float32)
        
        params = (bounds_arr, coeffs_matrix)
        
        self.engine = TemplateNode(
            kernel_fn=_piecewise_sgte_kernel,
            params=params,
            signal_indices=[T_index]
        )

    def _evaluate(self, signals):
        return self.engine(signals)
