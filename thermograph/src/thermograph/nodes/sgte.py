import jax
import jax.numpy as jnp
import equinox as eqx
from zgraph.core import PiecewiseNode
from zgraph.core.base import ZGraphNode, Ensemble
from typing import List, Tuple

class SGTESingleNode(ZGraphNode):
    """
    Evaluates a single SGTE polynomial equation for a specific temperature range.
    Computes: G = a + bT + cT*ln(T) + dT^2 + eT^-1 + fT^3 + iT^7 + jT^-9
    """
    coeffs: jax.Array
    T_index: int = eqx.field(static=True)

    def __init__(self, coeffs: jax.Array, T_index: int = 0):
        self.coeffs = jnp.array(coeffs, dtype=jnp.float32)
        self.T_index = T_index

    def evaluate(self, signals):
        T = signals[self.T_index]
        a, b, c, d, e, f, i, j = self.coeffs
        T_safe = jnp.maximum(T, 1e-10)
        return a + b*T + c*T_safe*jnp.log(T_safe) + d*T**2 + e*T_safe**-1 + f*T**3 + i*T**7 + j*T_safe**-9

class SGTENode(ZGraphNode):
    """
    Domain-specific model for the SGTE piecewise polynomial thermodynamic equation.
    """
    engine: PiecewiseNode
    
    def __init__(self, piecewise_data: List[Tuple[float, List[float]]], T_index: int = 0):
        """
        Args:
            piecewise_data: A list of tuples, each representing a temperature range.
                Format: [(T_max, [a, b, c, d, e, f, i, j]), ...]
                The last tuple should have T_max as jnp.inf or the highest valid temperature.
            T_index: The index of Temperature in the signals array.
        """
        bounds = []
        subgraphs = []
        
        for i, (t_max, coeffs_arr) in enumerate(piecewise_data):
            if len(coeffs_arr) != 8:
                raise ValueError(f"SGTE kernel requires exactly 8 coefficients. Got {len(coeffs_arr)}.")
                
            coeffs = jnp.array(coeffs_arr, dtype=jnp.float32)
            subgraphs.append(SGTESingleNode(coeffs=coeffs, T_index=T_index))
            
            # The last T_max isn't a transition bound, it's just the end of the domain
            if i < len(piecewise_data) - 1:
                bounds.append(t_max)
                
        self.engine = PiecewiseNode(
            bounds=bounds,
            subgraph_list=subgraphs,
            signal_index=T_index
        )

    def evaluate(self, signals):
        return self.engine(signals)
