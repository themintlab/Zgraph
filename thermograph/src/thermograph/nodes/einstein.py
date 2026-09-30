import jax
import jax.numpy as jnp
import equinox as eqx
from zgraph.core.base import ZGraphNode, Ensemble
from thermograph.core.constants import KB_R, SAFE_MIN_T

class GroundStateNode(ZGraphNode):
    """
    Domain model for the ground state energy E_0.
    """
    E_0: jax.Array
    
    def __init__(self, E_0):
        self.E_0 = jnp.array(E_0, dtype=jnp.float32)
        
    def evaluate(self, signals):
        return self.E_0

class EinsteinNode(ZGraphNode):
    """
    Domain model for a pure 1-DOF quantum harmonic oscillator.
    """
    Theta_E: jax.Array
    T_index: int = eqx.field(static=True)
    
    def __init__(self, Theta_E, T_index: int = 0):
        self.Theta_E = jnp.array(Theta_E, dtype=jnp.float32)
        self.T_index = T_index
        
    def evaluate(self, signals):
        T = signals[self.T_index]
        T_safe = jnp.maximum(T, SAFE_MIN_T)
        
        # 1-DOF Zero-Point Energy
        ZPE = 0.5 * KB_R * self.Theta_E
        
        # 1-DOF Thermal Energy
        G_thermal = KB_R * T_safe * jnp.log(1.0 - jnp.exp(-self.Theta_E / T_safe))
        
        return ZPE + G_thermal

