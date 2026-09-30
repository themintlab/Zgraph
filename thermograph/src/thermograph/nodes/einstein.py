import jax.numpy as jnp
import equinox as eqx
from zgraph.core.leaf_nodes import TemplateNode
from zgraph.core.base import ZGraphNode, Ensemble
from thermograph.core.constants import KB_R, SAFE_MIN_T

def _ground_state_kernel(signals, params):
    """
    Computes the 0 K structural energy of the lattice.
    Signals: [] (or [Volume] in the future)
    Params: [E_0]
    """
    E_0 = params[0]
    return E_0

class GroundStateNode(ZGraphNode):
    """
    Domain model for the ground state energy E_0.
    """
    engine: TemplateNode
    
    def __init__(self, E_0):
        self.engine = TemplateNode(
            kernel_fn=_ground_state_kernel,
            params=(E_0,),
            signal_indices=[]
        )
        
    def evaluate(self, signals):
        return self.engine(signals)

def _einstein_kernel(signals, params):
    """
    Computes the ZPE and thermal excitation for a single, normalized quantum harmonic oscillator (1 DOF).
    Signals: [T]
    Params: [Theta_E]
    """
    T = signals[0]
    Theta_E = params[0]
    
    T_safe = jnp.maximum(T, SAFE_MIN_T)
    
    # 1-DOF Zero-Point Energy
    ZPE = 0.5 * KB_R * Theta_E
    
    # 1-DOF Thermal Energy
    G_thermal = KB_R * T_safe * jnp.log(1.0 - jnp.exp(-Theta_E / T_safe))
    
    return ZPE + G_thermal

class EinsteinNode(ZGraphNode):
    """
    Domain model for a pure 1-DOF quantum harmonic oscillator.
    """
    engine: TemplateNode
    
    def __init__(self, Theta_E, T_index: int = 0):
        self.engine = TemplateNode(
            kernel_fn=_einstein_kernel,
            params=(Theta_E,),
            signal_indices=[T_index]
        )
        
    def evaluate(self, signals):
        return self.engine(signals)
