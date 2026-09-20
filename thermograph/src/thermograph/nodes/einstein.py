import jax.numpy as jnp
from zgraph.core import TemplateNode
from thermograph.core.constants import KB_R, SAFE_MIN_T

def _ground_state_kernel(signals, params):
    """
    Computes the 0 K structural energy of the lattice.
    Signals: [] (or [Volume] in the future)
    Params: [E_0]
    """
    E_0 = params[0]
    return E_0

class GroundStateNode:
    """
    Domain builder for the ground state energy E_0.
    """
    def __init__(self, E_0: float):
        self.E_0 = float(E_0)
        
    def compile_zgraph_engine(self) -> TemplateNode:
        return TemplateNode(
            kernel_fn=_ground_state_kernel,
            params=(self.E_0,),
            signal_indices=[]
        )

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

class EinsteinNode:
    """
    Domain builder for a pure 1-DOF quantum harmonic oscillator.
    """
    def __init__(self, Theta_E: float, T_index: int = 0):
        self.Theta_E = float(Theta_E)
        self.T_index = T_index
        
    def compile_zgraph_engine(self) -> TemplateNode:
        return TemplateNode(
            kernel_fn=_einstein_kernel,
            params=(self.Theta_E,),
            signal_indices=[self.T_index]
        )
