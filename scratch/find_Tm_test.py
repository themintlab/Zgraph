import os, sys
root_dir = os.path.abspath('.')
sys.path.insert(0, os.path.join(root_dir, 'zgraph', 'src'))
sys.path.insert(0, os.path.join(root_dir, 'thermograph', 'src'))
sys.path.insert(0, root_dir)

import jax
import jax.numpy as jnp
from zgraph import *
from thermograph.nodes.sgte import SGTENode
from thermograph.nodes.einstein import GroundStateNode, EinsteinNode
from external_data.si_ge_sgte import GLIQSI

T_grid = jnp.linspace(1400, 2000, 300)
sgte_liq_si = SGTENode(GLIQSI, T_index=0).compile_zgraph_engine()

def find_Tm(E_0, theta_E):
    e0_node = GroundStateNode(E_0).compile_zgraph_engine()
    osc_node = EinsteinNode(theta_E, T_index=0).compile_zgraph_engine()
    si_phase = FactorNode(jnp.array([[1.0, 3.0]]), [e0_node, osc_node], beta=0.0)
    
    G_dia = jax.vmap(lambda t: si_phase(jnp.atleast_1d(t)))(T_grid)
    G_liq = jax.vmap(lambda t: sgte_liq_si(jnp.atleast_1d(t)))(T_grid)
    
    delta_G = G_dia - G_liq
    return jnp.interp(0.0, delta_G, T_grid)

print("Tm:", find_Tm(-21352.98, 551.5))
