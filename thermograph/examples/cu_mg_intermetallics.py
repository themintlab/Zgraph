from thermograph import load_library
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go

from zgraph import *
from thermograph.config import LIBRARY_DIR
from thermograph.nodes.einstein import GroundStateNode, EinsteinNode
from thermograph.nodes.sgte import SGTENode
from thermograph.prediction import PhaseBoundaryPredictor


T, mu_Cu, mu_Mg = SignalNodes(0, 1, 2)
RT = FactorNode([[8.314]], [T])

# 1. Cu FCC Solid
cu_fcc_e0 = GroundStateNode(-13221.8)
cu_fcc_osc = EinsteinNode(244.0, T_index=0)
cu_fcc_energy = FactorNode([[1.0, 3.0]], [cu_fcc_e0, cu_fcc_osc], beta=0.0)
phase_FCC = FactorNode([[1.0, 0.0, -1.0]], [mu_Cu, mu_Mg, cu_fcc_energy], beta=0.0)

# 2. Mg HCP Solid
mg_hcp_e0 = GroundStateNode(-11000.0)
mg_hcp_osc = EinsteinNode(400.0, T_index=0)
mg_hcp_energy = FactorNode([[1.0, 3.0]], [mg_hcp_e0, mg_hcp_osc], beta=0.0)
phase_HCP = FactorNode([[0.0, 1.0, -1.0]], [mu_Cu, mu_Mg, mg_hcp_energy], beta=0.0)

# 3. Cu2Mg Intermetallic (Laves Phase, x_Mg = 1/3)
# Must have a deep negative formation energy to be stable
cu2mg_e0 = GroundStateNode(-25000.0)
cu2mg_osc = EinsteinNode(300.0, T_index=0)
cu2mg_energy = FactorNode([[1.0, 3.0]], [cu2mg_e0, cu2mg_osc], beta=0.0)
phase_CU2MG = FactorNode([[2/3, 1/3, -1.0]], [mu_Cu, mu_Mg, cu2mg_energy], beta=0.0)

# 4. CuMg2 Intermetallic (x_Mg = 2/3)
cumg2_e0 = GroundStateNode(-22000.0)
cumg2_osc = EinsteinNode(300.0, T_index=0)
cumg2_energy = FactorNode([[1.0, 3.0]], [cumg2_e0, cumg2_osc], beta=0.0)
phase_CUMG2 = FactorNode([[1/3, 2/3, -1.0]], [mu_Cu, mu_Mg, cumg2_energy], beta=0.0)


# G_liq = G_solid(T_m) + L - T*(L/T_m)
GLIQCU = [(3000.0, [2552.0, -1.908, 0, 0, 0, 0, 0, 0])] # Melt at 1337 K
GLIQMG = [(3000.0, [2000.0, -2.166, 0, 0, 0, 0, 0, 0])] # Melt at 923 K

cu_liq_node = load_library('unary/GLIQCU')
mg_liq_node = load_library('unary/GLIQMG')

# Ideal Liquid components
w_cu_liq = FactorNode([[1.0, -1.0]], [mu_Cu, cu_liq_node])
w_mg_liq = FactorNode([[1.0, -1.0]], [mu_Mg, mg_liq_node])

# To model a liquid with a negative interaction (L0 < 0), we can add an interacting microstate
# just like we did for the miscibility gap, but with a highly attractive (negative) energy.
L0_liq = -15000.0
liq_interact_node = GroundStateNode(L0_liq)
w_cumg_liq = FactorNode([[0.5, 0.5, -1.0]], [mu_Cu, mu_Mg, liq_interact_node])

M_liq = jnp.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 2.0] # Multiplicity 2 for AB/BA interactions
])
phase_LIQ = FactorNode(M_liq, [w_cu_liq, w_mg_liq, w_cumg_liq], beta=RT)


system = FactorNode(jnp.eye(5), [phase_FCC, phase_HCP, phase_CU2MG, phase_CUMG2, phase_LIQ], beta=0.0)


predictor = PhaseBoundaryPredictor(system)

T_vals_pd = jnp.linspace(600, 1400, 200)
mu_diff = jnp.linspace(-80000, 80000, 400)
T_grid, mu_grid = jnp.meshgrid(T_vals_pd, mu_diff, indexing='ij')
inputs = jnp.stack([T_grid, mu_grid/2, -mu_grid/2], axis=-1)

# Extract boundaries
batched_x = predictor.predict_compositions(inputs, mu_index=2)
x_left = jnp.minimum(batched_x[:, 0], batched_x[:, 1])
x_right = jnp.maximum(batched_x[:, 0], batched_x[:, 1])

# Remove nan/inf and filter out tiny numerical noise
mask = jnp.abs(x_left - x_right) > 1e-3

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.asarray(x_left)[mask], y=np.asarray(T_grid)[mask], mode='markers', name='Phase Boundaries', marker=dict(size=2, color='black')))
fig.add_trace(go.Scatter(x=np.asarray(x_right)[mask], y=np.asarray(T_grid)[mask], mode='markers', name='Phase Boundaries', marker=dict(size=2, color='black')))

fig.update_layout(title='Cu-Mg Phase Diagram (Intermetallics)', xaxis_title='Mole Fraction Mg', yaxis_title='Temperature (K)', width=800, height=600)
fig.show()

