import json

with open("thermograph/examples/einstein_oscillator.ipynb", "r") as f:
    nb = json.load(f)

new_cells = [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Constructing a Hybrid Phase Diagram\n",
    "We can seamlessly mix our physically consistent `EinsteinNode` solid phases with standard `SGTENode` liquid phases to create a complete Cu-Al Phase Diagram!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from thermograph.nodes.sgte import SGTENode\n",
    "from external_data.cu_ni_sgte import GLIQCU\n",
    "\n",
    "# Manually extracted GLIQAL\n",
    "GLIQAL = [\n",
    "    (933.473, [11005.045-11276.24, -11.84185+223.048446, -38.5844296, 18.531982e-3, 74092, -5.764227e-6, 79.337e-21, 0]),\n",
    "    (2900.0, [-795.991, 177.430209, -31.748192, 0, 0, 0, 0, 0])\n",
    "]\n",
    "\n",
    "cu_liq_node = SGTENode(GLIQCU, T_index=0).compile_zgraph_engine()\n",
    "al_liq_node = SGTENode(GLIQAL, T_index=0).compile_zgraph_engine()\n",
    "\n",
    "T, mu_Cu, mu_Al = SignalNodes(0, 1, 2)\n",
    "RT = FactorNode([[8.314]], [T])\n",
    "\n",
    "# Mix the Einstein Solid Phases\n",
    "w_cu_fcc = FactorNode([[1.0, -1.0]], [mu_Cu, cu_phase])\n",
    "w_al_fcc = FactorNode([[1.0, -1.0]], [mu_Al, al_phase])\n",
    "phase_FCC = FactorNode(jnp.eye(2), [w_cu_fcc, w_al_fcc], beta=RT)\n",
    "\n",
    "# Mix the SGTE Liquid Phases\n",
    "w_cu_liq = FactorNode([[1.0, -1.0]], [mu_Cu, cu_liq_node])\n",
    "w_al_liq = FactorNode([[1.0, -1.0]], [mu_Al, al_liq_node])\n",
    "phase_LIQ = FactorNode(jnp.eye(2), [w_cu_liq, w_al_liq], beta=RT)\n",
    "\n",
    "system = FactorNode(jnp.eye(2), [phase_FCC, phase_LIQ], beta=0.0)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Extract Boundaries\n",
    "We use the `PhaseDiagramCompiler` to compute the solidus and liquidus lines."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from thermograph.visualize import PhaseDiagramCompiler\n",
    "\n",
    "compiler = PhaseDiagramCompiler(system)\n",
    "T_vals_pd = jnp.linspace(800, 1400, 100)\n",
    "mu_diff = jnp.linspace(-80000, 80000, 200)\n",
    "T_grid, mu_grid = jnp.meshgrid(T_vals_pd, mu_diff, indexing='ij')\n",
    "inputs = jnp.stack([T_grid, mu_grid/2, -mu_grid/2], axis=-1)\n",
    "\n",
    "batched_x = compiler.extract_boundaries(inputs, mu_index=2)\n",
    "x_left = jnp.minimum(batched_x[:, 0], batched_x[:, 1])\n",
    "x_right = jnp.maximum(batched_x[:, 0], batched_x[:, 1])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig_pd = go.Figure()\n",
    "\n",
    "T_np = np.asarray(T_vals_pd)\n",
    "xl_np = np.asarray(x_left)\n",
    "xr_np = np.asarray(x_right)\n",
    "\n",
    "fig_pd.add_trace(go.Scatter(x=xl_np, y=T_np, mode='lines', name='Solidus (Einstein)', line=dict(color='red', width=3)))\n",
    "fig_pd.add_trace(go.Scatter(x=xr_np, y=T_np, mode='lines', name='Liquidus (SGTE)', line=dict(color='blue', width=3)))\n",
    "\n",
    "fig_pd.update_layout(\n",
    "    title='Hybrid Cu-Al Phase Diagram (Einstein Solid + SGTE Liquid)',\n",
    "    xaxis_title='Mole Fraction Al',\n",
    "    yaxis_title='Temperature (K)',\n",
    "    width=800,\n",
    "    height=600\n",
    ")\n",
    "fig_pd.show()\n"
   ]
  }
]

nb["cells"].extend(new_cells)

with open("thermograph/examples/einstein_oscillator.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

