# ZGraph

**A hardware-accelerated, fully differentiable, physics-free soft-tropical tensor graph engine.**

`zgraph` is a low-level, high-performance numerical kernel engine built natively on PyTorch 2.0 (`torch.compile`, `vmap`, `torch.func`). It reduces complex high-dimensional factor graph contractions, soft-minimum selections, and state space reductions into homogeneous chains of parameterized $\text{LogSumExp}$ matrix operations.

---

## 🏛️ Core Philosophy: Physics-Free Pure Math

`zgraph` is **strictly domain-agnostic and physics-free**.

The engine operates exclusively on unlabeled numerical tensors (`torch.Tensor`). It contains zero domain-specific logic, zero unit conversions, and zero Python object overhead (such as string matching or dictionary lookups inside `forward` loops). 

All physical nomenclature, thermodynamic ensembles, chemical potential mappings ($\mu$), state variables ($T, P$), phase definitions, and human-readable metadata are relegated to higher-level application layers such as [**Thermograph**](../thermograph).

### Separation of Concerns

| Feature / Responsibility | **`zgraph`** (Math Engine) | **`thermograph`** (Application Layer) |
| :--- | :--- | :--- |
| **Domain Scope** | Abstract factor graphs, soft-tropical semirings | Thermodynamics, statistical mechanics, CALPHAD |
| **Data Types** | Unlabeled PyTorch Tensors (`torch.Tensor`) | Named variables, species, phases, physical units |
| **Core Operations** | Matrix contraction, parameterized $\text{LogSumExp}$ | Ensemble definitions $(T, P, \mu)$, driving forces |
| **Optimization Target** | `torch.compile`, C++/CUDA fusion, `vmap` vectorization | Phase equilibrium, convex hulls, phase-field coupling |
| **Graph Mutations** | Structurally immutable execution graphs | Dynamic model assembly & configuration |

---

## ⚡ Key Pillars

### 1. Physics-Free Pure Tensor Math
By enforcing pure tensor-only signatures (`forward(local_signals: Tensor) -> Tensor`), `zgraph` guarantees zero graph breaks during JIT compilation. Nodes do not store string names, metadata dicts, or conditional Python logic.

### 2. Soft Tropical Tensor Algebra
Partition functions and soft-minimum energy landscapes are formulated as soft-tropical tensor contractions over factor matrices $\mathbf{M}$ and dynamic weight vectors $\mathbf{w}(\mathbf{x})$:
$$\mathcal{F}(\mathbf{x}) = \beta \cdot \text{LogSumExp}\left( \frac{\mathbf{M} \cdot \mathbf{w}(\mathbf{x})}{\beta} \right)$$
where $\beta$ controls the smoothing transition between exact hard-minimum selection ($\beta \to 0^+$) and soft statistical integration.

### 3. Hardware-Native Execution
Engine nodes (`FactorNode`, `LeafNode`, `SignalNode`) are engineered to compile seamlessly via `torch.compile` into fused C++/CUDA kernels. Vectorization across spatial grids or multi-dimensional parameter batches is achieved without overhead using PyTorch's `vmap`.

### 4. End-to-End Differentiability
Built for direct integration with `torch.func`, `zgraph` provides exact analytical Jacobians and Hessians ($\nabla_\mathbf{x} \mathcal{F}$, $\nabla^2_\mathbf{x} \mathcal{F}$) for gradient-based optimization, continuous sensitivity analysis, and autograd-driven parameter estimation.

---

## 💻 Engine Usage Example

At the `zgraph` level, models are built using purely mathematical node abstractions:

```python
import torch
import zgraph as zg

# 1. Define configuration matrix M (Microstates x Clusters)
M_matrix = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.5]
])

# 2. Define pure energy subgraphs (Leaf / Constant / Signal nodes)
subgraphs = [
    zg.ConstantNode(init_val=1.5),
    zg.SignalNode(0)
]

# 3. Instantiate the FactorNode engine
factor_node = zg.FactorNode(M_matrix=M_matrix, subgraph_list=subgraphs, beta=1.0)

# 4. Pure tensor input evaluation
signals = torch.tensor([0.2, 1.0])  # Unlabeled numerical input tensor
free_energy = factor_node(signals)   # Fully differentiable scalar tensor output
```

---

## 🔗 Application Layer

For thermodynamic potential evaluation, materials modeling, phase diagram construction, and user-facing APIs, please see the [**Thermograph**](../thermograph) repository.
