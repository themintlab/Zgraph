# Thermograph

**A domain compiler and application layer bridging classical thermodynamics with modern tensor networks.**

`thermograph` is the user-facing thermodynamic application layer built on top of [**ZGraph**](../zgraph). It acts as a domain compiler—translating human-readable physical concepts (elements, phases, CALPHAD polynomials, microstate ensembles, and state variables like $T, P, \mu$) into the strict, unlabelled tensor graphs required by the `zgraph` execution engine.

---

## 🏛️ Core Philosophy: Knows Physics, Delegates Math

`thermograph` **knows physics, but does zero math directly**.

All mathematical evaluation, tensor contraction, soft-minimum selection, and auto-differentiation are delegated to the underlying [`zgraph`](../zgraph) engine. `thermograph` handles model definition, variable index packing, unit conversions, and physical envelope management.

### Division of Responsibilities

| Feature / Responsibility | **`thermograph`** (Application Layer) | **`zgraph`** (Math Engine) |
| :--- | :--- | :--- |
| **Primary Focus** | Domain semantics, CALPHAD models, phase equilibrium | Soft-tropical tensor algebra, kernel execution |
| **Knowledge** | **Knows physics**, elements, phases, units | **Physics-free**, pure tensor math |
| **Data Representation** | Human-readable named objects, DataFrames, signals | Unlabeled PyTorch Tensors (`torch.Tensor`) |
| **Execution Role** | Graph compilation, index linker, envelope swapping | Tensor contraction, `torch.compile` JIT kernels |
| **Introspection** | Offline inspection (`detach().cpu()`), plots | High-throughput autograd & Hessian calculation |

---

## ⚡ Key Architecture & Features

### 1. The Linker & Compiler Pattern
`thermograph` maps complex thermodynamic systems (such as SGTE unary databases, Redlich-Kister interaction polynomials, or sublattice models) into structured input channel signal vectors. It compiles physical equations into lightweight `zgraph` factor and leaf nodes.

### 2. Envelope Replacement (Safe Physics Hot-Swapping)
Because `zgraph` tensor execution graphs are strictly immutable to maintain `torch.compile` JIT optimization, `thermograph` handles physical model changes (e.g., changing from an ideal solution to a sub-regular solution model) by rebuilding and replacing the lightweight `zgraph` node envelopes without compromising kernel stability.

### 3. Offline Introspection
All user-facing diagnostic tools and DataFrame exports (e.g., microstate configuration matrices, phase fraction summaries) safely detach from the PyTorch computation tree via `.detach().cpu()`, enabling rich interactive analysis without breaking autograd backpropagation.

---

## 💻 Application Usage Example

Using `thermograph` to construct and evaluate thermodynamic phase models backed by `zgraph`:

```python
import torch
import thermograph as tg

# 1. Define human-readable thermodynamic system (e.g., Binary System)
# thermograph handles signal index packing, SGTE parameters, and phase setup
system = tg.System(elements=["Ni", "Al"])

# 2. Compile physical phase models into zgraph execution nodes
fcc_phase = tg.PhaseModel(name="FCC_A1", components=["Ni", "Al"])
engine = fcc_phase.compile_zgraph_engine()

# 3. Evaluate physical states in intensive (T, P, mu) space
# Inputs are mapped into signals; computation runs entirely inside zgraph
T = 1200.0  # Kelvin
mu_Al = -50000.0  # J/mol
free_energy = engine.evaluate(temperature=T, chemical_potentials=[mu_Al])
```

---

## 🔗 Underlying Math Engine

For low-level tensor kernel operations, soft-tropical semirings, and custom differentiable graph construction, see the [**ZGraph**](../zgraph) engine repository.
