# Thermograph Architecture & Design Philosophy

`thermograph` is the Application and UI layer built on top of the `zgraph`. It acts as the bridge between human-readable thermodynamic concepts (Elements, Microstates, Enthalpies, Phases) and the strict, tensor-only requirements of the `zgraph` math engine.

`thermograph` knows physics, but does not do math. Rather, it constructs `zgraph` objects for calculation. 

## 1. The Flattened Hierarchy (Extending `ZGraphNode`)
Historically, `thermograph` classes were standard Python classes that "compiled" into `zgraph` nodes. 
**This pattern has been deprecated.**
Today, domain application nodes (e.g., `EinsteinNode`, `SGTENode`, `PhaseBoundaryPredictor`) inherit directly from `zgraph`'s `ZGraphNode` base class. 
*   They construct their raw `zgraph` mathematical topology in their `__init__` method, storing it as `self.engine`.
*   They expose a pristine `evaluate(self, signals)` method that simply delegates to `self.engine(signals)`.
*   As a result, the entire stack—from the macroscopic phase diagram predictor down to the lowest mathematical factor node—is a single, unified, executable JAX PyTree.

## 2. Automated Uncertainty Quantification (UQ)
Because all `thermograph` nodes inherit from `ZGraphNode`, they natively inherit the `__call__` auto-vectorization interceptor and the `.ensemble` BatchProxy. 
This means that *any* thermodynamic logic written in `thermograph` automatically and effortlessly scales across massive multidimensional parameter uncertainty ensembles (Parallel Universes) without the developer ever needing to write a `jax.vmap` loop or construct a PyTree mask.

## 3. Safe Physics Hot-Swapping & Immutability
Because `thermograph` nodes are `eqx.Module`s, they are strictly immutable to preserve `jax.jit` compatibility. 
You cannot swap a model's internal engine in-place (e.g., `phase.engine = new_model`).
If you need to dynamically hot-swap a physical model (e.g., swapping a Regular Solution model for a Redlich-Kister polynomial), you must use Equinox's functional PyTree mutation:
```python
new_phase = eqx.tree_at(lambda p: p.engine, old_phase, new_engine)
```

## 4. Introspection is Offline
All methods that return Pandas DataFrames or human-readable mappings (e.g., `get_configuration_matrix()`) must safely detach data from the JAX computation graph using `np.asarray()`. This ensures that data scientists can interrogate the thermodynamic states visually without accidentally breaking the JAX autodiff tracer.

---

## Core Design Principles

All developers and contributors modifying `thermograph` MUST adhere to the following rules:

1. **Domain Abstraction Only:** `thermograph` is a pure application wrapper. Do not implement complex tensor math routines directly in `thermograph`. If a new mathematical transformation is needed, implement it in `zgraph` and call it from `thermograph`.
2. **Inherit from `ZGraphNode`:** Never write `.compile_zgraph_engine()` factory methods. Inherit from `ZGraphNode`, compose your graph in `__init__`, and delegate `evaluate` to the engine.
3. **Immutable PyTrees:** Treat all `thermograph` nodes as immutable JAX primitives. Rely entirely on `eqx.tree_at` for configuration updates.
4. **Computational Efficiency:** Delegate all batching, spatial grids, and microstate sums directly to the `ZGraphNode` auto-vectorization pipeline. Do not introduce Python loops to iterate over data points in `thermograph`.
5. **Clean Code & Strict Typing:** Use Python's built-in `typing` module to document all compiler interfaces, ensuring seamless handoffs to the untyped tensors of `zgraph`.
