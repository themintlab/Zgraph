# ZGraph Architecture & Design Philosophy

`zgraph` is fundamentally a **Probabilistic Graphical Model (PGM)** engine designed to simulate log-categorical distributions and thermodynamic ensembles. It performs hyper-dimensional Softmin/LogSumExp contractions and calculates microstate energies using JAX.

To ensure speed and portability across platforms and hardware architectures, all code within this package MUST adhere to the following strict architectural directives. This ensures robust compilation via `jax.jit` and seamless compatibility with `jax`. 

`zgraph` is purely mathematical and has no knowledge of the application. All labelling of variables, modules, and physical significance (e.g., mapping gradients to CALPHAD composition tie-lines) are relegated to independent domain packages (like `thermograph`).

---

## Architectural Division

The `zgraph` codebase is strictly divided into three distinct categories based on their mathematical role:

1. **`core/` (The PGM Forward Pass)**
   Contains the structural math primitives (`FactorNode`, `ProductNode`, `marginalize`). These define the topology of the graphical model. They execute the __call__() pass to compute either the collapsed macroscopic state (the partition function via `.__call__()`) or the latent microscopic distribution (the logits via `.logits()`).

2. **`transforms/` (Graph-to-Graph Operators)**
   Contains functional operators (like `legendre_transform`) that reshape the topology or coordinates. A transform takes a graph (`eqx.Module`) and returns a *new* graph. They do not find solutions; they alter the geometry of the problem analytically.

3. **`solvers/` (Execution Routines)**
   Contains algorithmic search functions (like `extract_decision_boundary` and `gauge_fix`). Solvers do not return graphs. Instead, they take a **callable function** (e.g., a pre-batched model or logits method) and an **input domain tensor**, and execute algorithmic searches (e.g., finding roots, crossovers, or projecting onto level-sets) over the evaluated landscape. They return numerical sub-domain features (shifted coordinates, boundary indices). They explicitly do **not** take responsibility for `vmap` or `compile` mapping; that is the domain layer's responsibility.

---

## Core Design Principles

To prevent breaking JAX/Equinox native `vmap` and `jax.jit` compatibility, all contributors must strictly adhere to the following rules:

1. **Pure Math, No Python Objects:**
   **No standard Python objects (strings, lists of strings, dictionaries) or Python control flow (`if` statements based on string matching) may exist inside `zgraph` `eqx.Module` classes or their `__call__()` passes.**
   - *Reason:* TorchScript and `jax.jit` require strict static typing. Dictionaries or string parsing cause graph breaks and kernel compilation failures. All domain knowledge (names, metadata) must remain in the application layer.

2. **Structural Immutability:**
   `zgraph` routing nodes (e.g., `FactorNode`) are structurally immutable after creation. You may **NOT** include methods that mutate the underlying `list or tuple (within eqx.Module)` or swap subgraphs in-place (e.g., `self.subgraphs[2] = new_model`).
   - *Reason:* Swapping nodes in-place corrupts JAX's tracer graph and invalidates fused C++ kernels. If physics change, discard the node and instantiate a new one.

3. **GPU & Device Management (The Buffer Rule):**
   - **Static indices and constant tensors** (e.g., `signal_indices`, gauge target values) MUST be registered as integer/float tensor buffers using `eqx.field(static=True) for non-arrays or just assign as standard JAX array`.
   - **Learnable constants** must use `standard JAX arrays (Equinox treats all arrays as parameters unless marked static)`.
   - **Never** store lists of integers or floats as raw attributes (e.g., `self.indices = [0, 1]`) if they are used in the __call__() pass. 
   - *Reason:* Doing so ensures that when a user calls `jax.device_put(model)`, all buffers and parameters seamlessly migrate to the GPU. Python lists are ignored by `.to()` and will trigger a device mismatch crash.

4. **Tensor-Only Communication:**
   All inputs and outputs between `zgraph` modules must be `jax.Array` types. No custom classes, tuples of mixed types, or optional arguments are permitted in the `forward` signature.
   - *Reason:* `jax.jit` traces continuous streams of tensor operations. Non-tensor objects force a return to the Python interpreter (a "graph break"), destroying performance.

5. **Computational Efficiency (No Python Loops):**
   **Never** use Python `for` or `while` loops over spatial dimensions, batches, or microstates inside a `__call__()` pass. All operations must be vectorized using native JAX tensor operations, broadcasting, or `vmap`.
   - *Reason:* Python loops are extremely slow and defeat the purpose of using JAX. ZGraph is designed for high-throughput batch evaluations; loops cause a massive bottleneck.

6. **Clean Code & Strict Typing:**
   All functions and methods must use explicit type hints from the built-in `typing` module (e.g., `List`, `Optional`, `Tuple`, `Union`).
   - *Reason:* Strict typing ensures that downstream wrappers (like Thermograph), IDEs, and future AI agents can parse and safely interact with the math engine without ambiguity.

---

## The `**kwargs` Parameter Standard
All leaf node engines dynamically register trainable parameters to ensure ZGraph can optimize arbitrary equations without needing to hardcode specific variable shapes into the engine block.

*(May be subject to change as application layer matures)*
