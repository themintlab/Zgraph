# ZGraph Architecture & Design Philosophy

`zgraph` is fundamentally a **Probabilistic Graphical Model (PGM)** engine designed to simulate log-categorical distributions and thermodynamic ensembles. It performs hyper-dimensional Softmin/LogSumExp contractions and calculates microstate energies using PyTorch.

To ensure speed and portability across platforms and hardware architectures, all code within this package MUST adhere to the following strict architectural directives. This ensures robust compilation via `torch.compile` and seamless compatibility with `torch.func`. 

`zgraph` is purely mathematical and has no knowledge of the application. All labelling of variables, modules, and physical significance (e.g., mapping gradients to CALPHAD composition tie-lines) are relegated to independent domain packages (like `thermograph`).

---

## Architectural Division

The `zgraph` codebase is strictly divided into three distinct categories based on their mathematical role:

1. **`core/` (The PGM Forward Pass)**
   Contains the structural math primitives (`FactorNode`, `ProductNode`, `marginalize`). These define the topology of the graphical model. They execute the forward pass to compute either the collapsed macroscopic state (the partition function via `.forward()`) or the latent microscopic distribution (the logits via `.logits()`).

2. **`transforms/` (Graph-to-Graph Operators)**
   Contains functional operators (like `legendre_transform`) that reshape the topology or coordinates. A transform takes a graph (`nn.Module`) and returns a *new* graph. They do not find solutions; they alter the geometry of the problem analytically.

3. **`solvers/` (Execution Routines)**
   Contains algorithmic search functions (like `extract_decision_boundary` and `gauge_fix`). Solvers do not return graphs. Instead, they take a **callable function** (e.g., a pre-batched model or logits method) and an **input domain tensor**, and execute algorithmic searches (e.g., finding roots, crossovers, or projecting onto level-sets) over the evaluated landscape. They return numerical sub-domain features (shifted coordinates, boundary indices). They explicitly do **not** take responsibility for `vmap` or `compile` mapping; that is the domain layer's responsibility.

---

## Core Design Principles

To prevent breaking PyTorch 2.0 native `vmap` and `torch.compile` compatibility, all contributors must strictly adhere to the following rules:

1. **Pure Math, No Python Objects:**
   **No standard Python objects (strings, lists of strings, dictionaries) or Python control flow (`if` statements based on string matching) may exist inside `zgraph` `nn.Module` classes or their `forward()` passes.**
   - *Reason:* TorchScript and `torch.compile` require strict static typing. Dictionaries or string parsing cause graph breaks and kernel compilation failures. All domain knowledge (names, metadata) must remain in the application layer.

2. **Structural Immutability:**
   `zgraph` routing nodes (e.g., `FactorNode`) are structurally immutable after creation. You may **NOT** include methods that mutate the underlying `nn.ModuleList` or swap subgraphs in-place (e.g., `self.subgraphs[2] = new_model`).
   - *Reason:* Swapping nodes in-place corrupts PyTorch's Autograd graph and invalidates fused C++ kernels. If physics change, discard the node and instantiate a new one.

3. **GPU & Device Management (The Buffer Rule):**
   - **Static indices and constant tensors** (e.g., `signal_indices`, gauge target values) MUST be registered as integer/float tensor buffers using `self.register_buffer()`.
   - **Learnable constants** must use `nn.Parameter()`.
   - **Never** store lists of integers or floats as raw attributes (e.g., `self.indices = [0, 1]`) if they are used in the forward pass. 
   - *Reason:* Doing so ensures that when a user calls `model.to('cuda')`, all buffers and parameters seamlessly migrate to the GPU. Python lists are ignored by `.to()` and will trigger a device mismatch crash.

4. **Tensor-Only Communication:**
   All inputs and outputs between `zgraph` modules must be `torch.Tensor` types. No custom classes, tuples of mixed types, or optional arguments are permitted in the `forward` signature.
   - *Reason:* `torch.compile` traces continuous streams of tensor operations. Non-tensor objects force a return to the Python interpreter (a "graph break"), destroying performance.

5. **Computational Efficiency (No Python Loops):**
   **Never** use Python `for` or `while` loops over spatial dimensions, batches, or microstates inside a `forward()` pass. All operations must be vectorized using native PyTorch tensor operations, broadcasting, or `vmap`.
   - *Reason:* Python loops are extremely slow and defeat the purpose of using PyTorch. ZGraph is designed for high-throughput batch evaluations; loops cause a massive bottleneck.

6. **Clean Code & Strict Typing:**
   All functions and methods must use explicit type hints from the built-in `typing` module (e.g., `List`, `Optional`, `Tuple`, `Union`).
   - *Reason:* Strict typing ensures that downstream wrappers (like Thermograph), IDEs, and future AI agents can parse and safely interact with the math engine without ambiguity.

---

## The `**kwargs` Parameter Standard
All leaf node engines dynamically register trainable parameters to ensure ZGraph can optimize arbitrary equations without needing to hardcode specific variable shapes into the engine block.

*(May be subject to change as application layer matures)*
