# ZGraph AI Directives

**ATTENTION AI CODING ASSISTANTS:**
When modifying `zgraph`, you **MUST** strictly adhere to the following rules to prevent breaking PyTorch 2.0 native `vmap` and `torch.compile` compatibility. For full context and philosophy, read `zgraph/src/zgraph/ARCHITECTURE.md`.

1. **Pure Math, No Python Objects:**
   No standard Python objects (strings, lists of strings, dicts) or Python control flow (`if` string matches) may exist inside `nn.Module` classes or their `forward()` passes.

2. **Structural Immutability:**
   Routing nodes (e.g., `FactorNode`) are structurally immutable. Do NOT include methods that mutate `nn.ModuleList` or swap subgraphs in-place. Instantiate new nodes if physics change.

3. **GPU & Device Management:**
   - Static indices/constant tensors MUST be registered as buffers using `self.register_buffer()`.
   - Learnable constants must use `nn.Parameter()`.
   - Never store lists of integers/floats as raw attributes if used in `forward()`.

4. **Tensor-Only Communication:**
   All inputs/outputs between modules must be `torch.Tensor` types. No custom classes or tuples of mixed types in the `forward` signature.

5. **Computational Efficiency:**
   Never use Python `for` or `while` loops over spatial dimensions, batches, or microstates inside `forward()`. Vectorize using native tensor ops or `vmap`.

6. **Clean Code & Strict Typing:**
   Use explicit type hints from `typing` (e.g., `List`, `Tuple`).

7. **PyTorch-Native Idioms & PyTrees:**
   - Prefer `torch.utils._pytree.tree_map` for all recursive container mappings (lists, tuples, dicts). Do not write manual `isinstance(x, dict)` blocks.
   - Prefer `torch.func.vmap` for batching operations across node inputs.
   - The functional API (e.g., `legendre_transform`, `graph_to_function`) must support arbitrary PyTrees of modules.
