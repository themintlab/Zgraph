# ZGraph Architecture & Design Philosophy

`zgraph` is an auto-differentiable linear algebra library designed to perform hyper-dimensional LogSumExp contractions and microstate energy calculations using PyTorch.

To ensure speed and portability across platforms and hardware architectures all code within this package MUST adhere to the following strict architectural directives. This will ensure robust compilation via `torch.compile`. 

`zgraph` is purely mathematical and has no knowledge of the application. All labelling of variables, modules, graphs, etc., are relegated to independent application layers (e.g., `thermograph`, `trafficgraph`).

---

## 🤖 AI AGENT DIRECTIVES

**ATTENTION ALL AI CODING ASSISTANTS & DEVELOPERS:**
If you are instructed to modify `zgraph`, you **MUST** strictly adhere to the following rules to prevent breaking the PyTorch 2.0 native `vmap` and `torch.compile` compatibility.

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

---

## The `**kwargs` Parameter Standard
All leaf node engines dynamically register trainable parameters to ensure ZGraph can optimize arbitrary equations without needing to hardcode specific variable shapes into the engine block.

*(May be subject to change as application layer matures)*
