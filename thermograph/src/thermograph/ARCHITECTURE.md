# Thermograph Architecture & Design Philosophy

`thermograph` is the Application and UI layer built on top of the `zgraph`. It acts as the bridge between human-readable thermodynamic concepts (Elements, Microstates, Enthalpies, Phases) and the strict, tensor-only requirements of the `zgraph` math engine.

`thermograph` knows physics, but does not do math. Rather, it constructs `zgraph` objects for calculation. 

## 1. The Separation of Concerns
`thermograph` classes (e.g., `PhaseModel`) are standard Python classes, that contain `.engine` attributes which are zgraph components. `thermograph` **never** executes mathematical loops over microstates, rather delegating all heavy lifting to its internal `zgraph` engine instances.

## 2. The Linker / Compiler Pattern
`thermograph` acts as a compiler, mapping human-readable requirements of its physics modules to the input channels (indices). Indices are then bound through `zgraph` methods. 

## 3. Safe Physics Hot-Swapping
`thermograph` provides the user-facing API for dynamically changing thermodynamic models (e.g., swapping a Regular Solution model for a Redlich-Kister polynomial).

Because `zgraph` graph structures are immutable (to preserve `torch.compile` compatibility), `thermograph` handles hot-swapping through **Envelope Replacement**. 

## 4. Introspection is Offline
All methods that return Pandas DataFrames or human-readable mappings (e.g., `get_configuration_matrix()`) must safely detach data from the PyTorch computation graph using `.detach().cpu().numpy()`. This ensures that data scientists can interrogate the thermodynamic states visually without accidentally breaking the Autograd backpropagation tree.

---

## 🤖 AI AGENT DIRECTIVES

**ATTENTION ALL AI CODING ASSISTANTS & DEVELOPERS:**
If you are instructed to modify `thermograph`, you **MUST** adhere to the following rules:

1. **Domain Abstraction Only:** `thermograph` is a pure application wrapper around `zgraph`. Do not implement complex tensor math routines directly in `thermograph`. If a new mathematical transformation is needed, implement it in `zgraph` and call it from `thermograph`.
2. **Immutable Engines:** Do not attempt to modify the `self.engine` (which is a `zgraph` `nn.Module`) in place. If the user changes a phase's physical model, you must fully re-compile and replace the `.engine` attribute.
3. **Indices to Buffers:** When building `zgraph` models from `thermograph`, ensure all indices are passed in such a way that the `zgraph` nodes can properly register them as buffers (e.g. passing lists of ints which `zgraph` nodes then cast to `register_buffer`).
4. **Computational Efficiency:** Delegate all batching, spatial grids, and microstate sums directly to the compiled `zgraph` engine. Do not introduce Python loops to iterate over data points in `thermograph`.
5. **Clean Code & Strict Typing:** Use Python's built-in `typing` module to document all compiler interfaces, ensuring seamless handoffs to the untyped tensors of `zgraph`.