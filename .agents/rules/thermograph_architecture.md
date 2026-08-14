# Thermograph AI Directives

**ATTENTION AI CODING ASSISTANTS:**
When modifying `thermograph`, you **MUST** adhere to the following rules. For full context, read `thermograph/src/thermograph/ARCHITECTURE.md`.

1. **Domain Abstraction Only:** `thermograph` is a pure application wrapper around `zgraph`. Do not implement complex tensor math routines directly in `thermograph`. If a new mathematical transformation is needed, implement it in `zgraph` and call it from `thermograph`.
2. **Immutable Engines:** Do not attempt to modify the `self.engine` (which is a `zgraph` `nn.Module`) in place. If the user changes a phase's physical model, you must fully re-compile and replace the `.engine` attribute.
3. **Indices to Buffers:** When building `zgraph` models from `thermograph`, ensure all indices are passed in such a way that the `zgraph` nodes can properly register them as buffers (e.g. passing lists of ints which `zgraph` nodes then cast to `register_buffer`).
4. **Computational Efficiency:** Delegate all batching, spatial grids, and microstate sums directly to the compiled `zgraph` engine. Do not introduce Python loops to iterate over data points in `thermograph`.
5. **Clean Code & Strict Typing:** Use Python's built-in `typing` module to document all compiler interfaces, ensuring seamless handoffs to the untyped tensors of `zgraph`.