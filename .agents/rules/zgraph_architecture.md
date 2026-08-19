# ZGraph AI Directives

**ATTENTION AI CODING ASSISTANTS:**
Modify `zgraph` strictly following these rules (see `zgraph/src/zgraph/ARCHITECTURE.md`):

1. **Pure Math**: No Python objects (strings, lists, dicts) or control flow in `nn.Module.forward()`.
2. **Immutable Nodes**: Routing nodes are structurally immutable. Instantiate new nodes if physics change.
3. **GPU/Device**: Register static indices/constants via `self.register_buffer()`. Learnables use `nn.Parameter()`.
4. **Tensors Only**: All `forward` I/O must be `torch.Tensor`.
5. **Vectorize**: No Python loops over dimensions inside `forward()`. Use native tensor ops.
6. **Strict Typing**: Use explicit `typing` hints.
7. **Graph Transforms**: `vmap` and `compile` must be applied AFTER the graph is built, NEVER inside nodes or `forward()`.
8. **PyTrees**: Use `torch.utils._pytree.tree_map`.
9. **Style**: Be concise and token-efficient. Write efficient but clear code. Prefer Plotly for visualizations. Adhere to PEP 8 naming conventions (short, all-lowercase with underscores) for files and directories to ensure standard Python import compatibility.
