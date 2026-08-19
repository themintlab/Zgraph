# ZGraph AI Directives

**ATTENTION AI CODING ASSISTANTS:**
Modify `zgraph` strictly following these rules (see `zgraph/src/zgraph/ARCHITECTURE.md`):

1. **Probabilistic Graphical Model (PGM) First**: `zgraph` is fundamentally a PGM engine simulating log-categorical distributions.
2. **Architectural Division (Core vs Transforms vs Solvers)**:
   - **`core/`**: Math primitives (`FactorNode`). Defines the forward pass and topology.
   - **`transforms/`**: Graph-to-Graph operators (`legendre_transform`). They reshape the topology but return evaluatable functions/modules. They do not algorithmically search for solutions.
   - **`solvers/`**: Execution routines (`extract_decision_boundary`, `gauge_fix`). They take a **Callable Function** + **Input Domain Tensors**, execute an algorithmic search (roots, crossovers), and return solution sub-domain tensors. They explicitly do *not* manage `vmap` or `compile`.
3. **Domain Agnosticism**: Keep `zgraph` strictly mathematical. Domain-specific physical logic (like mapping gradients to CALPHAD composition tie-lines) must live in outer domain packages like `thermograph`.
4. **Pure Math**: No Python objects (strings, lists, dicts) or control flow in `nn.Module.forward()`.
5. **Immutable Nodes**: Routing nodes are structurally immutable. Instantiate new nodes if physics change.
6. **GPU/Device**: Register static indices/constants via `self.register_buffer()`. Learnables use `nn.Parameter()`.
7. **Tensors Only**: All `forward` I/O must be `torch.Tensor`.
8. **Vectorize**: No Python loops over dimensions inside `forward()`. Use native tensor ops.
9. **Strict Typing**: Use explicit `typing` hints.
10. **Graph Transforms**: `vmap` and `compile` must be applied AFTER the graph is built, NEVER inside nodes or `forward()`.
11. **PyTrees**: Use `torch.utils._pytree.tree_map`.
12. **Style**: Be concise and token-efficient. Write efficient but clear code. Prefer Plotly for visualizations. Adhere to PEP 8 naming conventions (short, all-lowercase with underscores) for files and directories to ensure standard Python import compatibility.
