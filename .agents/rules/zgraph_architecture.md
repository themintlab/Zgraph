# ZGraph AI Directives

**ATTENTION AI CODING ASSISTANTS:**
Modify `zgraph` strictly following these rules (see `zgraph/src/zgraph/ARCHITECTURE.md`):

1. **Probabilistic Graphical Model (PGM) First**: `zgraph` is fundamentally a PGM engine simulating log-categorical distributions.
2. **Architectural Division (Core vs Transforms vs Solvers)**:
   - **`core/`**: Math primitives (`FactorNode`). Defines the forward pass and topology.
   - **`transforms/`**: Graph-to-Graph operators (`legendre_transform`). They reshape the topology but return evaluatable functions/modules. They do not algorithmically search for solutions.
   - **`solvers/`**: Execution routines (`extract_decision_boundary`, `gauge_fix`). They take a **Callable Function** + **Input Domain Tensors**, execute an algorithmic search (roots, crossovers), and return solution sub-domain tensors. They explicitly do *not* manage `jax.vmap` or `jax.jit`.
3. **Domain Agnosticism**: Keep `zgraph` strictly mathematical. Domain-specific physical logic (like mapping gradients to CALPHAD composition tie-lines) must live in outer domain packages like `thermograph`.
4. **Pure Math**: No Python objects (strings, lists, dicts) or control flow in `eqx.Module.__call__()`.
5. **Immutable Nodes**: Routing nodes are structurally immutable. Instantiate new nodes if physics change.
6. **Static vs Dynamic**: Equinox handles static fields natively (e.g., `eqx.field(static=True)`).
7. **Arrays Only**: All `__call__` I/O must be `jax.Array`.
8. **Vectorize**: No Python loops over dimensions inside `__call__()`. Use native JAX ops.
9. **Strict Typing**: Use explicit `typing` hints.
10. **Graph Transforms**: `jax.vmap` and `jax.jit` must be applied AFTER the graph is built, NEVER inside nodes or `__call__()`.
11. **PyTrees**: Use `jax.tree_map`.
12. **Style**: Be concise and token-efficient. Write efficient but clear code. Prefer Plotly for visualizations. Adhere to PEP 8 naming conventions (short, all-lowercase with underscores) for files and directories to ensure standard Python import compatibility.
