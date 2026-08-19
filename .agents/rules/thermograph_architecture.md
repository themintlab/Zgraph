# Thermograph AI Directives

**ATTENTION AI CODING ASSISTANTS:**
Modify `thermograph` strictly following these rules (see `thermograph/src/thermograph/ARCHITECTURE.md`):

1. **Domain Wrapper**: Implement tensor math in `zgraph`, call from `thermograph`.
2. **Immutable Engines**: Re-compile and replace `self.engine` if physics change; do not mutate in-place.
3. **Buffers**: Pass indices so `zgraph` nodes can register them as buffers.
4. **Vectorize**: Delegate batching/loops to `zgraph`. No Python loops over data here.
5. **Strict Typing**: Document compiler interfaces with `typing`.
6. **Style**: Be concise and token-efficient. Write efficient but clear code. Prefer Plotly for visualizations. Adhere to PEP 8 naming conventions (short, all-lowercase with underscores) for files and directories to ensure standard Python import compatibility.