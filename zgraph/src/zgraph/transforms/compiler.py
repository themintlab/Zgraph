import jax
from jax import vmap, jit
from jax.tree_util import tree_map

import equinox as eqx

def graph_to_function(root_node, in_axes=0, compile=True):
    """
    Converts ZGraph graph node(s) into batched callables and optionally compiles them.

    Args:
        root_node (eqx.Module or container of eqx.Module): Single root node or an
            arbitrary PyTree of root nodes.
        in_axes (int or tuple): Dimension(s) to batch over for vmap. Default is 0.
        compile (bool): Whether to apply jax.jit. Default is True.

    Returns:
        Callable or container of Callables: Batched/compiled function(s)
        matching the input structure.
    """

    def _compile_single_node(node):
        batched = vmap(node, in_axes=in_axes)
        if compile:
            return jit(batched)
        return batched

    return tree_map(_compile_single_node, root_node, is_leaf=lambda x: isinstance(x, eqx.Module))
