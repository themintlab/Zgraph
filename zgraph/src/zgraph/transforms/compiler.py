import torch
from torch.func import vmap


def graph_to_function(root_node, in_dims=0, compile=True, compile_mode="default"):
    """
    Converts ZGraph graph node(s) into batched callables and optionally compiles them.

    Args:
        root_node (nn.Module or container of nn.Module): Single root node or an
            arbitrary PyTree of root nodes.
        in_dims (int or tuple): Dimension(s) to batch over for vmap. Default is 0.
        compile (bool): Whether to apply torch.compile. Default is True.
        compile_mode (str): Mode parameter passed to torch.compile. Default is 'default'.

    Returns:
        Callable or container of Callables: Batched/compiled function(s)
        matching the input structure.
    """
    from torch.utils._pytree import tree_map

    def _compile_single_node(node):
        batched = vmap(node, in_dims=in_dims)
        if compile:
            return torch.compile(batched, mode=compile_mode)
        return batched

    return tree_map(_compile_single_node, root_node)
