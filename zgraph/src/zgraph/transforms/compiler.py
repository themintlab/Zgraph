import torch
from torch.func import vmap


def graph_to_function(root_node, in_dims=0, compile=True, compile_mode="default"):
    """
    Converts ZGraph graph node(s) into batched callables and optionally compiles them.

    Args:
        root_node (nn.Module or container of nn.Module): Single root node or a
            list, tuple, or dict of root nodes.
        in_dims (int or tuple): Dimension(s) to batch over for vmap. Default is 0.
        compile (bool): Whether to apply torch.compile. Default is True.
        compile_mode (str): Mode parameter passed to torch.compile. Default is 'default'.

    Returns:
        Callable or container of Callables: Batched/compiled function(s)
        matching the input structure.
    """
    if isinstance(root_node, (list, tuple)):
        funcs = [
            graph_to_function(node, in_dims=in_dims, compile=compile, compile_mode=compile_mode)
            for node in root_node
        ]
        return type(root_node)(funcs)

    if isinstance(root_node, dict):
        return {
            key: graph_to_function(node, in_dims=in_dims, compile=compile, compile_mode=compile_mode)
            for key, node in root_node.items()
        }

    batched = vmap(root_node, in_dims=in_dims)
    if compile:
        return torch.compile(batched, mode=compile_mode)
    return batched
