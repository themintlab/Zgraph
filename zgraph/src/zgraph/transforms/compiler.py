import torch
from torch.func import vmap

def finalize(root_node, in_dims=0, compile_graph=True, compile_mode="default"):
    """
    Finalizes ZGraph graph node(s) by vectorizing (batching) and optionally compiling.

    Args:
        root_node (nn.Module or container of nn.Module): Single root node or a 
            list, tuple, or dict of root nodes.
        in_dims (int or tuple): Dimension(s) to batch over for vmap. Default is 0.
        compile_graph (bool): Whether to apply torch.compile. Default is True.
        compile_mode (str): Mode parameter passed to torch.compile. Default is 'default'.

    Returns:
        Callable or container of Callables: Finalized batched/compiled function(s) 
        matching the input structure.
    """
    if isinstance(root_node, (list, tuple)):
        funcs = [
            finalize(node, in_dims=in_dims, compile_graph=compile_graph, compile_mode=compile_mode)
            for node in root_node
        ]
        return type(root_node)(funcs)

    if isinstance(root_node, dict):
        return {
            key: finalize(node, in_dims=in_dims, compile_graph=compile_graph, compile_mode=compile_mode)
            for key, node in root_node.items()
        }


    batched = vmap(root_node, in_dims=in_dims)
    if compile_graph:
        return torch.compile(batched, mode=compile_mode)
    return batched
