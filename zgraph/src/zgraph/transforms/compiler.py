import torch
from torch.func import vmap, functional_call


def compile_graph(root_node, batched=True, in_dims=0, use_torch_compile=True, compile_mode="default"):
    """
    Compiles a ZGraph nn.Module into an efficient, stateless Python callable.

    The module (graph) remains the persistent object of interest. This function
    derives a pure execution function from it — suitable for inference, training
    loops, or functional composition — without creating a new persistent object.

    Buffers (graph structure: M matrices, signal indices, etc.) are captured
    once at compile-time and baked into the closure. They cannot change after
    compilation. Call ``compile_graph`` again if the module is moved to a new
    device or its structure changes.

    Parameters remain live: the returned function reads ``module.named_parameters()``
    on every call, so optimizer updates and gradient flow work transparently.

    The returned callable has signature:
        fn(x)        — when batched=False, x is a single 1-D signal tensor
        fn(x_batch)  — when batched=True,  x_batch is (N, channels)

    For functional composition (e.g. ``torch.func.grad``, ``jacrev``), the
    ``fn`` attribute exposes the pure ``(params, x) → scalar`` signature:

        g = torch.func.grad(compiled.fn, argnums=0)
        dparam = g(dict(module.named_parameters()), x_single)

    Args:
        root_node (nn.Module or container of nn.Module):
            Single root node or a list, tuple, or dict of root nodes.
        batched (bool):
            If True (default), wraps the function with vmap so it accepts a
            batch of inputs with shape (N, channels). If False, the returned
            callable operates on a single un-batched input.
        in_dims (int or tuple):
            Passed to vmap to control which dimension to batch over. Ignored
            when batched=False.
        use_torch_compile (bool):
            Whether to apply torch.compile after vmapping. Default is True.
        compile_mode (str):
            Mode parameter passed to torch.compile. Default is 'default'.

    Returns:
        callable or container of callables: A plain Python callable (with a
        ``.fn`` attribute for functional transforms), matching the structure of
        the input (list/tuple/dict passthrough).

    Note:
        Call ``compile_graph`` *after* moving the module to its target device.
    """
    if isinstance(root_node, (list, tuple)):
        funcs = [
            compile_graph(node, batched=batched, in_dims=in_dims,
                          use_torch_compile=use_torch_compile, compile_mode=compile_mode)
            for node in root_node
        ]
        return type(root_node)(funcs)

    if isinstance(root_node, dict):
        return {
            key: compile_graph(node, batched=batched, in_dims=in_dims,
                               use_torch_compile=use_torch_compile, compile_mode=compile_mode)
            for key, node in root_node.items()
        }

    # Buffers represent fixed graph structure — baked into the closure at
    # compile-time so they are never passed as arguments.
    buffers = dict(root_node.named_buffers(recurse=True))

    # Pure (params, x) → scalar function; exposed for functional transforms.
    def stateless_fn(params, x):
        return functional_call(root_node, {**params, **buffers}, (x,))

    # Build the execution path: optionally vmap, then optionally torch.compile.
    # vmap-then-compile is the correct ordering: the compiler sees the full
    # batched computation and can fuse across the batch dimension.
    if batched:
        _exec = vmap(stateless_fn, in_dims=(None, in_dims))
    else:
        _exec = stateless_fn

    if use_torch_compile:
        _exec = torch.compile(_exec, mode=compile_mode)

    def call(x):
        params = dict(root_node.named_parameters())
        return _exec(params, x)

    # Attach the raw (params, x) function for torch.func composition.
    call.fn = stateless_fn

    return call

