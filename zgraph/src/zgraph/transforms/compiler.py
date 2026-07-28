import torch
from torch.func import vmap, functional_call


def finalize(root_node, batched=True, in_dims=0, compile_graph=True, compile_mode="default"):
    """
    Finalizes a ZGraph nn.Module into an efficient, stateless callable.

    Buffers (graph structure: M matrices, signal indices, etc.) are captured
    once at finalize-time and baked into the closure — they cannot change after
    finalization. Parameters (learnable weights) remain live and are read from
    the module on every call, so gradient flow and optimizer updates work
    transparently.

    The returned callable has signature:
        fn(x)          — when batched=False, x is a single 1-D signal tensor
        fn(x_batch)    — when batched=True,  x_batch is (N, channels)

    For functional composition (e.g. torch.func.grad, jacrev), the lower-level
    ``fn`` attribute exposes the pure (params, x) → scalar signature:
        grad_fn = torch.func.grad(finalized.fn, argnums=0)
        dparam  = grad_fn(dict(module.named_parameters()), x_single)

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
        compile_graph (bool):
            Whether to apply torch.compile after vmapping. Default is True.
        compile_mode (str):
            Mode parameter passed to torch.compile. Default is 'default'.

    Returns:
        FinalizedGraph or container of FinalizedGraph: a callable wrapping the
        node, matching the structure of the input (list/tuple/dict passthrough).

    Note:
        Call finalize *after* moving the module to its target device. Buffers
        are captured by reference at call time; a subsequent ``.to(device)``
        will not update the baked-in copies.
    """
    if isinstance(root_node, (list, tuple)):
        funcs = [
            finalize(node, batched=batched, in_dims=in_dims,
                     compile_graph=compile_graph, compile_mode=compile_mode)
            for node in root_node
        ]
        return type(root_node)(funcs)

    if isinstance(root_node, dict):
        return {
            key: finalize(node, batched=batched, in_dims=in_dims,
                          compile_graph=compile_graph, compile_mode=compile_mode)
            for key, node in root_node.items()
        }

    return FinalizedGraph(root_node, batched=batched, in_dims=in_dims,
                          compile_graph=compile_graph, compile_mode=compile_mode)


class FinalizedGraph:
    """
    Wraps an nn.Module as a stateless, optionally vmapped and compiled callable.

    Buffers are baked in at construction time; parameters are read live from the
    module on every call so that optimizer updates and gradient computation work
    without any extra bookkeeping.
    """

    def __init__(self, module, batched=True, in_dims=0,
                 compile_graph=True, compile_mode="default"):
        self._module = module

        # Capture buffers once — these represent fixed graph structure and must
        # not change after finalization.
        self._buffers = {
            name: buf.detach()
            for name, buf in module.named_buffers(recurse=True)
        }

        # Pure (params, x) → scalar function; exposed for functional transforms.
        def _stateless(params, x):
            return functional_call(module, {**params, **self._buffers}, (x,))

        self.fn = _stateless

        # Build the execution path: optionally vmap, then optionally compile.
        if batched:
            _exec = vmap(_stateless, in_dims=(None, in_dims))
        else:
            _exec = _stateless

        if compile_graph:
            _exec = torch.compile(_exec, mode=compile_mode)

        self._exec = _exec

    def __call__(self, x):
        params = dict(self._module.named_parameters())
        return self._exec(params, x)

    def parameters(self):
        """Delegate to the underlying module for use with torch.optim."""
        return self._module.parameters()

    def named_parameters(self):
        """Delegate to the underlying module."""
        return self._module.named_parameters()

    @property
    def module(self):
        """The original nn.Module, e.g. for state_dict serialization."""
        return self._module
