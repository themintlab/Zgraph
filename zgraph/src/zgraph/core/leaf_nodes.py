import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Union, List, Callable, Dict, Any, Tuple
from jax.tree_util import tree_map

class TemplateNode(eqx.Module):
    """
    A purely mathematical leaf node that executes a static JAX function over registered tensor parameters.
    Designed to serve as an anonymous block for domain-specific kernels without violating zgraph's pure-tensor constraints.
    """
    params: Any
    kernel_fn: Callable = eqx.field(static=True)
    signal_indices: jax.Array

    def __init__(self, kernel_fn: Callable, params: Any, signal_indices: Optional[List[int]] = None):
        if isinstance(params, dict):
            raise TypeError("Dictionaries are forbidden in zgraph equinox modules. Pass arrays or tuples instead.")
        self.kernel_fn = kernel_fn
        self.params = params
        indices = signal_indices if signal_indices is not None else []
        self.signal_indices = jnp.array(indices, dtype=jnp.int32)

    def __call__(self, local_signals: jax.Array) -> jax.Array:
        sliced_signals = local_signals[self.signal_indices]
        return self.kernel_fn(sliced_signals, self.params)

class ConstantNode(eqx.Module):
    """The simplest physics model: a trainable constant (or constants)."""
    value: jax.Array

    def __init__(self, init_val: Union[float, int, jax.Array] = 1.0):
        if isinstance(init_val, jax.Array):
            self.value = init_val.astype(jnp.float32)
        else:
            self.value = jnp.array(init_val, dtype=jnp.float32)

    def __call__(self, signals: jax.Array) -> jax.Array:
        return self.value
    
class SignalNode(eqx.Module):
    """A node that extracts specific signal indices from the input."""
    signal_index: int = eqx.field(static=True)

    def __init__(self, signal_index: int):
        try:
            signal_index = int(signal_index)
        except (TypeError, ValueError):
            raise TypeError("signal_index must be an integer. Use SignalNodes() for multiple nodes.")
        self.signal_index = signal_index

    def __call__(self, local_signals: jax.Array) -> jax.Array:
        return local_signals[self.signal_index]

def SignalNodes(*indices: Any) -> Any:
    """
    Convenience factory for generating SignalNodes.
    Accepts flat args: mu1, mu2 = SignalNodes(1, 2)
    Or a PyTree: nodes = SignalNodes({'T': 0, 'mu': [1, 2]})
    """
    if len(indices) == 1 and isinstance(indices[0], (list, dict, tuple)):
        pytree = indices[0]
    else:
        pytree = indices
    return tree_map(lambda i: SignalNode(int(i)), pytree)
