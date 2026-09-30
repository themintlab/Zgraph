import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Union, List, Callable, Dict, Any, Tuple
from jax.tree_util import tree_map
from .base import ZGraphNode, Ensemble


class ConstantNode(ZGraphNode):
    """The simplest physics model: a trainable constant (or constants)."""
    value: jax.Array

    def __init__(self, init_val: Union[float, int, jax.Array] = 1.0):
        if isinstance(init_val, jax.Array):
            self.value = init_val.astype(jnp.float32)
        else:
            self.value = jnp.array(init_val, dtype=jnp.float32)

    def evaluate(self, signals: jax.Array) -> jax.Array:
        return self.value
    
class SignalNode(ZGraphNode):
    """A node that extracts specific signal indices from the input."""
    signal_index: int = eqx.field(static=True)

    def __init__(self, signal_index: int):
        try:
            signal_index = int(signal_index)
        except (TypeError, ValueError):
            raise TypeError("signal_index must be an integer. Use SignalNodes() for multiple nodes.")
        self.signal_index = signal_index

    def evaluate(self, local_signals: jax.Array) -> jax.Array:
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
