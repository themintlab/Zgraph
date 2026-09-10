import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Union, List, Callable, Dict, Any, Tuple
from jax.tree_util import tree_map

class BaseLeafNode(eqx.Module):
    """
    Abstract base class for all standard leaf nodes in ZGraph.
    Subclasses MUST explicitly define their mathematical __call__() method.
    """
    signal_indices: jax.Array

    def __init__(self, signal_indices: Optional[List[int]] = None):
        indices_to_register = signal_indices if signal_indices is not None else []
        self.signal_indices = jnp.array(indices_to_register, dtype=jnp.int32)

    def __call__(self, local_signals: jax.Array) -> jax.Array:
        raise NotImplementedError("Subclasses must implement __call__()")

class DynamicLeafNode(BaseLeafNode):
    """
    Evaluation-only node that accepts arbitrary pure JAX functions.
    Ideal for rapid prototyping. Should NOT be used for performance-critical training.
    """
    energy_function: Callable = eqx.field(static=True)
    constants: Dict[str, jax.Array]

    def __init__(self, energy_function: Callable[..., jax.Array], signal_indices: Optional[List[int]] = None, **constants: Any):
        """
        Args:
            energy_function (callable): The pure math equation.
            signal_indices (list[int], optional): Hardcoded indices for early testing.
            **constants: Constant parameters passed to the function.
        """
        super().__init__(signal_indices)
        self.energy_function = energy_function
        
        self.constants = {}
        for key, val in constants.items():
            if not isinstance(val, jax.Array):
                val = jnp.array(val, dtype=jnp.float32)
            self.constants[key] = val

    def __call__(self, full_local_signals: jax.Array) -> jax.Array:
        # Strictly vector input: (Channels,) -> scalar output: ()
        sliced_signals = full_local_signals[self.signal_indices]
        return self.energy_function(sliced_signals, **self.constants)

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
