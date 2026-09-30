import jax
import jax.numpy as jnp
import equinox as eqx
import functools

class Ensemble(eqx.Module):
    """
    A lightweight wrapper that tags a JAX array as an uncertainty distribution (ensemble) 
    rather than a physical vector constant.
    The @auto_vectorize decorator recursively scans ZGraph PyTrees for this wrapper 
    and automatically builds the in_axes mask to execute jax.vmap.
    """
    array: jax.Array
    
    @classmethod
    def normal(cls, mean, std, N: int, key: jax.Array):
        import jax.random as jr
        samples = mean + std * jr.normal(key, (N,) + jnp.shape(mean))
        return cls(samples)
    
    @classmethod
    def uniform(cls, low, high, N: int, key: jax.Array):
        import jax.random as jr
        samples = jr.uniform(key, (N,) + jnp.shape(low), minval=low, maxval=high)
        return cls(samples)

class ZGraphNode(eqx.Module):
    """
    The universal base class for all nodes in the ZGraph framework.
    Provides the @auto_vectorize decorator to automatically scale scalar math
    to multidimensional hardware-optimized tensor operations.
    """
    
    @classmethod
    def auto_vectorize(cls, _func=None, *, signal_ndim=1):
        """
        A decorator that wraps any ZGraphNode evaluation method.
        It intercepts the call, checks if the inputs or parameters are batched,
        and dynamically applies eqx.filter_vmap and jax.vmap to the inner function.
        
        Args:
            signal_ndim: The expected number of dimensions for a single input signal.
                         Default is 1 (e.g. a flat vector [T, P]). For predictors that 
                         operate on a mesh grid, this might be 2.
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, signals, **kwargs):
                # Check if signals is batched beyond the expected base dimensionality
                is_batched = getattr(signals, "ndim", 0) > signal_ndim
                
                has_ensemble = False
                def check_ensemble(x):
                    nonlocal has_ensemble
                    if isinstance(x, Ensemble):
                        has_ensemble = True
                    return None
                    
                jax.tree_util.tree_map(check_ensemble, self, is_leaf=lambda x: isinstance(x, Ensemble))
                
                def _inner(sys, sig):
                    def unwrap(x):
                        return x.array if isinstance(x, Ensemble) else x
                    unwrapped_sys = jax.tree_util.tree_map(unwrap, sys, is_leaf=lambda x: isinstance(x, Ensemble))
                    return func(unwrapped_sys, sig, **kwargs)

                mapped_func = _inner
                
                if is_batched:
                    # Determine how many batch dimensions there are
                    num_batch_dims = signals.ndim - signal_ndim
                    # We map over the very first axis repeatedly
                    for _ in range(num_batch_dims):
                        mapped_func = jax.vmap(mapped_func, in_axes=(None, 0))
                    
                if has_ensemble:
                    mask = jax.tree_util.tree_map(
                        lambda x: 0 if isinstance(x, Ensemble) else None, 
                        self, 
                        is_leaf=lambda x: isinstance(x, Ensemble)
                    )
                    mapped_func = eqx.filter_vmap(mapped_func, in_axes=(mask, None))
                    
                return mapped_func(self, signals)
            return wrapper
        
        if _func is None:
            return decorator
        else:
            return decorator(_func)

    def __call__(self, signals, **kwargs):
        """
        By default, evaluating the node directly delegates to the pure physics
        method `_evaluate`, which must be implemented by the subclass and decorated
        with @ZGraphNode.auto_vectorize.
        """
        return self._evaluate(signals, **kwargs)

    def _evaluate(self, signals, **kwargs):
        raise NotImplementedError("ZGraphNodes must implement pure scalar physics in _evaluate.")
