import jax
import jax.numpy as jnp
import equinox as eqx

class Ensemble(eqx.Module):
    """
    A lightweight wrapper that tags a JAX array as an uncertainty distribution (ensemble) 
    rather than a physical vector constant.
    The _execute_vectorized interceptor recursively scans ZGraph PyTrees for this wrapper 
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

class BatchProxy:
    """
    A magical proxy object that intercepts method calls and automatically applies 
    Uncertainty Quantification (UQ) vectorization.
    
    NOTE: This is intentionally designed to future-proof the graph for expansion.
    For example, if an X-Ray Spectrum node implements a secondary `phonon_broadening(E)` 
    method alongside its primary `evaluate(E)` method, users can effortlessly map 
    that secondary method over uncertainty ensembles without writing PyTree masks:
        spectrum.ensemble.phonon_broadening(E)
    """
    def __init__(self, node):
        self._node = node
        
    def __getattr__(self, name):
        def proxy_method(signals, signal_ndim=1, **kwargs):
            return self._node._execute_vectorized(name, signals, signal_ndim=signal_ndim, **kwargs)
        return proxy_method

class ZGraphNode(eqx.Module):
    """
    The universal base class for all nodes in the ZGraph framework.
    """
    
    def _execute_vectorized(self, method_name: str, signals: jax.Array, signal_ndim: int = 1, **kwargs):
        """
        The core interceptor that dynamically applies eqx.filter_vmap and jax.vmap.
        Inside the mapped function, Ensemble wrappers are stripped away, allowing
        the underlying method to remain purely scalar and pristine.
        """
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
            
            unbound_method = getattr(unwrapped_sys, method_name)
            return unbound_method(sig, **kwargs)

        mapped_func = _inner
        
        if is_batched:
            num_batch_dims = signals.ndim - signal_ndim
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

    def __call__(self, signals, **kwargs):
        """
        The primary execution interface. Automatically vectorizes `evaluate`
        over parameter ensembles and signal batches.
        """
        return self._execute_vectorized("evaluate", signals, signal_ndim=1, **kwargs)

    def evaluate(self, signals, **kwargs):
        raise NotImplementedError("ZGraphNodes must implement pure scalar physics in evaluate.")
        
    @property
    def ensemble(self):
        """
        Returns a BatchProxy. Any method called on this proxy will be automatically
        vectorized over the uncertainty ensemble and signal batches.
        Example: system.ensemble.logits(signals)
        """
        return BatchProxy(self)
