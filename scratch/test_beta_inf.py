import jax
import jax.numpy as jnp

def proposed_marginalize(energy_landscape, beta=1.0):
    energy_max = jnp.max(energy_landscape, axis=-1, keepdims=True)
    safe_beta = jnp.where(beta == 0.0, 1e-10, beta)
    centered = (energy_landscape - energy_max) / safe_beta
    sum_exp = jnp.sum(jnp.exp(centered), axis=-1, keepdims=True)
    res = energy_max + beta * jnp.log(sum_exp)
    return jnp.squeeze(res, axis=-1)

x = jnp.array([1.0, 2.0, 3.0])
print("Beta=inf:", proposed_marginalize(x, jnp.inf))
print("Beta=1000:", proposed_marginalize(x, 1000.0))
