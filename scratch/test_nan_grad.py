import jax
import jax.numpy as jnp

def proposed_marginalize(energy_landscape, beta=1.0):
    energy_max = jnp.max(energy_landscape, axis=-1, keepdims=True)
    safe_beta = jnp.where(beta == 0.0, 1e-10, beta)
    centered = (energy_landscape - energy_max) / safe_beta
    sum_exp = jnp.sum(jnp.exp(centered), axis=-1, keepdims=True)
    lse = jnp.log(sum_exp)
    term = jnp.where(lse == 0.0, 0.0, beta * lse)
    res = energy_max + term
    return jnp.squeeze(res, axis=-1)

x = jnp.array([1.0])
# grad w.r.t x at beta=inf
try:
    print("Grad x:", jax.grad(proposed_marginalize, argnums=0)(x, jnp.inf))
except Exception as e:
    print("Grad x error:", e)

# what if beta=1.0 but we hit lse==0 ?
try:
    print("Grad x beta=1.0:", jax.grad(proposed_marginalize, argnums=0)(x, 1.0))
except Exception as e:
    print("Grad x error:", e)

