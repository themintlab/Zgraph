import jax
import jax.numpy as jnp

def custom(x, beta):
    x_max = jnp.max(x, axis=-1, keepdims=True)
    safe_beta = jnp.where(beta == 0.0, 1.0, beta)
    z = (x - x_max) / safe_beta
    lse = jnp.log(jnp.sum(jnp.exp(z), axis=-1, keepdims=True))
    res = jnp.where(beta == 0.0, x_max, x_max + safe_beta * lse)
    return jnp.squeeze(res, axis=-1)

x = jnp.array([3.0, 3.0, 1.0])

grad_beta_0 = jax.grad(custom, argnums=1)(x, 0.0)
print("Grad beta at 0:", grad_beta_0)
