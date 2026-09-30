import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def custom2(x, beta):
    x_max = jnp.max(x, axis=-1, keepdims=True)
    safe_beta = jnp.where(beta == 0.0, 1e-10, beta)
    z = (x - x_max) / safe_beta
    lse = jnp.log(jnp.sum(jnp.exp(z), axis=-1, keepdims=True))
    res = x_max + beta * lse
    return jnp.squeeze(res, axis=-1)

x = jnp.array([1.0, 3.0, 3.0])
beta = 1e-4

grad_3 = jax.grad(lambda x: jnp.sum(jax.grad(lambda x: jnp.sum(jax.grad(custom2, argnums=0)(x, beta)))(x)))(x)
print("Grad 3:", grad_3)
