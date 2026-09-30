import jax
import jax.numpy as jnp

def custom2(x, beta):
    x_max = jnp.max(x, axis=-1, keepdims=True)
    # Use epsilon instead of exactly 1.0 to get better limit behavior?
    # Actually if beta is exactly 0.0, safe_beta = inf? No.
    # What if we just use safe_beta and multiply by beta?
    safe_beta = jnp.where(beta == 0.0, 1.0, beta)
    z = (x - x_max) / safe_beta
    lse = jnp.log(jnp.sum(jnp.exp(z), axis=-1, keepdims=True))
    # here multiply by beta directly
    res = x_max + beta * lse
    return jnp.squeeze(res, axis=-1)

x = jnp.array([3.0, 3.0, 1.0])

grad_beta_0 = jax.grad(custom2, argnums=1)(x, 0.0)
print("Grad beta at 0:", grad_beta_0)

grad_x_0 = jax.grad(custom2, argnums=0)(x, 0.0)
print("Grad x at 0:", grad_x_0)
