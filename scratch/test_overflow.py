import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

x = jnp.array([100.0, 200.0], dtype=jnp.float32)
beta = jnp.array(1e-35, dtype=jnp.float32)

def builtin(x, beta):
    return beta * logsumexp(x / beta)

def custom(x, beta):
    x_max = jnp.max(x)
    safe_beta = jnp.where(beta == 0.0, 1.0, beta)
    z = (x - x_max) / safe_beta
    lse = jnp.log(jnp.sum(jnp.exp(z)))
    return jnp.where(beta == 0.0, x_max, x_max + safe_beta * lse)

print("Builtin:", builtin(x, beta))
print("Custom:", custom(x, beta))
