import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

def builtin(x, beta):
    return beta * logsumexp(x / beta, axis=-1)

x = jnp.array([100.0], dtype=jnp.float32)
beta = jnp.array(1e-4, dtype=jnp.float32)
g3 = jax.grad(lambda x: jnp.sum(jax.grad(lambda x: jnp.sum(jax.grad(lambda x: jnp.sum(builtin(x, beta)))(x)))(x)))(x)
print("G3 builtin x=100 beta=1e-4:", g3)

x2 = jnp.array([10.0], dtype=jnp.float32)
beta2 = jnp.array(1e-6, dtype=jnp.float32)
g3_2 = jax.grad(lambda x: jnp.sum(jax.grad(lambda x: jnp.sum(jax.grad(lambda x: jnp.sum(builtin(x, beta2)))(x)))(x2)))(x2)
print("G3 builtin x=10 beta=1e-6:", g3_2)

