import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)

def builtin(x, beta):
    return beta * logsumexp(x / beta, axis=-1)

x = jnp.array([1000.0])
beta = 1e-4

# we need to test up to 3rd derivative
g3 = jax.grad(lambda x: jnp.sum(jax.grad(lambda x: jnp.sum(jax.grad(lambda x: jnp.sum(builtin(x, beta)))(x)))(x)))(x)
print("G3 builtin:", g3)

